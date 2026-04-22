"""Retention-guaranteed context compression.

Design mirrors LLMLingua-2: score each segment's informativeness, then solve a
budget-constrained knapsack that keeps the highest-signal segments while
preserving original order.

Two quality defenses, layered:

1. **Retention whitelist** (:mod:`vecr_compress.retention`) — segments
   containing numbers, IDs, URLs, citations, or code spans are *pinned*: kept
   regardless of budget. Silent data loss is the fastest way to lose trust.
2. **Filler hard-drop + heuristic informativeness scoring** — greetings,
   AI-scaffolding, one-word acknowledgements score 0 and are removed before
   the knapsack runs; remaining segments are ranked by a pure-Python entropy
   + structural-signal score.

Optional question-aware scoring: pass ``use_question_relevance=True`` to
:func:`compress` to blend :func:`vecr_compress.scorer.question_relevance`
into the heuristic (0.6 heuristic + 0.4 Jaccard). Off by default — the
synthetic structured-needle bench shows no uplift, but the HotpotQA probe
(``bench/hotpotqa_results.md``) shows ~+10pp supporting-fact survival on
real multi-hop NL-QA. Callers can also supply a fully custom ``scorer``.

Public entrypoint: :func:`compress`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .retention import DEFAULT_RULES, RetentionRules
from .scorer import ScorerFn, blended_score, heuristic_score
from .tokens import count as tcount

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")

Message = dict[str, Any]
"""A chat message — OpenAI or Anthropic shape.

Minimally: ``{"role": "user", "content": "..."}``. ``content`` may be a string
or a list of content blocks (Anthropic-style). Other keys are preserved
verbatim through the compression pipeline.
"""


# -----------------------------------------------------------------------------
# Public result / input shapes
# -----------------------------------------------------------------------------


@dataclass
class CompressResult:
    """The output of :func:`compress`.

    Attributes:
        messages: The compressed messages, in the same role shape as the
            input. Messages that were dropped entirely are omitted.
        original_tokens: Total tokens in the input messages (tiktoken
            o200k_base estimate).
        compressed_tokens: Total tokens in the compressed messages.
        ratio: ``compressed_tokens / original_tokens`` (1.0 if original was
            empty or compression was skipped).
        dropped_segments: One dict per segment the compressor dropped. Each
            dict contains ``{"message_index", "order", "text", "tokens",
            "reason"}``. Useful for debugging why the compressor was
            over-aggressive.
        retained_matches: One dict per segment pinned by a retention rule:
            ``{"message_index", "order", "text", "rule"}``. Tells you which
            facts the compressor promised to keep.
        skipped: True iff compression was bypassed entirely (empty input,
            budget ≥ original, or all messages contained structured blocks).
        kept_message_indices: Indices (into the original ``messages`` list) of
            every message that appears in the compressed output. In the same
            order as ``messages``. When ``skipped`` is True this is
            ``list(range(len(input_messages)))``. This is the authoritative
            source for reconstructing which input messages survived; adapters
            should use it rather than heuristic telemetry reconstruction.
    """

    messages: list[Message]
    original_tokens: int
    compressed_tokens: int
    ratio: float
    dropped_segments: list[dict[str, Any]] = field(default_factory=list)
    retained_matches: list[dict[str, Any]] = field(default_factory=list)
    skipped: bool = False
    kept_message_indices: list[int] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Internal segment representation
# -----------------------------------------------------------------------------


@dataclass
class _Segment:
    message_index: int
    order: int
    text: str
    tokens: int
    protected: bool = False
    pinned: bool = False
    pin_reason: str | None = None


# -----------------------------------------------------------------------------
# Structured-content detection
# -----------------------------------------------------------------------------

# Default block types; can be overridden per-call via
# `compress(structured_block_types=...)`.
_STRUCTURED_BLOCK_TYPES = frozenset(
    {"tool_use", "tool_result", "image", "input_image", "document"}
)
_TEXT_BLOCK_TYPES = frozenset({"text", "input_text", "output_text"})


def _is_structured_content(
    content: Any, types: frozenset[str] | None = None
) -> bool:
    """True when a message's content is **entirely** structured — i.e. has no
    compressible text at all.

    A mixed list (text + image, text + tool_use) returns False so that its
    text portion participates in compression while non-text blocks are
    preserved by :func:`_reassemble`. A pure-structured list (only tool_use
    / tool_result / image / document blocks) still returns True so the
    message short-circuits the pipeline.

    ``types`` optionally overrides the set of block ``type`` values treated as
    non-textual structured blocks. Defaults to :data:`_STRUCTURED_BLOCK_TYPES`
    when ``None``, preserving existing behaviour.

    Used by :func:`_extract_question` to skip tool-call user turns and by the
    ``all(skip_mask)`` early-return in :func:`compress`.
    """
    structured_types = types if types is not None else _STRUCTURED_BLOCK_TYPES
    if not isinstance(content, list):
        return False
    has_any_text = False
    has_structured = False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in _TEXT_BLOCK_TYPES:
            text = block.get("text")
            if isinstance(text, str) and text:
                has_any_text = True
        elif btype in structured_types:
            has_structured = True
    return has_structured and not has_any_text


def _split_content_blocks(content: Any) -> tuple[str, list[dict]]:
    """Split a message's content into (compressible_text, preserved_blocks).

    - str content: returns ``(content, [])``.
    - list content: text-type blocks are concatenated into the first element;
      every non-text block is preserved verbatim (in original order) in the
      second element.
    - unexpected types: logs a WARNING and coerces via ``str()``, returning
      ``(str(content), [])``.

    Preserved blocks are re-inserted by :func:`_reassemble` so that, for
    example, an image attached to a text turn survives compression alongside
    the (compressed) prose.
    """
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        text_parts: list[str] = []
        preserved: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in _TEXT_BLOCK_TYPES:
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            else:
                preserved.append(block)
        return "".join(text_parts), preserved
    if content is not None:
        logger.warning(
            "vecr_compress: unexpected content type %s; coercing to str. "
            "Pass a string or list of content blocks.",
            type(content).__name__,
        )
        return str(content), []
    return "", []


def _content_to_text(content: Any) -> str:
    """Best-effort extraction of the plain text portion of a message content.

    Used only for messages we've already confirmed are text-only. Joins
    Anthropic-style ``[{"type":"text","text":"..."}, ...]`` into a single
    string.

    For unexpected content types (not str, not list), logs a WARNING and
    coerces via ``str()`` rather than silently returning empty string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in {"text", "input_text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is not None:
        logger.warning(
            "vecr_compress: unexpected content type %s; coercing to str. "
            "Pass a string or list of content blocks.",
            type(content).__name__,
        )
        return str(content)
    return ""


def _is_content_list(content: Any) -> bool:
    return isinstance(content, list)


# -----------------------------------------------------------------------------
# Segmenting / reassembly
# -----------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text) if s.strip()]


def _extract_question(
    messages: list[Message], types: frozenset[str] | None = None
) -> str | None:
    """Pick the final non-empty user-role message as the reference question.

    Used only when a custom scorer reads the ``question`` argument. The
    default :func:`vecr_compress.scorer.heuristic_score` ignores it.

    ``types`` is forwarded to :func:`_is_structured_content` so callers can
    control which block types are considered non-textual.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if _is_structured_content(content, types):
            continue
        text = _content_to_text(content)
        if text.strip():
            return text
    return None


def _segment_messages(
    messages: list[Message],
    protect_tail: int,
    protect_system: bool,
    rules: RetentionRules,
    skip_mask: list[bool],
) -> list[_Segment]:
    """Produce a flat list of scoring units, one per sentence.

    ``skip_mask[i]`` = True means message ``i`` is structured / should not be
    split; it won't contribute any segments (compressor leaves it verbatim).
    """
    segs: list[_Segment] = []
    protected_start = max(0, len(messages) - protect_tail)
    for mi, msg in enumerate(messages):
        if skip_mask[mi]:
            continue
        role = msg.get("role")
        text, _preserved = _split_content_blocks(msg.get("content"))
        if not text.strip():
            continue
        protected = mi >= protected_start or (protect_system and role == "system")
        parts = _split_sentences(text)
        if not parts:
            parts = [text]
        for oi, part in enumerate(parts):
            reason = rules.reason(part)
            segs.append(
                _Segment(
                    message_index=mi,
                    order=oi,
                    text=part,
                    tokens=tcount(part),
                    protected=protected,
                    pinned=reason is not None,
                    pin_reason=reason,
                )
            )
    return segs


def _resolve_budget(original: int, budget_tokens: int | None, target_ratio: float | None) -> int:
    candidates: list[int] = []
    if budget_tokens is not None:
        candidates.append(budget_tokens)
    if target_ratio is not None:
        candidates.append(int(original * target_ratio))
    if not candidates:
        return original
    raw = min(candidates)
    # Floor at 32 tokens — a smaller budget is almost always user error.
    floored = max(32, raw)
    if raw < 32:
        logger.warning(
            "vecr_compress: budget_tokens=%d rewritten to %d minimum; "
            "likely caller error — did you mean a larger value?",
            raw,
            floored,
        )
    return min(original, floored)


def _budget_prune(
    segments: list[_Segment],
    budget: int,
    question: str | None,
    scorer: ScorerFn,
) -> tuple[list[_Segment], list[_Segment]]:
    """Return (kept, dropped) segment lists, both in original order."""
    must_keep = [s for s in segments if s.protected or s.pinned]
    prunable = [s for s in segments if not (s.protected or s.pinned)]

    reserved = sum(s.tokens for s in must_keep)
    if reserved > budget:
        logger.warning(
            "vecr_compress: must-keep content (%d tokens) exceeds budget (%d). "
            "Returning must-keep only; consider raising budget_tokens.",
            reserved,
            budget,
        )
        kept = sorted(must_keep, key=lambda s: (s.message_index, s.order))
        dropped = sorted(prunable, key=lambda s: (s.message_index, s.order))
        return kept, dropped

    remaining = budget - reserved
    scored = [(scorer(s.text, question), s) for s in prunable]
    scored = [(sc, s) for sc, s in scored if sc > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    chosen_ids: set[int] = set()
    chosen: list[_Segment] = []
    used = 0
    for _, seg in scored:
        if used + seg.tokens > remaining:
            continue
        chosen.append(seg)
        chosen_ids.add(id(seg))
        used += seg.tokens

    kept = must_keep + chosen
    kept.sort(key=lambda s: (s.message_index, s.order))

    dropped = [s for s in prunable if id(s) not in chosen_ids]
    dropped.sort(key=lambda s: (s.message_index, s.order))
    return kept, dropped


def _reassemble(
    original: list[Message],
    kept: list[_Segment],
    skip_mask: list[bool],
    preserved_blocks_by_index: dict[int, list[dict]] | None = None,
) -> tuple[list[Message], list[int]]:
    """Re-build the message list in original order, with kept sentences joined
    by single spaces. Structured (skip_mask) messages pass through untouched.

    For messages whose original content was a list of blocks, any non-text
    blocks (images, tool_use, tool_result, documents) are preserved verbatim
    from ``preserved_blocks_by_index``: the composed content is
    ``preserved_blocks + [{"type":"text","text": joined}]`` — non-text blocks
    first (they tend to be context resources — images, tool outputs), then the
    compressed prose. A message that had non-text blocks but whose text was
    fully budget-pruned still emits the preserved blocks so that images and
    tool calls are never silently dropped.

    Returns:
        (messages, kept_indices) — the compressed message list and the list of
        original indices that appear in the output, in the same order.
    """
    preserved_blocks_by_index = preserved_blocks_by_index or {}
    buckets: dict[int, list[_Segment]] = {}
    for seg in kept:
        buckets.setdefault(seg.message_index, []).append(seg)

    result: list[Message] = []
    kept_indices: list[int] = []
    for mi, msg in enumerate(original):
        if skip_mask[mi]:
            result.append(msg)
            kept_indices.append(mi)
            continue
        preserved = preserved_blocks_by_index.get(mi, [])
        parts = sorted(buckets.get(mi, []), key=lambda s: s.order)
        joined = " ".join(p.text.strip() for p in parts).strip()

        if not joined and not preserved:
            if mi not in buckets:
                logger.debug(
                    "vecr_compress: message[%d] absent from output and not in skip_mask; "
                    "all its segments were budget-pruned.",
                    mi,
                )
            continue

        new_msg: Message = {k: v for k, v in msg.items() if k != "content"}
        new_msg["role"] = msg.get("role")
        # Preserve caller's content shape. If the original was a list OR we
        # have preserved non-text blocks, emit a block list; otherwise a str.
        original_content = msg.get("content")
        if _is_content_list(original_content) or preserved:
            blocks: list[dict] = list(preserved)
            if joined:
                blocks.append({"type": "text", "text": joined})
            new_msg["content"] = blocks
        else:
            new_msg["content"] = joined
        result.append(new_msg)
        kept_indices.append(mi)
    return result, kept_indices


def _sum_tokens(messages: list[Message]) -> int:
    """Sum tiktoken estimates across every message, block-by-block.

    - String content: one ``tcount(content)``.
    - List content: per-block — text-type blocks use ``tcount(text)``;
      ``tool_use`` / ``tool_result`` blocks serialise their ``input`` /
      ``content`` payload via ``json.dumps(..., default=str)`` and count that
      string with ``tcount`` (same tokenizer as the compressible prose, so
      ``CompressResult.ratio`` stays honest for tool-heavy turns). Other
      block types (image, document, ...) contribute 0 text tokens.
    - Unexpected types fall back to ``_content_to_text`` + ``tcount``.
    """
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += tcount(content)
            continue
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype in _TEXT_BLOCK_TYPES:
                    text = block.get("text")
                    if isinstance(text, str):
                        total += tcount(text)
                elif btype in {"tool_use", "tool_result"}:
                    inp = block.get("input") or block.get("content")
                    if inp is not None:
                        try:
                            inp_str = json.dumps(inp, default=str)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "vecr_compress: json.dumps failed on %s block "
                                "(%s); falling back to str() — token count may undercount",
                                btype,
                                exc,
                            )
                            inp_str = str(inp)
                        total += tcount(inp_str)
            continue
        # Unexpected type — coerce via _content_to_text (which warns) and count.
        total += tcount(_content_to_text(content))
    return total


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------


def compress(
    messages: list[Message],
    budget_tokens: int | None = None,
    *,
    question: str | None = None,
    target_ratio: float | None = None,
    retention_rules: RetentionRules | None = None,
    scorer: ScorerFn | None = None,
    use_question_relevance: bool = False,
    protect_tail: int = 2,
    protect_system: bool = True,
    retain: bool = True,
    structured_block_types: frozenset[str] | None = None,
) -> CompressResult:
    """Compress ``messages`` to fit within ``budget_tokens``.

    Supports both OpenAI and Anthropic message role shapes. Messages containing
    structured content blocks (``tool_use``, ``tool_result``, ``image``, etc.)
    are passed through verbatim — the compressor never rewrites them.

    Args:
        messages: Chat messages. Each is a dict with at least ``role`` and
            ``content``. Content may be a string or a list of blocks.
        budget_tokens: Hard token budget for the compressed output. If omitted,
            ``target_ratio`` is required.
        question: Optional question string. Used when
            ``use_question_relevance=True`` or by any custom ``scorer`` that
            reads it. The default :func:`heuristic_score` ignores it. When
            omitted, :func:`compress` auto-extracts the last user-role
            message.
        target_ratio: Alternative to ``budget_tokens`` — compress to this
            fraction of the original token count (e.g. ``0.3``). If both are
            provided, the tighter of the two wins.
        retention_rules: Custom retention rule set. Defaults to
            :data:`vecr_compress.DEFAULT_RULES`.
        scorer: Custom scorer callable ``(text, question) -> float``. Defaults
            to :func:`vecr_compress.scorer.heuristic_score`. If both
            ``scorer`` and ``use_question_relevance=True`` are supplied, the
            explicit ``scorer`` wins and ``use_question_relevance`` is ignored.
        use_question_relevance: Opt-in question-aware scoring. When True and
            no custom ``scorer`` is supplied, uses
            :func:`vecr_compress.scorer.blended_score` (0.6 heuristic + 0.4
            Jaccard). Off by default; worth enabling for NL-QA workloads
            where the question is informative (see
            ``bench/hotpotqa_results.md`` for uplift numbers). Synthetic
            structured-needle workloads see no uplift.
        protect_tail: Never prune the last N messages (defaults to 2 — user
            turn + its immediate context).
        protect_system: If True, system messages are never pruned.
        retain: If False, turn off the retention whitelist entirely. Almost
            always a mistake outside of benchmarks.
        structured_block_types: Optional override of the block ``type`` values
            treated as non-textual and passed through verbatim. Defaults to
            ``{"tool_use", "tool_result", "image", "input_image", "document"}``.
            Provide a superset or custom set to support new content-block
            protocols (e.g. Gemini ``parts``, OpenAI ``output_audio``) without
            modifying the library.

    Returns:
        A :class:`CompressResult` with the compressed messages, token counts,
        and telemetry (dropped segments, retention matches).
    """
    if not isinstance(messages, list):
        raise TypeError(
            f"messages must be a list, got {type(messages).__name__}"
        )
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise TypeError(
                f"messages[{i}] must be a dict, got {type(m).__name__}"
            )

    rules = retention_rules if retention_rules is not None else DEFAULT_RULES
    if scorer is not None:
        score_fn: ScorerFn = scorer
    elif use_question_relevance:
        score_fn = blended_score
    else:
        score_fn = heuristic_score

    if not messages:
        return CompressResult(
            messages=[],
            original_tokens=0,
            compressed_tokens=0,
            ratio=1.0,
            skipped=True,
            kept_message_indices=[],
        )

    structured_types = (
        structured_block_types
        if structured_block_types is not None
        else _STRUCTURED_BLOCK_TYPES
    )
    skip_mask = [
        _is_structured_content(m.get("content"), structured_types) for m in messages
    ]

    # Extract non-text blocks once so _reassemble can re-insert them verbatim
    # even when the message's text was compressed or pruned.
    preserved_blocks_by_index: dict[int, list[dict]] = {}
    for mi, m in enumerate(messages):
        if skip_mask[mi]:
            continue
        _text, preserved = _split_content_blocks(m.get("content"))
        if preserved:
            preserved_blocks_by_index[mi] = preserved

    original_tokens = _sum_tokens(messages)

    # Pick question automatically if not supplied.
    q = (
        question
        if question is not None
        else _extract_question(messages, structured_types)
    )

    # If the request is trivial, bail out as pass-through.
    if budget_tokens is None and target_ratio is None:
        return CompressResult(
            messages=list(messages),
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            ratio=1.0,
            skipped=True,
            kept_message_indices=list(range(len(messages))),
        )

    # If every message is structured, there's nothing to compress.
    if all(skip_mask):
        return CompressResult(
            messages=list(messages),
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            ratio=1.0,
            skipped=True,
            kept_message_indices=list(range(len(messages))),
        )

    segments = _segment_messages(
        messages,
        protect_tail=protect_tail,
        protect_system=protect_system,
        rules=rules if retain else RetentionRules(()),
        skip_mask=skip_mask,
    )

    seg_total = sum(s.tokens for s in segments)

    # Compute budget against segment total (structured messages add their own
    # cost but we don't reduce them).
    budget = _resolve_budget(seg_total, budget_tokens, target_ratio)
    # If the budget already accommodates the original, return as-is.
    if budget >= seg_total:
        return CompressResult(
            messages=list(messages),
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            ratio=1.0,
            skipped=True,
            kept_message_indices=list(range(len(messages))),
        )

    kept, dropped = _budget_prune(segments, budget, q, score_fn)
    compressed_messages, kept_message_indices = _reassemble(
        messages, kept, skip_mask, preserved_blocks_by_index
    )
    compressed_tokens = _sum_tokens(compressed_messages)

    dropped_info = [
        {
            "message_index": s.message_index,
            "order": s.order,
            "text": s.text,
            "tokens": s.tokens,
            "reason": "budget",
        }
        for s in dropped
    ]
    retained_matches = [
        {
            "message_index": s.message_index,
            "order": s.order,
            "text": s.text,
            "rule": s.pin_reason,
        }
        for s in kept
        if s.pinned and not s.protected
    ]

    ratio = compressed_tokens / original_tokens if original_tokens else 1.0

    return CompressResult(
        messages=compressed_messages,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        ratio=ratio,
        dropped_segments=dropped_info,
        retained_matches=retained_matches,
        skipped=False,
        kept_message_indices=kept_message_indices,
    )
