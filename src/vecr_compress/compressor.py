"""Retention-guaranteed context compression.

Design mirrors LLMLingua-2: score each segment's informativeness, then solve a
budget-constrained knapsack that keeps the highest-signal segments while
preserving original order.

Three quality defenses, layered:

1. **Retention whitelist** (:mod:`vecr_compress.retention`) — segments
   containing numbers, IDs, URLs, citations, or code spans are *pinned*: kept
   regardless of budget. Silent data loss is the fastest way to lose trust.
2. **Question-aware scoring** — when the caller provides a question
   (typically the last user message), each segment is boosted by its lexical
   overlap with the question. Same budget, higher relevance retention.
3. **Filler hard-drop** — greetings, AI-scaffolding, one-word acknowledgements
   score 0 and are removed before the knapsack runs.

Public entrypoint: :func:`compress`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .retention import DEFAULT_RULES, RetentionRules
from .scorer import ScorerFn, heuristic_score
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

_STRUCTURED_BLOCK_TYPES = frozenset(
    {"tool_use", "tool_result", "image", "input_image", "document"}
)


def _is_structured_content(content: Any) -> bool:
    """True when a message's content includes non-text blocks we must not
    compress (tool calls, tool results, images, etc).

    Safety mode: if any block looks structured, the whole message is passed
    through verbatim.
    """
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in _STRUCTURED_BLOCK_TYPES:
            return True
    return False


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


def _extract_question(messages: list[Message]) -> str | None:
    """Pick the final non-empty user-role message as the reference question."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if _is_structured_content(content):
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
        text = _content_to_text(msg.get("content"))
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
) -> tuple[list[Message], list[int]]:
    """Re-build the message list in original order, with kept sentences joined
    by single spaces. Structured (skip_mask) messages pass through untouched.

    Returns:
        (messages, kept_indices) — the compressed message list and the list of
        original indices that appear in the output, in the same order.
    """
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
        if mi not in buckets:
            logger.debug(
                "vecr_compress: message[%d] absent from output and not in skip_mask; "
                "all its segments were budget-pruned.",
                mi,
            )
            continue
        parts = sorted(buckets[mi], key=lambda s: s.order)
        joined = " ".join(p.text.strip() for p in parts).strip()
        if not joined:
            continue
        new_msg: Message = {k: v for k, v in msg.items() if k != "content"}
        new_msg["role"] = msg.get("role")
        # Preserve whatever shape the caller used for content: if they passed
        # a list-of-blocks, give back a single text block; otherwise a string.
        if _is_content_list(msg.get("content")):
            new_msg["content"] = [{"type": "text", "text": joined}]
        else:
            new_msg["content"] = joined
        result.append(new_msg)
        kept_indices.append(mi)
    return result, kept_indices


def _sum_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if _is_structured_content(content):
            # Estimate tokens for structured content blocks. Sum embedded text
            # for text-type blocks; for tool_use/tool_result, approximate via
            # str(input) length ÷ 4 (standard char-per-token heuristic).
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if isinstance(block.get("text"), str):
                    total += tcount(block["text"])
                if btype in {"tool_use", "tool_result"}:
                    inp = block.get("input") or block.get("content")
                    if inp is not None:
                        import json
                        try:
                            inp_str = json.dumps(inp, default=str)
                        except Exception:  # noqa: BLE001
                            inp_str = str(inp)
                        # Approximate: 4 chars ≈ 1 token for structured data.
                        total += max(1, len(inp_str) // 4)
        else:
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
    protect_tail: int = 2,
    protect_system: bool = True,
    retain: bool = True,
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
        question: Optional user question used for question-aware Jaccard
            boosting. If None, :func:`compress` picks the last user-role
            message automatically.
        target_ratio: Alternative to ``budget_tokens`` — compress to this
            fraction of the original token count (e.g. ``0.3``). If both are
            provided, the tighter of the two wins.
        retention_rules: Custom retention rule set. Defaults to
            :data:`vecr_compress.DEFAULT_RULES`.
        scorer: Custom scorer callable ``(text, question) -> float``. Defaults
            to :func:`vecr_compress.scorer.heuristic_score`.
        protect_tail: Never prune the last N messages (defaults to 2 — user
            turn + its immediate context).
        protect_system: If True, system messages are never pruned.
        retain: If False, turn off the retention whitelist entirely. Almost
            always a mistake outside of benchmarks.

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
    score_fn: ScorerFn = scorer or heuristic_score

    if not messages:
        return CompressResult(
            messages=[],
            original_tokens=0,
            compressed_tokens=0,
            ratio=1.0,
            skipped=True,
            kept_message_indices=[],
        )

    skip_mask = [_is_structured_content(m.get("content")) for m in messages]

    original_tokens = _sum_tokens(messages)

    # Pick question automatically if not supplied.
    q = question if question is not None else _extract_question(messages)

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
    compressed_messages, kept_message_indices = _reassemble(messages, kept, skip_mask)
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
