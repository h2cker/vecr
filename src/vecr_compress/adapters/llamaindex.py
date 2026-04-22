"""LlamaIndex adapter.

Usage:

    from vecr_compress.adapters.llamaindex import VecrNodePostprocessor
    from llama_index.core.schema import NodeWithScore, TextNode

    processor = VecrNodePostprocessor(budget_tokens=2000)
    kept = processor.postprocess_nodes(nodes, query_str="...")

Installing: ``pip install vecr-compress[llamaindex]``.
"""

from __future__ import annotations

from typing import Any

from ..compressor import CompressResult, compress
from ..retention import RetentionRules
from ..scorer import ScorerFn


_INSTALL_HINT = (
    "llama-index-core is required for the LlamaIndex adapter. "
    "Install with: pip install vecr-compress[llamaindex]"
)


def _require_llamaindex():
    try:
        from llama_index.core.schema import NodeWithScore, TextNode  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover - covered by test_adapters
        raise ImportError(_INSTALL_HINT) from exc


class VecrNodePostprocessor:
    """Compress LlamaIndex retrieved nodes to fit a token budget.

    Each node's text is treated as a user-role message; the compressor runs
    retention-aware pruning and returns only the nodes whose text survived in
    full. Nodes whose text was entirely dropped are removed; nodes partially
    dropped have their ``.node.text`` replaced with the compressed text.

    Query-aware mode: pass ``query_str`` (or ``query_bundle.query_str``) to
    :meth:`postprocess_nodes` — it's forwarded to the compressor's
    question-aware scorer.

    Note: this is a thin utility, not a subclass of
    ``BaseNodePostprocessor`` — that class depends on pydantic v1 internals
    that we don't want to pin. It quacks the same way for practical use.
    """

    def __init__(
        self,
        budget_tokens: int | None = None,
        *,
        target_ratio: float | None = None,
        retention_rules: RetentionRules | None = None,
        scorer: ScorerFn | None = None,
        retain: bool = True,
    ):
        _require_llamaindex()
        self.budget_tokens = budget_tokens
        self.target_ratio = target_ratio
        self.retention_rules = retention_rules
        self.scorer = scorer
        self.retain = retain

    def postprocess_nodes(
        self,
        nodes: list[Any],
        query_bundle: Any | None = None,
        query_str: str | None = None,
    ) -> list[Any]:
        """Return the pruned / rewritten node list."""
        if not nodes:
            return nodes

        question = query_str
        if question is None and query_bundle is not None:
            # Match LlamaIndex's BaseNodePostprocessor signature loosely.
            question = getattr(query_bundle, "query_str", None)

        messages = [
            {"role": "user", "content": _node_text(n)} for n in nodes
        ]
        result = compress(
            messages,
            budget_tokens=self.budget_tokens,
            target_ratio=self.target_ratio,
            question=question,
            retention_rules=self.retention_rules,
            scorer=self.scorer,
            protect_tail=0,
            protect_system=False,
            retain=self.retain,
        )

        # Re-derive which original indices survived by text equality.
        # Compressor preserves order, so the i-th surviving message maps to
        # the i-th surviving input index — but we need to be more precise
        # because the compressor may have dropped whole messages.
        # Use message_index from dropped_segments + retained_matches to
        # reconstruct.
        dropped_by_idx: dict[int, list[str]] = {}
        for seg in result.dropped_segments:
            dropped_by_idx.setdefault(seg["message_index"], []).append(seg["text"])

        surviving_indices = _surviving_indices(
            original_count=len(messages),
            result=result,
        )

        kept_nodes: list[Any] = []
        for original_idx in surviving_indices:
            node = nodes[original_idx]
            # Find this node's compressed text.
            position = surviving_indices.index(original_idx)
            if position < len(result.messages):
                new_text = _message_text(result.messages[position])
                _set_node_text(node, new_text)
            kept_nodes.append(node)
        return kept_nodes

    def compress_with_report(
        self, nodes: list[Any], query_str: str | None = None
    ) -> CompressResult:
        """Compress nodes and return the raw :class:`CompressResult` without
        mutating them. Use this when you need telemetry
        (``original_tokens``, ``retained_matches``).
        """
        messages = [{"role": "user", "content": _node_text(n)} for n in nodes]
        return compress(
            messages,
            budget_tokens=self.budget_tokens,
            target_ratio=self.target_ratio,
            question=query_str,
            retention_rules=self.retention_rules,
            scorer=self.scorer,
            protect_tail=0,
            protect_system=False,
            retain=self.retain,
        )


def _node_text(node: Any) -> str:
    """Extract text from a LlamaIndex node or NodeWithScore."""
    inner = getattr(node, "node", None)
    target = inner if inner is not None else node
    text_fn = getattr(target, "get_content", None)
    if callable(text_fn):
        try:
            return text_fn()
        except TypeError:
            pass
    text_attr = getattr(target, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(target)


def _set_node_text(node: Any, text: str) -> None:
    inner = getattr(node, "node", None)
    target = inner if inner is not None else node
    if hasattr(target, "text"):
        try:
            target.text = text
        except AttributeError:
            pass


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return str(content)


def _surviving_indices(original_count: int, result: CompressResult) -> list[int]:
    """Given a CompressResult, return which original message indices survived.

    Uses ``result.kept_message_indices`` — the authoritative list populated by
    ``_reassemble`` in the compressor. This replaces the previous heuristic
    that reconstructed survival from dropped_segments / retained_matches
    telemetry and was demonstrably incorrect in mixed-drop scenarios.
    """
    return list(result.kept_message_indices)


__all__ = ["VecrNodePostprocessor"]
