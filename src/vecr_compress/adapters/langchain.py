"""LangChain adapter.

Usage:

    from vecr_compress.adapters.langchain import VecrContextCompressor
    from langchain_core.messages import HumanMessage, SystemMessage

    compressor = VecrContextCompressor(budget_tokens=2000)
    compressed = compressor.compress_messages([
        SystemMessage(content="You are a refund analyst."),
        HumanMessage(content="Long history..."),
    ])

Installing: ``pip install vecr-compress[langchain]``.
"""

from __future__ import annotations

from typing import Any

from ..compressor import CompressResult, compress
from ..retention import RetentionRules
from ..scorer import ScorerFn


_INSTALL_HINT = (
    "langchain-core is required for the LangChain adapter. "
    "Install with: pip install vecr-compress[langchain]"
)


def _require_langchain():
    try:
        from langchain_core.messages import (  # type: ignore[import-not-found]
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
    except ImportError as exc:  # pragma: no cover - covered by test_adapters
        raise ImportError(_INSTALL_HINT) from exc
    return AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


_ROLE_MAP_IN = {
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "SystemMessage": "system",
    "ToolMessage": "tool",
    "ChatMessage": "user",
}


def _lc_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a LangChain BaseMessage to an OpenAI/Anthropic-style dict.

    For AIMessage with tool_calls, converts to Anthropic-style content blocks
    so the compressor's skip_mask sees ``tool_use`` and passes through untouched.
    """
    role = _ROLE_MAP_IN.get(type(msg).__name__, "user")
    content = msg.content

    # Preserve tool_calls on AIMessage as Anthropic-style content blocks.
    tool_calls = getattr(msg, "tool_calls", None)
    if role == "assistant" and tool_calls:
        blocks: list[dict[str, Any]] = []
        if content:
            blocks.append({"type": "text", "text": content})
        for tc in tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "input": tc.get("args", {}),
                }
            )
        return {"role": role, "content": blocks}

    return {"role": role, "content": content}


def _dict_to_lc(msg: dict[str, Any]):
    AIMessage, _Base, HumanMessage, SystemMessage, ToolMessage = _require_langchain()
    role = msg.get("role")
    content = msg.get("content", "")

    # Detect Anthropic-style content blocks with tool_use — reconstruct
    # AIMessage(content=text, tool_calls=[...]).
    if role == "assistant" and isinstance(content, list):
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                if isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                        "type": "tool_call",
                    }
                )
        if tool_calls:
            return AIMessage(content="".join(text_parts), tool_calls=tool_calls)
        # No tool_use blocks — fall through to plain text.
        content = "".join(text_parts)

    # LangChain can't serialize structured-content lists cleanly across roles;
    # fall back to string for text messages.
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        content = "".join(parts)

    if role == "assistant":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        tool_call_id = msg.get("tool_call_id", "")
        return ToolMessage(content=content, tool_call_id=tool_call_id)
    return HumanMessage(content=content)


class VecrContextCompressor:
    """Compress a list of LangChain ``BaseMessage`` objects.

    This is not a LangChain ``BaseDocumentCompressor`` / ``Runnable`` (those
    integrate against retrievers or chains). It is a plain utility for
    compressing the chat history before you send it to an LLM.
    """

    def __init__(
        self,
        budget_tokens: int | None = None,
        *,
        target_ratio: float | None = None,
        question: str | None = None,
        retention_rules: RetentionRules | None = None,
        scorer: ScorerFn | None = None,
        protect_tail: int = 2,
        protect_system: bool = True,
        retain: bool = True,
    ):
        _require_langchain()  # fail fast with a friendly message
        self.budget_tokens = budget_tokens
        self.target_ratio = target_ratio
        self.question = question
        self.retention_rules = retention_rules
        self.scorer = scorer
        self.protect_tail = protect_tail
        self.protect_system = protect_system
        self.retain = retain

    def compress_messages(self, messages: list[Any]) -> list[Any]:
        """Return compressed LangChain messages. See :meth:`compress_with_report`
        if you need the full :class:`CompressResult`.
        """
        result = self.compress_with_report(messages)
        return [_dict_to_lc(m) for m in result.messages]

    def compress_with_report(self, messages: list[Any]) -> CompressResult:
        """Return the raw :class:`CompressResult`. Use this when you want
        access to ``original_tokens``, ``ratio``, ``retained_matches``, etc.
        """
        as_dicts = [_lc_to_dict(m) for m in messages]
        return compress(
            as_dicts,
            budget_tokens=self.budget_tokens,
            question=self.question,
            target_ratio=self.target_ratio,
            retention_rules=self.retention_rules,
            scorer=self.scorer,
            protect_tail=self.protect_tail,
            protect_system=self.protect_system,
            retain=self.retain,
        )


__all__ = ["VecrContextCompressor"]
