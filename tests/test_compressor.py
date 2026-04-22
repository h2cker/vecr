"""Compressor tests — adapted from vecr/tests/test_compression.py."""

from __future__ import annotations

import logging

import pytest

from vecr_compress import compress


def test_budget_drops_fillers_first():
    messages = [
        {"role": "system", "content": "You are a helpful senior engineer."},
        {
            "role": "user",
            "content": (
                "Hi! Hello! "
                "Please review this algorithm: the BFS traversal uses a deque "
                "and visits each node exactly once in O(V+E) time on adjacency lists. "
                "It needs O(V) extra space for the visited set and queue. "
                "The implementation handles disconnected graphs by iterating over all roots. "
                "Edge cases include self-loops and parallel edges. "
            ),
        },
        {"role": "user", "content": "What is the time complexity?"},
    ]
    result = compress(messages, target_ratio=0.7, protect_tail=1)

    assert result.compressed_tokens < result.original_tokens
    assert len(result.dropped_segments) > 0
    # The tail (question) must survive.
    assert any(
        "complexity" in m["content"] for m in result.messages if isinstance(m.get("content"), str)
    )
    # Standalone bare greetings must be pruned (whole-segment filler).
    body = " ".join(
        m["content"] for m in result.messages if isinstance(m.get("content"), str)
    )
    assert "Hi!" not in body
    assert "Hello!" not in body
    # Some technical content must survive — at least one sentence about the algorithm.
    technical_signals = ["BFS", "O(V", "disconnected graphs", "self-loops"]
    assert any(sig in body for sig in technical_signals), body


def test_no_budget_is_passthrough():
    messages = [{"role": "user", "content": "hello world " * 100}]
    result = compress(messages)
    assert result.messages == messages
    assert result.ratio == 1.0
    assert result.skipped is True


def test_hard_token_budget_respected():
    long = " ".join(f"sentence number {i} with some padding content." for i in range(200))
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": long},
        {"role": "user", "content": "final question"},
    ]
    result = compress(messages, budget_tokens=300, protect_tail=1)
    assert result.compressed_tokens <= result.original_tokens
    assert any(
        "final question" in m["content"]
        for m in result.messages
        if isinstance(m.get("content"), str)
    )


def test_anthropic_style_list_content_preserved():
    """Anthropic-style content-block lists are accepted and returned in the same shape."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello! Thanks. "},
                {"type": "text", "text": "The order is ORD-12345. Please refund $99.00. "},
                {"type": "text", "text": "Sure thing! Anything else you need?"},
            ],
        },
        {"role": "user", "content": "What is the order id?"},
    ]
    result = compress(messages, target_ratio=0.4, protect_tail=1, protect_system=False)
    # Content blocks preserved
    assert isinstance(result.messages[0]["content"], list)
    # Retention of order id survives
    flattened = "".join(
        block.get("text", "")
        for m in result.messages
        for block in (m["content"] if isinstance(m.get("content"), list) else [])
    )
    assert "ORD-12345" in flattened
    assert "$99.00" in flattened


def test_structured_content_skipped():
    """Messages with tool_use / tool_result / image blocks pass through verbatim."""
    tool_call = {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "foo"}}
    tool_result = {"type": "tool_result", "tool_use_id": "t1", "content": "result data"}
    messages = [
        {"role": "user", "content": "Find foo bar baz."},
        {"role": "assistant", "content": [tool_call]},
        {"role": "user", "content": [tool_result]},
        {"role": "user", "content": "Summarize the result."},
    ]
    result = compress(messages, budget_tokens=20, protect_tail=1, protect_system=False)
    # Structured messages are untouched.
    assert result.messages[1]["content"] == [tool_call]
    assert result.messages[2]["content"] == [tool_result]


def test_empty_messages():
    result = compress([], budget_tokens=100)
    assert result.messages == []
    assert result.original_tokens == 0
    assert result.ratio == 1.0
    assert result.skipped is True


def test_question_inferred_from_last_user_message():
    """When no explicit question is passed, the last user message is used."""
    messages = [
        {
            "role": "user",
            "content": (
                "Goldfish live in freshwater environments. "
                "Quantum computers use qubits rather than classical bits. "
                "Qubits can exist in superpositions of 0 and 1. "
                "Bicycles typically have two wheels and a chain drive."
            ),
        },
        {"role": "user", "content": "Explain qubits."},
    ]
    result = compress(messages, target_ratio=0.4, protect_tail=1, protect_system=False)
    body = " ".join(
        m["content"] for m in result.messages if isinstance(m.get("content"), str)
    )
    assert "qubit" in body.lower() or "superposition" in body.lower()


def test_ratio_field_computed():
    messages = [
        {"role": "user", "content": "Long prose " * 200},
        {"role": "user", "content": "final question"},
    ]
    result = compress(messages, budget_tokens=50, protect_tail=1, protect_system=False)
    assert 0 < result.ratio <= 1.0
    # Field is consistent with the token counts.
    if result.original_tokens:
        assert abs(result.ratio - result.compressed_tokens / result.original_tokens) < 1e-6


def test_nonstring_content_coerced_with_warning(caplog):
    """Non-string, non-list content is coerced to str and a WARNING is logged."""
    messages_under_test = [
        {"role": "user", "content": 42},
        {"role": "user", "content": 3.14},
        {"role": "user", "content": {"not": "expected"}},
    ]
    with caplog.at_level(logging.WARNING, logger="vecr_compress"):
        result = compress(messages_under_test, budget_tokens=500)
    # All messages should appear in output (coerced).
    body = " ".join(
        str(m.get("content", "")) for m in result.messages
    )
    assert "42" in body
    assert "3.14" in body
    assert "expected" in body
    # A warning must have been logged for each non-string content.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 3


def test_nondict_message_raises_typed_error():
    """Non-dict entries in messages raise TypeError with a clear message."""
    with pytest.raises(TypeError, match=r"messages\[1\]"):
        compress([{"role": "user", "content": "ok"}, "bad msg"])


def test_invalid_budget_logs_warning(caplog):
    """budget_tokens=0 or negative must log a WARNING when rewritten to 32."""
    messages = [{"role": "user", "content": "Hello world, this is a test message."}]
    with caplog.at_level(logging.WARNING, logger="vecr_compress"):
        result = compress(messages, budget_tokens=0)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "rewritten" in r.message]
    assert len(warnings) >= 1
    # Also verify negative budget triggers the same.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="vecr_compress"):
        result2 = compress(messages, budget_tokens=-5)
    warnings2 = [r for r in caplog.records if r.levelno == logging.WARNING and "rewritten" in r.message]
    assert len(warnings2) >= 1


def test_structured_input_counted_in_original_tokens():
    """tool_use blocks with large input must contribute to original_tokens."""
    large_input = {"key": "value" * 500}  # ~3000 chars of payload
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "analyze",
                    "input": large_input,
                }
            ],
        }
    ]
    result = compress(messages, budget_tokens=500)
    # original_tokens must not be 0 — large input must contribute.
    assert result.original_tokens > 0
    # Should reflect at least some portion of the ~3000 char payload.
    assert result.original_tokens > 50


def test_kept_message_indices_reflects_surviving_messages():
    """kept_message_indices must exactly match the indices of messages in result.messages."""
    # Build a conversation where aggressive budget forces drops.
    filler = "Sure thing! Happy to help! Thanks for asking! Hi there! "
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": filler * 20},   # heavy filler, should be dropped
        {"role": "user", "content": "What is the capital of France?"},
    ]
    result = compress(messages, budget_tokens=20, protect_tail=1, protect_system=False)
    # The kept_message_indices must be a subset of [0,1,2].
    assert all(0 <= i < len(messages) for i in result.kept_message_indices)
    # The indices must correspond exactly to the messages that appear in output.
    # We verify by checking the output message count matches.
    assert len(result.kept_message_indices) == len(result.messages)
    # The tail message must have survived (protect_tail=1).
    assert (len(messages) - 1) in result.kept_message_indices


# -----------------------------------------------------------------------------
# Section 4 extra tests
# -----------------------------------------------------------------------------


def test_contract_stress_at_budget_zero(caplog):
    """budget=0 → warning logged, pinned tokens appear in output, compressed_tokens > 0."""
    messages = [
        {
            "role": "user",
            "content": (
                "The order id is ORD-99172. "
                "Customer email: buyer@example.com. "
                "Total: $1,499.00. "
                "Random filler to pad. " * 30
            ),
        },
        {"role": "user", "content": "What is the order id?"},
    ]
    with caplog.at_level(logging.WARNING, logger="vecr_compress"):
        result = compress(messages, budget_tokens=0, protect_tail=1, protect_system=False)
    # Warning about budget rewrite must be present.
    assert any("rewritten" in r.message for r in caplog.records if r.levelno == logging.WARNING)
    # Pinned facts must survive.
    body = " ".join(
        m["content"] for m in result.messages if isinstance(m.get("content"), str)
    )
    assert "ORD-99172" in body
    assert "buyer@example.com" in body
    assert "$1,499.00" in body
    # compressed_tokens must be positive (pinned content always returned).
    assert result.compressed_tokens > 0


def test_mixed_content_array_passthrough():
    """content=[text_block, tool_use_block] passes through verbatim (skip_mask)."""
    tool_use = {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}}
    text_block = {"type": "text", "text": "Let me search for that."}
    messages = [
        {
            "role": "assistant",
            "content": [text_block, tool_use],
        },
        {"role": "user", "content": "Thanks."},
    ]
    result = compress(messages, budget_tokens=5, protect_tail=1, protect_system=False)
    # The assistant message with mixed content must pass through verbatim.
    assistant_msgs = [m for m in result.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == [text_block, tool_use]
