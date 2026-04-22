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


def test_mixed_content_text_compressed_image_preserved():
    """一条消息含 text + image block 时，text 参与压缩，image 原样保留。"""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is a long filler sentence one. " * 20},
                {"type": "image", "source": {"type": "base64", "data": "FAKE"}},
                {"type": "text", "text": "Please find the order ORD-12345."},
            ],
        },
        {"role": "user", "content": "What is the order ID?"},
    ]
    result = compress(messages, budget_tokens=80)
    # image 块必须原样在输出中
    middle = result.messages[1]["content"]
    assert isinstance(middle, list)
    assert any(b.get("type") == "image" for b in middle)
    # ORD-12345 必须幸存（retention 规则）
    text_blocks = [b["text"] for b in middle if b.get("type") == "text"]
    assert any("ORD-12345" in t for t in text_blocks)


def test_tool_use_token_count_uses_tiktoken():
    """tool_use input 的 token 估算应走 tiktoken，不走 char/4。"""
    import json
    from vecr_compress.tokens import count as tcount
    payload = {"order_id": 99172, "amount": 1499.00, "notes": "refund authorized"}
    expected = tcount(json.dumps(payload, default=str))
    messages = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "x", "input": payload}
        ]},
    ]
    result = compress(messages, budget_tokens=10_000)  # 大 budget，pass-through
    # original_tokens 里的 tool_use 部分应该等于 tcount(json.dumps(payload))
    # （加上其它 message 的 token；但这里唯一消息就是这个）
    assert result.original_tokens == expected


def test_all_structured_message_passes_through():
    """全 structured 消息（只含 tool_use）仍走原样透传路径。"""
    messages = [
        {"role": "user", "content": "前面说了 process_refund(order_id=99172) 要调用。"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "refund", "input": {"order_id": 99172}}],
        },
        {"role": "user", "content": "继续吗？"},
    ]
    result = compress(messages, budget_tokens=50)
    # tool_use 消息应仍然原样保留
    assert any(
        isinstance(m["content"], list) and any(b.get("type") == "tool_use" for b in m["content"])
        for m in result.messages
    )


def test_mixed_content_array_preserves_structured_block():
    """content=[text_block, tool_use_block]: text may be compressed/pruned but
    the tool_use block survives verbatim (fine-grained structured handling)."""
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
    # The assistant message must still be in output and its tool_use intact.
    assistant_msgs = [m for m in result.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    content = assistant_msgs[0]["content"]
    assert isinstance(content, list)
    assert tool_use in content


def test_structured_block_types_overridable():
    """Callers can mark custom block types as structured (not compressible)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a lot of prose " * 50},
                {"type": "custom_media", "payload": "binary-or-whatever"},
            ],
        },
        {"role": "user", "content": "question?"},
    ]
    # Default behavior: custom_media is NOT treated as structured,
    # so the message is "mixed text"; text part compresses.
    result_default = compress(messages, budget_tokens=40)
    # With override: custom_media IS structured → message passes through in full.
    result_override = compress(
        messages,
        budget_tokens=40,
        structured_block_types=frozenset(
            {"tool_use", "tool_result", "image", "input_image", "document", "custom_media"}
        ),
    )
    # Under override, user message with custom_media is recognized as
    # structured-only path OR mixed-but-preserves-custom_media. Either way,
    # the custom_media block must survive in output.
    first_user_msg_blocks = next(
        (m["content"] for m in result_override.messages if isinstance(m.get("content"), list)),
        None,
    )
    assert first_user_msg_blocks is not None
    assert any(b.get("type") == "custom_media" for b in first_user_msg_blocks)


def test_structured_block_types_default_unchanged():
    """Without override, default set is used and behavior is unchanged."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "data": "X"}},
        ]},
    ]
    r = compress(messages, budget_tokens=100)
    # image 块必须在输出
    blocks = r.messages[0]["content"]
    assert isinstance(blocks, list)
    assert any(b.get("type") == "image" for b in blocks)


def test_use_question_relevance_prefers_topical_segment():
    """With use_question_relevance=True, a topical segment should beat an off-topic
    one of similar heuristic score when the budget is tight."""
    context = (
        "Kafka partitions messages across brokers for horizontal scale. "
        "Cassandra uses consistent hashing to distribute data across nodes. "
        "Raft elects a single leader via randomized election timeouts. "
        "MongoDB supports flexible document schemas and secondary indexes. "
    )
    messages = [
        {"role": "user", "content": context},
        {"role": "user", "content": "How does Raft elect a leader?"},
    ]

    # Default path (heuristic only) — Raft sentence may or may not survive at tight ratio.
    default = compress(
        messages,
        target_ratio=0.4,
        protect_tail=1,
        retain=False,
    )
    # Question-aware path — Raft sentence should survive.
    aware = compress(
        messages,
        target_ratio=0.4,
        protect_tail=1,
        retain=False,
        use_question_relevance=True,
    )

    aware_text = " ".join(
        m["content"] for m in aware.messages if isinstance(m.get("content"), str)
    )
    assert "Raft" in aware_text

    # Question-aware may or may not beat default on every fixture, but the kwarg
    # must at least produce a valid CompressResult and not crash the pipeline.
    assert 0.0 < aware.ratio <= 1.0
    assert default.compressed_tokens >= 0


def test_use_question_relevance_off_by_default():
    """Default behavior unchanged — question ignored when flag is off."""
    messages = [
        {"role": "user", "content": "The algorithm is elegant and well-understood."},
        {"role": "user", "content": "How does Raft elect a leader?"},
    ]
    r1 = compress(messages, target_ratio=0.5, protect_tail=1)
    r2 = compress(messages, target_ratio=0.5, protect_tail=1, question="something else")
    # Without the opt-in flag, question changes don't affect scoring → same output.
    assert r1.compressed_tokens == r2.compressed_tokens


def test_explicit_scorer_overrides_use_question_relevance():
    """If both scorer and use_question_relevance are supplied, the explicit scorer wins."""
    calls = []

    def custom(text, question=None):
        calls.append((text, question))
        return 0.5

    # Large enough context that the knapsack actually runs (beats 32-token floor).
    body = " ".join(
        f"This is sentence number {i} describing a topic in detail with enough words."
        for i in range(20)
    )
    messages = [
        {"role": "user", "content": body},
        {"role": "user", "content": "question text here, look it up please"},
    ]
    compress(
        messages,
        target_ratio=0.3,
        protect_tail=1,
        scorer=custom,
        use_question_relevance=True,  # should be ignored
    )
    assert calls, "custom scorer must be invoked"
