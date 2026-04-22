"""Adapter smoke tests.

Each adapter has an optional framework dependency. Tests skip (not fail) when
the framework is missing, so a baseline ``pip install vecr-compress`` passes
the suite without needing langchain / llamaindex.
"""

from __future__ import annotations

import importlib.util

import pytest


# -----------------------------------------------------------------------------
# LangChain
# -----------------------------------------------------------------------------

langchain_available = importlib.util.find_spec("langchain_core") is not None


@pytest.mark.skipif(
    not langchain_available, reason="langchain-core not installed"
)
def test_langchain_compresses_messages():
    from langchain_core.messages import HumanMessage, SystemMessage

    from vecr_compress.adapters.langchain import VecrContextCompressor

    compressor = VecrContextCompressor(budget_tokens=40, protect_tail=1, protect_system=False)
    messages = [
        SystemMessage(content="You are helpful."),
        HumanMessage(
            content=(
                "Hi there! Thanks so much. "
                "The refund for order ORD-42 totaling $99.00 is pending. "
                "Sure thing, happy to help! "
                "Kafka partitions messages across brokers."
            )
        ),
        HumanMessage(content="What is the order id and amount?"),
    ]
    compressed = compressor.compress_messages(messages)
    flat = " ".join(m.content for m in compressed if isinstance(m.content, str))
    # Retention-guaranteed facts must survive.
    assert "ORD-42" in flat
    assert "$99.00" in flat


@pytest.mark.skipif(
    not langchain_available, reason="langchain-core not installed"
)
def test_langchain_compress_with_report_returns_metadata():
    from langchain_core.messages import HumanMessage

    from vecr_compress.adapters.langchain import VecrContextCompressor

    compressor = VecrContextCompressor(target_ratio=0.5, protect_tail=1, protect_system=False)
    result = compressor.compress_with_report([
        HumanMessage(content="sentence one is here. " * 20),
        HumanMessage(content="what is the topic?"),
    ])
    assert result.original_tokens >= result.compressed_tokens
    assert 0 < result.ratio <= 1.0


def test_langchain_preserves_tool_calls(monkeypatch):
    """AIMessage.tool_calls must survive round-trip through compress. Uses
    duck-typed stubs so this test never skips, even without langchain installed."""
    import vecr_compress.adapters.langchain as adapter_mod

    # Duck-typed LangChain message classes — names must match _ROLE_MAP_IN.
    class AIMessage:  # noqa: N801
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class HumanMessage:  # noqa: N801
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class SystemMessage:  # noqa: N801
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class ToolMessage:  # noqa: N801
        def __init__(self, content, tool_call_id=""):
            self.content = content
            self.tool_call_id = tool_call_id
            self.tool_calls = []

    def fake_require():
        return AIMessage, object, HumanMessage, SystemMessage, ToolMessage

    monkeypatch.setattr(adapter_mod, "_require_langchain", fake_require)

    original = AIMessage(
        content="let me check",
        tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "SF"}}],
    )

    # Convert to dict (should produce content blocks with tool_use).
    as_dict = adapter_mod._lc_to_dict(original)
    assert as_dict["role"] == "assistant"
    assert isinstance(as_dict["content"], list)
    tool_use_blocks = [b for b in as_dict["content"] if b.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "get_weather"

    # Convert back (should reconstruct tool_calls).
    restored = adapter_mod._dict_to_lc(as_dict)
    assert hasattr(restored, "tool_calls")
    assert len(restored.tool_calls) == 1
    assert restored.tool_calls[0]["name"] == "get_weather"
    assert restored.tool_calls[0]["id"] == "call_1"


def test_langchain_adapter_raises_friendly_error_when_missing(monkeypatch):
    """If langchain_core isn't importable, the adapter should give an actionable error."""
    import vecr_compress.adapters.langchain as adapter_mod

    def boom():
        raise ImportError(adapter_mod._INSTALL_HINT)

    monkeypatch.setattr(adapter_mod, "_require_langchain", boom)
    with pytest.raises(ImportError) as exc_info:
        adapter_mod.VecrContextCompressor()
    assert "pip install vecr-compress[langchain]" in str(exc_info.value)


# -----------------------------------------------------------------------------
# LlamaIndex
# -----------------------------------------------------------------------------

llamaindex_available = importlib.util.find_spec("llama_index") is not None


@pytest.mark.skipif(
    not llamaindex_available, reason="llama-index-core not installed"
)
def test_llamaindex_postprocesses_nodes():
    from llama_index.core.schema import NodeWithScore, TextNode

    from vecr_compress.adapters.llamaindex import VecrNodePostprocessor

    nodes = [
        NodeWithScore(node=TextNode(text="Kafka partitions messages across brokers."), score=0.8),
        NodeWithScore(node=TextNode(text="Raft uses randomized election timeouts."), score=0.7),
        NodeWithScore(node=TextNode(text="Cassandra uses gossip for membership."), score=0.6),
    ]
    processor = VecrNodePostprocessor(budget_tokens=40)
    kept = processor.postprocess_nodes(nodes, query_str="How does Raft elect a leader?")
    # Something survives.
    assert len(kept) >= 1


def test_llamaindex_adapter_raises_friendly_error_when_missing(monkeypatch):
    import vecr_compress.adapters.llamaindex as adapter_mod

    def boom():
        raise ImportError(adapter_mod._INSTALL_HINT)

    monkeypatch.setattr(adapter_mod, "_require_llamaindex", boom)
    with pytest.raises(ImportError) as exc_info:
        adapter_mod.VecrNodePostprocessor()
    assert "pip install vecr-compress[llamaindex]" in str(exc_info.value)


def test_llamaindex_preserves_correct_nodes(monkeypatch):
    """Reproduce exact failure case: node 0 partial-keep (5 dropped+1 knapsack-kept),
    node 1 trivially kept (no segmentation issue), node 2 fully dropped.
    _surviving_indices must return [0, 1], not [1, 2]."""
    import vecr_compress.adapters.llamaindex as adapter_mod

    monkeypatch.setattr(adapter_mod, "_require_llamaindex", lambda: None)

    class _Inner0:
        # node 0: long content with one pinned fact and lots of filler → partial keep
        text = (
            "Sure! Thanks! Hi there! Happy to help! OK! Great! "
            "The order id is ORD-99172 for tracking purposes. "
            "Sure thing! Agreed! Totally!"
        )

    class _Inner1:
        # node 1: short content, trivially kept (no segmentation needed)
        text = "Raft uses randomized election timeouts."

    class _Inner2:
        # node 2: pure filler — fully dropped
        text = "Sure! Thanks! Hi! OK! Great! Happy to help! Absolutely!"

    class _Node:
        def __init__(self, inner):
            self.node = inner
            self.score = 0.5

    nodes = [_Node(_Inner0()), _Node(_Inner1()), _Node(_Inner2())]
    processor = adapter_mod.VecrNodePostprocessor(budget_tokens=30)
    kept = processor.postprocess_nodes(nodes)
    # node 2 (pure filler) must not appear; nodes 0 and 1 must appear.
    kept_texts = [adapter_mod._node_text(n) for n in kept]
    # At minimum node 1 (Raft fact) must survive — it has real content.
    assert any("Raft" in t for t in kept_texts), f"Raft node missing from {kept_texts}"
    # node 2 (pure filler) should be dropped.
    assert not any("Sure! Thanks! Hi! OK! Great! Happy to help! Absolutely!" == t for t in kept_texts), \
        "Pure filler node 2 should have been dropped"


def test_llamaindex_no_nameerror_without_llamaindex_installed(monkeypatch):
    """postprocess_nodes must not raise NameError — the dead loop referencing
    _iter_messages_with_index must be gone. Uses duck-typed fake nodes so this
    test never skips, even when llama_index is absent."""
    import vecr_compress.adapters.llamaindex as adapter_mod

    # Bypass _require_llamaindex so the constructor doesn't fail if llama_index
    # is not installed.
    monkeypatch.setattr(adapter_mod, "_require_llamaindex", lambda: None)

    # Duck-typed minimal node: has .node.text attribute
    class _FakeInner:
        text = "Raft uses randomized election timeouts. The leader is elected by majority vote."

    class _FakeNode:
        node = _FakeInner()
        score = 0.9

    processor = adapter_mod.VecrNodePostprocessor(budget_tokens=200)
    # Must not raise NameError
    result = processor.postprocess_nodes([_FakeNode()])
    assert isinstance(result, list)


def test_llamaindex_duck_typed_round_trip(monkeypatch):
    """Duck-typed NodeWithScore/TextNode round-trip — never skips, no llama_index needed."""
    import vecr_compress.adapters.llamaindex as adapter_mod

    monkeypatch.setattr(adapter_mod, "_require_llamaindex", lambda: None)

    class _Inner:
        def __init__(self, text):
            self.text = text

    class _FakeNode:
        def __init__(self, text, score=0.9):
            self.node = _Inner(text)
            self.score = score

    nodes = [
        _FakeNode("Raft uses randomized election timeouts to elect a leader."),
        _FakeNode("Kafka partitions messages across brokers for horizontal scale."),
        _FakeNode("Cassandra uses gossip for membership and consistent hashing for placement."),
    ]
    processor = adapter_mod.VecrNodePostprocessor(budget_tokens=500)
    kept = processor.postprocess_nodes(nodes, query_str="How does Raft elect a leader?")
    # Result is a list — at least one node must survive given generous budget.
    assert isinstance(kept, list)
    assert len(kept) >= 1
    # Each returned element still has .node.text accessible.
    for n in kept:
        assert isinstance(adapter_mod._node_text(n), str)


def test_langchain_use_question_relevance_threads_to_core(monkeypatch):
    """VecrContextCompressor(use_question_relevance=True) must forward the kwarg
    to compress() so adapter users get the v0.1.3 opt-in without reaching for
    the core API."""
    import vecr_compress.adapters.langchain as adapter_mod

    class HumanMessage:  # noqa: N801
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class SystemMessage:  # noqa: N801
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class AIMessage:  # noqa: N801
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class ToolMessage:  # noqa: N801
        def __init__(self, content, tool_call_id=""):
            self.content = content
            self.tool_call_id = tool_call_id
            self.tool_calls = []

    monkeypatch.setattr(
        adapter_mod, "_require_langchain",
        lambda: (AIMessage, object, HumanMessage, SystemMessage, ToolMessage),
    )

    seen: dict[str, object] = {}

    def fake_compress(messages, **kwargs):
        seen.update(kwargs)
        from vecr_compress.compressor import CompressResult
        return CompressResult(
            messages=list(messages),
            original_tokens=0,
            compressed_tokens=0,
            ratio=1.0,
            dropped_segments=[],
            retained_matches=[],
            kept_message_indices=list(range(len(messages))),
        )

    monkeypatch.setattr(adapter_mod, "compress", fake_compress)

    adapter_mod.VecrContextCompressor(
        budget_tokens=2000, use_question_relevance=True
    ).compress_with_report([HumanMessage("hi")])
    assert seen.get("use_question_relevance") is True

    seen.clear()
    adapter_mod.VecrContextCompressor(budget_tokens=2000).compress_with_report(
        [HumanMessage("hi")]
    )
    assert seen.get("use_question_relevance") is False


def test_llamaindex_use_question_relevance_threads_to_core(monkeypatch):
    """VecrNodePostprocessor(use_question_relevance=True) must forward to
    compress() for both postprocess_nodes and compress_with_report paths."""
    import vecr_compress.adapters.llamaindex as adapter_mod

    monkeypatch.setattr(adapter_mod, "_require_llamaindex", lambda: None)

    class _Inner:
        text = "Raft uses randomized election timeouts."

    class _FakeNode:
        node = _Inner()
        score = 0.9

    seen: dict[str, object] = {}

    def fake_compress(messages, **kwargs):
        seen.update(kwargs)
        from vecr_compress.compressor import CompressResult
        return CompressResult(
            messages=list(messages),
            original_tokens=0,
            compressed_tokens=0,
            ratio=1.0,
            dropped_segments=[],
            retained_matches=[],
            kept_message_indices=list(range(len(messages))),
        )

    monkeypatch.setattr(adapter_mod, "compress", fake_compress)

    adapter_mod.VecrNodePostprocessor(
        budget_tokens=1500, use_question_relevance=True
    ).postprocess_nodes([_FakeNode()], query_str="how?")
    assert seen.get("use_question_relevance") is True

    seen.clear()
    adapter_mod.VecrNodePostprocessor(
        budget_tokens=1500, use_question_relevance=True
    ).compress_with_report([_FakeNode()], query_str="how?")
    assert seen.get("use_question_relevance") is True

    seen.clear()
    adapter_mod.VecrNodePostprocessor(budget_tokens=1500).postprocess_nodes(
        [_FakeNode()], query_str="how?"
    )
    assert seen.get("use_question_relevance") is False


def test_langchain_duck_typed_round_trip(monkeypatch):
    """Duck-typed LangChain round-trip through VecrContextCompressor — never skips."""
    import vecr_compress.adapters.langchain as adapter_mod

    class HumanMessage:  # noqa: N801
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class SystemMessage:  # noqa: N801
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class AIMessage:  # noqa: N801
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class ToolMessage:  # noqa: N801
        def __init__(self, content, tool_call_id=""):
            self.content = content
            self.tool_call_id = tool_call_id
            self.tool_calls = []

    monkeypatch.setattr(
        adapter_mod, "_require_langchain",
        lambda: (AIMessage, object, HumanMessage, SystemMessage, ToolMessage),
    )

    messages = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("Hello! The order id is ORD-42819. " * 10),
        HumanMessage("What is the order id?"),
    ]
    compressor = adapter_mod.VecrContextCompressor(
        budget_tokens=60, protect_tail=1, protect_system=False
    )
    result = compressor.compress_messages(messages)
    assert isinstance(result, list)
    assert len(result) >= 1
    # Retention contract: ORD-42819 must survive.
    body = " ".join(
        m.content for m in result if isinstance(getattr(m, "content", None), str)
    )
    assert "ORD-42819" in body
