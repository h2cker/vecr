"""Retention whitelist tests — adapted from vecr/tests/test_retention.py."""

from __future__ import annotations

from vecr_compress import compress
from vecr_compress.retention import is_pinned, retention_reason


def test_retention_catches_structured_data():
    assert retention_reason("The order id is ORD-42819 for tracking.") == "code-id"
    assert retention_reason("Contact us at help@vecr.ai please") == "email"
    assert retention_reason("Docs at https://vecr.ai/docs/api available") == "url"
    assert retention_reason("Deploy occurred on 2026-04-21.") == "date"
    assert retention_reason("git rev 9f3ab2c4") == "hash"
    assert retention_reason("Call `client.invoke()` for this") == "code-span"
    assert retention_reason("Citation [12] shows the method") == "citation"
    assert retention_reason("Net revenue was $1,299.00 last month") == "number"


def test_retention_ignores_plain_prose():
    assert not is_pinned("Hello there, how are you doing today")
    assert not is_pinned("The algorithm is elegant and efficient")


def test_pinned_segments_survive_aggressive_compression():
    body = (
        "Hello! Thanks for reaching out. "
        "The refund request references order ORD-99172 placed on 2026-03-15. "
        "The customer email is buyer@example.com. "
        "We are reviewing it carefully. "
        "Totally agree this is important. "
        "The total charge was $1,499.00 on card ending 4242. "
    )
    messages = [
        {"role": "system", "content": "You are a refund analyst."},
        {"role": "user", "content": body},
        {"role": "user", "content": "What is the order ID and refund amount?"},
    ]
    # Very aggressive — would normally drop most sentences.
    result = compress(
        messages,
        target_ratio=0.1,
        protect_tail=1,
        protect_system=False,
    )
    body_out = " ".join(
        m["content"] for m in result.messages if isinstance(m.get("content"), str)
    )

    # Every critical reference must survive, even at ratio=0.1.
    assert "ORD-99172" in body_out
    assert "2026-03-15" in body_out
    assert "buyer@example.com" in body_out
    assert "$1,499.00" in body_out


def test_hash_requires_digit():
    """Hash rule must require at least one digit — pure-alpha hex words must not match."""
    # Pure alpha (no digits) — should NOT be pinned.
    assert not is_pinned("deadbeef"), "pure-alpha hex word should not be pinned"
    assert not is_pinned("cabbaged"), "pure-alpha hex word should not be pinned"
    # Has at least one digit — should be pinned.
    assert is_pinned("a1b2c3d4e5f6"), "mixed hex with digits must be pinned"
    assert is_pinned("feedface1234"), "mixed hex with digits must be pinned"
    assert is_pinned("1234567a"), "hex with leading digits must be pinned"
    assert is_pinned("9f3ab2c4"), "real git SHA prefix must be pinned"


def test_question_aware_boosts_relevant_sentences():
    # A body with two distinct topics; the question mentions only one.
    messages = [
        {
            "role": "user",
            "content": (
                "Raft uses randomized election timeouts to elect a leader. "
                "Raft leaders replicate log entries to followers via AppendEntries. "
                "Kafka partitions messages across brokers for horizontal scale. "
                "Kafka consumer groups track offsets in an internal topic. "
                "Cassandra uses gossip for membership and consistent hashing for placement. "
                "Cassandra tunable consistency ranges from ONE to ALL. "
            ),
        },
        {"role": "user", "content": "How does Raft handle leader election?"},
    ]
    result = compress(
        messages,
        target_ratio=0.35,
        protect_tail=1,
        protect_system=False,
    )
    body = " ".join(
        m["content"] for m in result.messages if isinstance(m.get("content"), str)
    )

    # Raft content about election should survive; unrelated Kafka/Cassandra
    # should be pruned preferentially.
    assert "Raft" in body
    assert "election" in body.lower() or "leader" in body.lower()
    # At ratio 0.35 over 6 sentences, at least one off-topic must be cut.
    off_topic_kept = sum(kw in body for kw in ["Kafka", "Cassandra"])
    assert off_topic_kept <= 1  # no more than one off-topic sentence survives
