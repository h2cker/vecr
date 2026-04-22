"""Retention whitelist tests — adapted from vecr/tests/test_retention.py."""

from __future__ import annotations

import pytest

import re

from vecr_compress import compress
from vecr_compress.retention import (
    DEFAULT_RULES,
    RetentionRule,
    RetentionRules,
    is_pinned,
    retention_reason,
)


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


# ---------------------------------------------------------------------------
# P0.B tightening — fn-call / hash / integer false-positive regression gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,should_match",
    [
        # --- good: code-like identifiers with argument-shaped content ---
        ("call process_refund(order_id, 100) on the queue", True),  # snake_case + comma
        ("see ProcessRefund(x=1) in the handler", True),            # CamelCase + kwarg
        ("obj.method(a, b) should return fast", True),              # dotted + comma
        ("wrap snake_case_fn('x') in a try block", True),           # snake + quoted arg
        ("re.compile(r'x') is called once at import", True),        # dotted + quoted arg
        # --- bad: bare English verbs + prose inside parens ---
        ("please run(quickly) and then come back", False),
        ("note(that it fails) even under low load", False),
        ("think(about this) before deciding", False),
        ("see(below) for the follow-up steps", False),
        ("check(the docs) and let me know", False),
    ],
)
def test_fn_call_regex(text: str, should_match: bool) -> None:
    rule = next(r for r in DEFAULT_RULES if r.name == "fn-call")
    assert bool(rule.pattern.search(text)) is should_match


@pytest.mark.parametrize(
    "text,should_match",
    [
        # --- good: 8+ hex chars with at least two digits ---
        ("git rev 9f3ab2c4 landed", True),              # real SHA prefix, 4 digits
        ("digest a1b2c3d4e5f6 mismatch", True),         # 6 digits, 12 chars
        ("payload feedface1234 seen", True),            # 4 digits
        ("pseudo-sha deadbeef00 in fixture", True),     # 8 alpha + 2 digits
        ("sha 9f3ab2c4e18d7a6b committed", True),       # long SHA, multiple digits
        # --- bad: too short, one-digit low-entropy, or no digits ---
        ("deadbeef is not a real hash", False),          # 0 digits
        ("cabbaged is an English word", False),          # 0 digits (actually has g, no match)
        ("aaaaaaa1 is a single-digit alpha run", False), # exactly 1 digit
        ("bbbbb1bb has only one digit too", False),      # exactly 1 digit
        ("abcdef1 has only seven hex chars", False),     # too short (<8)
    ],
)
def test_hash_regex(text: str, should_match: bool) -> None:
    rule = next(r for r in DEFAULT_RULES if r.name == "hash")
    assert bool(rule.pattern.search(text)) is should_match


@pytest.mark.parametrize(
    "text,should_match",
    [
        # --- good: 4+ digit bare integers (real IDs, years, reference numbers) ---
        ("order 9172 was shipped", True),
        ("invoice 99172 is overdue", True),
        ("scheduled for 2026 launch", True),       # year
        ("ticket 12345 escalated", True),          # 5-digit id
        ("card ending in 4242 was declined", True),
        # --- bad: 2- or 3-digit quantities in normal prose ---
        ("42 items remain in the queue", False),
        ("120 users joined today", False),
        ("85 percent of requests succeeded", False),
        ("Chapter 3 begins on the next page", False),  # single digit, still bare
        ("survey shows 45 responded", False),          # 2-digit
    ],
)
def test_integer_regex(text: str, should_match: bool) -> None:
    rule = next(r for r in DEFAULT_RULES if r.name == "integer")
    assert bool(rule.pattern.search(text)) is should_match


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


def test_retention_rules_with_extra_extends_defaults():
    """User-supplied extra rules append to DEFAULT_RULES without shadowing built-ins."""
    # Custom pattern no built-in catches: lowercase-prefixed tag with colon.
    extra = RetentionRule("ticket", re.compile(r"\btkt:\d+\b"))
    extended = DEFAULT_RULES.with_extra([extra])

    # Extra rule matches what defaults miss.
    assert extended.reason("Reference tkt:42 in the changelog.") == "ticket"
    assert DEFAULT_RULES.reason("Reference tkt:42 in the changelog.") is None

    # Built-in rules still win on their own territory — extras appended after.
    assert extended.reason("Order ORD-99172 shipped.") == "code-id"

    # Original rules object is not mutated.
    assert len(DEFAULT_RULES) + 1 == len(extended)
    assert DEFAULT_RULES.reason("Reference tkt:42 in the changelog.") is None
