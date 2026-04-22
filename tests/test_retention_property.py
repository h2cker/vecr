"""Property-based regression gate for the retention contract.

The core promise of :func:`vecr_compress.compress` is that segments containing
structured facts (order IDs, URLs, dates, code, citations, numbers, ...) are
pinned and survive any budget. This file turns that promise into Hypothesis
properties so the full retention rule set is exercised on every CI run, not
just the hand-picked fixtures in ``test_retention.py``.

Properties covered:

* **P1 - hit span survives**: if any :data:`DEFAULT_RULES` pattern matches a
  generated text, that same pattern still matches the concat of the compressed
  output at the minimum (32-token) budget.
* **P2 - soft idempotence**: running ``compress`` twice never introduces a new
  pinned rule that was not present in the first pass.
* **P3 - malformed content**: empty / None / list-of-blocks / unknown-typed
  content does not crash ``compress``.

Known P0.B follow-ups: if any ``xfail`` is added below it should be linked to
a specific pattern that is either over-segmented (a pinned span is split mid-
match) or a false positive in the rule itself. Wave 1 only adds the net; the
P0.B wave tightens the rules and resegmentation to flip any xfails to pass.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vecr_compress import compress
from vecr_compress.retention import DEFAULT_RULES

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Seeded samples — every entry is chosen to hit at least one DEFAULT_RULES
# pattern. Pure random ``st.text()`` almost never produces these shapes, so we
# mix them in explicitly to give the properties meaningful coverage.
_STRUCTURED_SAMPLES = st.sampled_from(
    [
        "ORD-99172",
        "INV_2024_A",
        "CUST#42",
        "2026-04-22",
        "2026-04-22T09:30:00",
        "3f6e4b1a-23cd-4e5f-9012-abcdef012345",
        "buyer@example.com",
        "https://api.example.com/v2/orders",
        "/var/log/app/error.log",
        "C:/data/report.csv",
        "`raise ValueError(msg)`",
        "process_refund(order_id, amount)",
        "[12]",
        "[Smith 2023]",
        '"status": "pending_review"',
        "a3f9b2c1deadbeef",
        "$1,499.00",
        "12.4%",
        "v3.2.1",
        "4242",
        "99172",
    ]
)
_TEXT_BODY = st.text(min_size=0, max_size=200)
_SEGMENT_STRATEGY = st.lists(
    st.one_of(_STRUCTURED_SAMPLES, _TEXT_BODY),
    min_size=0,
    max_size=10,
).map(" ".join)


def _concat_output(result_messages: list[dict[str, Any]]) -> str:
    """Flatten the compressed messages into a single string for regex checks."""
    parts: list[str] = []
    for msg in result_messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# P1 — a pattern that hit the input still hits the output
# ---------------------------------------------------------------------------


@given(body=_SEGMENT_STRATEGY)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_hit_spans_survive_budget_prune(body: str) -> None:
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": body},
    ]
    result = compress(messages, budget_tokens=32)
    output = _concat_output(result.messages)

    for rule in DEFAULT_RULES:
        if rule.pattern.search(body) is None:
            continue
        assert rule.pattern.search(output) is not None, (
            f"rule {rule.name!r} matched input but not output; "
            f"input={body!r} output={output!r}"
        )


# ---------------------------------------------------------------------------
# P2 — soft idempotence: second pass never grows the pinned-rule set
# ---------------------------------------------------------------------------


@given(body=_SEGMENT_STRATEGY)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_idempotent_pinned_rules(body: str) -> None:
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": body},
    ]
    budget = 32
    first = compress(messages, budget_tokens=budget)
    second = compress(first.messages, budget_tokens=budget)

    first_rules = {m["rule"] for m in first.retained_matches}
    second_rules = {m["rule"] for m in second.retained_matches}

    assert second_rules <= first_rules, (
        f"second compression introduced new pinned rules: "
        f"{second_rules - first_rules}; input={body!r}"
    )


# ---------------------------------------------------------------------------
# P3 — malformed / empty content must not crash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "",
        None,
        [],
        [{"type": "text", "text": ""}],
        [{"type": "unknown"}],
        [{"type": "text"}],
        [{"not": "a block"}],
        [{"type": "text", "text": "ok"}, {"type": "unknown"}],
    ],
)
def test_malformed_content_does_not_crash(content: Any) -> None:
    messages = [{"role": "user", "content": content}]
    # Should return a CompressResult without raising.
    result = compress(messages, budget_tokens=100)
    assert result is not None
