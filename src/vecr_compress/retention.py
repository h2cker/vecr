"""Strong-retention rules.

Some spans MUST survive compression regardless of budget or scorer output.
Losing a number, order-id, or URL silently is the class of failure that kills
trust the fastest. These patterns are conservative — false positives (over-
retention) are fine; false negatives (losing a fact) are not.

Matching semantics: a segment containing ANY retention pattern is `pinned`.
Pinned segments bypass the knapsack — they are kept even if it blows the
budget. The compressor emits a warning when that happens so the caller can
raise the budget.

Callers can supply their own rules by building a :class:`RetentionRules`
object. The default set, :data:`DEFAULT_RULES`, matches the patterns the
reference vecr gateway uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RetentionRule:
    """A single named regex pattern. If ``pattern.search(text)`` matches, the
    containing segment is pinned (kept regardless of budget)."""

    name: str
    pattern: re.Pattern[str]


class RetentionRules:
    """Ordered collection of retention rules.

    Pattern order matters — specific patterns are listed before general ones
    so a code-id like "ORD-42819" is classified as ``code-id`` rather than
    ``integer``.
    """

    def __init__(self, rules: Iterable[RetentionRule]):
        self._rules: tuple[RetentionRule, ...] = tuple(rules)

    def __iter__(self):
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def reason(self, text: str) -> str | None:
        """Return the first matched rule name, or None if nothing pinned."""
        for rule in self._rules:
            if rule.pattern.search(text):
                return rule.name
        return None

    def is_pinned(self, text: str) -> bool:
        return self.reason(text) is not None

    def with_extra(self, extra: Iterable[RetentionRule]) -> "RetentionRules":
        """Return a new RetentionRules with ``extra`` appended.

        Extra rules run after the built-ins so user patterns don't shadow
        the stricter structural matches.
        """
        return RetentionRules(tuple(self._rules) + tuple(extra))


_DEFAULT_RULE_LIST: tuple[RetentionRule, ...] = (
    # --- Specific structural patterns first (so they win over generic numerics). ---
    RetentionRule(
        "uuid",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
    ),
    RetentionRule(
        "date",
        re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"),
    ),
    # Identifiers that look like codes: ORDER-123, INV_2024_A, CUST#42
    RetentionRule(
        "code-id",
        re.compile(r"\b[A-Z][A-Z0-9]{1,}[-_#]?\d+[A-Z0-9_-]*\b"),
    ),
    RetentionRule("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    RetentionRule("url", re.compile(r"https?://\S+")),
    # Filesystem paths (abs or deep relative) with extension.
    RetentionRule(
        "path",
        re.compile(r"(?:[A-Za-z]:)?/[\w\-./]+\.[A-Za-z0-9]{1,6}\b"),
    ),
    # Backticked code spans.
    RetentionRule("code-span", re.compile(r"`[^`\n]+`")),
    # Function calls: identifier(...) with at least one letter before the paren.
    RetentionRule(
        "fn-call",
        re.compile(r"\b[a-zA-Z_][\w]*\([^)]{0,80}\)"),
    ),
    # Citation markers: [12], [Smith 2023], (Knuth, 1997)
    RetentionRule(
        "citation",
        re.compile(r"\[(?:\d{1,3}|[A-Z][a-zA-Z]+\s+\d{4})\]"),
    ),
    # JSON-looking keys with values.
    RetentionRule(
        "json-kv",
        re.compile(r'"[\w_]+"\s*:\s*"[^"]{1,120}"'),
    ),
    # Hex hashes (git SHAs, crypto digests) — 8+ hex chars on a word boundary
    # with at least one digit (prevents matching pure-alpha words like "deadbeef").
    RetentionRule("hash", re.compile(r"\b(?=[0-9a-f]{8,}\b)(?=[a-f]*[0-9])[0-9a-f]{8,}\b")),
    # --- Generic numerics last. ---
    # Formatted: "$1,299.00", "12.4%", "v3.2.1".
    RetentionRule(
        "number",
        re.compile(r"(?<![A-Za-z])[\$€£¥]?-?\d{1,3}(?:[,\.]\d+)+[%kKmMbB]?\b"),
    ),
    # Bare integers of 2+ digits.
    RetentionRule(
        "integer",
        re.compile(r"(?<![A-Za-z_])\d{2,}(?![A-Za-z_])"),
    ),
)


DEFAULT_RULES: RetentionRules = RetentionRules(_DEFAULT_RULE_LIST)
"""The built-in retention rule set.

Covers UUIDs, ISO dates, ORDER-style codes, emails, URLs, paths, backticked
code, function calls, citations, JSON key-value pairs, hex hashes, formatted
numbers, and bare integers with 2+ digits.
"""


def retention_reason(text: str, rules: RetentionRules | None = None) -> str | None:
    """Return the first matched pattern name, or None if nothing pinned."""
    return (rules or DEFAULT_RULES).reason(text)


def is_pinned(text: str, rules: RetentionRules | None = None) -> bool:
    return (rules or DEFAULT_RULES).is_pinned(text)
