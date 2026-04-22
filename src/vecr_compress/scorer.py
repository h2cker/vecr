"""Segment informativeness scoring.

Three scorers ship here:

- :func:`heuristic_score` — pure-Python entropy + structural-signal score.
  Default scorer; ignores ``question``.
- :func:`question_relevance` — lexical Jaccard overlap between a segment and a
  question's content words.
- :func:`blended_score` — 0.6 * heuristic + 0.4 * Jaccard. Opt-in question-aware
  scorer; enable via ``compress(..., use_question_relevance=True)``.

The synthetic needle bench (594 trials, ``docs/BENCHMARK.md``) showed that
question-aware blending adds no uplift over L2 retention alone — but the
HotpotQA probe (``bench/hotpotqa_probe.py``) on real multi-hop NL-QA shows
~+10pp supporting-fact survival at aggressive ratios. So blending is off by
default (keeps the deterministic narrative clean) but a one-kwarg opt-in for
QA-style workloads where it pays off.

No model dependency. MLX / embedding scorers are intentionally not in this
package — they live in the reference gateway.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

_FILLER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Whole-segment greetings / sign-offs — MUST be anchored to the full
        # stripped segment to avoid dropping prose that merely starts with one
        # of these words (e.g. "Please review this algorithm.").
        r"^(hi|hello|hey|greetings|thanks|thank you|please)[\s,!.]*$",
        r"\b(as an ai|i'm happy to|i'd be glad to|let me know if)\b.*",
        r"^(sure|okay|ok|got it|understood)[\s,!.]*$",
    ]
]

_STOPWORDS = frozenset(
    "a an the of to in for on at by with from is are was were be been being do does did "
    "has have had will would could should may might can and or but if then else when where "
    "why how what which who whom this that these those it its their there here".split()
)
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


def content_words(text: str) -> frozenset[str]:
    """Lowercased content words (stopwords removed). Exposed for scorer reuse."""
    return frozenset(
        w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS
    )


def question_relevance(segment: str, question_words: frozenset[str]) -> float:
    """Jaccard overlap of content words between segment and question.

    Cheap but surprisingly strong — lexical overlap catches a large fraction of
    the question-aware uplift documented in LongLLMLingua (EMNLP 2024) at
    near-zero compute cost. For production quality you'd swap this for an
    embedding-cosine scorer, but at v0.1 we keep the dependency footprint
    minimal.
    """
    if not question_words:
        return 0.0
    seg_words = content_words(segment)
    if not seg_words:
        return 0.0
    inter = len(seg_words & question_words)
    union = len(seg_words | question_words)
    return inter / union if union else 0.0


def heuristic_score(segment: str, question: str | None = None) -> float:
    """Informativeness proxy in [0, 1].

    The ``question`` argument is accepted for API compatibility with the
    :data:`ScorerFn` signature but is ignored. Use :func:`blended_score`
    (or pass ``use_question_relevance=True`` to :func:`compress`) when you
    want question-aware blending.

    Returns exactly 0.0 for filler/greetings so the budget packer drops them
    unconditionally.
    """
    del question  # retained for API compatibility; intentionally unused
    s = segment.strip()
    if not s:
        return 0.0
    for pat in _FILLER_PATTERNS:
        if pat.search(s):
            return 0.0

    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    entropy_norm = min(entropy / 5.0, 1.0)

    signal = 0.0
    if re.search(r"\d", s):
        signal += 0.15
    if re.search(r"[{}()\[\]=<>/;]", s):
        signal += 0.15
    if re.search(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", s):
        signal += 0.10

    length_prior = 0.0 if n < 12 else min(math.log10(n) / 3.0, 0.4)
    base = 0.5 * entropy_norm + signal + length_prior

    return max(0.0, min(1.0, base))


# Type alias for callers passing a custom scorer.
ScorerFn = Callable[[str, "str | None"], float]


def blended_score(segment: str, question: str | None = None) -> float:
    """Question-aware scorer: ``0.6 * heuristic + 0.4 * question_relevance``.

    Falls back to pure heuristic when ``question`` is empty or None. The
    weights match the formula removed from the default path in v0.1.2, kept
    here as an opt-in extension. HotpotQA probe shows ~+10pp supporting-fact
    survival at ratio 0.5 vs. heuristic alone — see
    ``bench/hotpotqa_results.md``.

    Enable via ``compress(..., use_question_relevance=True)``.
    """
    base = heuristic_score(segment)
    if base == 0.0:
        # Filler stays filler — don't let Jaccard resurrect greetings.
        return 0.0
    if not question or not question.strip():
        return base
    q_words = content_words(question)
    rel = question_relevance(segment, q_words)
    return max(0.0, min(1.0, 0.6 * base + 0.4 * rel))
