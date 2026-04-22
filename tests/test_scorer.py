"""Scorer tests — covers Jaccard overlap, heuristic, and edge cases."""

from __future__ import annotations

from vecr_compress.scorer import (
    blended_score,
    content_words,
    heuristic_score,
    question_relevance,
)


def test_question_relevance_jaccard_overlap():
    """Overlapping content words produce a non-zero Jaccard score, and more
    overlap yields a higher score."""
    q_words = content_words("How does Raft handle leader election?")
    relevant = question_relevance(
        "Raft uses randomized election timeouts to elect a leader.", q_words
    )
    off_topic = question_relevance(
        "Kafka partitions messages across brokers for horizontal scale.", q_words
    )
    empty_q = question_relevance("anything", frozenset())

    assert relevant > 0.0
    assert relevant > off_topic
    assert empty_q == 0.0


def test_heuristic_score_drops_fillers():
    """Greeting/scaffolding patterns score exactly zero.

    Only whole-segment filler is dropped — see P1 #6 fix. Sentences that start
    with a greeting word but contain substantive content are NOT dropped.
    """
    # Standalone greetings / sign-offs (whole segment = filler).
    assert heuristic_score("Hi!") == 0.0
    assert heuristic_score("Thanks") == 0.0
    assert heuristic_score("Sure") == 0.0
    assert heuristic_score("As an AI, I'm happy to help.") == 0.0
    # Empty string is also zero.
    assert heuristic_score("") == 0.0
    assert heuristic_score("   ") == 0.0


def test_heuristic_score_ignores_question_by_default():
    """The default scorer must ignore ``question`` entirely.

    Question-aware Jaccard blending was removed from the default path after
    the 594-trial benchmark in ``docs/BENCHMARK.md`` showed zero uplift. The
    ``question`` argument is still accepted for API compatibility but any
    string (or None) must produce identical scores.
    """
    segments = [
        "The refund is for ORD-99172 totaling $1,499.",
        "Raft uses randomized election timeouts to elect a leader.",
        "Kafka partitions messages across brokers for horizontal scale.",
    ]
    questions = [
        None,
        "What order was refunded?",
        "How does Raft elect a leader?",
        "unrelated noise words here",
    ]
    for s in segments:
        base = heuristic_score(s, None)
        for q in questions:
            assert heuristic_score(s, q) == base, (
                f"heuristic_score should ignore question (segment={s!r}, question={q!r})"
            )
        # Bounded in [0, 1] regardless of question.
        assert 0.0 <= base <= 1.0


def test_heuristic_score_rewards_structural_signals():
    """Sentences with numbers / code-punctuation / CamelCase get a signal bump."""
    plain = heuristic_score("the response was fast and clean for the user")
    structural = heuristic_score("The HttpClient returned 200 OK in 42ms via GET /api/users;")
    assert structural > plain


def test_whitespace_question_equivalent_to_none():
    """heuristic_score must accept a whitespace-only ``question`` without error.

    Now trivially true because the default scorer ignores ``question``
    entirely, but kept as a regression guard against ever re-introducing
    question-aware logic that mishandles whitespace.
    """
    segment = "Raft uses randomized election timeouts to elect a leader."
    assert heuristic_score(segment, question="   ") == heuristic_score(segment)
    assert heuristic_score(segment, question="  \t\n  ") == heuristic_score(segment)


def test_filler_does_not_drop_substantive_prose_starting_with_please():
    """'please' at start of a sentence with content must NOT score 0.0.

    Regression guard: the old regex used a prefix match that killed any
    sentence beginning with 'please', dropping real instructions.
    """
    assert heuristic_score("Please review this algorithm carefully.") > 0.0
    assert heuristic_score("Please see the attached report for details.") > 0.0
    assert heuristic_score("Thanks for the context about the migration plan.") > 0.0
    # Standalone filler words still score 0.
    assert heuristic_score("Please.") == 0.0
    assert heuristic_score("Thanks.") == 0.0
    assert heuristic_score("Please!") == 0.0


def test_blended_score_reorders_by_question_relevance():
    """With a question, the relevant segment must score above the off-topic one,
    even if the heuristic alone would rank them equal."""
    question = "How does Raft handle leader election?"
    relevant = "Raft uses randomized election timeouts to elect a leader."
    off_topic = "Kafka partitions messages across brokers for horizontal scale."

    # Heuristic alone ranks them similarly (both are well-formed prose).
    # Blended scorer must prefer the relevant one.
    assert blended_score(relevant, question) > blended_score(off_topic, question)


def test_blended_score_falls_back_to_heuristic_on_empty_question():
    """Blended scorer with None / empty / whitespace question == heuristic_score."""
    segment = "Raft uses randomized election timeouts to elect a leader."
    base = heuristic_score(segment)
    assert blended_score(segment, None) == base
    assert blended_score(segment, "") == base
    assert blended_score(segment, "   ") == base


def test_blended_score_does_not_resurrect_filler():
    """Filler segments must score 0.0 even when the question trivially overlaps."""
    # Greeting with matching question words still scores 0.
    assert blended_score("Hi!", "Hi how are you") == 0.0
    assert blended_score("Thanks", "Thanks for the thanks") == 0.0


def test_blended_score_bounded_in_unit_interval():
    """Output must stay in [0, 1] for any input."""
    for seg in ["", "x", "A normal sentence.", "The order ORD-42 shipped on 2026-04-22."]:
        for q in [None, "", "what is it", "ORD-42 shipped"]:
            s = blended_score(seg, q)
            assert 0.0 <= s <= 1.0
