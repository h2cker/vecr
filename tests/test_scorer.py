"""Scorer tests — covers Jaccard overlap, heuristic, and edge cases."""

from __future__ import annotations

from vecr_compress.scorer import content_words, heuristic_score, question_relevance


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


def test_heuristic_score_reorders_by_question_relevance():
    """Relative ordering between two sentences flips when a question is asked."""
    relevant = "Raft uses randomized election timeouts to elect a leader."
    off_topic = "Kafka partitions messages across brokers for horizontal scale."
    q = "How does Raft elect a leader?"

    # With the question, the relevant sentence should outrank the off-topic one
    # by a bigger margin than it does without one.
    delta_with_q = heuristic_score(relevant, question=q) - heuristic_score(off_topic, question=q)
    delta_without_q = heuristic_score(relevant) - heuristic_score(off_topic)
    assert delta_with_q > delta_without_q

    # And scores remain bounded in [0, 1].
    for s in (relevant, off_topic):
        assert 0.0 <= heuristic_score(s) <= 1.0
        assert 0.0 <= heuristic_score(s, question=q) <= 1.0


def test_heuristic_score_rewards_structural_signals():
    """Sentences with numbers / code-punctuation / CamelCase get a signal bump."""
    plain = heuristic_score("the response was fast and clean for the user")
    structural = heuristic_score("The HttpClient returned 200 OK in 42ms via GET /api/users;")
    assert structural > plain


def test_whitespace_question_equivalent_to_none():
    """heuristic_score with question='   ' must equal score with question=None."""
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
