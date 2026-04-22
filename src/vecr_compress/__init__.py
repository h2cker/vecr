"""vecr-compress — retention-guaranteed LLM context compression.

Public API:

    from vecr_compress import compress, CompressResult, DEFAULT_RULES, RetentionRules

    result = compress(
        messages=[{"role": "user", "content": "..."}, ...],
        budget_tokens=4000,
        question="the latest user query",
    )
    result.messages            # compressed messages (same role shape as input)
    result.original_tokens     # int
    result.compressed_tokens   # int
    result.ratio               # float in (0, 1]
    result.dropped_segments    # list[dict]
    result.retained_matches    # list[dict]

See `vecr_compress.compressor.compress` for the full signature.
"""

from __future__ import annotations

from .compressor import CompressResult, compress
from .retention import DEFAULT_RULES, RetentionRule, RetentionRules, is_pinned, retention_reason
from .scorer import blended_score, heuristic_score, question_relevance

__all__ = [
    "compress",
    "CompressResult",
    "DEFAULT_RULES",
    "RetentionRule",
    "RetentionRules",
    "is_pinned",
    "retention_reason",
    "heuristic_score",
    "blended_score",
    "question_relevance",
]

__version__ = "0.1.3"
