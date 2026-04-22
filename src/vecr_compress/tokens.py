"""Token counting. Uses tiktoken's o200k_base as a provider-agnostic proxy.

We do not need exact per-provider token counts for pruning decisions — only a
stable monotonic estimate. Callers doing billing should use the native
tokenizer of their provider; this module is strictly for internal budgeting.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)
_FALLBACK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


@lru_cache(maxsize=1)
def _fallback_mode() -> bool:
    try:
        _enc()
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("vecr_compress tokenizer fallback enabled: %s", exc)
        return True


def count(text: str) -> int:
    """Return an estimated token count for ``text``."""
    if _fallback_mode():
        return len(_FALLBACK_RE.findall(text))
    return len(_enc().encode(text, disallowed_special=()))


def encode(text: str) -> list[int]:
    if _fallback_mode():
        return list(range(len(_FALLBACK_RE.findall(text))))
    return _enc().encode(text, disallowed_special=())


def decode(ids: list[int]) -> str:
    if _fallback_mode():
        return "<fallback-tokenizer-no-decode>"
    return _enc().decode(ids)
