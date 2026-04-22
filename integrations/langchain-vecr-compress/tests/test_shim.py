"""Shim tests for langchain-vecr-compress.

Tests use duck-typed fakes where possible so langchain_core is not required
for tests 1 and 2. Test 3 is skipped when langchain_core is not installed.
"""
import importlib
import sys

import pytest

import langchain_vecr_compress
from langchain_vecr_compress import VecrContextCompressor
import vecr_compress.adapters.langchain as _core_adapter


def test_shim_reexports_compressor():
    """The shim must re-export the exact same class from the core adapter."""
    assert VecrContextCompressor is _core_adapter.VecrContextCompressor


def test_shim_version_matches():
    """Partner-package version must be 0.1.0."""
    assert langchain_vecr_compress.__version__ == "0.1.0"


_langchain_available = importlib.util.find_spec("langchain_core") is not None


@pytest.mark.skipif(
    not _langchain_available,
    reason="langchain_core not installed; skipping tool_calls round-trip test",
)
def test_shim_preserves_tool_calls_roundtrip():
    """AIMessage tool_calls must survive a compress/decompress round-trip."""
    from langchain_core.messages import AIMessage  # type: ignore[import-not-found]

    ai_msg = AIMessage(
        content="let me check",
        tool_calls=[{"id": "c1", "name": "search", "args": {"q": "x"}, "type": "tool_call"}],
    )

    compressor = VecrContextCompressor(budget_tokens=500)
    result = compressor.compress_messages([ai_msg])

    assert len(result) == 1, "AIMessage with tool_calls should be kept"
    out_msg = result[0]
    assert hasattr(out_msg, "tool_calls"), "Reconstructed AIMessage must have tool_calls"
    assert len(out_msg.tool_calls) == 1
    tc = out_msg.tool_calls[0]
    assert tc["id"] == "c1"
    assert tc["name"] == "search"
    assert tc["args"] == {"q": "x"}
