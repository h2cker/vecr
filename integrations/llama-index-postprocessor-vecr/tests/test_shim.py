"""Shim tests for llama-index-postprocessor-vecr.

Tests use duck-typed fakes where possible so llama_index_core is not required
for tests 1 and 2. Test 3 is skipped when llama_index.core.schema is not installed.
"""
import importlib
import sys

import pytest

import llama_index.postprocessor.vecr as shim_module
from llama_index.postprocessor.vecr import VecrNodePostprocessor
import vecr_compress.adapters.llamaindex as _core_adapter


def test_shim_reexports_postprocessor():
    """The shim must re-export the exact same class from the core adapter."""
    assert VecrNodePostprocessor is _core_adapter.VecrNodePostprocessor


def test_shim_version_matches():
    """Partner-package version must be 0.1.0."""
    assert shim_module.__version__ == "0.1.0"


_llamaindex_available = (
    importlib.util.find_spec("llama_index") is not None
    and importlib.util.find_spec("llama_index.core") is not None
)


@pytest.mark.skipif(
    not _llamaindex_available,
    reason="llama_index.core not installed; skipping node postprocessor test",
)
def test_shim_returns_correct_kept_nodes():
    """Nodes containing ORD-99172 must survive aggressive compression."""
    from llama_index.core.schema import NodeWithScore, TextNode  # type: ignore[import-not-found]

    nodes = [
        NodeWithScore(
            node=TextNode(
                id_="needle",
                text=(
                    "The refund for order ORD-99172 was approved on 2026-03-15. "
                    "Total charge was $1,499.00."
                ),
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(
                id_="filler",
                text="Hi! Hope you are having a wonderful day. Thanks for reaching out.",
            ),
            score=0.1,
        ),
    ]

    processor = VecrNodePostprocessor(budget_tokens=40)
    kept = processor.postprocess_nodes(nodes, query_str="refund for ORD-99172")

    kept_ids = [n.node.id_ for n in kept]
    assert "needle" in kept_ids, (
        f"Node with ORD-99172 must survive aggressive budget. Kept: {kept_ids}"
    )
