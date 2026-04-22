"""LlamaIndex postprocessor for vecr-compress.

Re-exports VecrNodePostprocessor from vecr_compress.adapters.llamaindex
so users can `from llama_index.postprocessor.vecr import VecrNodePostprocessor`
following LlamaIndex's partner-package convention.
"""
from vecr_compress.adapters.llamaindex import VecrNodePostprocessor

__all__ = ["VecrNodePostprocessor"]
__version__ = "0.1.0"
