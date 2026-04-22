"""LlamaIndex postprocessor for vecr-compress.

Re-exports VecrNodePostprocessor from vecr_compress.adapters.llamaindex
so users can `from llama_index.postprocessor.vecr import VecrNodePostprocessor`
following LlamaIndex's partner-package convention.
"""
import warnings

warnings.warn(
    "llama-index-postprocessor-vecr is deprecated; install vecr-compress[llamaindex] instead. "
    "This shim will stop receiving updates in the next major release.",
    DeprecationWarning,
    stacklevel=2,
)

from vecr_compress.adapters.llamaindex import VecrNodePostprocessor

__all__ = ["VecrNodePostprocessor"]
__version__ = "0.1.0"
