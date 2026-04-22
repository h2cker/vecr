"""LangChain integration for vecr-compress.

Re-exports VecrContextCompressor from vecr_compress.adapters.langchain
so users can `from langchain_vecr_compress import VecrContextCompressor`
following LangChain's partner-package convention.

The underlying implementation lives in the vecr-compress library.
"""
from vecr_compress.adapters.langchain import VecrContextCompressor

__all__ = ["VecrContextCompressor"]
__version__ = "0.1.0"
