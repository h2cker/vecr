# Shim Package Deprecation

## What changed

As of 2026-04-22, the standalone PyPI shim packages `langchain-vecr-compress`
and `llama-index-postprocessor-vecr` are deprecated. The full adapter code
already lives inside the main `vecr-compress` package and is available via
install extras — maintaining three separate version numbers was pure overhead.

## Migration

### LangChain users

Old:

```bash
pip install langchain-vecr-compress
```

New:

```bash
pip install vecr-compress[langchain]
```

Import path is unchanged:

```python
from vecr_compress.adapters.langchain import VecrContextCompressor
```

The `from langchain_vecr_compress import VecrContextCompressor` import still
works on existing installations but now emits a `DeprecationWarning`.

### LlamaIndex users

Old:

```bash
pip install llama-index-postprocessor-vecr
```

New:

```bash
pip install vecr-compress[llamaindex]
```

Import path is unchanged:

```python
from vecr_compress.adapters.llamaindex import VecrNodePostprocessor
```

The `from llama_index.postprocessor.vecr import VecrNodePostprocessor` import
still works on existing installations but now emits a `DeprecationWarning`.

## For maintainer (checklist — not automated)

- [ ] Tag current shim package source in git as `shim-archive-v0.1.0`
- [ ] Publish one final patch version of each shim (0.1.1) with
      DeprecationWarning + README notice
- [ ] (Optional, manual) On PyPI, mark `langchain-vecr-compress` and
      `llama-index-postprocessor-vecr` metadata classifier
      `Development Status :: 7 - Inactive`
- [ ] Do NOT yank existing versions — keeps existing `pip install` commands
      working
- [ ] Update GitHub repo topic tags to remove the shim-specific keywords if any

## Timeline

- **v0.1.1 (shims)**: deprecation warning only, code unchanged.
- **v0.2.0 (main)**: shims formally archived; `vecr-compress[langchain]` and
  `vecr-compress[llamaindex]` extras remain the canonical install path.
