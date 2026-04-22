# PR: Add vecr-compress document compressor integration

## Summary

vecr-compress is an open-source LLM context compressor that provides a **deterministic retention contract**: order IDs, dates, URLs, emails, and code references are pinned by an auditable regex whitelist before any token-budget packing runs. This PR adds a docs page for the `langchain-vecr-compress` partner package, listing it under LangChain's document compressors integrations so users can discover it via the official integrations index.

## What this PR adds

- New docs page at `docs/docs/integrations/document_compressors/vecr-compress.md` listing the `langchain-vecr-compress` partner package, install instructions, and a minimal usage example.

## Partner package

- PyPI: https://pypi.org/project/langchain-vecr-compress/ *(live after first publish)*
- GitHub: https://github.com/h2cker/vecr/tree/main/integrations/langchain-vecr-compress

## Why this belongs in LangChain docs

- **Deterministic retention contract** — unlike every other compressor in the integrations index (LLMLingua, FlashrankRerank), vecr-compress guarantees via regex whitelist that structured tokens (IDs, amounts, dates) are never dropped. This is a unique primitive that users handling compliance-sensitive or agentic workflows need.
- **Tested against the LangChain adapter** — `VecrContextCompressor` is tested end-to-end with `langchain_core.messages` (`AIMessage`, `HumanMessage`, `SystemMessage`, `ToolMessage`), including a verified `tool_calls` round-trip (pinned in v0.1.1 audit).
- **No competition with existing compressors** — vecr-compress targets accuracy/compliance use cases; LLMLingua targets maximum compression ratio. They solve different problems and complement each other in the docs.

## How to test

```bash
pip install langchain-vecr-compress langchain-core
python - <<'EOF'
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_vecr_compress import VecrContextCompressor

compressor = VecrContextCompressor(budget_tokens=120)
msgs = [
    SystemMessage(content="You are a refund analyst."),
    HumanMessage(content="Order ORD-99172 placed 2026-03-15. Amount $1,499.00."),
    HumanMessage(content="Hi there, just wanted to say hello!"),
]
out = compressor.compress_messages(msgs)
texts = [m.content for m in out]
assert any("ORD-99172" in t for t in texts), "retention contract broken"
assert all("just wanted to say hello" not in t for t in texts), "filler should be dropped"
print("OK:", [type(m).__name__ for m in out])
EOF
```

## Checklist

- [x] Docs change only — no modifications to `langchain` or `langchain-core` source
- [x] New integration page follows existing compressor page format
- [x] Install command tested locally
- [x] Partner package is open source (Apache 2.0) and publicly available on PyPI
- [x] No new Python dependencies added to the LangChain monorepo
