# PR: Add vecr-compress node postprocessor integration

## Summary

vecr-compress is an open-source LLM context compressor that provides a **deterministic retention contract**: order IDs, dates, URLs, emails, and code references are pinned by an auditable regex whitelist before any token-budget packing runs. This PR adds a docs page for the `llama-index-postprocessor-vecr` partner package, listing it under LlamaIndex's node postprocessor integrations so users can discover it via the official integrations index.

## What this PR adds

- New docs page at `docs/docs/integrations/node_postprocessor/vecr.md` listing the `llama-index-postprocessor-vecr` partner package, install instructions, and a minimal usage example.

## Partner package

- PyPI: https://pypi.org/project/llama-index-postprocessor-vecr/ *(live after first publish)*
- GitHub: https://github.com/h2cker/vecr/tree/main/integrations/llama-index-postprocessor-vecr

## Why this belongs in LlamaIndex docs

- **Deterministic retention contract** — unlike rerankers and other postprocessors in the integrations index, vecr-compress guarantees via regex whitelist that structured tokens (IDs, amounts, dates) in retrieved nodes are never dropped, regardless of token budget. This is a unique guarantee for RAG pipelines handling compliance-sensitive data.
- **Tested against the LlamaIndex adapter** — `VecrNodePostprocessor` is tested end-to-end with `llama_index.core.schema.NodeWithScore` and `TextNode`, with a verified test that nodes containing `ORD-99172` survive aggressive 40-token budgets.
- **Complementary to rerankers** — vecr-compress is not a semantic reranker (no embeddings); it is a node-level compression postprocessor. The two can be chained: rerank first, then compress into the token budget before synthesis.

## How to test

```bash
pip install llama-index-postprocessor-vecr llama-index-core
python - <<'EOF'
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.postprocessor.vecr import VecrNodePostprocessor

nodes = [
    NodeWithScore(
        node=TextNode(id_="n1", text="Refund for ORD-99172 approved 2026-03-15. Amount $1,499.00."),
        score=0.9,
    ),
    NodeWithScore(
        node=TextNode(id_="n2", text="Hi! Thanks for reaching out, hope you're well."),
        score=0.2,
    ),
]

processor = VecrNodePostprocessor(budget_tokens=40)
kept = processor.postprocess_nodes(nodes, query_str="refund for ORD-99172")
ids = [n.node.id_ for n in kept]
assert "n1" in ids, "retention contract broken"
print("kept:", ids)
EOF
```

## Checklist

- [x] Docs change only — no modifications to `llama-index-core` source
- [x] New integration page follows existing node postprocessor page format
- [x] Install command tested locally
- [x] Partner package follows LlamaIndex namespace convention (`llama_index.postprocessor.vecr`)
- [x] Open source (Apache 2.0), publicly available on PyPI
- [x] No new Python dependencies added to the LlamaIndex monorepo
