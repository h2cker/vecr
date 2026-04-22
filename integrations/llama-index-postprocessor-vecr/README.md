# llama-index-postprocessor-vecr

> **DEPRECATED (2026-04-22)**: This shim package is deprecated in favor of the main
> `vecr-compress` package's extras. Install the integration directly with
> `pip install vecr-compress[llamaindex]` (or `[langchain]`). This standalone
> package will stop receiving updates; existing installations keep working but
> should migrate before the next major release. See
> [DEPRECATION_SHIMS.md](../../docs/DEPRECATION_SHIMS.md) for details.

Retention-guaranteed node compression for LlamaIndex RAG pipelines. This partner package wraps [vecr-compress](https://github.com/h2cker/vecr) — an auditable, deterministic LLM context compressor that makes a **retention contract**: order IDs, dates, URLs, emails, and code references are pinned by an explicit regex whitelist before any token-budget packing runs. Filler prose is dropped, high-signal sentences are scored by entropy and structural signals (digits, braces, capitalization), and your retrieved nodes arrive at the synthesizer with all structured facts intact.

## Install

```bash
pip install llama-index-postprocessor-vecr
```

This installs `vecr-compress` and `llama-index-core` automatically.

## 30-second example

```python
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.postprocessor.vecr import VecrNodePostprocessor

nodes = [
    NodeWithScore(
        node=TextNode(
            id_="node-1",
            text=(
                "The refund for order ORD-99172 was approved on 2026-03-15. "
                "The total amount was $1,499.00. "
                "Please see https://refunds.example.com/ORD-99172 for details."
            ),
        ),
        score=0.92,
    ),
    NodeWithScore(
        node=TextNode(
            id_="node-2",
            text="Hi! I hope this message finds you well. Have a great day.",
        ),
        score=0.31,
    ),
]

processor = VecrNodePostprocessor(budget_tokens=60)
kept = processor.postprocess_nodes(nodes, query_str="refund status for ORD-99172")

for n in kept:
    print(n.node.id_, "->", n.node.text[:80])
# node-1 -> The refund for order ORD-99172 was approved ...
# (node-2 dropped: pure filler, no retention match)
```

The node containing `ORD-99172`, `2026-03-15`, `$1,499.00`, and the URL is kept because each of those tokens fires a retention rule. The filler-only node is dropped even at an aggressive 60-token budget.

## Query string

Pass `query_str` to provide question context:

```python
kept = processor.postprocess_nodes(
    nodes,
    query_str="what is the refund amount?",
)
```

The default scorer uses heuristic signals (entropy, structural patterns). Callers implementing a custom `ScorerFn` can use `scorer.question_relevance` (Jaccard overlap) to blend question-aware ranking if desired.

## Compression telemetry

```python
result = processor.compress_with_report(nodes, query_str="refund amount")
print(f"{result.original_tokens} -> {result.compressed_tokens} tokens ({result.ratio:.1%})")
print(f"Pinned facts: {len(result.retained_matches)}")
```

## Custom retention rules

```python
import re
from vecr_compress import RetentionRule, DEFAULT_RULES

custom_rules = DEFAULT_RULES.with_extra([
    RetentionRule(name="sku", pattern=re.compile(r"\bSKU-[A-Z0-9]{6}\b")),
])
processor = VecrNodePostprocessor(budget_tokens=1500, retention_rules=custom_rules)
```

## Links

- Main repo and full docs: [https://github.com/h2cker/vecr](https://github.com/h2cker/vecr)
- Retention contract details: [RETENTION.md](https://github.com/h2cker/vecr/blob/main/RETENTION.md)
- Issues: [https://github.com/h2cker/vecr/issues](https://github.com/h2cker/vecr/issues)

## License

Apache 2.0 — see [LICENSE](LICENSE).
