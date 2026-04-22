# langchain-vecr-compress

> **DEPRECATED (2026-04-22)**: This shim package is deprecated in favor of the main
> `vecr-compress` package's extras. Install the integration directly with
> `pip install vecr-compress[langchain]` (or `[llamaindex]`). This standalone
> package will stop receiving updates; existing installations keep working but
> should migrate before the next major release. See
> [DEPRECATION_SHIMS.md](../../docs/DEPRECATION_SHIMS.md) for details.

Drop-in LangChain integration for [vecr-compress](https://github.com/h2cker/vecr) — an auditable, deterministic LLM context compressor with a **retention contract**. Before your chat history reaches the model, vecr-compress pins every order ID, URL, date, email, and code reference using an explicit regex whitelist, then packs the remaining budget with heuristically scored sentences (entropy + structural signals). Structured data never disappears silently; tool calls round-trip intact. This partner package is a thin shim so you can install it with the standard LangChain pattern and get started immediately, with all logic staying in the `vecr-compress` core.

## Install

```bash
pip install langchain-vecr-compress
```

This installs `vecr-compress` and `langchain-core` automatically.

## 30-second example

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_vecr_compress import VecrContextCompressor

compressor = VecrContextCompressor(budget_tokens=120)

history = [
    SystemMessage(content="You are a refund analyst."),
    HumanMessage(content="Hi! I have a question about order ORD-99172."),
    AIMessage(
        content="Let me look that up.",
        tool_calls=[{"id": "c1", "name": "lookup_order", "args": {"order_id": "ORD-99172"}}],
    ),
    HumanMessage(content="The charge was $1,499.00 on 2026-03-15. Please advise."),
    HumanMessage(content="Also, totally just saying hi — hope you're having a great day!"),
]

compressed = compressor.compress_messages(history)

for msg in compressed:
    print(type(msg).__name__, "->", msg.content[:80])
```

The compressor will drop the filler greeting while preserving `ORD-99172`, `$1,499.00`, `2026-03-15`, and the `AIMessage` with its `tool_calls` list fully intact.

## tool_calls round-trip guarantee

`AIMessage` objects carrying `tool_calls` are converted to Anthropic-style `tool_use` content blocks internally. The compressor's skip-mask treats any message containing a `tool_use` block as must-keep and passes it through verbatim. On the way out, the blocks are converted back to `AIMessage(content=..., tool_calls=[...])`. This round-trip was audited and tested in vecr-compress v0.1.1.

```python
ai_msg = AIMessage(
    content="searching now",
    tool_calls=[{"id": "t1", "name": "web_search", "args": {"q": "refund policy"}}],
)
[out] = compressor.compress_messages([ai_msg])
assert out.tool_calls[0]["name"] == "web_search"  # always true
```

## Advanced: access compression telemetry

```python
result = compressor.compress_with_report(history)
print(f"{result.original_tokens} -> {result.compressed_tokens} tokens ({result.ratio:.1%})")
print(f"Pinned facts: {len(result.retained_matches)}")
for seg in result.dropped_segments:
    print("dropped:", seg["text"][:60])
```

## Extending the retention contract

```python
import re
from vecr_compress import RetentionRule, DEFAULT_RULES

custom_rules = DEFAULT_RULES.with_extra([
    RetentionRule(name="ticket", pattern=re.compile(r"\bTICKET-\d{4,8}\b")),
])
compressor = VecrContextCompressor(budget_tokens=2000, retention_rules=custom_rules)
```

## Links

- Main repo and full docs: [https://github.com/h2cker/vecr](https://github.com/h2cker/vecr)
- Retention contract details: [RETENTION.md](https://github.com/h2cker/vecr/blob/main/RETENTION.md)
- Issues: [https://github.com/h2cker/vecr/issues](https://github.com/h2cker/vecr/issues)

## License

Apache 2.0 — see [LICENSE](LICENSE).
