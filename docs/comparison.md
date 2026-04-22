# vecr vs Similar Projects

## TL;DR

Choose **vecr** when you need *contractually guaranteed retention* of structured data (order IDs, URLs, citations, code) while cutting input tokens on long, partially-templated prompts in a Python stack — and you're OK with v0.1 limits (text-only, no streaming, no tool calls). Choose **[Compresr](https://www.ycombinator.com/companies/compresr)** (YC W26) when you want a production-ready Go gateway with streaming, tool-call support, and LLM-based summarization claiming up to 100× ratios. Choose **[LLMLingua-2](https://github.com/microsoft/LLMLingua)** when you want research-grade token-classification compression and can afford to run an extra small LM for scoring. Choose **OpenAI native compaction** when you live on one provider and their built-in behavior already fits. Choose **LangChain DeepAgents** when you're on LangChain and want the agent to decide when to compact.

## Landscape (2026)

Context compression went from research-only in 2023-2024 to commodity in 2026:

- **Compresr** (YC W26, EPFL 4-person team) launched February 2026 — a Go-based Apache 2.0 Context Gateway with proprietary `hcc_espresso_v1` summarization, ~450 GitHub stars at six weeks, claiming up to 100× compression and 76% cost reduction via preemptive summarization + tool output compression + phantom tools.
- **Portkey** fully open-sourced its AI gateway under Apache 2.0 in March 2026.
- **Helicone** was acquired by Mintlify and is now in maintenance mode — effectively out of the active-development race.
- **OpenAI** shipped a provider-side compaction API in addition to the older prefix cache + `prompt_cache_key` pattern (see [developers.openai.com/api/docs/guides/compaction](https://developers.openai.com/api/docs/guides/compaction)).
- **LangChain DeepAgents** released an autonomous context-compression tool for agents in March 2026.
- **LLMLingua / LLMLingua-2** (Microsoft Research) remain the open-source research baseline: token-classification compressors reporting 3-6× ratios at 95-98% accuracy retention.

vecr's deliberate niche in that landscape is narrow and disciplined: *retention-first, not ratio-first*.

## Feature matrix

| Feature | vecr | Compresr | LLMLingua-2 | OpenAI native compaction | LangChain DeepAgents |
|---|---|---|---|---|---|
| Open source | Apache 2.0 | Apache 2.0 | MIT | N/A (provider) | MIT |
| Language | Python | Go | Python | — | Python / TS |
| Approach | Retention whitelist + Jaccard + knapsack | LLM summarization + phantom tools | Token classification (small LM) | Provider-side prefix cache + compaction API | Agent-driven `compact` tool |
| Typical compression ratio | 3-10× (sentence-level) | Claimed up to 100× | 3-6× | Variable / opaque | Variable |
| Structured-data retention | Regex whitelist — contractual | Best-effort via summarization | Best-effort via token keep-probability | Opaque to caller | Agent-driven heuristic |
| Streaming | Planned (not v0.1) | Yes | No | Yes | Yes |
| Tool calls / function calling | Passthrough only (planned) | Yes (phantom tools) | No | Yes | Yes |
| Managed service | No | Yes (hosted `hcc_espresso_v1`) | No | Yes (built-in) | No |
| Extra model required | No | Hosted LLM call | Yes (small classifier LM) | No | Yes (agent LLM) |
| Multi-provider gateway | OpenAI + Anthropic | Multi-provider | N/A (library) | N/A | N/A |
| Maturity | v0.1 experimental | Public since Feb 2026 (~6 wks) | Research-grade, 2+ years | Production | Production |

## When to choose each

### Choose vecr when

- You need **guaranteed retention of structured data** — regulatory / audit / RAG correctness constraints where silently dropping an order ID, URL, citation, or code snippet is unacceptable.
- You're Python-first and want a small, dependency-light library (`vecr-compress`) you can call inline — not a hosted service, not a proxy.
- Your workload has long, partially-templated prompts (RAG chunks, chat history, repeated system instructions) where filler is real and sentence-level pruning is enough.
- You can live with v0.1 limits: text-only, no streaming, no tool-call rewriting, +20-60 ms overhead on the median prompt.

### Choose Compresr when

- You want maximum compression ratio and are comfortable with LLM-based summarization paraphrasing your prompt.
- You need a production-ready proxy in Go with streaming, tool calls, and phantom-tool support *today*.
- You're OK depending on their hosted API for `hcc_espresso_v1`, or running their Go gateway in your infra.

### Choose LLMLingua-2 when

- You want research-grade token-classification compression with published accuracy numbers.
- You're willing to run an additional small LM for scoring and can tolerate the extra latency.
- You don't need structured-data retention guarantees — the token classifier is best-effort.

### Choose OpenAI native compaction when

- You only use one provider and the built-in compaction + prefix cache already fit your workload.
- You don't need per-segment control, retention guarantees, or cross-provider consistency.
- Sub-10 ms overhead matters more than explicit compression accounting.

### Choose LangChain DeepAgents compact tool when

- You're already on LangChain / LangGraph and want the agent itself to decide when to compact its own context.
- Your workflow is agent-first rather than prompt-first.

## Honest caveats

- vecr's 100% structured-fact survival applies to retention-whitelist regex hits (UUIDs, order codes, URLs, dates, emails, code spans, numbers, citations, hashes). If your "structured fact" doesn't match any of the ~13 built-in patterns, it's not pinned.
- Compresr's 100× claim is a *maximum* under favorable conditions, not a universal guarantee. Compare on your own workload before committing.
- LLMLingua-2 runs an extra model — measure end-to-end latency, not just compression time.
- Provider-native compaction behavior is opaque by design; you trade control for zero integration work.

## Sources

- Compresr (YC W26): <https://www.ycombinator.com/companies/compresr>
- LLMLingua / LLMLingua-2: <https://github.com/microsoft/LLMLingua>
- OpenAI compaction guide: <https://developers.openai.com/api/docs/guides/compaction>
- LangChain DeepAgents autonomous context compression (March 2026 release).
- Portkey open-sourcing (March 2026) and Helicone / Mintlify acquisition (2026) are referenced from public announcements.

## Last updated

2026-04-22
