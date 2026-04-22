# When to Use vecr (and When Not To)

vecr is a **specialized** tool with a deliberately narrow niche. This doc exists so you can stop reading in 90 seconds if it doesn't fit.

## Use vecr when

- Long system prompt with structured facts that must not be silently dropped (IDs, URLs, citations, code spans, dates, numbers).
- RAG pipeline where retrieved chunks contain IDs, URLs, or code that matter for answer correctness.
- Regulatory / audit / compliance workloads where "the compressor quietly dropped the order number" is unacceptable.
- Python project with non-critical latency tolerance (p95 ~10 ms on 5k-token contexts, ~100–125 ms on 50k-token contexts — see [BENCHMARK.md#latency](BENCHMARK.md#latency)).
- Willing to self-host or import the library inline — there is no managed service in v0.1.
- You want to layer compression on top of provider-native prompt caching and measure both effects separately.

## Do NOT use vecr when

- You need **streaming** today (planned for Phase 1 / v0.2, not shipped).
- Your calls rely on **function / tool calls** or structured outputs in middle messages (planned, not shipped — currently passed through verbatim, which is safe but means zero gain on those turns).
- You need **multi-modal** (image / audio) compression.
- You need **sub-10 ms overhead** — use provider-native caching or compaction instead.
- You need **>10× compression ratios** — try LLM-based summarization (Compresr) or accept paraphrasing loss.
- You don't care about structured-data retention — cheaper alternatives exist.
- Your prompts are already short; there's nothing to compress.
- You need a **production SLA** or **managed service** — neither exists in v0.1.

## Workload fit matrix

| Workload | Fit | Why |
|---|---|---|
| Long system prompt with structured facts | Strong | Retention whitelist directly targets this. |
| RAG with ID-heavy chunks | Strong | Structured-data regex pins IDs/URLs before knapsack. |
| Customer-support multi-turn chat | OK — test on your data first | Benefit depends on filler density and history length. |
| Agentic workflows with tool outputs | Wait | Tool-call rewriting is planned, not v0.1. |
| Coding copilot with heavy tool use | Wait | Same reason — tool calls pass through verbatim today. |
| Streaming chat UIs | Wait | Streaming is planned for v0.2 / Phase 1. |
| Real-time voice / audio | Out of scope | No multimodal support. |
| Short single-turn prompts | Poor | Nothing to compress; overhead without benefit. |

## Integration checklist

Before flipping vecr on for real traffic, walk through this list:

- [ ] Pin a specific `vecr-compress` version (library is pre-1.0; breaking changes likely).
- [ ] Run your top 20 representative real prompts through `compress()` offline. Diff input vs output.
- [ ] Verify that every structured fact that matters to you (IDs, URLs, citations, numeric totals) survives in the compressed output.
- [ ] Measure added latency on your median prompt. Published numbers: p95 ~1 ms on 500-token, ~10 ms on 5k-token, ~100–125 ms on 50k-token contexts (Apple M3 Max, tokenization-dominated). Re-measure on your hardware with `python -m bench.latency`.
- [ ] Confirm budget safety: when must-keep content exceeds the budget, vecr overshoots budget rather than drop pinned facts — decide whether that's acceptable.
- [ ] For the gateway: test the bypass path (`x-vecr-bypass: true`) so you know requests still succeed if vecr errors internally.
- [ ] A/B a 5% traffic slice before full rollout. Compare answer quality, not just tokens.
- [ ] If you're on one provider with native compaction, run it alongside vecr and measure whether vecr adds anything on your workload.

## Decision tree (90 seconds)

1. Do you need streaming or tool-call support today? Yes → **not vecr** (wait for v0.2, or use Compresr / provider-native).
2. Are your prompts short or structured-fact-free? Yes → **not vecr** (overhead without benefit).
3. Do you need guaranteed retention of IDs / URLs / code / citations? Yes → **vecr is a strong fit**.
4. Do you want a Python library you can call inline? Yes → use `vecr-compress`. Want an HTTP gateway? Use the reference `oss_gateway`.
5. Are you stuck on one provider and their native compaction already fits? Yes → start there; only add vecr if you hit a retention correctness bug.

## See also

- [comparison.md](./comparison.md) — vs Compresr / LLMLingua-2 / OpenAI native compaction / LangChain DeepAgents.
- [BENCHMARK.md](./BENCHMARK.md) — what numbers we can cite honestly.
- [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md) — roadmap and 2026 market positioning.

## Last updated

2026-04-22
