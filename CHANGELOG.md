# Changelog

## 0.1.0 — 2026-04-22

Initial public release.

### Core

- `compress(messages, budget_tokens=..., target_ratio=..., question=..., retention_rules=..., retain=..., protect_tail=..., protect_system=...)` public API returning `CompressResult`.
- `CompressResult` dataclass exposes `messages`, `original_tokens`, `compressed_tokens`, `ratio`, `dropped_segments`, `retained_matches`, and `kept_message_indices` (authoritative list of original message indices present in the compressed output — use this for downstream node-tracking instead of telemetry heuristics).
- 13 built-in retention rules (see [RETENTION.md](RETENTION.md)) — UUIDs, ISO dates, code-IDs, emails, URLs, file paths, backticked code, function calls, citations, JSON keys, hex hashes (digit-required), formatted numbers, and bare integers.
- Question-aware Jaccard scorer (auto-picks last user message when `question=None`).
- Filler hard-drop — greetings and AI scaffolding phrases score 0 before the knapsack runs. Anchored to whole stripped segment, so substantive sentences starting with "Please..." or "Thanks..." are **not** dropped.
- Greedy knapsack budget packing that respects retention pins and `protect_tail` / `protect_system`.
- OpenAI and Anthropic message shape support.
- Structured content passthrough — `tool_use`, `tool_result`, `image`, `document` blocks transit verbatim; mixed-content messages auto-skip compression.

### Adapters

- LangChain: `VecrContextCompressor` — round-trips `HumanMessage` / `SystemMessage` / `AIMessage` including `AIMessage.tool_calls` preserved via Anthropic-style `tool_use` content blocks.
- LlamaIndex: `VecrNodePostprocessor` — duck-typed `postprocess_nodes(nodes, query_bundle=None, query_str=None)` that uses `CompressResult.kept_message_indices` for deterministic node mapping.

### Partner packages (in `integrations/`)

- `langchain-vecr-compress` — LangChain partner-package shim that re-exports `VecrContextCompressor`. `pip install langchain-vecr-compress`.
- `llama-index-postprocessor-vecr` — LlamaIndex partner-package shim under the `llama_index.postprocessor.vecr` namespace. `pip install llama-index-postprocessor-vecr`.

### Benchmark and docs

- `bench/needle.py` — reproducible NeedleInHaystack benchmark (11 needles × 3 positions × 6 ratios × 3 configs = 594 trials, no API keys, ~5 seconds).
- `docs/BENCHMARK.md`, `docs/comparison.md`, `docs/when-to-use.md` — honest methodology, vs. Compresr / LLMLingua-2 / native compaction / DeepAgents, use / don't-use matrix.

### Tests

- 36 core tests pass; 3 shim tests per partner package (6 total) pass. Adapter tests run via duck typing without requiring LangChain or LlamaIndex to be installed.

### Input validation

- Non-`dict` entries in `messages` raise `TypeError` with the offending index.
- Non-string, non-list content values are coerced via `str()` with a `WARNING` log (never silently dropped).
- `budget_tokens` of 0 or negative logs a `WARNING` before flooring to 32.

### Known limits for v0.1

- Text only — no streaming response compression.
- No tool-call argument rewriting (passed through verbatim, which is safe but gives no compression on tool turns).
- English-tuned regex / stopword list.
- Heuristic + Jaccard scorer only (no embedding scorer).
- Sentence-level granularity (no token-level pruning).
