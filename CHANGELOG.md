# Changelog

## [0.1.3] - 2026-04-22

### Added
- `compress(..., use_question_relevance=True)` — opt-in question-aware scoring. Blends `question_relevance` (Jaccard) into `heuristic_score` at 0.6/0.4 weights. Off by default.
- `vecr_compress.scorer.blended_score` — the composed scorer; exported publicly so callers can reuse or wrap it.
- `heuristic_score` / `blended_score` / `question_relevance` are now top-level exports on `vecr_compress`.
- `bench/hotpotqa_probe.py` — research-tier spike over N=100 HotpotQA dev examples. Probes supporting-fact survival on real multi-hop NL-QA.
- `[bench]` optional-dependency extra (`pip install vecr-compress[bench]`) declaring `datasets>=2.14` for the HotpotQA loader.
- New docs section in `docs/BENCHMARK.md` ("HotpotQA spike") documenting where the synthetic needle bench hits its ceiling.

### Changed
- Reversal of the v0.1.2 "no question-aware uplift" claim for natural-language workloads. HotpotQA probe shows +9.9pp supporting-fact survival at ratio 0.5 from restoring the Jaccard blend (58.0% → 67.9%). The synthetic needle bench did not see this because it saturates at 100% on structured needles. v0.1.2's removal was too broad; v0.1.3 makes blending opt-in rather than default.
- Docstrings across `scorer.py` / `compressor.py` now describe the opt-in explicitly.

### Notes
- Default behavior is unchanged — passing no new kwarg gives you v0.1.2 semantics bit-for-bit. Enabling `use_question_relevance=True` is a caller-side decision appropriate for NL-QA workloads.

## [0.1.2] - 2026-04-22

### Changed
- Removed L3 question-aware Jaccard from default scorer path. Benchmark (594 trials) showed zero uplift over L2 alone; keeping it in docs and code path was misleading.
- `question_relevance` function remains exposed as a helper for callers implementing custom `ScorerFn` who want to re-enable Jaccard blending.
- Updated `docs/BENCHMARK.md` to reflect 2-config sweep (baseline + L2), 396 trials.
- Documentation (comparison.md, when-to-use.md, RETENTION.md, compressor.py module docstring) reworded from "three layers" to "two layers: retention + heuristic".

### Documentation
- README rewritten: headline emphasizes "auditable" over "the only…" claim; "Three layers" architecture collapsed to two to match the P1.A scorer change; removed the repeated "Try to get this guarantee…" tagline (was overselling).
- pyproject.toml description updated to match the new README headline.
- vs. alternatives table updated: "Jaccard knapsack" → "heuristic knapsack" to reflect the P1.A scorer change.
- "Choose vecr-compress when..." paragraph reworded to avoid "compliance or correctness risk you cannot accept" (overpromise for v0.1 alpha) — now emphasizes "auditable, extensible whitelist you can reason about end-to-end."
- Added `bench/latency.py` and a Latency section to `docs/BENCHMARK.md` (3 sizes × 2 budgets, p50/p95/p99 on Apple M3 Max). Replaces the old hand-wavy "+20-60ms" figure, which was wrong in both directions — small inputs are faster, 50k-token inputs run ~100-125 ms because tokenization dominates.

### Dev dependencies
- Added `hypothesis>=6` to `[project.optional-dependencies].dev`. Required by `tests/test_retention_property.py` added earlier in this release cycle; was installed locally but missing from the declared dev-deps.

### Notes
- No API break. `compress()` `question` parameter is preserved and accepted for backward compatibility; default scorer just ignores it now. Custom scorers can still read it.

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
