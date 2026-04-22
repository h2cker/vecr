# vecr Quality Benchmark

## What this measures

**Fact survival rate** — does a specific "needle" fact still appear in the compressed output? This is the closest offline proxy we have to "did the compressor silently drop something the downstream model needs?" — the question vecr's retention whitelist exists to answer.

All numbers in this document trace to a single script: [`bench/needle.py`](../bench/needle.py). If a number is here and not reproducible from that script, it is a bug — please file an issue.

## Methodology

Source: [`bench/needle.py`](../bench/needle.py) (single file, no external dependencies beyond `vecr`).

- **Haystack construction** — 120 filler sentences from `_FILLER_BANK` (20 plain-prose + 8 "decoy" sentences with digits/percentages, to prevent a digit-biased scorer from trivially winning). One needle is inserted at the `start`, `middle`, or `end` of the haystack (`build_haystack`).
- **Needles** — 11 total:
  - 7 **structured** needles (order ID, URL, ISO date, commit hash, customer email, config key, percentage). Each matches the retention whitelist.
  - 4 **stealth** needles (plain prose about migration / escalation / policy / architecture). These contain no retention-pattern features — only semantic relevance can save them.
  - Each needle carries a `survival_probe` regex anchored on a resilient token; the needle is counted as surviving iff the regex matches the joined compressed output.
- **Ratios swept** — `[1.0, 0.5, 0.3, 0.15, 0.08, 0.04]` (`main()` in `bench/needle.py`).
- **Configurations reported here** (2 of 3 in `bench/needle.py`):
  1. `baseline` — heuristic scorer, no retention whitelist, no question (`retain=False`).
  2. `+ L2 retention whitelist` — adds the structured-data regex pinning (`retain=True`).
  - The script still runs a third configuration that used to toggle L3 question-aware Jaccard; it now produces identical numbers to (2) because the default scorer ignores `question` (see Changelog 2026-04-22 P1.A). The third table is intentionally not reproduced below.
- **Total trials** = 11 needles × 3 positions × 6 ratios × 2 reported configs = **396 trials**.
- **Trials per ratio per config** = 11 needles × 3 positions = 33. Reported cell is the mean survival rate per needle across the 3 positions; the "overall" column is the mean across needles.
- **Fixtures are frozen** (`build_fixtures` seeds `random.seed(42)` and pre-generates one shared haystack per `(needle.topic, position)`) so both configs see identical filler for a fair comparison.
- **Offline** — no API keys, no model calls, no network.

## Results

Numbers below are taken from the output table printed by `bench/needle.py` on the fixture set described above. Each cell is the mean survival rate across 3 positions (`start` / `middle` / `end`).

### (1) Baseline — heuristic scorer, no retention, no question

| target ratio | actual | structured (7 needles) | stealth (4 needles) | overall |
|---:|---:|:---:|:---:|:---:|
| 1.00 | 1.00 | 100% | 100% | 100.0% |
| 0.50 | 0.50 | 100% | 50% (migration 100%, escalation 0%, policy 100%, architecture 0%) | 81.8% |
| 0.30 | 0.30 | 69% (each structured needle 67%) | 25% (migration 100%, others 0%) | 54.5% |
| 0.15 | 0.15 | 33% | 25% (migration 100%, others 0%) | 30.3% |
| 0.08 | 0.08 | 33% | 25% | 30.3% |
| 0.04 | 0.04 | 33% | 25% | 30.3% |

### (2) + L2 retention whitelist

| target ratio | actual | structured (7 needles) | stealth (4 needles) | overall |
|---:|---:|:---:|:---:|:---:|
| 1.00 | 1.00 | 100% | 100% | 100.0% |
| 0.50 | 0.50 | 100% | 50% | 81.8% |
| 0.30 | 0.30 | 100% | 25% (migration 100%, others 0%) | 72.7% |
| 0.15 | **0.16** (must-keep slightly exceeds target) | 100% | 25% | 72.7% |
| 0.08 | 0.15 | 100% | 25% | 72.7% |
| 0.04 | 0.15 | 100% | 25% | 72.7% |

Source: `bench/needle.py` → `eval_config(...)` output tables.

## Two findings from the data

1. **L2 turns structured-needle survival into a contract.** Baseline without retention loses 33–67% of structured needles once the budget gets tight (ratio ≤ 0.30); L2 keeps them at 100% across every ratio. The whitelist is the primary quality defense. Source: `bench/needle.py` STRUCTURED-only aggregate, all ratios.

2. **L2's visible cost: budget pinned to must-keep content squeezes stealth needles.** When the target ratio is below the natural must-keep floor (e.g., target 0.15 → actual 0.16), the retained structured content fills the budget and plain-prose stealth needles lose their budget slot. This is intentional and honest behaviour — under the current fixture only the "migration" stealth needle (which happens to carry date/number tokens the retention rules pin) survives reliably. Source: `bench/needle.py` `actual` column and stealth-needle breakdown at target ≤ 0.30.

## HotpotQA spike — where the synthetic bench hits its ceiling

Source: [`bench/hotpotqa_probe.py`](../bench/hotpotqa_probe.py) — a research-tier probe on N=100 HotpotQA `distractor` dev examples (243 supporting-fact sentences total). This is **not** a production bench (too small, no answerability measurement, no human eval); it exists to cross-check the conclusions drawn from the synthetic needle fixture.

**Headline: on real NL multi-hop QA, blended question-aware scoring materially uplifts supporting-fact survival. The synthetic needle bench did not see this because it saturates at 100% on structured needles.**

| config | target 0.5 survival | actual ratio |
|---|:---:|---:|
| heuristic only (no L2) | 53.1% | 0.50 |
| heuristic + L2 retention (default in v0.1.2) | 58.0% | 0.53 |
| heuristic + L2 + Jaccard blend (opt-in in v0.1.3) | **67.9%** | 0.53 |

Interpretation:

1. **L2 alone adds only +4.9pp** on HotpotQA prose. On structured-ID needles L2 was the primary defense (+0 → +30pp). On natural-language Wikipedia prose, the sentences that L2 pins are mostly sentences the heuristic would keep anyway (proper nouns, dates, high-entropy tokens).
2. **L3 Jaccard blend adds +9.9pp over L2** on HotpotQA — a genuine signal, not bench noise. The v0.1.2 decision to drop L3 as "no uplift" was measured entirely on synthetic structured-ID fixtures where the supporting content is already saturated at 100%; it did not generalize.
3. **Latency cost is acceptable**: p50 2.9 ms / p95 4.7 ms per 100-paragraph HotpotQA context.

Response in v0.1.3: `compress(..., use_question_relevance=True)` restores the 0.6-heuristic + 0.4-Jaccard blend as an opt-in. Default remains heuristic-only so the deterministic narrative is intact for structured workloads. Rule of thumb — pass `use_question_relevance=True` when the retrieved context is long NL prose and you have a real question; leave it off when the context is structured data (logs, records, code) and quality comes from L2 pinning.

Reproduce:

```bash
pip install -e .[bench]        # adds `datasets` for the HotpotQA loader
python -m bench.hotpotqa_probe
```

See `bench/hotpotqa_results.md` for the raw table and the 5-example synthetic fallback used when the HuggingFace download is unavailable.

## Reading a cell

Each percentage is the rate at which that specific needle survived across the 3 haystack positions at that ratio. `100%` = 3/3, `33%` = 1/3, `0%` = always lost. The "overall" column averages across all 11 needles.

## Reproducing

```bash
pip install -e .
python -m bench.needle
```

This runs the full sweep (3 configs × 6 ratios × 11 needles × 3 positions = 594 trials) in a few seconds. The third configuration is the legacy L3 slot and now produces identical numbers to (2); only the first two are reproduced in this document. No API keys required.

## vs LLMLingua-2 (pending)

We do not currently run a head-to-head against LLMLingua-2 in `bench/`. A head-to-head would require:

- Wiring `llmlingua` into `bench/` as an optional baseline.
- Running the same fixture set through it.
- Reporting token ratios and needle survival side-by-side.

**Status: pending future run.** No comparison number is cited here until the script lands — per this repo's rule that every benchmark number must be reproducible from a committed script.

## Limitations — honest

- **This is a synthetic fixture with only 11 needles.** The needle set is not diverse — your domain IDs may have different token distributions, casing, or affixes that interact differently with the scorer or retention regexes.
- **Fillers are English prose, synthetic.** Real workloads have different filler distributions. A workload dense with numeric data (logs, financial records) will exercise the scorer differently than this bench does.
- **String survival is not answer quality.** A model might paraphrase a needle correctly even without the exact text appearing; conversely, it might hallucinate around a surviving fact. A dedicated LLM-as-judge eval over gold answers is on the P1 list.
- **Stealth-needle recovery is weak.** Under L2, only the "migration" stealth needle (which carries incidentally-pinnable date/number tokens) survives reliably below target 0.5. Prose-only stealth recovery requires a custom scorer — the `question_relevance` helper remains exposed for callers who want to build one.
- **English-only.** The filler bank and retention regexes are English-first. Multilingual retention patterns (CJK numbers, full-width punctuation) are not yet tested.
- **Latency bench is synthetic too.** The numbers in the Latency section below come from synthesized conversations on a single machine (Apple M3 Max). Real workloads with very different sentence-length distributions or non-English text may tokenize differently; re-run `bench/latency.py` on your own hardware for a number you can trust.

## Latency

Wall-clock overhead of `compress()` on synthetic conversations. Source: [`bench/latency.py`](../bench/latency.py).

- `time.perf_counter`, 100 iters per trial, 1 warmup, single-threaded on CPU (no GPU / no network).
- Trials sweep message size (~500 / ~5k / ~50k input tokens) against budget ratio (50% — light compression; 10% — aggressive). Reported as median (p50) / 95th / 99th percentile / worst-case in milliseconds. Budget values in the table are absolute token counts derived from each synthesized conversation's actual token total (so the "5k" row is ~4.7k real tokens, the "50k" row is ~46.6k, etc. — see the `budget` column printed by the script).

| trial | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| 500t / budget 245     |   1.05 |   1.36 |   1.73 |   1.94 |
| 500t / budget 49      |   1.06 |   1.33 |   1.58 |   1.58 |
| 5kt / budget 2341     |  11.31 |  11.91 |  13.35 |  15.13 |
| 5kt / budget 468      |   9.56 |   9.86 |   9.99 |  10.11 |
| 50kt / budget 23302   | 118.10 | 123.48 | 124.85 | 125.78 |
| 50kt / budget 4660    |  93.71 | 100.61 | 102.39 | 102.47 |

**Reading**: for a 5k-token chat history compressed to ~10% budget, p95 overhead is ~10 ms — well under a typical LLM call's round-trip. For a 50k-token context, expect p95 ~100–125 ms; tiktoken tokenization dominates that range (one full pass to count input tokens, plus per-segment encoding during pruning). Aggressive (10%) budgets run faster than light (50%) budgets at larger sizes because the knapsack finishes earlier once the remaining budget runs out.

**Machine**: measured on Darwin arm64 / Apple M3 Max / 48 GB RAM (Python 3.x, tiktoken o200k_base). Re-run locally to get your own numbers:

```bash
pip install -e .
python -m bench.latency
```

## Changelog

- **2026-04-22 (v0.1.3)** — HotpotQA retention probe (`bench/hotpotqa_probe.py`) landed. On N=100 real dev examples at ratio 0.5, supporting-fact survival is 58.0% with L2 alone vs. 67.9% with L2 + Jaccard blend. This contradicts the v0.1.2 conclusion that Jaccard blending added "no uplift" — which was measured only on the saturated synthetic needle fixture. Response: restored the blend as an opt-in via `compress(use_question_relevance=True)`. Default remains heuristic-only.
- **2026-04-22 (P1.C)**: Added latency benchmark (`bench/latency.py`). Previously unverified "+20-60 ms" claim is now replaced with real p50/p95/p99 numbers for 3 message sizes × 2 budget ratios.
- **2026-04-22 (P1.A)** — Removed L3 question-aware Jaccard from the default scoring path; `question_relevance` remains an exposed helper for custom scorers. Bench now covers 2 reported configurations (baseline + L2) instead of 3, totalling 396 trials. The third configuration in `bench/needle.py` still runs but produces the same numbers as (2) because the default scorer ignores `question`. Previously-reported numbers that relied on the Jaccard blend (e.g. baseline 0.30 overall 93.9%, L2 0.30 overall 93.9%) have been updated to the true no-blend values (54.5% / 72.7%).
- **2026-04-22** — P0.B tightened the `fn-call`, `hash`, and `integer` retention regexes to reduce false-positive pinning (code-like identifiers only for `fn-call`; 2+ digits for `hash`; 4+ digits for `integer`). Structured-needle survival under L2 is unchanged at 100% across every ratio.

## Last updated

2026-04-22
