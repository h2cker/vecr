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
- **Configurations**:
  1. `baseline` — heuristic scorer, no retention whitelist, no question (`retain=False`).
  2. `+ L2 retention whitelist` — adds the structured-data regex pinning (`retain=True`).
  3. `+ L2 + L3 question-aware` — adds Jaccard overlap between the needle's `question` and each candidate sentence.
- **Total trials** = 11 needles × 3 positions × 6 ratios × 3 configs = **594 trials**.
- **Trials per ratio per config** = 11 needles × 3 positions = 33. Reported cell is the mean survival rate per needle across the 3 positions; the "overall" column is the mean across needles.
- **Fixtures are frozen** (`build_fixtures` seeds `random.seed(42)` and pre-generates one shared haystack per `(needle.topic, position)`) so all three configs see identical filler for a fair comparison.
- **Offline** — no API keys, no model calls, no network.

## Results

Numbers below are taken from the output table printed by `bench/needle.py` on the fixture set described above. Each cell is the mean survival rate across 3 positions (`start` / `middle` / `end`).

### (1) Baseline — heuristic scorer, no retention, no question

| target ratio | actual | structured (7 needles) | stealth (4 needles) | overall |
|---:|---:|:---:|:---:|:---:|
| 1.00 | 1.00 | 100% | 100% | 100.0% |
| 0.50 | 0.50 | 100% | 100% | 100.0% |
| 0.30 | 0.30 | 100% | 83% (migration 33%, others 100%) | 93.9% |
| 0.15 | 0.15 | 100% | 75% (migration 0%, others 100%) | 90.9% |
| 0.08 | 0.08 | 100% | 75% | 90.9% |
| 0.04 | 0.04 | 100% | 75% | 90.9% |

### (2) + L2 retention whitelist

| target ratio | actual | structured (7 needles) | stealth (4 needles) | overall |
|---:|---:|:---:|:---:|:---:|
| 1.00 | 1.00 | 100% | 100% | 100.0% |
| 0.50 | 0.50 | 100% | 100% | 100.0% |
| 0.30 | **0.36** (must-keep exceeds target) | 100% | 16.5% (migration 33%, escalation 0%, policy 0%, architecture 33%) | 78.8% |
| 0.15 | 0.36 | 100% | 0% | 72.7% |
| 0.08 | 0.36 | 100% | 0% | 72.7% |
| 0.04 | 0.36 | 100% | 0% | 72.7% |

### (3) + L2 + L3 question-aware

| target ratio | actual | structured (7 needles) | stealth (4 needles) | overall |
|---:|---:|:---:|:---:|:---:|
| All ratios | same as L2 | same as L2 | same as L2 | same as L2 |

L3 (question-aware Jaccard blend at 0.4 weight) gives **no additional improvement over L2** in this synthetic bench. The stealth-needle recovery that was present in earlier versions of the algorithm has regressed post-extraction. This is documented as a known gap — see the Limitations section below.

Source: `bench/needle.py` → `eval_config(...)` output tables.

## Three findings from the data

1. **In this synthetic fixture, the heuristic scorer alone keeps all structured tokens at every ratio tested, down to 0.04.** Baseline reaches 100% structured survival without any retention whitelist. Source: structured-needle aggregate from `bench/needle.py`.

2. **L2 provides a deterministic contract, not a statistical uplift here.** The scorer's 100% structured survival is a measurement over this specific synthetic fixture; L2 makes it a contract that holds across any workload, scorer, or distribution — if `ORD-\d+` appears in the input, it will appear in the output, unconditionally. Source: `bench/needle.py` STRUCTURED-only aggregate, all ratios.

3. **L2's visible cost: budget pinned to must-keep content squeezes stealth needles.** When target ratio < actual required ratio (e.g., target 0.30 → actual 0.36), the retained structured content fills the budget and plain-prose stealth needles lose their budget slot. This is intentional and honest behaviour. Source: `bench/needle.py` `actual` column and `actual ratio` overshoot at target ≤ 0.30.

## Reading a cell

Each percentage is the rate at which that specific needle survived across the 3 haystack positions at that ratio. `100%` = 3/3, `33%` = 1/3, `0%` = always lost. The "overall" column averages across all 11 needles.

## Reproducing

```bash
pip install -e .
python -m bench.needle
```

This runs the full sweep (3 configs × 6 ratios × 11 needles × 3 positions = 594 trials) in a few seconds. No API keys required.

## vs LLMLingua-2 (pending)

We do not currently run a head-to-head against LLMLingua-2 in `bench/`. A head-to-head would require:

- Wiring `llmlingua` into `bench/` as an optional baseline.
- Running the same 594-trial fixture set through it.
- Reporting token ratios and needle survival side-by-side.

**Status: pending future run.** No comparison number is cited here until the script lands — per this repo's rule that every benchmark number must be reproducible from a committed script.

## Limitations — honest

- **This is a synthetic fixture with only 11 needles.** The needle set is not diverse — your domain IDs may have different token distributions, casing, or affixes that interact differently with the scorer or retention regexes.
- **Fillers are English prose, synthetic.** Real workloads have different filler distributions. A workload dense with numeric data (logs, financial records) will exercise the scorer differently than this bench does.
- **String survival is not answer quality.** A model might paraphrase a needle correctly even without the exact text appearing; conversely, it might hallucinate around a surviving fact. A dedicated LLM-as-judge eval over gold answers is on the P1 list.
- **L3 question-aware Jaccard shows no uplift in this bench.** The Jaccard blend at 0.4 weight does not overcome the filler-vs-needle gap for stealth needles. Future work: raise the Jaccard weight, or replace Jaccard with embedding similarity (e.g. `all-MiniLM-L6-v2`). The "+24 pp stealth uplift from question-aware" claim made in earlier versions of this document was based on a different version of the algorithm and is no longer accurate.
- **English-only.** The filler bank and retention regexes are English-first. Multilingual retention patterns (CJK numbers, full-width punctuation) are not yet tested.
- **No latency numbers in this doc.** Benchmark is correctness-only. Median added-latency numbers belong in a separate perf bench (not yet written).

## Last updated

2026-04-22
