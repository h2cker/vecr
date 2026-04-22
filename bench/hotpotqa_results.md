# HotpotQA retention probe — vecr-compress

Research spike (NOT a production bench). Measures whether supporting-fact
sentences survive compression; does NOT measure answerability.

- source: `hotpot_qa/distractor`
- requested N: 100  |  usable examples: 100  |  supporting facts: 243
- fuzzy match threshold: SequenceMatcher ratio >= 0.85

## Results — retention ON (default, `retain=True`)

source: `hotpot_qa/distractor`  |  N=100 examples  |  243 supporting facts  |  retain=True (L2 pins ON)

| ratio_target | ratio_actual | supp_fact_survival | p50_ms | p95_ms |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 100.0% | 2.19 | 3.94 |
| 0.50 | 0.53 | 58.0% | 2.86 | 4.42 |
| 0.20 | 0.43 | 43.6% | 2.69 | 4.31 |

## Results — retention OFF (isolates the pure heuristic scorer)

source: `hotpot_qa/distractor`  |  N=100 examples  |  243 supporting facts  |  retain=False (pure scorer)

| ratio_target | ratio_actual | supp_fact_survival | p50_ms | p95_ms |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 100.0% | 1.43 | 2.41 |
| 0.50 | 0.50 | 53.1% | 2.28 | 3.72 |
| 0.20 | 0.20 | 21.4% | 2.03 | 3.36 |

## Results — L3 question-aware scorer (0.6 heuristic + 0.4 Jaccard)

Reintroduces the question-Jaccard blend removed in v0.1.2 after the
synthetic NeedleInHaystack bench showed no uplift. If this row beats the
L2 row on HotpotQA, the L3 removal was premature for NL QA.

source: `hotpot_qa/distractor`  |  N=100 examples  |  243 supporting facts  |  retain=True + L3 question-aware

| ratio_target | ratio_actual | supp_fact_survival | p50_ms | p95_ms |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 100.0% | 2.20 | 3.53 |
| 0.50 | 0.53 | 67.9% | 3.16 | 5.18 |
| 0.20 | 0.43 | 46.5% | 2.69 | 4.28 |

## Verdict (spike)

- Headline: supporting-fact survival at `target_ratio=0.50` is **58.0%** (L2 default) at ratio_actual=0.53.
- L2 uplift over pure scorer: **+4.9pp** (53.1% -> 58.0%).
- L3 uplift over L2: **+9.9pp** (58.0% -> 67.9%).

- Go/no-go: **GO** — survival at 0.5 is far below the 90% "already
  good" line, so this is a useful optimisation surface the synthetic
  needle bench cannot provide (it saturates at 100%).
- L3 Jaccard decision: L3 recovers ~10pp of lost facts on HotpotQA.
  The v0.1.2 L3 removal was premature for NL-QA workloads; the
  docs/BENCHMARK.md "no uplift" claim needs a caveat that it was
  measured on structured-ID needles only.


## Notes

- `ratio_target` is the compressor's `target_ratio` kwarg.
- `ratio_actual` is the achieved mean. L2 pins can prevent hitting the target.
- `supp_fact_survival` = fraction of HotpotQA supporting sentences that appear
  in the compressed output (exact substring OR SequenceMatcher >= 0.85).
- `question` kwarg is passed to `compress` to exercise the full public API,
  even though the default scorer ignores it post-v0.1.2.
- Message layout: `[system, user(context), user(question)]` with
  `protect_tail=1` so the context is compressed, not pinned by tail-protection.
