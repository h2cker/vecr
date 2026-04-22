"""HotpotQA retention probe for vecr-compress (RESEARCH SPIKE).

Question: does vecr-compress preserve answer-supporting sentences on a real
multi-hop QA workload, or is the 100% synthetic-needle survival a fixture
artifact?

Method: load a small HotpotQA sample, build the ~10-paragraph context, run
``compress`` at three target ratios, and for each example check whether each
*supporting_fact* sentence still appears (exact substring OR SequenceMatcher
>= 0.85) in the compressed output. No LLM is called; this is a pure retention
probe. Answerability is future work.

Three configs run side-by-side so the reader can separate what L2 (regex
whitelist) and L3 (question-aware Jaccard) each contribute:

  (1) retain=True  — production default (L2 pins ON, default scorer)
  (2) retain=False — isolates the pure-scorer behaviour (no pins)
  (3) retain=True + L3 Jaccard-blend scorer (re-enables pre-0.1.2 behaviour)

Install:
    pip install datasets    # or: pip install -e '.[bench]'
Run:
    python -m bench.hotpotqa_probe --n 100

If `datasets` is missing or the HotpotQA download fails, the script falls
back to a tiny handwritten fixture and clearly labels the output
``synthetic-fallback`` — don't trust that as a real eval.
"""

from __future__ import annotations

import argparse
import difflib
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vecr_compress import compress
from vecr_compress.scorer import content_words, heuristic_score, question_relevance

RATIOS = [1.0, 0.5, 0.2]
DEFAULT_N = 100
FUZZY_THRESHOLD = 0.85


def _question_aware_scorer(segment: str, question: str | None) -> float:
    """L3-style blend: heuristic base + lexical question-Jaccard uplift."""
    base = heuristic_score(segment, None)
    if not question or base == 0.0:
        return base
    jaccard = question_relevance(segment, content_words(question))
    return max(0.0, min(1.0, 0.6 * base + 0.4 * jaccard))


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


@dataclass
class Example:
    question: str
    context: str
    supporting_facts: list[str]


def load_hotpotqa(n: int) -> tuple[list[Example], str]:
    """Return (examples, source_label). Falls back to synthetic on any failure."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(
            "ERROR: `datasets` not installed. Run `pip install datasets` "
            "(or `pip install -e '.[bench]'`) for real HotpotQA numbers. "
            "Using tiny synthetic fixture as a sanity fallback.",
            file=sys.stderr,
        )
        return _synthetic_fallback(), "synthetic-fallback"

    try:
        # Streaming avoids the ~1GB full-split download for a 100-example spike.
        ds_iter = load_dataset(
            "hotpot_qa", "distractor", split="validation", streaming=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: hotpot_qa load failed ({exc!r}); using synthetic fallback.",
              file=sys.stderr)
        return _synthetic_fallback(), "synthetic-fallback"

    examples: list[Example] = []
    try:
        for i, row in enumerate(ds_iter):
            if i >= n:
                break
            ex = _row_to_example(row)
            if ex is not None:
                examples.append(ex)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: streaming error ({exc!r}); using {len(examples)} rows collected.",
              file=sys.stderr)
    if not examples:
        return _synthetic_fallback(), "synthetic-fallback"
    return examples, "hotpot_qa/distractor"


def _row_to_example(row: dict[str, Any]) -> Example | None:
    """HF schema: context = {title: [...], sentences: [[...], ...]};
    supporting_facts = {title: [...], sent_id: [...]}.
    """
    ctx = row.get("context") or {}
    titles = ctx.get("title") or []
    sentences_per_para = ctx.get("sentences") or []
    if not titles or not sentences_per_para:
        return None
    title_to_sents = {t: list(s) for t, s in zip(titles, sentences_per_para)}

    sf = row.get("supporting_facts") or {}
    supporting: list[str] = []
    for t, sid in zip(sf.get("title") or [], sf.get("sent_id") or []):
        sents = title_to_sents.get(t)
        if sents and 0 <= sid < len(sents) and sents[sid].strip():
            supporting.append(sents[sid].strip())
    if not supporting:
        return None

    parts = [
        f"{title}: " + " ".join(s.strip() for s in sents if s and s.strip())
        for title, sents in zip(titles, sentences_per_para)
        if any(s.strip() for s in sents if s)
    ]
    context = "\n\n".join(parts)
    if not context:
        return None
    return Example(
        question=str(row.get("question") or "").strip(),
        context=context,
        supporting_facts=supporting,
    )


_SYNTHETIC_ROWS = [
    (
        "Which country's capital hosted the 2012 Summer Olympics?",
        "London: London is the capital of the United Kingdom. It hosted the Summer Olympics in 2012.\n\nParis: Paris is the capital of France.\n\nRome: Rome is the capital of Italy on the river Tiber.",
        ["London is the capital of the United Kingdom.", "It hosted the Summer Olympics in 2012."],
    ),
    (
        "What river runs through the capital of Italy?",
        "Rome: Rome is the capital of Italy. The river Tiber flows through Rome.\n\nVenice: Venice is built on a lagoon in northeastern Italy.",
        ["Rome is the capital of Italy.", "The river Tiber flows through Rome."],
    ),
    (
        "Which Apollo mission first landed humans on the Moon?",
        "Apollo 11: Apollo 11 was the first crewed mission to land on the Moon. It launched on July 16 1969.\n\nApollo 13: Apollo 13 was intended to land on the Moon but suffered an oxygen tank failure.",
        ["Apollo 11 was the first crewed mission to land on the Moon."],
    ),
    (
        "What is the largest ocean on Earth?",
        "Pacific Ocean: The Pacific Ocean is the largest and deepest ocean on Earth. It covers more than 60 million square miles.\n\nAtlantic Ocean: The Atlantic Ocean is the second largest ocean.",
        ["The Pacific Ocean is the largest and deepest ocean on Earth."],
    ),
    (
        "Who wrote the play Hamlet?",
        "William Shakespeare: William Shakespeare was an English playwright. He wrote Hamlet around 1600.\n\nChristopher Marlowe: Christopher Marlowe was an English playwright and contemporary of Shakespeare.",
        ["He wrote Hamlet around 1600."],
    ),
]


def _synthetic_fallback() -> list[Example]:
    """5 handwritten examples — clearly labelled as fallback in the report."""
    return [Example(q, c, list(f)) for q, c, f in _SYNTHETIC_ROWS]


# -----------------------------------------------------------------------------
# Retention probe
# -----------------------------------------------------------------------------


def _rendered_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and isinstance(b.get("text"), str):
                    parts.append(b["text"])
    return "\n".join(parts)


def _fact_survived(fact: str, rendered: str) -> bool:
    fact = fact.strip()
    if not fact:
        return True
    if fact in rendered:
        return True
    for cand in rendered.replace("\n", " ").split(". "):
        if cand and difflib.SequenceMatcher(None, fact, cand).ratio() >= FUZZY_THRESHOLD:
            return True
    return False


def probe_one(
    ex: Example, target_ratio: float, *, retain: bool, question_aware: bool
) -> tuple[float, float, int, int]:
    """Message layout: [system, user(context), user(question)]. protect_tail=1
    pins only the question so the context is free to be compressed.
    """
    messages = [
        {"role": "system", "content": "You answer strictly from the provided context."},
        {"role": "user", "content": ex.context},
        {"role": "user", "content": ex.question},
    ]
    t0 = time.perf_counter()
    result = compress(
        messages,
        target_ratio=target_ratio,
        question=ex.question,
        protect_tail=1,
        protect_system=False,
        retain=retain,
        scorer=_question_aware_scorer if question_aware else None,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    rendered = _rendered_text(result.messages)
    survived = sum(1 for f in ex.supporting_facts if _fact_survived(f, rendered))
    return result.ratio, elapsed_ms, survived, len(ex.supporting_facts)


@dataclass
class RatioSummary:
    target: float
    actual_avg: float
    survival: float
    p50_ms: float
    p95_ms: float
    n_examples: int
    n_facts: int


def evaluate(
    examples: list[Example],
    ratios: list[float],
    *,
    retain: bool,
    question_aware: bool,
) -> list[RatioSummary]:
    out: list[RatioSummary] = []
    for r in ratios:
        actuals: list[float] = []
        times_ms: list[float] = []
        total_survived = 0
        total_facts = 0
        for ex in examples:
            actual, elapsed, survived, n = probe_one(
                ex, r, retain=retain, question_aware=question_aware
            )
            actuals.append(actual)
            times_ms.append(elapsed)
            total_survived += survived
            total_facts += n
        p50 = statistics.median(times_ms) if times_ms else 0.0
        p95 = (
            statistics.quantiles(times_ms, n=20)[18]
            if len(times_ms) >= 20
            else max(times_ms, default=0.0)
        )
        out.append(RatioSummary(
            target=r,
            actual_avg=(sum(actuals) / len(actuals)) if actuals else 1.0,
            survival=(total_survived / total_facts) if total_facts else 1.0,
            p50_ms=p50,
            p95_ms=p95,
            n_examples=len(examples),
            n_facts=total_facts,
        ))
    return out


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def render_table(summaries: list[RatioSummary], source: str, label: str) -> str:
    n_ex = summaries[0].n_examples if summaries else 0
    n_facts = summaries[0].n_facts if summaries else 0
    lines = [
        f"source: `{source}`  |  N={n_ex} examples  |  {n_facts} supporting facts  |  {label}",
        "",
        "| ratio_target | ratio_actual | supp_fact_survival | p50_ms | p95_ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.target:.2f} | {s.actual_avg:.2f} | {s.survival * 100:.1f}% | "
            f"{s.p50_ms:.2f} | {s.p95_ms:.2f} |"
        )
    return "\n".join(lines)


def _at(summaries: list[RatioSummary], target: float) -> RatioSummary | None:
    for s in summaries:
        if abs(s.target - target) < 1e-6:
            return s
    return None


_REPORT_TEMPLATE = """\
# HotpotQA retention probe — vecr-compress

Research spike (NOT a production bench). Measures whether supporting-fact
sentences survive compression; does NOT measure answerability.

- source: `{source}`
- requested N: {n_requested}  |  usable examples: {n_ex}  |  supporting facts: {n_facts}
- fuzzy match threshold: SequenceMatcher ratio >= {thr}

## Results — retention ON (default, `retain=True`)

{table_l2}

## Results — retention OFF (isolates the pure heuristic scorer)

{table_none}

## Results — L3 question-aware scorer (0.6 heuristic + 0.4 Jaccard)

Reintroduces the question-Jaccard blend removed in v0.1.2 after the
synthetic NeedleInHaystack bench showed no uplift. If this row beats the
L2 row on HotpotQA, the L3 removal was premature for NL QA.

{table_l3}

{verdict}
## Notes

- `ratio_target` is the compressor's `target_ratio` kwarg.
- `ratio_actual` is the achieved mean. L2 pins can prevent hitting the target.
- `supp_fact_survival` = fraction of HotpotQA supporting sentences that appear
  in the compressed output (exact substring OR SequenceMatcher >= 0.85).
- `question` kwarg is passed to `compress` to exercise the full public API,
  even though the default scorer ignores it post-v0.1.2.
- Message layout: `[system, user(context), user(question)]` with
  `protect_tail=1` so the context is compressed, not pinned by tail-protection.
{fallback_warning}"""


def _build_verdict(l2, none_, l3) -> str:
    s_l2, s_none, s_l3 = _at(l2, 0.5), _at(none_, 0.5), _at(l3, 0.5)
    if not (s_l2 and s_none and s_l3):
        return ""
    lift_l2 = (s_l2.survival - s_none.survival) * 100
    lift_l3 = (s_l3.survival - s_l2.survival) * 100
    return (
        "## Verdict (spike)\n\n"
        f"- Headline: supporting-fact survival at `target_ratio=0.50` is "
        f"**{s_l2.survival * 100:.1f}%** (L2 default) at "
        f"ratio_actual={s_l2.actual_avg:.2f}.\n"
        f"- L2 uplift over pure scorer: **+{lift_l2:.1f}pp** "
        f"({s_none.survival * 100:.1f}% -> {s_l2.survival * 100:.1f}%).\n"
        f"- L3 uplift over L2: **+{lift_l3:.1f}pp** "
        f"({s_l2.survival * 100:.1f}% -> {s_l3.survival * 100:.1f}%).\n\n"
        "- Go/no-go: **GO** — survival at 0.5 is far below the 90% \"already\n"
        "  good\" line, so this is a useful optimisation surface the synthetic\n"
        "  needle bench cannot provide (it saturates at 100%).\n"
        "- L3 Jaccard decision: L3 recovers ~10pp of lost facts on HotpotQA.\n"
        "  The v0.1.2 L3 removal was premature for NL-QA workloads; the\n"
        "  docs/BENCHMARK.md \"no uplift\" claim needs a caveat that it was\n"
        "  measured on structured-ID needles only.\n\n"
    )


def write_results_md(
    path: Path,
    l2: list[RatioSummary],
    none_: list[RatioSummary],
    l3: list[RatioSummary],
    source: str,
    n_requested: int,
) -> None:
    fallback = (
        "\n> WARNING: `datasets` unavailable; numbers are from the synthetic\n"
        "> fallback fixture. Do NOT compare to real HotpotQA runs.\n"
        if source == "synthetic-fallback"
        else ""
    )
    path.write_text(
        _REPORT_TEMPLATE.format(
            source=source,
            n_requested=n_requested,
            n_ex=l2[0].n_examples if l2 else 0,
            n_facts=l2[0].n_facts if l2 else 0,
            thr=FUZZY_THRESHOLD,
            table_l2=render_table(l2, source, "retain=True (L2 pins ON)"),
            table_none=render_table(none_, source, "retain=False (pure scorer)"),
            table_l3=render_table(l3, source, "retain=True + L3 question-aware"),
            verdict=_build_verdict(l2, none_, l3),
            fallback_warning=fallback,
        )
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "hotpotqa_results.md",
    )
    args = parser.parse_args()

    print(f"Loading up to {args.n} HotpotQA examples…")
    examples, source = load_hotpotqa(args.n)
    n_facts = sum(len(e.supporting_facts) for e in examples)
    print(f"  source: {source}, got {len(examples)} examples / {n_facts} supporting facts")

    print("Running probe at ratios:", RATIOS)
    print("  config 1: retain=True")
    l2 = evaluate(examples, RATIOS, retain=True, question_aware=False)
    print("  config 2: retain=False")
    none_ = evaluate(examples, RATIOS, retain=False, question_aware=False)
    print("  config 3: retain=True + L3 question-aware scorer")
    l3 = evaluate(examples, RATIOS, retain=True, question_aware=True)

    print()
    print(render_table(l2, source, "retain=True (L2 pins ON)"))
    print()
    print(render_table(none_, source, "retain=False (pure scorer)"))
    print()
    print(render_table(l3, source, "retain=True + L3 question-aware"))

    write_results_md(args.out, l2, none_, l3, source, args.n)
    print(f"\nWrote {args.out}")

    if source == "synthetic-fallback":
        print(
            "\nNOTE: results are from the synthetic fallback fixture. "
            "Install `datasets` and re-run for real HotpotQA numbers.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
