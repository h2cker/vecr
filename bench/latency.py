"""Latency benchmark for vecr-compress.

Measures wall-clock overhead of ``compress()`` across representative message
sizes and budget targets. No API keys; runs on a laptop in a few seconds.

    python -m bench.latency

Reported numbers are per-call wall-clock latency for a single invocation of
:func:`vecr_compress.compress` on a synthetic conversation. tiktoken (o200k
encoding) tokenization dominates for large payloads; the knapsack and
retention regex work are O(n) in sentences.

The bench is intentionally pure Python (``statistics`` only) so re-running it
on another machine needs nothing beyond ``pip install -e .``.
"""

from __future__ import annotations

import logging
import random
import statistics
import time
from dataclasses import dataclass

from vecr_compress import compress
from vecr_compress.tokens import count as tcount

random.seed(42)

# Under tight budget ratios the retention whitelist's must-keep set can
# exceed the budget — that's an expected, user-facing warning the library
# emits once per call. Silencing it here keeps the bench output clean; the
# behaviour under test is the latency path, not the warning path.
logging.getLogger("vecr_compress.compressor").setLevel(logging.ERROR)


# Filler sentences intentionally mirror ``bench/needle.py`` in spirit but are
# kept as a local copy so this bench does not couple to the needle fixtures.
_FILLER_BANK = [
    "The system provides high availability across multiple regions.",
    "Users appreciate the responsive interface and clear documentation.",
    "Performance has been consistently strong in production environments.",
    "The team meets weekly to discuss upcoming features and priorities.",
    "Customer feedback drives most of the roadmap decisions each quarter.",
    "Data is encrypted at rest using industry standard algorithms.",
    "The service scales horizontally by adding worker nodes to the cluster.",
    "Caching layers reduce load on the primary database during peak hours.",
    "Monitoring dashboards track latency percentiles and error rates.",
    "Deployment pipelines run automated tests before promoting to production.",
    "Backups are taken nightly and retained according to compliance policy.",
    "The vendor offers premium support with faster response time guarantees.",
    "Configuration changes are audited and versioned for rollback capability.",
    "Release cadence has increased since adopting continuous delivery practices.",
    "Engineers rotate through on-call shifts to handle production incidents.",
    "Most new users find the onboarding flow intuitive and self-service friendly.",
    "The design system ensures visual consistency across product surfaces.",
    "Internationalization support covers more than twenty locales today.",
    "Accessibility is treated as a requirement rather than a late-stage checklist.",
    "The knowledge base gets updated whenever a common support question recurs.",
    "Over 1,200 teams tried the preview last quarter with mixed feedback.",
    "Response times averaged 45 milliseconds, well inside the stated target.",
    "The offsite brought together 80 engineers for two days of planning.",
    "Morale surveys show a 4.2 rating on average for team collaboration.",
    "Meeting hours dropped by 15 percent after the no-meeting-Wednesday rule.",
    "Coffee consumption in the office exceeded 3,000 cups last month.",
    "The swag budget was raised to 75 dollars per new hire this year.",
    "About 300 attendees joined the webinar held on the third Thursday.",
]


def _structured_span(rng: random.Random) -> str:
    """Return a short string with a retention-pinned token (ID, citation, URL,
    email). Sprinkled into filler to make sure the retention whitelist has
    real work to do in every trial — otherwise the bench under-reports the
    regex cost for retention-heavy payloads.
    """
    kind = rng.randint(0, 4)
    if kind == 0:
        return f"Order ORD-{rng.randint(10000, 99999)}-{rng.randint(100, 999)} is pending."
    if kind == 1:
        return f"See reference [{rng.randint(1, 99)}] for the derivation."
    if kind == 2:
        return f"The endpoint https://api.example.com/v2/resource/{rng.randint(1000, 9999)} returns JSON."
    if kind == 3:
        return f"Contact user-{rng.randint(100, 9999)}@example.com for details."
    return f"Deploy at 2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T12:00:00Z completed."


def synthesize_conversation(total_tokens_target: int) -> list[dict]:
    """Build a synthetic conversation summing to roughly ``total_tokens_target``
    tokens across alternating user/assistant turns.

    Each turn is a paragraph of filler with an occasional structured span
    (~1 in 8 sentences) so the retention whitelist finds real work. The final
    message is always a short user question so :func:`compress` has a sensible
    tail to protect.
    """
    rng = random.Random(0xC0FFEE ^ total_tokens_target)
    # Budget turns of varying length — short/long/short/long/... — so the
    # segmentation stage sees realistic message-size variance rather than
    # a single uniform blob.
    turn_sizes: list[int] = []
    remaining = total_tokens_target
    toggle = True
    while remaining > 0:
        if toggle:
            chunk = min(remaining, rng.randint(150, 400))
        else:
            chunk = min(remaining, rng.randint(500, 1500))
        turn_sizes.append(chunk)
        remaining -= chunk
        toggle = not toggle

    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    for i, target in enumerate(turn_sizes):
        role = "user" if i % 2 == 0 else "assistant"
        parts: list[str] = []
        acc = 0
        while acc < target:
            if rng.randint(0, 7) == 0:
                sent = _structured_span(rng)
            else:
                sent = rng.choice(_FILLER_BANK)
            parts.append(sent)
            acc += tcount(sent) + 1  # +1 for the joining space
        messages.append({"role": role, "content": " ".join(parts)})

    # Final short user question — protect_tail keeps this verbatim.
    messages.append(
        {"role": "user", "content": "Given the above, what should I do next?"}
    )
    return messages


@dataclass
class Trial:
    label: str
    total_tokens: int
    budget: int
    samples_ms: list[float]


def run_one(messages: list[dict], budget: int, iters: int) -> list[float]:
    timings_ms: list[float] = []
    # Warm up once so tiktoken's LRU cache and any first-call overhead are not
    # charged to sample #1.
    compress(messages, budget_tokens=budget)
    for _ in range(iters):
        t0 = time.perf_counter()
        compress(messages, budget_tokens=budget)
        timings_ms.append((time.perf_counter() - t0) * 1000)
    return timings_ms


def _percentile(sorted_samples: list[float], p: float) -> float:
    """Nearest-rank percentile. ``p`` in [0, 1]."""
    if not sorted_samples:
        return float("nan")
    k = max(1, int(round(p * len(sorted_samples))))
    k = min(k, len(sorted_samples))
    return sorted_samples[k - 1]


def main() -> None:
    # 3 message-size targets × 2 budget ratios = 6 trials. The sizes span
    # short (~500 tok — single-turn Q&A), medium (~5k — mid chat), and long
    # (~50k — filled context window).
    size_targets = [500, 5_000, 50_000]
    budget_ratios = [0.5, 0.1]  # 50% budget = light compression; 10% = aggressive.
    iters = 100
    trials: list[Trial] = []

    for size in size_targets:
        msgs = synthesize_conversation(size)
        # Actual original tokens will be close-but-not-exactly ``size`` because
        # sentence boundaries are discrete; compute budget off the real total
        # so the ratio column lands where the label says.
        actual_tokens = sum(tcount(m["content"]) for m in msgs)
        for ratio in budget_ratios:
            budget = max(32, int(actual_tokens * ratio))
            label = f"{size // 1000 if size >= 1000 else size}{'k' if size >= 1000 else ''}t / budget {budget}"
            samples = run_one(msgs, budget, iters)
            trials.append(Trial(label, actual_tokens, budget, samples))

    print("=" * 80)
    print("vecr-compress latency benchmark")
    print(
        f"  Python time.perf_counter, {iters} iters after 1 warmup, p50/p95/p99 in ms"
    )
    print("=" * 80)
    print(
        f"{'trial':<28}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'max':>8}"
    )
    print("-" * 80)
    for t in trials:
        s = sorted(t.samples_ms)
        p50 = statistics.median(s)
        p95 = _percentile(s, 0.95)
        p99 = _percentile(s, 0.99)
        mx = max(s)
        print(f"{t.label:<28}  {p50:>7.2f}  {p95:>7.2f}  {p99:>7.2f}  {mx:>7.2f}")
    print("=" * 80)
    print(
        "Reading: p50/p95/p99 are per-call wall-clock ms for one compress()\n"
        "invocation on a synthetic conversation at the given token size and\n"
        "budget. tiktoken tokenization dominates at large sizes."
    )


if __name__ == "__main__":
    main()
