"""NeedleInHaystack benchmark for vecr-compress.

Measures **fact survival rate** — does the critical needle text still appear
in the compressed output? This is the quantitative answer to "does the
retention primitive actually preserve what you care about?"

Sweeps:
  - Compression target ratio (1.0, 0.5, 0.3, 0.15, 0.08, 0.04)
  - Three configurations:
      (1) baseline — heuristic scorer, no retention, no question
      (2) + retention whitelist (L2)
      (3) + retention + question-aware scoring (L2 + L3)
  - Needle position (start / middle / end)

No API keys. Runs in a few seconds on a laptop.

    python -m bench.needle
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from vecr_compress import compress

random.seed(42)


# Filler sentences that read like real distractor content but carry no
# retention-pattern matches. Keeps the benchmark honest — the retention
# whitelist can't cheat by pinning distractors.
_FILLER_BANK = [
    # Plain-prose fillers — no structural retention features.
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
    # "Decoy" fillers — contain digits / numbers that a naive scorer loves
    # but which carry no real information. Strips the digit-bias advantage
    # from baseline so the retention guarantee has to stand on its own.
    "Over 1,200 teams tried the preview last quarter with mixed feedback.",
    "Response times averaged 45 milliseconds, well inside the stated target.",
    "The offsite brought together 80 engineers for two days of planning.",
    "Morale surveys show a 4.2 rating on average for team collaboration.",
    "Meeting hours dropped by 15 percent after the no-meeting-Wednesday rule.",
    "Coffee consumption in the office exceeded 3,000 cups last month.",
    "The swag budget was raised to 75 dollars per new hire this year.",
    "About 300 attendees joined the webinar held on the third Thursday.",
]


@dataclass
class Needle:
    topic: str
    fact_text: str
    question: str
    survival_probe: re.Pattern[str]


# Structured needles — each contains a retention-pattern match (ID, URL,
# date, hash, email, code-ish token, percentage). Baseline can drop these;
# L2 must keep them.
STRUCTURED = [
    Needle(
        topic="order id",
        fact_text="The refund is authorized for order ORD-99172 totaling $1,499.00.",
        question="What order ID was refunded and for how much?",
        survival_probe=re.compile(r"ORD-99172.*\$1,499\.00|\$1,499\.00.*ORD-99172", re.S),
    ),
    Needle(
        topic="server address",
        fact_text="The primary API endpoint is https://api.vecr.ai/v1/completions.",
        question="What is the primary API endpoint URL?",
        survival_probe=re.compile(r"https://api\.vecr\.ai/v1/completions"),
    ),
    Needle(
        topic="date",
        fact_text="The incident began at 2026-04-12T03:17:00Z and lasted 42 minutes.",
        question="When did the incident start?",
        survival_probe=re.compile(r"2026-04-12T03:17:00Z"),
    ),
    Needle(
        topic="commit hash",
        fact_text="The regression was introduced in commit 9f3ab2c4e18d7a6b by the platform team.",
        question="Which commit introduced the regression?",
        survival_probe=re.compile(r"9f3ab2c4e18d7a6b"),
    ),
    Needle(
        topic="customer email",
        fact_text="Reach out to buyer-3392@example.com for confirmation before proceeding.",
        question="Which email should be contacted for confirmation?",
        survival_probe=re.compile(r"buyer-3392@example\.com"),
    ),
    Needle(
        topic="config key",
        fact_text='The service reads the flag `max_retries: 7` at startup from config.',
        question="What value is set for the max retries configuration?",
        survival_probe=re.compile(r"max_retries:\s*7|`max_retries: 7`"),
    ),
    Needle(
        topic="percentage",
        fact_text="Conversion improved by 23.7% after the June 2026 checkout redesign.",
        question="By what percent did conversion improve?",
        survival_probe=re.compile(r"23\.7%"),
    ),
]

# Stealth needles — distinctive semantic content but NO retention-pattern
# matches. Success depends entirely on the scorer finding them via question
# relevance, not retention pinning.
STEALTH = [
    Needle(
        topic="migration",
        fact_text=(
            "Legacy billing accounts will migrate to the new ledger service during the "
            "overnight deploy window."
        ),
        question="When will legacy billing accounts migrate to the new ledger service?",
        survival_probe=re.compile(r"legacy billing.*migrate|migrate.*legacy billing", re.I),
    ),
    Needle(
        topic="escalation",
        fact_text=(
            "Escalations for refund disputes should go directly to the customer success lead "
            "rather than the generic support queue."
        ),
        question="Who handles escalations for refund disputes?",
        survival_probe=re.compile(r"customer success lead", re.I),
    ),
    Needle(
        topic="policy",
        fact_text=(
            "Internal policy forbids sharing customer contact details with third-party "
            "marketing affiliates under any circumstances."
        ),
        question="What does internal policy say about sharing customer contact details?",
        survival_probe=re.compile(r"third-party marketing affiliates", re.I),
    ),
    Needle(
        topic="architecture",
        fact_text=(
            "Every new microservice must register itself with the central service catalog "
            "before accepting production traffic."
        ),
        question="What must a new microservice do before accepting production traffic?",
        survival_probe=re.compile(r"service catalog", re.I),
    ),
]

NEEDLES = STRUCTURED + STEALTH


def build_haystack(needle: Needle, position: str, n_filler: int = 120) -> str:
    fillers = [random.choice(_FILLER_BANK) for _ in range(n_filler)]
    if position == "start":
        insert_at = 1
    elif position == "middle":
        insert_at = n_filler // 2
    else:
        insert_at = n_filler - 2
    fillers.insert(insert_at, needle.fact_text)
    return " ".join(fillers)


@dataclass
class Config:
    label: str
    retain: bool
    question_aware: bool


def build_fixtures(positions: list[str]) -> dict[tuple[str, str], str]:
    """One shared haystack per (needle.topic, position), used for every config.

    Without this, each config's trials consume different random state and end
    up with different fillers — making the config comparison apples-to-
    oranges. We pre-generate once and freeze.
    """
    random.seed(42)
    return {(n.topic, pos): build_haystack(n, pos) for n in NEEDLES for pos in positions}


def run_trial(
    needle: Needle,
    haystack: str,
    ratio: float,
    cfg: Config,
) -> tuple[bool, float]:
    messages = [
        {"role": "system", "content": "You answer strictly from the provided context."},
        {"role": "user", "content": haystack},
        {"role": "user", "content": needle.question},
    ]
    result = compress(
        messages,
        target_ratio=ratio,
        question=needle.question if cfg.question_aware else None,
        retain=cfg.retain,
        protect_tail=1,
        protect_system=False,
    )
    rendered = " ".join(
        m["content"] if isinstance(m["content"], str) else str(m["content"])
        for m in result.messages
    )
    survived = bool(needle.survival_probe.search(rendered))
    return survived, result.ratio


def eval_config(
    cfg: Config, ratios: list[float], fixtures: dict[tuple[str, str], str]
) -> None:
    positions = ["start", "middle", "end"]
    print(f"\n{cfg.label}")
    print("-" * 110)
    print(
        f"{'ratio':>6}  {'actual':>7}  "
        + "  ".join(f"{n.topic[:10]:>10}" for n in NEEDLES)
        + "   overall"
    )
    for ratio in ratios:
        per_needle_survival: list[float] = []
        actual_ratios: list[float] = []
        for needle in NEEDLES:
            hits = 0
            for pos in positions:
                hay = fixtures[(needle.topic, pos)]
                survived, actual = run_trial(needle, hay, ratio, cfg)
                if survived:
                    hits += 1
                actual_ratios.append(actual)
            per_needle_survival.append(hits / len(positions))

        overall = sum(per_needle_survival) / len(per_needle_survival)
        actual_avg = sum(actual_ratios) / len(actual_ratios)
        cells = "  ".join(f"{v * 100:>9.0f}%" for v in per_needle_survival)
        print(f"{ratio:>6.2f}  {actual_avg:>6.2f}   {cells}   {overall * 100:>5.1f}%")


def main() -> None:
    print("=" * 110)
    print("vecr-compress NeedleInHaystack benchmark")
    print(f"  structured needles: {len(STRUCTURED)} (with IDs/numbers/URLs)")
    print(f"  stealth needles:    {len(STEALTH)} (plain prose, no retention features)")
    print("  positions: start/middle/end   fillers per trial: 120")
    print("=" * 110)

    ratios = [1.0, 0.5, 0.3, 0.15, 0.08, 0.04]

    configs = [
        Config(
            label="(1) baseline — heuristic scorer, no retention, no question",
            retain=False,
            question_aware=False,
        ),
        Config(
            label="(2) + retention whitelist (L2)",
            retain=True,
            question_aware=False,
        ),
        Config(
            label="(3) + retention + question-aware scoring (L2 + L3)",
            retain=True,
            question_aware=True,
        ),
    ]
    fixtures = build_fixtures(["start", "middle", "end"])
    for cfg in configs:
        eval_config(cfg, ratios, fixtures)

    print("\n" + "=" * 110)
    print("Reading the table:")
    print("  - Each cell: share of trials (start/middle/end positions) where the")
    print(f"    needle survived compression at that ratio. {len(NEEDLES)} needles total.")
    print("  - 'overall': mean across all needles.")
    print("  - Trust metric: structured needles should stay at 100% under L2 at every ratio.")
    print("  - Quality metric: stealth 'overall' under L3 should clearly beat baseline at")
    print("    mid-range ratios (0.3-0.5). Extreme ratios (0.04) have no budget to recover.")
    print("=" * 110)


if __name__ == "__main__":
    main()
