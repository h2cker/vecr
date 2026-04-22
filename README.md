# vecr-compress

English | [中文](README.zh-CN.md)

**Auditable, deterministic context compression for LLMs.** Structured data — order IDs, URLs, dates, citations, code — survives compression by an explicit regex whitelist you can inspect, extend, and audit. Every pin and every drop is logged: you get a `retained_matches` list and a `dropped_segments` list on every call.

[![PyPI version](https://img.shields.io/pypi/v/vecr-compress)](https://pypi.org/project/vecr-compress/)
[![Python versions](https://img.shields.io/pypi/pyversions/vecr-compress)](https://pypi.org/project/vecr-compress/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#benchmark-reproducible)
[![Downloads](https://img.shields.io/pypi/dm/vecr-compress)](https://pypi.org/project/vecr-compress/)

## Why this exists

A 2026 Factory.ai production study found that "artifact tracking" (IDs, file paths, error codes) is the worst-compressed category across every compressor tested — scoring just 2.19–2.45 out of 5.0, worse even than OpenAI's native compaction (3.43/5.0). No shipped library offers a deterministic retention primitive: they all rely on LLM judgment or learned scoring that can silently drop a customer ID, a transaction amount, or a compliance citation. vecr-compress solves exactly that gap. It does not claim the highest compression ratio — that is Compresr's lane. It offers an auditable, extensible whitelist-based compressor you can reason about end-to-end.

## 30-second example

```python
from vecr_compress import compress

messages = [
    {"role": "system", "content": "You are a refund analyst."},
    {"role": "user", "content":
        "Hello! Thanks for reaching out. "
        "The refund request references order ORD-99172 placed on 2026-03-15. "
        "The customer email is buyer@example.com. "
        "We are reviewing it carefully. "
        "Totally agree this is important. "
        "The total charge was $1,499.00 on card ending 4242."},
    {"role": "user", "content": "What is the order ID and refund amount?"},
]

result = compress(messages, budget_tokens=80)

for m in result.messages:
    print(m["role"], "->", m["content"])

print(f"\n{result.original_tokens} -> {result.compressed_tokens} tokens "
      f"({result.ratio:.2%}); pinned {len(result.retained_matches)} facts")
```

Every structured fact in the input — `ORD-99172`, `2026-03-15`, `buyer@example.com`, `$1,499.00` — survives, because each is pinned by the retention whitelist before the knapsack budget packing runs. Filler phrases like "Hello! Thanks for reaching out" and "Totally agree this is important" are dropped.

## The retention contract

vecr-compress ships 13 built-in rules. Any segment containing a match is **pinned** — kept regardless of token budget. If total pinned content exceeds the budget, the compressor returns all pinned segments and logs a warning rather than silently dropping facts.

| Pattern | Example match | Why it matters |
|---|---|---|
| `uuid` | `3f6e4b1a-23cd-4e5f-9012-abcdef012345` | Trace IDs, session IDs, correlation keys |
| `date` | `2026-03-15`, `2026-03-15T09:30:00` | Deadlines, timestamps, audit trails |
| `code-id` | `ORD-99172`, `INV_2024_A`, `CUST#42` | Order, invoice, customer identifiers |
| `email` | `buyer@example.com` | Contact records, PII audit |
| `url` | `https://api.example.com/v2/orders` | Endpoints, evidence links, sources |
| `path` | `/var/log/app/error.log`, `C:/data/report.csv` | File references, error locations |
| `code-span` | `` `raise ValueError(msg)` `` | Inline code in prose |
| `fn-call` | `process_refund(order_id, amount)`, `obj.method(a, b)` (code-like identifiers only) | Function references in code review |
| `citation` | `[12]`, `[Smith 2023]` | Academic and legal citations |
| `json-kv` | `"status": "pending_review"` | Structured payload fields |
| `hash` | `9f3ab2c4` (8+ hex chars, 2+ digits) | Git SHAs, content digests |
| `number` | `$1,499.00`, `12.4%`, `v3.2.1` | Amounts, rates, version strings |
| `integer` | `9172`, `99172`, `2026` (4+ digits) | IDs, reference numbers, years |

Extend the contract with your own rules:

```python
from vecr_compress import compress, RetentionRule, DEFAULT_RULES

custom_rules = DEFAULT_RULES.with_extra([
    RetentionRule(name="invoice", pattern=re.compile(r"INV-\d{6}")),
])
result = compress(messages, budget_tokens=2000, rules=custom_rules)
```

Details on testing and extending rules: see [RETENTION.md](RETENTION.md).

## Benchmark (reproducible)

Needle-in-haystack survival: 11 needles × 3 positions × 6 ratios × 3 configs = 594 trials (`bench/needle.py`).

**Structured needles (7) — baseline vs. L2 retention**

| ratio | baseline | + L2 retention |
|---:|:---:|:---:|
| 1.00 | 100% | 100% |
| 0.50 | 100% | 100% |
| 0.30 | 100% | 100% |
| 0.15 | 100% | 100% |
| 0.08 | 100% | 100% |
| 0.04 | 100% | 100% |

The baseline heuristic scorer keeps all structured tokens in this synthetic fixture. L2 turns that measurement into a **deterministic contract** — the same 100% holds across any workload, scorer, or distribution, not just this fixture. If `ORD-\d+` appears in the input, it will appear in the output.

**Stealth needles (4, plain prose) — where the tradeoff shows**

| ratio | baseline | + L2 retention |
|---:|:---:|:---:|
| 1.00 | 100% | 100% |
| 0.50 | 100% | 100% |
| 0.30 | 83% | 83% |
| 0.15 | 75% | 67% |
| 0.08 | 75% | 0% |
| 0.04 | 75% | 0% |

L2's cost: must-keep structured content pins the budget, leaving little room for plain-prose stealth needles at aggressive ratios (target 0.15 → actual 0.16 because the whitelist overrides the budget). On natural-language QA (HotpotQA probe, N=100) a blended question-aware scorer adds **+9.9pp supporting-fact survival** at ratio 0.5 over L2 alone — opt in with `compress(..., use_question_relevance=True)` (v0.1.3+). Off by default so the deterministic contract stays loud; worth turning on when your context is long prose and you have a real question. See [docs/BENCHMARK.md](docs/BENCHMARK.md) for details. Notes: filler detection was tightened in v0.1.1 to only drop whole-segment greetings, so prose starting with "please" / "thanks" is no longer discarded; the 2026-04-22 P0.B pass tightened `fn-call` / `hash` / `integer` regexes to reduce false-positive pinning, which lifted stealth survival at 0.30 / 0.15 without any regression in structured-needle survival (still 100% at every ratio).

Note: actual compression ratio may exceed the target when must-keep content is large — this is intentional and honest behaviour, not a bug.

To reproduce:

```bash
pip install -e .
python -m bench.needle
```

## Install

```bash
pip install vecr-compress                  # core only (requires tiktoken)
pip install vecr-compress[langchain]       # + LangChain adapter
pip install vecr-compress[llamaindex]      # + LlamaIndex adapter
```

Requires Python 3.10+.

## LangChain / LlamaIndex

Framework adapters are opt-in via extras (`[langchain]`, `[llamaindex]`). Core compression has no framework dependency.

**LangChain** — compress a chat history before passing it to any chat model:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from vecr_compress.adapters.langchain import VecrContextCompressor

compressor = VecrContextCompressor(budget_tokens=2000)
compressed = compressor.compress_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Long conversation history..."),
    HumanMessage(content="The actual question."),
])
```

**LlamaIndex** — postprocess retrieved nodes before synthesis:

```python
from llama_index.core.schema import NodeWithScore, TextNode
from vecr_compress.adapters.llamaindex import VecrNodePostprocessor

processor = VecrNodePostprocessor(budget_tokens=1500)
kept = processor.postprocess_nodes(nodes, query_str="the user's question")
```

## How it works (30-second tour)

Two layers applied in order:

1. **Retention whitelist** — segments matching any built-in rule are pinned and bypass the budget knapsack entirely.
2. **Heuristic packing** — remaining segments are scored by entropy and structural signal (digits, braces, capitalization); filler lines like `Hi!`, `Thanks!`, `As an AI…` score 0.0 and are dropped before any budget math; the rest are packed greedily into the token budget.

Callers can provide a custom `ScorerFn` to re-enable question-aware blending — `scorer.question_relevance` remains exported as a Jaccard helper. See [RETENTION.md](RETENTION.md) for details.

## vs. alternatives

| | Approach | Open source | Retention contract |
|---|---|---|---|
| **Compresr (YC W26)** | LLM summarization, hosted model | No | None — JSON atomic treatment is planned |
| **LLMLingua-2** | Probabilistic token classifier | Yes | None |
| **LangChain DeepAgents compact** | Autonomous agent-triggered | Yes (LangChain) | None |
| **Provider-native compaction** (OpenAI/Google) | Opaque, single-provider | No | None |
| **vecr-compress** | Regex whitelist + heuristic knapsack | Yes | **Deterministic, auditable** |

Choose Compresr for maximum compression ratio. Choose LLMLingua-2 for pure-Python research. Choose vecr-compress when you want an auditable, extensible whitelist-based compressor you can reason about end-to-end, and you can live with v0.1 limits (Python-only, sentence-level granularity, no streaming).

## What this does NOT do

- **No streaming.** `compress()` is synchronous and one-shot.
- **No tool-call rewriting.** `tool_use` / `tool_result` blocks pass through verbatim — safe, zero gain.
- **Sentence-level granularity only.** No token-level pruning or learned rewrites.
- **English-tuned.** Stopword list and regex patterns are English-first. Multilingual quality is untested.
- **No embedding scorer.** Jaccard overlap is lexical. Semantic relevance scoring lives in the reference gateway.

## Contributing / License / Links

Apache 2.0. Contributions welcome via the main repo.

- Main repo: [https://github.com/h2cker/vecr](https://github.com/h2cker/vecr)
- Issues: [https://github.com/h2cker/vecr/issues](https://github.com/h2cker/vecr/issues)
- Retention contract details: [RETENTION.md](RETENTION.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
