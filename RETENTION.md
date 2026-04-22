# Retention Contract

vecr-compress guarantees that content matching any of the built-in retention rules will be preserved through compression, regardless of the token budget. If total must-keep content exceeds the budget, the compressor returns all must-keep segments and logs a warning — it never silently drops structured data.

The contract is deterministic and auditable: the rules are plain Python regex patterns you can read, test, and extend.

## Built-in rules

13 rules ship with v0.1.0. Pattern order matters — specific structural patterns run before generic numerics, so `ORD-42819` is classified as `code-id` rather than `integer`.

| Name | Regex (simplified) | Example match | Rationale |
|---|---|---|---|
| `uuid` | `[0-9a-fA-F]{8}-...-[0-9a-fA-F]{12}` | `3f6e4b1a-23cd-4e5f-9012-abcdef012345` | Trace/session/correlation IDs |
| `date` | `\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}...)?` | `2026-03-15`, `2026-03-15T09:30:00` | Timestamps, deadlines, audit trails |
| `code-id` | `[A-Z][A-Z0-9]{1,}[-_#]?\d+...` | `ORD-99172`, `INV_2024_A`, `CUST#42` | Order, invoice, customer identifiers |
| `email` | `[\w.+-]+@[\w-]+\.[\w.-]+` | `buyer@example.com` | Contact records, PII audit |
| `url` | `https?://\S+` | `https://api.example.com/v2/orders` | Endpoints, evidence links, sources |
| `path` | `([A-Za-z]:)?/[\w\-./]+\.[A-Za-z0-9]{1,6}` | `/var/log/app/error.log` | File references, error locations |
| `code-span` | `` `[^`\n]+` `` | `` `raise ValueError(msg)` `` | Inline code in prose |
| `fn-call` | `[a-zA-Z_]\w*\([^)]{0,80}\)` | `process_refund(order_id, amount)` | Function references in review |
| `citation` | `\[(\d{1,3}\|[A-Z][a-zA-Z]+ \d{4})\]` | `[12]`, `[Smith 2023]` | Academic and legal citations |
| `json-kv` | `"[\w_]+"\s*:\s*"[^"]{1,120}"` | `"status": "pending_review"` | Structured payload fields |
| `hash` | `[0-9a-f]{8,}` (word boundary) | `a3f9b2c1d4e5f678` | Git SHAs, content digests |
| `number` | `[\$€£¥]?-?\d{1,3}([,\.]\d+)+[%kKmMbB]?` | `$1,499.00`, `12.4%`, `v3.2.1` | Amounts, rates, version strings |
| `integer` | `\d{2,}` (not adjacent to letters) | `4242`, `99172` | Reference numbers, IDs |

Full regex source: `src/vecr_compress/retention.py`.

## Extending the contract

Use `DEFAULT_RULES.with_extra(...)` to append custom rules. Extra rules run after the built-ins so user patterns cannot shadow the stricter structural matches.

```python
import re
from vecr_compress import compress, RetentionRule, DEFAULT_RULES

# Pin all 6-digit invoice numbers in your workload
invoice_rule = RetentionRule(
    name="invoice",
    pattern=re.compile(r"\bINV-\d{6}\b"),
)

custom_rules = DEFAULT_RULES.with_extra([invoice_rule])

result = compress(messages, budget_tokens=2000, rules=custom_rules)
```

You can also build a completely custom rule set from scratch:

```python
from vecr_compress import RetentionRule, RetentionRules

my_rules = RetentionRules([
    RetentionRule("ticket", re.compile(r"\bTICKET-\d{4,8}\b")),
    RetentionRule("sha256", re.compile(r"\b[0-9a-f]{64}\b")),
])
```

## What the contract does NOT guarantee

- **Paraphrased versions of structured data.** If the text says "the order ID mentioned above" rather than `ORD-99172`, the literal match never fires. Only exact pattern matches are pinned.
- **Low-structure content.** Prose sentences that contain no retention-matching token go through the heuristic scorer and may be dropped if they are low-signal. The default scorer does not use the `question` argument (see [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — question-Jaccard showed no uplift); to re-enable question-aware blending, pass a custom `scorer` callable and reuse `vecr_compress.scorer.question_relevance` as a helper.
- **Semantic meaning.** Dropping a prose sentence that provides necessary context for a pinned fact will not be caught by retention rules. For prose recovery beyond what retention + the heuristic scorer deliver, implement a custom scorer (the `question_relevance` helper is exposed for exactly this extension point).

## Testing your rules

Use `is_pinned()` and `retention_reason()` to verify a rule fires on intended inputs:

```python
from vecr_compress import is_pinned, retention_reason, RetentionRule, DEFAULT_RULES
import re

invoice_rule = RetentionRule(name="invoice", pattern=re.compile(r"\bINV-\d{6}\b"))
rules = DEFAULT_RULES.with_extra([invoice_rule])

# Should be True
assert is_pinned("Payment for INV-004821 is overdue.", rules)
assert retention_reason("INV-004821", rules) == "invoice"

# Should be False (5 digits, not 6)
assert not is_pinned("Reference INV-0482.", rules)
```

You can also test the built-in rules directly:

```python
from vecr_compress import is_pinned, retention_reason

assert is_pinned("Order ORD-99172 placed 2026-03-15")
assert retention_reason("ORD-99172") == "code-id"
assert retention_reason("2026-03-15") == "date"
assert retention_reason("buyer@example.com") == "email"
assert not is_pinned("Hello, thanks for reaching out.")
```

## Failure modes

**Over-retention:** your rules keep more than the budget allows. The compressor will log a warning (`WARNING: must-keep content exceeds budget`) and return all pinned segments, potentially exceeding the requested token count. Raise the budget or tighten your rules.

**Under-retention:** a new data format appears in your workload that no rule matches — a new ID scheme, a non-standard date format, a custom hash type. Add a custom rule via `with_extra()` to cover it. Monitor for `dropped_segments` in `CompressResult` to catch gaps.

**Regex catastrophic backtracking:** built-in rules use O(n) patterns with no nested quantifiers. If you write your own rules, test them for ReDoS before deploying — a slow regex on a large message can block the event loop or cause timeouts. Tools like `redos-detector` or the `re` module's timeout support can help.

## Changelog of built-in rules

| Version | Change |
|---|---|
| v0.1.0 | Initial 13 rules: `uuid`, `date`, `code-id`, `email`, `url`, `path`, `code-span`, `fn-call`, `citation`, `json-kv`, `hash`, `number`, `integer` |
