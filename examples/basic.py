"""Minimal vecr-compress usage. No framework, just plain dict messages.

Run: ``python examples/basic.py``
"""

from vecr_compress import compress

messages = [
    {"role": "system", "content": "You are a refund analyst."},
    {
        "role": "user",
        "content": (
            "Hello! Thanks for reaching out, we really appreciate it. "
            "The refund request references order ORD-99172 placed on 2026-03-15. "
            "The customer email is buyer@example.com for any follow-ups. "
            "We are reviewing it carefully and will get back to you. "
            "Totally agree this is important, thanks again for flagging. "
            "The total charge was $1,499.00 on card ending 4242. "
            "Let us know if there is anything else we can help with. "
            "Please also note the case number is CASE-2026-0042 for reference."
        ),
    },
    {"role": "user", "content": "What is the order ID and refund amount?"},
]

# Aggressive budget — forces most prose to be dropped, but retention-pinned
# facts (order ID, date, email, amount, case number) must survive.
result = compress(messages, target_ratio=0.3, protect_tail=1, protect_system=False)

print(f"Compressed {result.original_tokens} -> {result.compressed_tokens} tokens "
      f"(ratio {result.ratio:.2%}); dropped {len(result.dropped_segments)} segments")
for m in result.messages:
    content = m["content"] if isinstance(m["content"], str) else str(m["content"])
    print(f"  [{m['role']}] {content}")
