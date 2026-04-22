"""LangChain adapter example.

Install the extra first:

    pip install vecr-compress[langchain]

Run: ``python examples/with_langchain.py``
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("langchain-core not installed — run: pip install vecr-compress[langchain]")
        # Exit 0 so example runs cleanly in CI without the optional dep.
        return 0

    from vecr_compress.adapters.langchain import VecrContextCompressor

    compressor = VecrContextCompressor(
        budget_tokens=120,
        protect_tail=1,
        protect_system=False,
    )
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(
            content=(
                "Hi! Thanks so much. "
                "The refund for order ORD-42 totaling $99.00 is pending. "
                "Sure thing, happy to help! "
                "Kafka partitions messages across brokers for horizontal scale. "
                "Raft uses randomized election timeouts to elect a leader."
            )
        ),
        HumanMessage(content="What is the order id and refund amount?"),
    ]

    compressed = compressor.compress_messages(messages)
    print("Compressed messages:")
    for m in compressed:
        print(f"  [{type(m).__name__}] {m.content}")

    report = compressor.compress_with_report(messages)
    print(f"\nReport: {report.original_tokens} -> {report.compressed_tokens} tokens "
          f"(ratio {report.ratio:.2%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
