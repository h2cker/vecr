"""LlamaIndex adapter example.

Install the extra first:

    pip install vecr-compress[llamaindex]

Run: ``python examples/with_llamaindex.py``
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from llama_index.core.schema import NodeWithScore, TextNode
    except ImportError:
        print("llama-index-core not installed — run: pip install vecr-compress[llamaindex]")
        # Exit 0 so example runs cleanly in CI without the optional dep.
        return 0

    from vecr_compress.adapters.llamaindex import VecrNodePostprocessor

    nodes = [
        NodeWithScore(
            node=TextNode(text="Kafka partitions messages across brokers for horizontal scale."),
            score=0.8,
        ),
        NodeWithScore(
            node=TextNode(
                text="Raft uses randomized election timeouts to elect a leader. "
                "Raft leaders replicate entries via AppendEntries."
            ),
            score=0.7,
        ),
        NodeWithScore(
            node=TextNode(text="Cassandra uses gossip for membership and consistent hashing."),
            score=0.6,
        ),
    ]

    processor = VecrNodePostprocessor(budget_tokens=40)
    kept = processor.postprocess_nodes(nodes, query_str="How does Raft elect a leader?")
    print(f"Kept {len(kept)} / {len(nodes)} nodes:")
    for n in kept:
        text = n.node.text if hasattr(n, "node") else n.text
        print(f"  - {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
