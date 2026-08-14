"""Generate a retrieval query set from the room labels present in a map.

Emits the ``queries.json`` consumed by ``evaluate_run.py --queries``: for
every COLD room class that actually appears on the map's nodes, a fixed set
of natural-language phrasings (bare room name, imperative sentence, object
description). Deriving the query set from the map itself keeps the protocol
honest and reproducible: every answerable room class is queried, none that
is absent from the map is, and the phrasings are fixed in code rather than
cherry-picked per run.

    python -m vts_evaluation.make_queries \
        --graph output/freiburg_a/final_graph.pkl \
        --out output/freiburg_a/queries.json
"""

from __future__ import annotations

import argparse
import json

from vts_core.topo_graph import TopoGraph

# Three fixed phrasings per COLD room class: bare noun phrase, imperative
# sentence, and an indirect/object-based description. Keys are the COLD
# place-annotation labels attached to nodes at mapping time.
_PHRASES: dict[str, list[str]] = {
    "1PO": [
        "the one-person office",
        "take me to the single-person office",
        "a small office for one person",
    ],
    "2PO": [
        "the two-people office",
        "go to the office shared by two people",
        "an office with two desks",
    ],
    "CNR": [
        "the conference room",
        "take me to the meeting room",
        "the room where meetings are held",
    ],
    "CR": [
        "the corridor",
        "bring me to the hallway",
        "a long corridor with doors",
    ],
    "KT": [
        "the kitchen",
        "take me to the kitchen",
        "the room with the fridge",
    ],
    "LO": [
        "the large office",
        "go to the big office",
        "a large office with many desks",
    ],
    "PA": [
        "the printer area",
        "I need to print something",
        "where the printers are",
    ],
    "RL": [
        "the robotics laboratory",
        "take me to the robotics lab",
        "the lab where the robots are",
    ],
    "ST": [
        "the stairs",
        "can you take me to the stairs",
        "the staircase area",
    ],
    "TL": [
        "the toilet",
        "take me to the bathroom",
        "the restroom",
    ],
    "TR": [
        "the terminal room",
        "go to the computers room",
        "the room full of computer terminals",
    ],
}


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="Path to a TopoGraph pickle")
    parser.add_argument("--out", required=True, help="Where to write queries.json")
    args: argparse.Namespace = parser.parse_args()

    graph: TopoGraph = TopoGraph.load(args.graph)
    labels: set[str] = {
        node.room_label
        for node in graph.nodes.values()
        if node.room_label is not None
    }
    unknown: set[str] = labels - _PHRASES.keys()
    if unknown:
        print(f"WARNING: no phrasings for map labels {sorted(unknown)}; skipped.")

    queries: list[dict[str, str]] = [
        {"query": phrase, "label": label}
        for label in sorted(labels & _PHRASES.keys())
        for phrase in _PHRASES[label]
    ]
    with open(args.out, "w") as f:
        json.dump(queries, f, indent=2)
    print(
        f"Wrote {len(queries)} queries for {len(labels & _PHRASES.keys())} "
        f"room classes to {args.out}"
    )


if __name__ == "__main__":
    main()
