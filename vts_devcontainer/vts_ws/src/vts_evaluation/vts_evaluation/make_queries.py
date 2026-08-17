"""Generate a fixed COLD text-to-node retrieval query set.

Emits the ``queries.json`` consumed by ``evaluate_run.py --queries``: for
every room class present in the sequence's ground-truth annotation file, a
fixed set of natural-language phrasings (bare room name, imperative sentence,
and indirect description). Crucially, classes are selected independently of
the generated map: if mapping misses a room entirely, its queries remain in
the benchmark and count as retrieval failures.

    python -m vts_evaluation.make_queries \
        --labels-file <sequence>/localization/places.lst \
        --out output/revised/freiburg_a/queries.json
"""

from __future__ import annotations

import argparse
import json

from vts_core.topo_graph import TopoGraph

_COLD_CLASSES: tuple[str, ...] = (
    "CR", "2PO", "RL", "TL", "TR", "LO", "1PO", "KT", "CNR", "PA", "LAB", "ST",
)

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


def _canonical_label(raw: str) -> str:
    """Normalize instance annotations such as ``2PO1-A`` to ``2PO``."""
    raw = raw.strip()
    for class_name in _COLD_CLASSES:
        if class_name in raw:
            return class_name
    return raw.split("-")[0].rstrip("0123456789")


def _labels_from_annotations(path: str) -> set[str]:
    """Read all room classes encountered in a COLD ``places.lst`` file."""
    labels: set[str] = set()
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels.add(_canonical_label(parts[1]))
    return labels


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labels-file",
        help="COLD places.lst; preferred because it is independent of map output",
    )
    source.add_argument(
        "--graph",
        help="Legacy fallback: derive classes from labels surviving in a map",
    )
    parser.add_argument("--out", required=True, help="Where to write queries.json")
    args: argparse.Namespace = parser.parse_args()

    if args.labels_file:
        labels = _labels_from_annotations(args.labels_file)
        source_description = "sequence annotations"
    else:
        graph: TopoGraph = TopoGraph.load(args.graph)
        labels = {
            node.room_label
            for node in graph.nodes.values()
            if node.room_label is not None
        }
        source_description = "generated map (legacy fallback)"
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
        f"room classes from {source_description} to {args.out}"
    )


if __name__ == "__main__":
    main()
