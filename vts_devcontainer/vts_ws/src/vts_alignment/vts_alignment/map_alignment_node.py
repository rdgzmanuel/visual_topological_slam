"""Map alignment ROS2 node.

Listens to ``/mapping/graph_ready`` events; when the event marked
``all_done`` arrives, loads every announced graph (any number, recommended
<= 5), fuses them with the order-invariant :class:`MultiMapAligner`, saves
``final_graph.pkl`` and announces it on ``/alignment/map_ready``. Can also be
run on pre-existing graph files via the ``graph_paths`` parameter (the old
"_start_directly" debugging path, now a first-class option).
"""

from __future__ import annotations

import json
import os
from contextlib import suppress

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vts_core.alignment import MultiMapAligner
from vts_core.topo_graph import TopoGraph


class MapAlignmentNode(Node):
    """Fuses N per-run graphs into a single topological map."""

    def __init__(self) -> None:
        super().__init__("map_alignment")

        self.declare_parameter("output_dir", "output/graphs")
        self.declare_parameter("graph_paths", [""])

        self._output_dir: str = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        preset: list[str] = [
            p
            for p in self.get_parameter("graph_paths")
            .get_parameter_value()
            .string_array_value
            if p
        ]

        self._collected: list[str] = []
        self._aligner: MultiMapAligner = MultiMapAligner()

        self.create_subscription(String, "/mapping/graph_ready", self._on_ready, 10)
        self._map_pub = self.create_publisher(String, "/alignment/map_ready", 10)

        if preset:
            self._collected = preset
            self._align_and_publish()

    def _on_ready(self, msg: String) -> None:
        event: dict[str, object] = json.loads(msg.data)
        graph_path: str = str(event["graph_path"])
        self._collected.append(graph_path)
        self.get_logger().info(
            f"Collected {len(self._collected)} graph(s): {graph_path}"
        )
        if bool(event.get("all_done", False)):
            self._align_and_publish()

    def _align_and_publish(self) -> None:
        graphs: list[TopoGraph] = [TopoGraph.load(p) for p in self._collected]
        if len(graphs) == 1:
            fused: TopoGraph = graphs[0]
            self.get_logger().info("Single graph received; nothing to align.")
        else:
            self.get_logger().info(f"Aligning {len(graphs)} graphs...")
            fused = self._aligner.align(graphs)

        os.makedirs(self._output_dir, exist_ok=True)
        final_path: str = os.path.join(self._output_dir, "final_graph.pkl")
        fused.save(final_path)

        out: String = String()
        out.data = json.dumps(
            {
                "final_graph_path": final_path,
                "n_nodes": len(fused.nodes),
                "n_edges": len(fused.edges()),
            }
        )
        self._map_pub.publish(out)
        self.get_logger().info(f"Final map ready: {out.data}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: MapAlignmentNode = MapAlignmentNode()
    with suppress(KeyboardInterrupt):
        rclpy.spin(node)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
