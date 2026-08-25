"""Dataset-agnostic online visual topological mapper.

Consumes synchronized (image, pose, covariance) samples — wherever they come
from (a real robot's odometry or a dataset player) — and incrementally builds
a :class:`TopoGraph`.

Design (deliberately simplified to a single, reliable run):

- Node creation: a node is emitted retrospectively at each confirmed valley of
  the visual stream's windowed algebraic connectivity. Frames received during
  the detector's confirmation latency are carried into the following segment,
  and the final segment is flushed explicitly at end-of-run.
- Revisit / loop closure: interval-relative odometry uncertainty generates and
  ranks non-local candidates first. A unique compatible place is accepted
  directly; when odometry is ambiguous, a bidirectional three-node sequence
  compares training-free von Mises--Fisher place distributions traversed in
  either direction.
- Every emitted keyframe remains a distinct pose variable. Sequential motion
  creates an odometry factor; an accepted revisit creates a separate
  probabilistic co-location factor. Live odometry and covariance are never
  overwritten or reset by place recognition.
- End-of-run SE(2) optimization refines a copy of the accumulated pose graph;
  it never feeds corrections back into the live mapping state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vts_core.directional import fit_von_mises_fisher, vmf_log_overlap_ratio
from vts_core.matching import gaussian_position_nll, mahalanobis_gate
from vts_core.node_detection import AdaptiveValleyDetector, ConnectivityMonitor
from vts_core.pose_graph import OptimizationResult, optimize_se2
from vts_core.topo_graph import (
    LOOP_TEMPORAL_EXCLUSION_NODES,
    LoopClosureEvent,
    Pose2D,
    TopoGraph,
    TopoNode,
)

# Numerical floor applied to reported odometry covariance before inversion.
# It prevents an unrealistically exact factor when a driver publishes zeros.
_MIN_EDGE_SIGMA: float = 0.05
_PLACE_LOCALIZATION_SIGMA: float = 0.5

# Visual revisit gate. A loop closure is accepted only if the candidate's vMF
# overlap evidence is a robust *outlier* among candidates that remain
# plausible after the independent context/geometric checks: at least the
# median plus k robust standard deviations
# (k * 1.4826 * MAD), with k the default below (tunable per run via the
# ``visual_outlier_k`` constructor argument). This is the same MAD-relative
# criterion used by the valley detector, applied to visual
# similarity. It is what stops the "pollution snowball": a place that is
# similar to *many* nodes (e.g. a generic corridor) produces no clear outlier
# and is rejected. LOWER k -> easier loop closure, RAISE it -> stricter.
_VISUAL_OUTLIER_K: float = 2.0
# Below this many nodes there is not enough evidence to judge an outlier;
# stay conservative and do not close loops yet.
_MIN_NODES_FOR_REVISIT: int = 5
# Existing nodes this close in temporal order are traversal neighbours, not
# loop-closure candidates. This structural exclusion adds no metric threshold.

# Revisit gate ablation modes. "both" is the full system (visual outlier test
# AND geometric gate must agree); "visual" and "geometric" disable one gate
# each; "threshold" replaces the whole dual gate with a naive absolute
# cosine-similarity threshold — the baseline the dual gate is argued against.
GATE_MODES: tuple[str, ...] = ("both", "visual", "geometric", "threshold")
VISUAL_MODELS: tuple[str, ...] = ("vmf", "cosine")
# Absolute cosine-similarity threshold used by the "threshold" ablation mode.
_NAIVE_THRESHOLD: float = 0.7


@dataclass
class FrameRecord:
    """A buffered frame: descriptor, pose, covariance snapshot and image.

    ``gt_pose`` is an optional ground-truth pose carried through for
    evaluation only; the mapping algorithm never reads it.
    """

    descriptor: np.ndarray
    pose: Pose2D
    covariance: np.ndarray
    image: np.ndarray
    gt_pose: Pose2D | None = None
    room_label: str | None = None
    frame_index: int = 0


class TopologicalMapper:
    """Incremental topological mapper for a single run."""

    def __init__(
        self,
        window_size: int = 30,
        frame_id: str = "run",
        valley_k: float = 1.5,
        visual_outlier_k: float = _VISUAL_OUTLIER_K,
        optimize: bool = True,
        optimizer_backend: str = "gtsam",
        gate_mode: str = "both",
        naive_threshold: float = _NAIVE_THRESHOLD,
        visual_model: str = "vmf",
    ) -> None:
        """
        Args:
            window_size: Sliding window size for the connectivity monitor.
                Larger = smoother lambda_2 = fewer, coarser nodes.
            frame_id: Identifier of this run's odometry frame.
            valley_k: Sensitivity of node creation. The valley detector fires
                when lambda_2 rises ``valley_k`` robust deviations above its
                running minimum, so a LARGER value yields FEWER, more separated
                nodes (raise it if nodes cluster too densely).
            visual_outlier_k: Strictness of the visual revisit gate (robust
                MAD multiplier). LOWER it to close loops more aggressively, RAISE for
                stricter loop closures. See ``_VISUAL_OUTLIER_K``.
            optimize: Run the end-of-run SE(2) pose-graph optimization in
                :meth:`finalize` (corrects accumulated odometry drift in the
                node poses).
            optimizer_backend: ``gtsam`` for production or ``numpy`` for the
                dependency-free reference solver.
            gate_mode: Revisit-gate ablation mode, one of :data:`GATE_MODES`.
                ``both`` (default) is the odometry-first dual gate;
                ``visual`` / ``geometric`` keep only one gate; ``threshold``
                is the naive absolute-similarity baseline.
            naive_threshold: Cosine-similarity threshold used when
                ``gate_mode == "threshold"``.
            visual_model: Evidence model used to resolve ambiguous geometric
                candidates: ``vmf`` for the final mapper or ``cosine`` for
                controlled external descriptor baselines.
        """
        if gate_mode not in GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}"
            )
        if visual_model not in VISUAL_MODELS:
            raise ValueError(
                f"visual_model must be one of {VISUAL_MODELS}, "
                f"got {visual_model!r}"
            )
        self.graph: TopoGraph = TopoGraph(frame_id=frame_id)
        self.current_node_id: int | None = None
        self._visual_outlier_k: float = visual_outlier_k
        self._optimize: bool = optimize
        self._optimizer_backend: str = optimizer_backend
        self._gate_mode: str = gate_mode
        self._naive_threshold: float = naive_threshold
        self._visual_model = visual_model

        self._monitor: ConnectivityMonitor = ConnectivityMonitor(window_size)
        self._detector: AdaptiveValleyDetector = AdaptiveValleyDetector(k=valley_k)
        self._frame_count: int = 0
        self._pending_frames: list[FrameRecord] = []

        # Odometry pose and covariance at each emitted node. Edge measurements
        # and information matrices are built from these unmodified inputs.
        self._last_visit_odom: dict[int, Pose2D] = {}
        self._last_visit_covariance: dict[int, np.ndarray] = {}
        # Wall-clock instrumentation for map maintenance and loop search.
        self._update_time_s: float = 0.0
        self._revisit_search_time_s: float = 0.0
        self._revisit_search_count: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        image_bgr: np.ndarray,
        descriptor: np.ndarray,
        pose: Pose2D,
        covariance: np.ndarray,
        gt_pose: Pose2D | None = None,
        room_label: str | None = None,
    ) -> int | None:
        """Process one synchronized sample.

        Thin timing wrapper around :meth:`_process_frame_impl` so the full
        pipeline cost can be reported separately from descriptor extraction.

        Args:
            image_bgr: Camera image.
            descriptor: L2-normalized place descriptor of the image.
            pose: Odometry pose (x, y, theta) at capture time.
            covariance: 3x3 odometry covariance at capture time.
            gt_pose: Optional ground-truth pose at capture time, carried
                through for evaluation only (never used to map). When given,
                the node inherits the ground-truth pose of its representative
                (medoid) frame, so placement error is measured against the
                exact frame that fixed the node's pose.
            room_label: Optional evaluation-only place label for this frame.

        Returns:
            The id of a node created/updated by this frame, or None.
        """
        start: float = time.perf_counter()
        try:
            return self._process_frame_impl(
                image_bgr, descriptor, pose, covariance, gt_pose, room_label
            )
        finally:
            self._update_time_s += time.perf_counter() - start

    def _process_frame_impl(
        self,
        image_bgr: np.ndarray,
        descriptor: np.ndarray,
        pose: Pose2D,
        covariance: np.ndarray,
        gt_pose: Pose2D | None = None,
        room_label: str | None = None,
    ) -> int | None:
        record: FrameRecord = FrameRecord(
            descriptor=descriptor.astype(np.float32),
            pose=pose,
            covariance=covariance.astype(np.float64).copy(),
            image=image_bgr,
            gt_pose=gt_pose,
            room_label=room_label,
            frame_index=self._frame_count,
        )
        self._pending_frames.append(record)
        self._frame_count += 1
        lambda_2: float | None = self._monitor.update(record.descriptor)
        if lambda_2 is None:
            return None

        valley_index: int | None = self._detector.step(lambda_2)
        if valley_index is None:
            return None

        # Connectivity sample zero is produced by camera frame one, so the
        # detector's lambda-series index maps to camera index + 1.
        boundary_frame_index = valley_index + 1
        split = 0
        while (
            split < len(self._pending_frames)
            and self._pending_frames[split].frame_index <= boundary_frame_index
        ):
            split += 1
        if split == 0:
            return None
        segment = self._pending_frames[:split]
        self._pending_frames = self._pending_frames[split:]
        return self._create_node(segment)

    def finalize_nodes(self) -> int | None:
        """Flush the final visual segment exactly once before graph saving."""
        if not self._pending_frames:
            return None
        segment = self._pending_frames
        self._pending_frames = []
        return self._create_node(segment)

    def optimize_graph(self) -> OptimizationResult:
        """Optimize the completed graph without modifying the live trajectory."""
        if not self._optimize:
            return OptimizationResult(0.0, 0.0, 0, True, "disabled")
        return optimize_se2(self.graph, backend=self._optimizer_backend)

    def finalize(self) -> OptimizationResult:
        """End-of-run hook. Optimize the pose graph (unless ``optimize=False``).

        The live odometry trajectory is never modified during mapping. The
        end-of-run solver uses the stored odometry and loop factors only;
        disable via ``optimize=False`` to retain raw odometry poses.

        Returns:
            Optimization diagnostics.
        """
        self.finalize_nodes()
        return self.optimize_graph()

    def performance_stats(self) -> dict[str, float]:
        """Computational-cost summary for map maintenance.

        Times are wall-clock and exclude descriptor extraction (the encoder's
        cost, incurred by the caller). ``map_update_time_ms`` is the mean
        per-frame map-maintenance cost; ``loop_closure_time_ms`` is the mean
        cost of one revisit (loop-closure) search at node creation.
        """
        frames: int = max(self._frame_count, 1)
        searches: int = max(self._revisit_search_count, 1)
        latencies = self._detector.latencies
        return {
            "frames": float(self._frame_count),
            "n_nodes": float(len(self.graph.nodes)),
            "n_edges": float(len(self.graph.edges())),
            "map_update_time_ms": 1000.0 * self._update_time_s / frames,
            "loop_closure_time_ms": (
                1000.0 * self._revisit_search_time_s / searches
            ),
            "total_map_time_s": self._update_time_s,
            "detector_warmup_samples": float(self._detector.warmup),
            "mean_detection_latency_samples": (
                float(np.mean(latencies)) if latencies else 0.0
            ),
            "max_detection_latency_samples": (
                float(np.max(latencies)) if latencies else 0.0
            ),
        }

    # ------------------------------------------------------------------ #
    # Edge bookkeeping
    # ------------------------------------------------------------------ #
    def _record_visit(
        self, node_id: int, pose: Pose2D, covariance: np.ndarray
    ) -> None:
        self._last_visit_odom[node_id] = pose
        self._last_visit_covariance[node_id] = covariance.copy()

    def _add_odometry_edge(
        self,
        from_id: int,
        to_id: int,
        to_odom_pose: Pose2D,
        to_covariance: np.ndarray,
    ) -> None:
        """Create an edge with an at-creation odometric SE(2) measurement.

        The measurement is the relative odometry pose between the most recent
        visit to ``from_id`` and the robot's current odometry pose (which
        physically corresponds to ``to_id``). The covariance is the increase
        in the additive odometry uncertainty budget between those two visits,
        rather than the sum of two correlated cumulative snapshots.
        """
        self.graph.add_edge(from_id, to_id)
        from_odom: Pose2D | None = self._last_visit_odom.get(from_id)
        if from_odom is None:
            return
        self.graph.set_edge_measurement(from_id, to_id, from_odom, to_odom_pose)
        from_covariance = self._last_visit_covariance.get(from_id)
        if from_covariance is None:
            return
        global_covariance = self._interval_covariance(
            from_covariance, to_covariance
        )
        theta = from_odom[2]
        c, s = float(np.cos(theta)), float(np.sin(theta))
        transform = np.array(
            [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        local_covariance = transform @ global_covariance @ transform.T
        local_covariance = 0.5 * (local_covariance + local_covariance.T)
        local_covariance += np.eye(3) * _MIN_EDGE_SIGMA**2
        information = np.linalg.pinv(local_covariance, hermitian=True)
        key: tuple[int, int] = (min(from_id, to_id), max(from_id, to_id))
        self.graph.edge_sigmas[key] = float(
            np.sqrt(np.mean(np.diag(local_covariance)[:2]))
        )
        self.graph.edge_information[key] = information
        self.graph.edge_types[key] = "odometry"

    @staticmethod
    def _interval_covariance(
        earlier: np.ndarray, later: np.ndarray
    ) -> np.ndarray:
        """PSD uncertainty accumulated between two cumulative snapshots."""
        difference = np.asarray(later, dtype=np.float64) - np.asarray(
            earlier, dtype=np.float64
        )
        difference = 0.5 * (difference + difference.T)
        eigenvalues, eigenvectors = np.linalg.eigh(difference)
        # Tiny negative eigenvalues can arise from serialization/roundoff.
        # Clipping also makes legacy, non-monotonic covariance streams safe.
        return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T

    def _add_loop_edge(self, matched_id: int, query_id: int) -> None:
        """Add a co-location factor without inventing an orientation match.

        Translation uncertainty is the same relative-odometry uncertainty
        used to validate the revisit. Angular information is exactly zero
        because global DINO descriptors do not estimate yaw.
        """
        self.graph.add_edge(matched_id, query_id)
        matched = self.graph.nodes[matched_id]
        query = self.graph.nodes[query_id]
        global_covariance = self._revisit_covariance(query, matched)
        theta = matched.pose[2]
        c, s = float(np.cos(theta)), float(np.sin(theta))
        rotation = np.array([[c, s], [-s, c]], dtype=np.float64)
        local_covariance = rotation @ global_covariance @ rotation.T
        try:
            xy_information = np.linalg.inv(local_covariance)
        except np.linalg.LinAlgError:
            xy_information = np.eye(2) / _PLACE_LOCALIZATION_SIGMA**2
        information = np.zeros((3, 3), dtype=np.float64)
        information[:2, :2] = xy_information
        self.graph.set_edge_constraint(
            matched_id,
            query_id,
            measurement=(0.0, 0.0, 0.0),
            information=information,
            edge_type="loop",
        )

    # ------------------------------------------------------------------ #
    # Node creation / revisit
    # ------------------------------------------------------------------ #
    def _segment_medoid(
        self, segment: list[FrameRecord], mean_descriptor: np.ndarray
    ) -> FrameRecord:
        """Frame of the current node segment closest to its mean descriptor."""
        similarities: list[float] = [
            float(frame.descriptor @ mean_descriptor)
            for frame in segment
        ]
        return segment[int(np.argmax(similarities))]

    def _create_node(self, segment: list[FrameRecord]) -> int:
        """Create one node from a completed, non-empty visual segment."""
        if not segment:
            raise ValueError("cannot create a node from an empty segment")
        stacked: np.ndarray = np.stack([frame.descriptor for frame in segment])
        directional_model = fit_von_mises_fisher(stacked)
        mean_descriptor = directional_model.mean_direction

        representative = self._segment_medoid(segment, mean_descriptor)
        positions = np.asarray([frame.pose[:2] for frame in segment], dtype=np.float64)
        offsets = positions - np.asarray(representative.pose[:2], dtype=np.float64)
        extent_covariance = offsets.T @ offsets / float(len(segment))
        extent_covariance = 0.5 * (extent_covariance + extent_covariance.T)

        candidate: TopoNode = TopoNode(
            node_id=self.graph.new_node_id(),
            pose=representative.pose,
            visual_features=mean_descriptor,
            visual_concentration=directional_model.concentration,
            visual_resultant_length=directional_model.mean_resultant_length,
            visual_sample_count=directional_model.sample_count,
            pose_covariance=representative.covariance[:2, :2].copy(),
            extent_covariance=extent_covariance,
            # Evaluation-only: pin the node's GT to the SAME frame whose pose
            # became the node pose, so placement error reflects odometric
            # distortion, not the medoid/segment-boundary frame mismatch.
            gt_pose=representative.gt_pose,
            room_label=representative.room_label,
        )
        # Medoid + segment endpoints provide viewpoint diversity at a fixed,
        # tiny storage cost. Deduplicate frames for very short segments.
        selected_frames: list[FrameRecord] = []
        selected_indices: set[int] = set()
        for frame in (representative, segment[0], segment[-1]):
            if frame.frame_index not in selected_indices:
                selected_frames.append(frame)
                selected_indices.add(frame.frame_index)
        candidate.set_views(
            [frame.image for frame in selected_frames],
            np.stack([frame.descriptor for frame in selected_frames]),
        )

        if self.current_node_id is None:
            self.graph.add_node(candidate)
            self.current_node_id = candidate.node_id
            self._record_visit(
                candidate.node_id, candidate.pose, representative.covariance
            )
            return candidate.node_id

        search_start: float = time.perf_counter()
        revisited_id, proposed_id, similarity, reason = self._find_revisit(candidate)
        self._revisit_search_time_s += time.perf_counter() - search_start
        self._revisit_search_count += 1
        self.graph.add_node(candidate)
        self._add_odometry_edge(
            self.current_node_id,
            candidate.node_id,
            candidate.pose,
            representative.covariance,
        )
        self.graph.loop_events.append(
            LoopClosureEvent(
                query_node_id=candidate.node_id,
                matched_node_id=proposed_id,
                accepted=revisited_id is not None,
                similarity=similarity,
                reason=reason,
            )
        )
        if revisited_id is not None:
            self._add_loop_edge(revisited_id, candidate.node_id)
        self.current_node_id = candidate.node_id
        self._record_visit(
            candidate.node_id, candidate.pose, representative.covariance
        )
        return candidate.node_id

    def _visual_outlier(self, similarities: np.ndarray, match_index: int) -> bool:
        """True if the match is a robust visual outlier in its evidence pool.

        Rejects matches that are not clearly distinctive — the core defence
        against perceptual-aliasing welds and descriptor-pollution snowballs.
        The strictness is ``self._visual_outlier_k`` (configurable per run).
        """
        # Search-level warm-up already guarantees enough map evidence. After
        # structural/geometric filtering, two surviving scores are sufficient
        # to assess whether the selected place is visually distinctive.
        if similarities.shape[0] < 2:
            return False
        match_sim: float = float(similarities[match_index])
        others: np.ndarray = np.delete(similarities, match_index)
        median: float = float(np.median(others))
        mad: float = float(np.median(np.abs(others - median)))
        threshold: float = median + self._visual_outlier_k * 1.4826 * mad
        # Strict inequality makes the zero-MAD/all-equal case deterministic:
        # an indistinctive tie is rejected rather than accepted.
        return match_sim > threshold

    def _passes_gates(
        self,
        delta_xy: np.ndarray,
        combined_cov: np.ndarray,
    ) -> bool:
        """Single uncertainty-aware chi-square gate for co-location."""
        return mahalanobis_gate(delta_xy, combined_cov)

    def _distinctive_within_geometry(
        self,
        similarities: np.ndarray,
        selected: int,
        geometric_indices: list[int],
    ) -> bool:
        """Verify appearance without letting impossible aliases veto a loop.

        The decisive pool contains every geometrically compatible candidate.
        If that pool is too small for a robust statistic, it is supplemented
        only with weaker visual candidates. A visually stronger node already
        disproved by odometry is deliberately irrelevant: appearance verifies
        an odometric hypothesis; it does not globally retrieve the hypothesis.
        """
        background = list(geometric_indices)
        if len(background) < _MIN_NODES_FOR_REVISIT:
            visual_order = np.argsort(-similarities, kind="stable")
            background.extend(
                int(index)
                for index in visual_order
                if (
                    similarities[index] <= similarities[selected]
                    and int(index) not in background
                )
            )
        pool_scores = similarities[background]
        return self._visual_outlier(pool_scores, background.index(selected))

    @staticmethod
    def _view_feature_matrix(node: TopoNode) -> np.ndarray:
        """Return normalized per-view features, with legacy mean fallback."""
        features = getattr(node, "view_features", None)
        if features is None or np.asarray(features).size == 0:
            features = np.asarray(node.visual_features, dtype=np.float32)[None, :]
        matrix = np.asarray(features, dtype=np.float32)
        return matrix / np.maximum(
            np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12
        )

    @classmethod
    def _node_similarity(cls, first: TopoNode, second: TopoNode) -> float:
        """Symmetric multi-view similarity between two topological places.

        Averaging each view's best counterpart rewards agreement across the
        stored viewpoints and avoids letting one accidental image pair decide
        a loop closure.
        """
        first_features = cls._view_feature_matrix(first)
        second_features = cls._view_feature_matrix(second)
        pairwise = first_features @ second_features.T
        return float(
            0.5 * (np.mean(np.max(pairwise, axis=1))
                   + np.mean(np.max(pairwise, axis=0)))
        )

    @staticmethod
    def _node_vmf_evidence(first: TopoNode, second: TopoNode) -> float:
        """Information-geometric visual evidence between two places."""
        return vmf_log_overlap_ratio(
            first.visual_features,
            first.visual_concentration,
            second.visual_features,
            second.visual_concentration,
        )

    def _odometry_step(self, node_id: int, direction: int) -> int | None:
        """Return an adjacent sequential node in ``direction`` if it exists.

        Node identifiers follow emission order. Checking the stored edge type
        prevents a loop edge from ever becoming visual sequence context.
        """
        neighbor = node_id + direction
        if neighbor not in self.graph.nodes:
            return None
        key = (min(node_id, neighbor), max(node_id, neighbor))
        if neighbor not in self.graph.adjacency.get(node_id, set()):
            return None
        if self.graph.edge_types.get(key, "odometry") != "odometry":
            return None
        return neighbor

    def _sequence_score(
        self,
        candidate: TopoNode,
        matched_id: int,
        scorer: Callable[[TopoNode, TopoNode], float],
    ) -> float:
        """Agreement of recent observations with a historical place.

        The candidate itself is compared with ``matched_id``. Up to two recent
        live nodes are then compared with historical sequential neighbors.
        Both historical directions are evaluated and the stronger mean is
        returned. Consequently, ``room -> door -> corridor`` can match the
        reverse traversal ``corridor -> door -> room`` without requiring the
        two door-facing images to be near-identical.
        """
        direct = scorer(candidate, self.graph.nodes[matched_id])
        if self.current_node_id is None:
            return direct

        live_context: list[int] = [self.current_node_id]
        cursor = self.current_node_id
        for _ in range(1):
            previous = self._odometry_step(cursor, -1)
            if previous is None:
                break
            live_context.append(previous)
            cursor = previous

        directional_scores: list[float] = []
        for direction in (-1, 1):
            scores = [direct]
            historical = matched_id
            for live_id in live_context:
                historical = self._odometry_step(historical, direction)
                if historical is None:
                    break
                scores.append(
                    scorer(
                        self.graph.nodes[live_id], self.graph.nodes[historical]
                    )
                )
            directional_scores.append(float(np.mean(scores)))
        return max(directional_scores)

    def _sequence_similarity(self, candidate: TopoNode, matched_id: int) -> float:
        """Bidirectional sequence score using multi-view cosine similarity."""
        return self._sequence_score(candidate, matched_id, self._node_similarity)

    def _sequence_vmf_evidence(
        self, candidate: TopoNode, matched_id: int
    ) -> float:
        """Bidirectional sequence score using vMF overlap evidence."""
        return self._sequence_score(candidate, matched_id, self._node_vmf_evidence)

    def _revisit_covariance(
        self, candidate: TopoNode, matched: TopoNode
    ) -> np.ndarray:
        """Uncertainty of the relative odometric displacement.

        A node's spatial extent describes the region it covers; it is not
        measurement uncertainty. Mixing the two made long corridor segments
        increasingly easy to accept as co-located. The cumulative covariance
        snapshots already encode uncertainty at the current and revisited
        observations through their interval-relative difference.
        """
        return (
            self._interval_covariance(
                matched.pose_covariance, candidate.pose_covariance
            )
            + np.eye(2) * _PLACE_LOCALIZATION_SIGMA**2
        )

    def _find_revisit(
        self, candidate: TopoNode
    ) -> tuple[int | None, int | None, float | None, str]:
        """Find a revisit using odometry-first, bidirectional sequence gating.

        The full method intentionally gives the two signals asymmetric jobs:
        odometry generates and ranks every statistically compatible place. A
        sole compatible place needs no appearance tie-break; when several
        remain, visual sequence evidence disambiguates them in odometric order.
        Appearance therefore cannot suppress an unambiguous geometric revisit
        or promote a geometrically implausible alias.
        """
        ids = sorted(self.graph.nodes)
        eligible_count = len(ids) - LOOP_TEMPORAL_EXCLUSION_NODES
        if eligible_count < _MIN_NODES_FOR_REVISIT:
            return None, None, None, "no_distinctive_match"

        eligible_ids = ids[:eligible_count]
        direct_similarities = np.asarray(
            [
                self._node_similarity(candidate, self.graph.nodes[node_id])
                for node_id in eligible_ids
            ],
            dtype=np.float64,
        )
        sequence_similarities = np.asarray(
            [self._sequence_similarity(candidate, node_id) for node_id in eligible_ids],
            dtype=np.float64,
        )

        if self._gate_mode == "threshold":
            selected = int(np.argmax(direct_similarities))
            proposed_id = eligible_ids[selected]
            proposed_similarity = float(direct_similarities[selected])
            if direct_similarities[selected] < self._naive_threshold:
                return None, proposed_id, proposed_similarity, "rejected_by_gate"
            accepted_id = eligible_ids[selected]
            return accepted_id, accepted_id, proposed_similarity, "accepted"

        if self._gate_mode == "visual":
            selected = int(np.argmax(sequence_similarities))
            proposed_id = eligible_ids[selected]
            proposed_similarity = float(sequence_similarities[selected])
            if not self._visual_outlier(sequence_similarities, selected):
                return None, proposed_id, proposed_similarity, "rejected_by_gate"
            return proposed_id, proposed_id, proposed_similarity, "accepted"

        sequence_evidence = (
            np.asarray(
                [
                    self._sequence_vmf_evidence(candidate, node_id)
                    for node_id in eligible_ids
                ],
                dtype=np.float64,
            )
            if self._visual_model == "vmf"
            else sequence_similarities
        )

        geometric_candidates: list[tuple[int, float]] = []
        odometry_ranking: list[tuple[int, float]] = []
        for index, node_id in enumerate(eligible_ids):
            matched = self.graph.nodes[node_id]
            delta = np.asarray(candidate.pose[:2]) - np.asarray(matched.pose[:2])
            covariance = self._revisit_covariance(candidate, matched)
            nll = gaussian_position_nll(delta, covariance)
            odometry_ranking.append((index, nll))
            if self._passes_gates(delta, covariance):
                geometric_candidates.append((index, nll))
        odometry_ranking.sort(key=lambda item: (item[1], item[0]))
        geometric_candidates.sort(key=lambda item: (item[1], item[0]))

        proposed_index = (
            geometric_candidates[0][0]
            if geometric_candidates else odometry_ranking[0][0]
        )
        proposed_id = eligible_ids[proposed_index]
        proposed_similarity = float(sequence_evidence[proposed_index])

        if self._gate_mode == "geometric":
            if not geometric_candidates:
                return None, proposed_id, proposed_similarity, "rejected_by_gate"
            selected = geometric_candidates[0][0]
            accepted_id = eligible_ids[selected]
            return (
                accepted_id,
                accepted_id,
                float(sequence_similarities[selected]),
                "accepted",
            )

        if not geometric_candidates:
            return None, proposed_id, proposed_similarity, "rejected_by_gate"

        # When uncertainty-aware geometry leaves exactly one hypothesis, the
        # observation is not ambiguous and visual appearance should not veto
        # it merely because the camera revisited from the opposite direction.
        if len(geometric_candidates) == 1:
            selected = geometric_candidates[0][0]
            accepted_id = eligible_ids[selected]
            return (
                accepted_id,
                accepted_id,
                float(sequence_evidence[selected]),
                "accepted",
            )

        # Odometry owns candidate order. When it is ambiguous, visual evidence
        # resolves the tie but cannot promote a geometrically impossible alias.
        geometric_indices = [index for index, _nll in geometric_candidates]
        for selected, _nll in geometric_candidates:
            if self._distinctive_within_geometry(
                sequence_evidence, selected, geometric_indices
            ):
                accepted_id = eligible_ids[selected]
                return (
                    accepted_id,
                    accepted_id,
                    float(sequence_evidence[selected]),
                    "accepted",
                )
        return None, proposed_id, proposed_similarity, "rejected_by_gate"
