"""Dataset-agnostic topological mapper (single-run, no pose-graph solver).

Consumes synchronized (image, pose, covariance) samples — wherever they come
from (a real robot's odometry or a dataset player) — and incrementally builds
a :class:`TopoGraph`.

Design (deliberately simplified to a single, reliable run):

- Node creation: a node is emitted whenever the windowed algebraic
  connectivity (lambda_2) of the visual stream dips through a valley
  (:class:`AdaptiveValleyDetector`). Its descriptor is the L2-normalized mean
  of the frames assigned to it; its representative view/pose is the segment
  *medoid* (the frame closest to that mean), which is stable and needs no
  fragile sliding-window index.
- Revisit / loop closure: a candidate node is fused into an existing one when
  it is a mutual visual nearest neighbour (Lowe-ratio verified) AND lies
  within a chi-square gate under the accumulated odometry covariance AND
  within a data-driven Euclidean radius (a few median node-spacings). The
  Euclidean cap is the key robustness fix: odometry covariance grows without
  bound along a run, which on its own eventually lets the chi-square gate weld
  visually-similar but physically distant places. Capping the search radius to
  the map's own spatial scale removes that failure mode.
- Continuous loop closures are debounced: the same candidate must pass the
  gates on ``_LOOP_DEBOUNCE`` consecutive frames before the closure commits.
- Anchored odometry: odometry is a guideline for inter-node distance and
  traversability, not a globally consistent position estimate — so it is not
  allowed to accumulate forever. Whenever a revisit is confirmed, the
  recognized node is taken as a trusted place: the robot's position estimate
  snaps to the node's stored pose (translation only; a place can be
  re-entered from any direction, so the stored heading carries no correction
  information) and the odometry covariance baseline restarts from zero. Both
  drift and uncertainty are thereby bounded by the loop structure of the
  environment instead of growing with total path length.

During the run, node poses live in the anchored-odometry frame and
topological correction happens through node fusion and re-anchoring. At the
end of the run an SE(2) pose-graph optimization (:mod:`vts_core.pose_graph`,
enabled by default via ``optimize``) corrects the residual metric drift of
the node poses from the stored relative edge measurements; multi-map fusion
remains removed to keep a single run robust and dependency-light.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from vts_core.matching import (
    mahalanobis_gate,
    median_spacing,
    mutual_nearest_neighbors,
)
from vts_core.node_detection import AdaptiveValleyDetector, ConnectivityMonitor
from vts_core.pose_graph import optimize_se2
from vts_core.topo_graph import Pose2D, TopoGraph, TopoNode

# Translational odometry sigma prior per sqrt-metre of travelled path. A single
# interpretable random-walk constant used only to weight pose-graph edges
# relative to each other.
_ODOM_SIGMA_PER_SQRT_M: float = 0.15
_MIN_EDGE_SIGMA: float = 0.05

# A revisit may not sit farther than this many median node-spacings from the
# odometry estimate. Bounds the chi-square gate so unbounded odometry
# covariance cannot weld distant places; it is a relative, data-driven scale
# rather than a fixed metric threshold.
_MAX_REVISIT_SPACINGS: float = 5.0

# Visual revisit gate. A loop closure is accepted only if the candidate's
# similarity to the matched node is a robust *outlier* above its similarity to
# every other node: at least the median plus k robust standard deviations
# (k * 1.4826 * MAD), with k the default below (tunable per run via the
# ``visual_outlier_k`` constructor argument). This is the same threshold-free,
# MAD-based criterion used by the valley detector, applied to visual
# similarity. It is what stops the "pollution snowball": a place that is
# similar to *many* nodes (e.g. a generic corridor) produces no clear outlier
# and is rejected. LOWER k -> easier merges (more loop closures), RAISE it ->
# stricter. The snowball is independently contained because ``_merge_into`` no
# longer averages the descriptor, so a single bad merge can no longer pollute a
# node and cascade; that is what makes a lower k safe here.
_VISUAL_OUTLIER_K: float = 2.0
# Below this many nodes there is not enough evidence to judge an outlier;
# stay conservative and do not close loops yet.
_MIN_NODES_FOR_REVISIT: int = 5
# Absolute similarity floor (safety net for the degenerate all-similar case
# where the robust spread collapses). Sits between the measured inter-place
# and intra-place similarity of the contrastive encoder.
_MIN_REVISIT_SIMILARITY: float = 0.5

# Revisit gate ablation modes. "both" is the full system (visual outlier test
# AND geometric gate must agree); "visual" and "geometric" disable one gate
# each; "threshold" replaces the whole dual gate with a naive absolute
# cosine-similarity threshold — the baseline the dual gate is argued against.
GATE_MODES: tuple[str, ...] = ("both", "visual", "geometric", "threshold")
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


class TopologicalMapper:
    """Incremental topological mapper for a single run."""

    _LOOP_DEBOUNCE: int = 3

    def __init__(
        self,
        window_size: int = 30,
        frame_buffer_size: int = 85,
        frame_id: str = "run",
        valley_k: float = 1.5,
        merge_radius: float = 2.0,
        visual_outlier_k: float = _VISUAL_OUTLIER_K,
        optimize: bool = True,
        gate_mode: str = "both",
        naive_threshold: float = _NAIVE_THRESHOLD,
    ) -> None:
        """
        Args:
            window_size: Sliding window size for the connectivity monitor.
                Larger = smoother lambda_2 = fewer, coarser nodes.
            frame_buffer_size: How many recent frames are kept for node
                creation (must exceed window_size).
            frame_id: Identifier of this run's odometry frame.
            valley_k: Sensitivity of node creation. The valley detector fires
                when lambda_2 rises ``valley_k`` robust deviations above its
                running minimum, so a LARGER value yields FEWER, more separated
                nodes (raise it if nodes cluster too densely).
            merge_radius: Absolute radius (m) within which a visually-matched
                revisit MERGES regardless of covariance. RAISE for more merges
                of co-located nodes.
            visual_outlier_k: Strictness of the visual revisit gate (robust
                MAD multiplier). LOWER it to merge more aggressively, RAISE for
                stricter loop closures. See ``_VISUAL_OUTLIER_K``.
            optimize: Run the end-of-run SE(2) pose-graph optimization in
                :meth:`finalize` (corrects accumulated odometry drift in the
                node poses).
            gate_mode: Revisit-gate ablation mode, one of :data:`GATE_MODES`.
                "both" (default) is the full dual gate; "visual" /
                "geometric" keep only one gate; "threshold" is the naive
                absolute-similarity baseline.
            naive_threshold: Cosine-similarity threshold used when
                ``gate_mode == "threshold"``.
        """
        if gate_mode not in GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}"
            )
        self.graph: TopoGraph = TopoGraph(frame_id=frame_id)
        self.current_node_id: int | None = None
        self._merge_radius: float = merge_radius
        self._visual_outlier_k: float = visual_outlier_k
        self._optimize: bool = optimize
        self._gate_mode: str = gate_mode
        self._naive_threshold: float = naive_threshold

        self._monitor: ConnectivityMonitor = ConnectivityMonitor(window_size)
        self._detector: AdaptiveValleyDetector = AdaptiveValleyDetector(k=valley_k)
        self._frames: deque[FrameRecord] = deque(maxlen=frame_buffer_size)
        self._frame_count: int = 0
        self._frames_in_node: list[FrameRecord] = []

        self._loop_candidate_id: int | None = None
        self._loop_candidate_hits: int = 0

        # Odometry pose / cumulative path length of the most recent visit to
        # each node. Edge measurements are built from these so they stay in the
        # consistent (drifting) odometry frame; the optimizer then corrects the
        # drift globally in finalize().
        self._last_visit_odom: dict[int, Pose2D] = {}
        self._last_visit_pathlen: dict[int, float] = {}
        self._path_length: float = 0.0
        self._previous_pose: Pose2D | None = None

        # Anchored odometry state. ``_odom_offset`` is the cumulative
        # translation that maps raw odometry into the map frame (updated at
        # every confirmed revisit); ``_cov_offset`` is the raw covariance at
        # the last anchor, so the effective covariance restarts from ~zero
        # there instead of accumulating over the whole run.
        self._odom_offset: np.ndarray = np.zeros(2, dtype=np.float64)
        self._cov_offset: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self._latest_raw_covariance: np.ndarray | None = None

        # Performance instrumentation (PRISM Table V analogue). Wall-clock time
        # of the per-frame map update (excludes descriptor extraction, which is
        # the encoder's cost and lives in the caller) and of the loop-closure
        # search performed at node creation.
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
    ) -> int | None:
        """Process one synchronized sample.

        Thin timing wrapper around :meth:`_process_frame_impl` so the mean
        map-update time can be reported for the PRISM performance comparison.

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

        Returns:
            The id of a node created/updated by this frame, or None.
        """
        start: float = time.perf_counter()
        try:
            return self._process_frame_impl(
                image_bgr, descriptor, pose, covariance, gt_pose
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
    ) -> int | None:
        # Anchored odometry: shift the raw pose into the map frame and restart
        # the covariance from the last anchor (confirmed revisit).
        self._latest_raw_covariance = covariance.astype(np.float64)
        pose = (
            pose[0] + float(self._odom_offset[0]),
            pose[1] + float(self._odom_offset[1]),
            pose[2],
        )
        record: FrameRecord = FrameRecord(
            descriptor=descriptor.astype(np.float32),
            pose=pose,
            covariance=self._effective_covariance(self._latest_raw_covariance),
            image=image_bgr,
            gt_pose=gt_pose,
        )
        self._frames.append(record)
        self._frames_in_node.append(record)
        self._frame_count += 1
        if self._previous_pose is not None:
            self._path_length += float(
                np.hypot(pose[0] - self._previous_pose[0],
                         pose[1] - self._previous_pose[1])
            )
        self._previous_pose = pose

        lambda_2: float | None = self._monitor.update(record.descriptor)
        if lambda_2 is None:
            return None

        valley_index: int | None = self._detector.step(lambda_2)
        if valley_index is None:
            if len(self.graph.nodes) > 1:
                self._continuous_loop_check(record)
            return None

        return self._create_or_merge_node()

    def finalize(self) -> tuple[float, float]:
        """End-of-run hook. Optimize the pose graph (unless ``optimize=False``).

        Anchored odometry (position updates + covariance restarts at every
        confirmed revisit) bounds the drift topologically, but the residual
        metric error between anchors remains in the node poses. The end-of-run
        SE(2) solver corrects it globally from the stored relative edge
        measurements; disable via ``optimize=False`` to keep the raw
        anchored-odometry poses (e.g. to report the unoptimized RMSE).

        Returns:
            (initial_error, final_error); ``(0.0, 0.0)`` when optimization is
            off.
        """
        if not self._optimize:
            return (0.0, 0.0)
        return optimize_se2(self.graph)

    def performance_stats(self) -> dict[str, float]:
        """Computational-cost summary for the PRISM Table V comparison.

        Times are wall-clock and exclude descriptor extraction (the encoder's
        cost, incurred by the caller). ``map_update_time_ms`` is the mean
        per-frame map-maintenance cost; ``loop_closure_time_ms`` is the mean
        cost of one revisit (loop-closure) search at node creation.
        """
        frames: int = max(self._frame_count, 1)
        searches: int = max(self._revisit_search_count, 1)
        return {
            "frames": float(self._frame_count),
            "n_nodes": float(len(self.graph.nodes)),
            "n_edges": float(len(self.graph.edges())),
            "map_update_time_ms": 1000.0 * self._update_time_s / frames,
            "loop_closure_time_ms": (
                1000.0 * self._revisit_search_time_s / searches
            ),
            "total_map_time_s": self._update_time_s,
        }

    # ------------------------------------------------------------------ #
    # Anchored odometry
    # ------------------------------------------------------------------ #
    def _effective_covariance(self, raw: np.ndarray) -> np.ndarray:
        """Covariance accumulated since the last anchor (PSD-clipped)."""
        effective: np.ndarray = raw - self._cov_offset
        effective = 0.5 * (effective + effective.T)
        values, vectors = np.linalg.eigh(effective)
        return vectors @ np.diag(np.clip(values, 0.0, None)) @ vectors.T

    def _anchor_at(self, node_pose: Pose2D, estimate: Pose2D) -> None:
        """Re-anchor odometry on a recognized node.

        A confirmed revisit makes the node's stored pose the best available
        estimate of where the robot is, so the translational error
        accumulated since the last anchor — the gap between the odometric
        estimate and the stored node pose — is corrected, and the covariance
        baseline restarts. Heading is left untouched: a place can be
        re-entered from any direction, so the stored theta carries no
        correction information. Buffered frames are shifted too, so a node
        created just after the anchor gets a consistent pose.
        """
        delta: np.ndarray = np.array(node_pose[:2], dtype=np.float64) - np.array(
            estimate[:2], dtype=np.float64
        )
        self._odom_offset += delta
        shifted: set[int] = set()
        for frame in list(self._frames) + self._frames_in_node:
            if id(frame) in shifted:
                continue
            shifted.add(id(frame))
            frame.pose = (
                frame.pose[0] + float(delta[0]),
                frame.pose[1] + float(delta[1]),
                frame.pose[2],
            )
        if self._previous_pose is not None:
            self._previous_pose = (
                self._previous_pose[0] + float(delta[0]),
                self._previous_pose[1] + float(delta[1]),
                self._previous_pose[2],
            )
        if self._latest_raw_covariance is not None:
            self._cov_offset = self._latest_raw_covariance.copy()

    # ------------------------------------------------------------------ #
    # Edge bookkeeping
    # ------------------------------------------------------------------ #
    def _record_visit(self, node_id: int, pose: Pose2D) -> None:
        self._last_visit_odom[node_id] = pose
        self._last_visit_pathlen[node_id] = self._path_length

    def _add_measured_edge(
        self, from_id: int, to_id: int, to_odom_pose: Pose2D
    ) -> None:
        """Create an edge with an at-creation odometric SE(2) measurement.

        The measurement is the relative odometry pose between the most recent
        visit to ``from_id`` and the robot's current odometry pose (which
        physically corresponds to ``to_id``). Its sigma scales with the square
        root of the path length travelled between the two (random-walk model).
        """
        self.graph.add_edge(from_id, to_id)
        from_odom: Pose2D | None = self._last_visit_odom.get(from_id)
        if from_odom is None:
            return
        self.graph.set_edge_measurement(from_id, to_id, from_odom, to_odom_pose)
        segment: float = max(
            self._path_length - self._last_visit_pathlen.get(from_id, 0.0), 0.0
        )
        sigma: float = max(
            _ODOM_SIGMA_PER_SQRT_M * float(np.sqrt(segment)), _MIN_EDGE_SIGMA
        )
        key: tuple[int, int] = (min(from_id, to_id), max(from_id, to_id))
        self.graph.edge_sigmas[key] = sigma

    # ------------------------------------------------------------------ #
    # Node creation / revisit
    # ------------------------------------------------------------------ #
    def _segment_medoid(self, mean_descriptor: np.ndarray) -> FrameRecord:
        """Frame of the current node segment closest to its mean descriptor."""
        if not self._frames_in_node:
            return self._frames[-1]
        similarities: list[float] = [
            float(frame.descriptor @ mean_descriptor)
            for frame in self._frames_in_node
        ]
        return self._frames_in_node[int(np.argmax(similarities))]

    def _create_or_merge_node(self) -> int:
        # Robust node descriptor: mean of frames assigned since the last node.
        if self._frames_in_node:
            stacked: np.ndarray = np.stack(
                [f.descriptor for f in self._frames_in_node]
            )
            mean_descriptor: np.ndarray = stacked.mean(axis=0)
        else:
            mean_descriptor = self._frames[-1].descriptor
        mean_descriptor = mean_descriptor / max(
            float(np.linalg.norm(mean_descriptor)), 1e-12
        )

        representative: FrameRecord = self._segment_medoid(mean_descriptor)
        segment: list[FrameRecord] = self._frames_in_node or [representative]

        candidate: TopoNode = TopoNode(
            node_id=self.graph.new_node_id(),
            pose=representative.pose,
            visual_features=mean_descriptor,
            pose_covariance=representative.covariance[:2, :2].copy(),
            # Evaluation-only: pin the node's GT to the SAME frame whose pose
            # became the node pose, so placement error reflects odometric
            # distortion, not the medoid/segment-boundary frame mismatch.
            gt_pose=representative.gt_pose,
        )
        candidate.add_view(representative.image)
        candidate.add_view(segment[0].image)
        candidate.add_view(segment[-1].image)
        self._frames_in_node = []

        if self.current_node_id is None:
            self.graph.add_node(candidate)
            self.current_node_id = candidate.node_id
            self._record_visit(candidate.node_id, candidate.pose)
            return candidate.node_id

        search_start: float = time.perf_counter()
        revisited_id: int | None = self._find_revisit(candidate)
        self._revisit_search_time_s += time.perf_counter() - search_start
        self._revisit_search_count += 1
        if revisited_id is not None:
            self._merge_into(candidate, revisited_id)
            self.current_node_id = revisited_id
            # The revisited node is a trusted place: snap the position
            # estimate onto it and restart odometry accumulation there.
            matched_pose: Pose2D = self.graph.nodes[revisited_id].pose
            self._anchor_at(matched_pose, candidate.pose)
            self._record_visit(
                revisited_id,
                (matched_pose[0], matched_pose[1], candidate.pose[2]),
            )
            return revisited_id

        self.graph.add_node(candidate)
        self._add_measured_edge(
            self.current_node_id, candidate.node_id, candidate.pose
        )
        self.current_node_id = candidate.node_id
        self._record_visit(candidate.node_id, candidate.pose)
        return candidate.node_id

    def _revisit_radius(self) -> float:
        """Data-driven cap on how far a revisit may sit from the estimate."""
        positions: np.ndarray = self.graph.positions()
        return _MAX_REVISIT_SPACINGS * median_spacing(positions)

    def _visual_outlier(self, similarities: np.ndarray, match_index: int) -> bool:
        """True if the match is a robust visual outlier above all other nodes.

        Rejects matches that are not clearly distinctive — the core defence
        against perceptual-aliasing welds and descriptor-pollution snowballs.
        The strictness is ``self._visual_outlier_k`` (configurable per run).
        """
        if similarities.shape[0] < _MIN_NODES_FOR_REVISIT:
            return False
        match_sim: float = float(similarities[match_index])
        others: np.ndarray = np.delete(similarities, match_index)
        median: float = float(np.median(others))
        mad: float = float(np.median(np.abs(others - median)))
        threshold: float = median + self._visual_outlier_k * 1.4826 * mad
        return match_sim >= max(threshold, _MIN_REVISIT_SIMILARITY)

    def _passes_gates(
        self,
        delta_xy: np.ndarray,
        combined_cov: np.ndarray,
        radius: float,
    ) -> bool:
        """Geometric merge gate: capped at ``radius``; accept if either the
        nodes are within an absolute ``merge_radius`` (the "same location"
        scale — what makes co-located nodes merge when odometry covariance is
        ~0) or they are chi-square consistent under the accumulated covariance
        (which widens the gate once odometry drifts).
        """
        distance: float = float(np.linalg.norm(delta_xy))
        if distance > radius:
            return False
        if distance <= self._merge_radius:
            return True
        return mahalanobis_gate(delta_xy, combined_cov)

    def _revisit_gates(
        self,
        similarities: np.ndarray,
        match_index: int,
        delta_xy: np.ndarray,
        combined_cov: np.ndarray,
    ) -> bool:
        """Full revisit decision on a mutual-NN candidate, honouring the
        ablation ``gate_mode``: the dual gate ("both"), a single gate
        ("visual"/"geometric"), or the naive absolute-similarity baseline
        ("threshold"). Candidate selection (mutual nearest neighbour) is
        common to every mode; only the acceptance criterion changes.
        """
        if self._gate_mode == "threshold":
            return float(similarities[match_index]) >= self._naive_threshold
        if self._gate_mode == "visual":
            return self._visual_outlier(similarities, match_index)
        if self._gate_mode == "geometric":
            return self._passes_gates(
                delta_xy, combined_cov, self._revisit_radius()
            )
        return self._visual_outlier(similarities, match_index) and (
            self._passes_gates(delta_xy, combined_cov, self._revisit_radius())
        )

    def _find_revisit(self, candidate: TopoNode) -> int | None:
        """Probabilistic + visual revisit test against all existing nodes.

        A node is a revisit if it is a mutual visual nearest neighbour of the
        candidate AND passes the gates selected by ``gate_mode`` (by default
        both the visual outlier test and the geometric gate).
        """
        ids, features = self.graph.feature_matrix()
        matches: list[tuple[int, int, float]] = mutual_nearest_neighbors(
            candidate.visual_features[None, :], features
        )
        if not matches:
            return None

        _, match_index, _ = matches[0]
        matched_id: int = ids[match_index]
        if matched_id == self.current_node_id:
            return None

        similarities: np.ndarray = features @ candidate.visual_features
        matched: TopoNode = self.graph.nodes[matched_id]
        delta: np.ndarray = np.array(candidate.pose[:2]) - np.array(
            matched.pose[:2]
        )
        combined_cov: np.ndarray = (
            candidate.pose_covariance + matched.pose_covariance
        )
        if not self._revisit_gates(similarities, match_index, delta, combined_cov):
            return None
        return matched_id

    def _merge_into(self, candidate: TopoNode, target_id: int) -> None:
        """Fuse the candidate into an existing node.

        The target's visual descriptor is deliberately left UNCHANGED. Averaging
        it with each revisit (the previous behaviour) let a single mistaken
        merge drag the descriptor toward a generic average, which then matched
        even more places — a runaway pollution snowball that produced a few
        hub nodes welding the whole map. Keeping the descriptor stable bounds
        the damage of any single bad merge to that one edge.
        """
        target: TopoNode = self.graph.nodes[target_id]
        for view in candidate.views:
            target.add_view(view)
        # Revisit observation reduces positional uncertainty: keep the
        # smaller covariance (trace-wise) of the two.
        if float(np.trace(candidate.pose_covariance)) < float(
            np.trace(target.pose_covariance)
        ):
            target.pose_covariance = candidate.pose_covariance.copy()
        if self.current_node_id is not None and self.current_node_id != target_id:
            # Loop-closure edge: the robot stands at candidate.pose (odometry),
            # which is physically the place of target_id.
            self._add_measured_edge(
                self.current_node_id, target_id, candidate.pose
            )

    def _continuous_loop_check(self, record: FrameRecord) -> None:
        """Between valleys, check whether the robot drifted onto an old node.

        Committing a loop closure on a single frame's mutual-NN match proved
        fragile (one noisy match under low light + a wide late-run covariance
        gate welds distant places). The check therefore requires the same
        candidate node to pass all gates on ``_LOOP_DEBOUNCE`` consecutive
        frames before the closure is committed.
        """
        ids, features = self.graph.feature_matrix()
        matches: list[tuple[int, int, float]] = mutual_nearest_neighbors(
            record.descriptor[None, :], features
        )
        if not matches:
            self._reset_loop_candidate()
            return
        match_index: int = matches[0][1]
        matched_id: int = ids[match_index]
        if matched_id == self.current_node_id:
            self._reset_loop_candidate()
            return
        similarities: np.ndarray = features @ record.descriptor
        matched: TopoNode = self.graph.nodes[matched_id]
        delta: np.ndarray = np.array(record.pose[:2]) - np.array(matched.pose[:2])
        combined_cov: np.ndarray = (
            record.covariance[:2, :2] + matched.pose_covariance
        )
        if not self._revisit_gates(similarities, match_index, delta, combined_cov):
            self._reset_loop_candidate()
            return

        if matched_id == self._loop_candidate_id:
            self._loop_candidate_hits += 1
        else:
            self._loop_candidate_id = matched_id
            self._loop_candidate_hits = 1
        if self._loop_candidate_hits < self._LOOP_DEBOUNCE:
            return
        self._reset_loop_candidate()

        if self.current_node_id is not None:
            self._add_measured_edge(
                self.current_node_id, matched_id, record.pose
            )
        self.current_node_id = matched_id
        # Same anchoring as the node-creation revisit path: the recognized
        # node is a trusted place, so position is updated and odometry
        # accumulation restarts.
        self._anchor_at(matched.pose, record.pose)
        self._record_visit(
            matched_id, (matched.pose[0], matched.pose[1], record.pose[2])
        )

    def _reset_loop_candidate(self) -> None:
        self._loop_candidate_id = None
        self._loop_candidate_hits = 0
