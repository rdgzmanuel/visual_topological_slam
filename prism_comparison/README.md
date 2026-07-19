# PRISM-TopoMap on COLD — comparison runs

This folder runs the PRISM-TopoMap baseline (Muravyev et al., RA-L 2025) on
the four COLD sequences used in the paper, inside the authors' own Docker
image, and evaluates the result with the same metrics as the paper.

## Setup from a fresh clone

Three inputs are not stored in this repository and must be produced once:

1. **Upstream code** — PRISM-TopoMap carries no license, so its sources are
   not redistributed here. Fetch them at the pinned commit and apply the
   device-selection patch (the only modification, see
   `vendor/device_autodetect.patch`):

   ```bash
   bash vendor/fetch_upstream.sh
   ```

2. **Data bundles** — after downloading the [COLD database](https://www.cas.kth.se/COLD/)
   sequences into `../encoder/seq_data/` (see the root README), convert them:

   ```bash
   python3 tools/convert_cold.py     # writes data/<env>.npz
   ```

3. **Place-recognition weights** — download the authors'
   [minkloc3d_nclt.pth](https://drive.google.com/file/d/19uPohPxQUa71jQzApjGPVbSyJeVdggwF/view)
   into `weights/minkloc3d_nclt.pth`.

Then run everything with `./run_all.sh` (Linux/macOS) or `run_all.bat`
(Windows), and evaluate with `python3 eval/evaluate_prism.py --all`.

## Simple instructions (no technical knowledge needed)

You need a computer with an **NVIDIA graphics card** and **Docker**.

1. **Install Docker Desktop**: https://www.docker.com/products/docker-desktop/
   - On Windows, accept the default WSL2 option during installation.
   - After installing, open Docker Desktop once and leave it running.
2. Copy this whole `prism_comparison` folder onto the computer (USB stick or
   OneDrive), somewhere simple like the Desktop.
3. **Windows**: double-click `run_all.bat`.
   **Linux/macOS**: open a terminal in the folder and run `./run_all.sh`.
4. The first start downloads a large (~15 GB) environment — this can take a
   while depending on the connection. After that, the four runs take roughly
   10–40 minutes total on a gaming GPU.
5. When it prints `All done`, send the whole `results` folder back (zip it).

If it says the GPU could not be used, it automatically retries on the
processor (slower but fine — just let it finish).

## What it does (technical)

- `data/*.npz` — four COLD sequences (2D SICK laser + odometry + ground truth),
  produced by `tools/convert_cold.py`. Odometry is synthesized from ground
  truth with the same probabilistic motion model, noise coefficients
  (`alpha = [0.025, 0.005, 0.01, 0.0025]`) and seed (17) as the VTS runs, so
  both systems receive the identical odometry realization.
- `vendor/prism-topomap/` — the authors' code, pinned at upstream commit
  `9cf8806e` (see `UPSTREAM_COMMIT.txt`). The only modification is automatic
  device selection (CUDA if available, else CPU); the full diff is in
  `vendor/device_autodetect.patch`.
- `driver/run_prism.py` — offline replay driver. It calls the unmodified
  `TopoSLAMModel.update(...)` per frame exactly as the authors' ROS node does,
  and fires `localizer.localize()` every `localization_frequency` seconds of
  *data* time (deterministic replacement of the ROS wall-clock timer).
  Each input point cloud is the last 1 s of 2D scans motion-compensated with
  odometry (a 181-beam planar scan alone is much sparser than the 3D LiDAR
  clouds PRISM-TopoMap expects; the aggregation is input preprocessing only).
- `config/cold.yaml` — PRISM-TopoMap parameters copied from the authors' own
  indoor experiment with noised odometry (`habitat_mp3d_noised_odom.yaml`),
  including the same `minkloc3d_nclt.pth` place-recognition weights
  (`weights/`). Only the input section is adapted to the planar cloud.
- `eval/evaluate_prism.py` — computes the same metrics as the VTS paper
  (structure, coverage, false merges, online localization RMSE, AR@k place
  recognition, map size, timings) from the run outputs. Run on the host after
  the runs finish: `python3 eval/evaluate_prism.py --all`.

## Outputs (per environment, in `results/<env>/`)

- `graph/` — the saved topological map (authors' format: `graph.json` +
  per-vertex grids)
- `frames.jsonl` — per-frame log (poses, vcur, vertex/edge counts, timings)
- `localizations.jsonl` — every localization call with matched vertices
- `frame_descriptors.npy` — per-frame minkloc3d descriptors (for AR@k)
- `summary.json` — run summary (counts, mean timings, RAM, device)
- `metrics.json`, `map.png` — written by the evaluation script

## Caveats recorded for the paper

- COLD has no 3D LiDAR; PRISM-TopoMap runs here on planar clouds from the
  2D SICK scanner. Its place-recognition model (MinkLoc3D trained on NCLT
  3D LiDAR) is therefore out of its training domain — this is an inherent
  sensor-modality limitation of applying the method to camera+2D-laser robots,
  and exactly what the comparison is about.
- COLD provides a single forward camera, so the multimodal (front+back camera)
  variant of their place-recognition model cannot be used; the point-cloud-only
  variant is the one their own noised-odometry indoor experiment uses anyway.
- CPU runs produce identical maps but timings are not comparable to GPU.
- **Vertical-extrusion ablation** (`config/cold_zrep.yaml`, results in
  `results_zrep/`): each planar scan replicated at z = {-0.4..0.4} m to mimic
  the vertical wall structure a 3D LiDAR would see. Within-run AR@1 changes by
  only -3.8 to +8.6 points (AR@5 unchanged), i.e. the place-recognition gap is
  a genuine modality/domain limitation, not an artifact of the flat cloud.
