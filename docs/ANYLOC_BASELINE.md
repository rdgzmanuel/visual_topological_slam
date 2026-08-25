# Literature baseline: AnyLoc-GeM

The selected external comparison is **AnyLoc: Towards Universal Visual Place
Recognition** (Keetha et al., IEEE RA-L 2023, presented at ICRA 2024). It is a
close methodological match because it is training-free at deployment, uses
off-the-shelf self-supervised ViT patch features, targets visual place
recognition, and was evaluated in indoor environments.

The integration replaces the image descriptor and uses cosine similarity, as
in standard AnyLoc retrieval. It retains the same topological mapper, recorded
odometry, geometric candidate generation, adaptive outlier rule,
hyperparameters and evaluation code. It therefore does not borrow the final
mapper's vMF evidence model. The implemented descriptor uses
DINOv2 attention **value** facets and signed GeM pooling with fixed `p=3`, then
L2 normalization. It does not fit a vocabulary and does not inspect the test
trajectory in advance.

Two configurations are exposed:

- `AnyLoc-GeM (ViT-S adaptation)`: ViT-S/14 block 11. This is the default and
  is computationally matched to the final mapper's ViT-S/14 CLS encoder. It is
  an adaptation of the published method, not the exact paper configuration.
- `AnyLoc-GeM (published encoder)`: ViT-G/14 block 31. This matches the paper's
  model/layer/facet configuration but is impractical on a CPU-only Mac Docker
  container; CUDA is strongly recommended.

After building and sourcing the ROS workspace, run all seven datasets without
gate ablations:

```bash
./run_anyloc_baseline.sh
```

Or run one suite/environment:

```bash
./run_anyloc_baseline.sh cold freiburg_a
./run_anyloc_baseline.sh cid-sims apartment1_1
```

For the exact published encoder on a CUDA machine:

```bash
ANYLOC_EXACT=1 ./run_anyloc_baseline.sh
```

Outputs are isolated beside the final results using the suffix
`_anyloc_gem_vits` or `_anyloc_gem_vitg`. The performance JSON records the
backend, DINO model and layer to prevent the adapted and exact configurations
from being conflated.
