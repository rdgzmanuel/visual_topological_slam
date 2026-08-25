# vMF directional visual validator

The final mapper fits one training-free von Mises--Fisher (vMF) distribution
to all normalized DINO descriptors in each node segment. Odometry still
generates and ranks loop candidates. Only when several geometric candidates
remain does bidirectional sequence evidence use the vMF log-overlap ratio.
No learned parameters, external dependency, or new decision threshold is
introduced.

## Final comparison

Across the four COLD and three CID-SIMS evaluations, the cosine validator
obtained 20 true positives, 1 false positive and 28 false negatives
(precision 0.9524, recall 0.4167, F1 0.5797). The vMF validator obtained 23
true positives, 1 false positive and 25 false negatives (precision 0.9583,
recall 0.4792, F1 0.6389).

The three additional correct closures all occur in CID-SIMS Apartment 1. The
other six datasets have unchanged closure decisions, so the result should be
reported as an aggregate improvement whose observed gain is localized rather
than universal.

Geometry-only obtained 24 true positives and 5 false positives (F1 0.6234).
The vMF dual gate therefore has slightly higher aggregate F1 while retaining
substantially better precision, an important property because false loop
closures can corrupt the whole pose graph.

## Complexity

The implementation adds roughly 260 production lines and no dependency. In
the recorded runs, vMF added about 0.64 ms to an average loop search; DINO
inference still dominates total runtime. Fitting stores the existing mean
direction plus three scalar diagnostics per node: concentration, mean
resultant length and sample count. The descriptors-only footprint avoids
serializing the mean direction twice.

## Preserved artifacts

- `output/revised/`: active vMF results.
- `output/vmf_2026-08-20/`: immutable original vMF result and source archive.
- `output/cosine_2026-08-20_final/`: complete cosine results and ablations.
- `output/cosine_baseline_2026-08-20_interrupted/`: earlier interrupted cosine
  checkpoint retained for provenance.

The cosine result is therefore a reproducible visual-gate ablation, not a
discarded experiment.
