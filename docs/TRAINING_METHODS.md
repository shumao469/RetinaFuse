# Training and analysis methods

## Analysis series

Archived development predictions, the source-disjoint nested benchmark and the genuine acquisition-date hold-out are separate series. Use the fixed source identifiers and split manifests; do not pool their predictions or treat repeated files, derived views, seeds or checkpoints as independent observations. Cohort accounting is documented in the manuscript Supplementary Information.

## Inputs

The full-export condition `graychange_4fps_t002` decodes native-resolution fundus frames at the confirmed 4-fps export rate. Grayscale intensities are divided by 255 before adding a log-intensity offset of 1e-6. Successive log-intensity changes above +0.02 or below −0.02 generate positive or negative events within estimated first-frame vascular support. Timestamps use original frame indices divided by four. Simultaneous events do not add independent temporal resolution. No event/frame cap truncates this condition. Sixty normalized bins form the two-polarity 64 × 64 event volume; the parallel input contains twelve event descriptors.

The static/video controls directly resize grayscale frames to 64 × 64. Single-frame takes the midpoint. Mean-image uses ECC translation registration to the first resized frame (35 iterations, epsilon 1e-4, reflected boundaries); failed registration retains the unregistered frame. Unordered-frames averages independently encoded frame embeddings. Raw-video encodes sixty uniformly selected frames in order. Frame-difference-video takes signed differences of those frames with a zero first bin. These compare complete pipelines, not an isolated causal effect of temporal ordering. Display-only padding is described in [input geometry](INPUT_GEOMETRY.md).

## Fitting and aggregation

Five outer source-disjoint folds each contain five inner folds. AdamW uses learning rate 2e-4, weight decay 1e-3 and batch size eight. Training permits 120 epochs, ten warm-up epochs and patience 15. Focal loss uses gamma 1.5 and healthy/MCAO weights 0.55/0.45, with inverse-frequency sampling. The rounded median inner-selected duration determines final outer fitting. Inner-OOF balanced accuracy selects a threshold on a 0.01 grid; ties use F1, accuracy and proximity to 0.5. Final checkpoint logits are averaged before softmax within seed. Source probabilities and corresponding selected thresholds are averaged across seeds 42, 1042 and 2042. No test-time augmented inputs enter this benchmark.

Independently trained Equal-fusion, Global-weight-fusion and Concat-fusion share encoder architectures and classifier-input width, but not fitted parameters. Concat-fusion projects 512 concatenated values to 256 with GELU and dropout 0.20. These are not frozen-network gate replacements.

## Entry points

Follow the README cache preparation first. Set `RETINAFUSE_WORKDIR` to the newly prepared run for all entry points, for example in PowerShell `$env:RETINAFUSE_WORKDIR='D:/new_retinafuse_run'` or bash `export RETINAFUSE_WORKDIR=/path/to/new_run`.

```bash
python scripts/run_task.py --workdir /path/to/new_run --model RetinaFuse --condition graychange_4fps_t002 --fold 1 --seed 42
python benchmark/code/analyze.py
python benchmark/audit_tools/analyze_representation_comparisons.py
python benchmark/followup/gate_controls.py --input-condition graychange_4fps_t002
```

Repeat the prespecified folds, seeds and model conditions before pooled analysis. GPU fitting/inference requires CUDA and prepared caches. The gate routine reads the resulting bundles; it does not train models. Never run it against unrelated checkpoint bundles merely because names match. The workdir preparation records the portable protocol hash separately from the original frozen protocol.

For the company comparator implementations, initialize the pinned submodules. Their original entry points are retained rather than falsely presented as a common training API: inspect the tracked Python entry points in `components/rf_ensemble`, `components/set_attention`, `components/snn` and `components/retinafuse` and provide the original feature/event formats and configurations required there. Their archived results are not part of the newly trained source-disjoint static/video comparison.

## Frozen-network interventions

Temporal homogenization blends the event volume with its temporal mean. Statistics neutralization scales standardized descriptors toward zero. Five strengths for each form a joint grid. Event thinning binomially removes event counts while holding descriptors fixed. Learned weights are recomputed; frozen-clean retains the corresponding source/checkpoint clean-input weights; equal replacement uses 0.5/0.5. Each condition has its own matched baseline. Learned-minus-control Brier below zero favours learned weighting; positive values favour the control. These experiments do not establish universal reliability sensing.

## Numerical uncertainty

AUROC and AP use empirical probabilities; Brier is mean squared probability error. Paired class-stratified bootstrap uses 5,000 source-level resamples and identical indices across paired models. Pipeline/decoder comparisons use pointwise 95% percentile intervals. The independently trained fusion-family comparison reports the released 99% intervals per endpoint, not simultaneous coverage of all analyses. Seeds/checkpoints are combined within source before resampling. Smoothed display coordinates never determine a reported score.

## Source portability and runtime

The release makes work directories configurable and renames a locally fetched vendor module to the manuscript-consistent `retinafuse_core.py`; the adaptation ledger records hashes of both versions. Upstream blobs retain their original rights and are fetched by pinned Git revision with SHA-256 verification. The runtime lock is an observed environment snapshot; historical nightly packages may require a compatible replacement that must be recorded. Checkpoint hashes identify states, but binaries are supplied separately and are necessary for exact fitted-model re-inference.
