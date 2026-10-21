# ReGVE module naming and implementation

The canonical implementation of **retinal grayscale-variation event encoding (ReGVE)** is [benchmark/code/regve.py](../benchmark/code/regve.py), previously named decoder.py.

This change renames the executable module and updates the queue worker and synthetic contract-test references. The convert() and main() function bodies, numerical settings, condition names, output keys and cache directories are unchanged. Existing data/decoder caches remain compatible.

Historical provenance files, including training_freeze.json, benchmark_adaptations.json and release_sha256.json, retain the original path and hashes as records of their original experiment/release. They are historical snapshots, not hash manifests for the renamed source in the current commit. Do not rewrite or mix an existing frozen training work directory; prepare a fresh work directory for the renamed entry point.

Eight synthetic contract checks passed after the rename: frame sampling, duplicate-removal timing, polarity counts, event-count conservation, mask support, threshold dependence, empty-input handling, and simultaneous timestamps. These test numerical encoding behavior, without retraining.

See the root [README](../README.md) for methods, location, execution and the complete twelve-feature SHAP mapping.
