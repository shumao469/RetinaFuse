# Exact descriptor SHAP for RetinaFuse

This analysis explains the **full fitted RetinaFuse probability**, retaining each recording's observed event-volume input. It is not a tree surrogate, an explanation of the Statistics-only comparator, or a new training experiment. The released entry points reproduce the archived original-recording functional analysis, not the multiview benchmark or the later source-disjoint training series. Use the corresponding inputs and checkpoint bundles without mixing analysis series.

## Inputs and setup

Install the compute dependencies and a compatible PyTorch build as described in the root README. Run `python scripts/fetch_sources.py` to retrieve and hash-check the pinned upstream architecture. No SHAP package is required: the included implementation evaluates the closed-form exact Shapley expression.

Copy `benchmark/explain/config.example.json` to `benchmark/explain/config.local.json` and set the local paths. Paths are relative to the configuration file, unless absolute. All paths and run settings come from this configuration; no local drive names are embedded in the analysis code.

| Input | Required contents |
|---|---|
| `predictions` | Original-recording held-out CSV: `animal_id`, `label`, `outer_fold`, `prediction`, `probability`; an optional `model` column selects RetinaFuse |
| `lineage` | CSV containing unique `animal_id` and `source_family`; content-identical aliases share a family |
| `input_cache` | NPZ: `{animal_id}_frames` and `{animal_id}_stats`, the latter in the feature order below |
| `embedding_cache` | Optional NPZ: `{animal_id}_checkpoint{k}_frame`; when absent, frame embeddings are recomputed |
| `checkpoint_root` and `checkpoint_pattern` | Original final bundles containing `model_state_dicts`, training `stat_mean`/`stat_std`, `ablation`, and `outer_fold` |
| `checkpoint_manifest` | CSV with `model`, `fold`, `sha256`; the checkpoint must match its recorded digest |
| `model_source` | Pinned `benchmark/vendor/retinafuse_core.py`; its digest is checked before loading architecture definitions |

Raw inputs, cached tensors, checkpoint binaries, local configurations and generated results are not committed. Choose a fresh output directory when changing configuration or inputs. Coalition caching is guarded by a fingerprint of the configuration and numerical input hashes. The first held-out cached frame embedding in each fold is checked against its actual encoder.

```bash
python benchmark/explain/run_descriptor_shap.py --config benchmark/explain/config.local.json
python benchmark/explain/summarize_descriptor_shap.py --input-dir /path/to/descriptor_shap --lineage /path/to/lineage.csv --output-dir /path/to/shap_tables --supplementary-figure 6
python benchmark/explain/plot_descriptor_shap.py --data /path/to/shap_tables/plot_data.json --output-dir /path/to/shap_plots
python -m unittest discover -s tests -v
```

`--limit` on the compute entry point is for diagnostics only. Its status is explicitly incomplete, and the summary entry point refuses incomplete runs. The device defaults to CPU. CUDA is optional and must be selected explicitly in the configuration. The plot entry point creates standalone panel sets as SVG with editable text and vector PDF, not replacements for an entire existing manuscript deck.

## Model function and exact attribution

For each held-out recording, fix the event-volume embedding separately for each checkpoint. Select sixteen unique-source reference donors from the corresponding outer-training identifiers, with four Healthy and twelve MCAO records, using seed `20261015 + fold`. Exclude every family alias of the held-out recordings before donor selection. These are reference-background counts, not a statement of cohort size.

Enumerate all 4,096 descriptor coalitions. Retain the target's features in the coalition and replace the other features using one intact donor vector. Apply that fold's training standardization. Recompute the statistics encoder, gate and head for every hybrid vector, retaining the observed frame embedding. Average checkpoint logit margins, apply the sigmoid, then average probabilities over donors. Averaging checkpoint probabilities instead would define a different model function.

For descriptor set $\mathcal F$ of size $M$, use

$$\phi_{ij}=\sum_{S\subseteq\mathcal F\setminus\{j\}}\frac{|S|!(M-|S|-1)!}{M!}\left[v_i(S\cup\{j\})-v_i(S)\right].$$

Here $i$ indexes the recording, $j$ the descriptor, $S$ a coalition excluding $j$, and $v_i(S)$ the donor-averaged MCAO probability with the event volume retained. Positive $\phi$ raises the MCAO probability. The baseline $v_i(\varnothing)$ depends on the same recording's retained frame input and is **not** the unconditional class prevalence. The implementation verifies baseline prediction replay and exact additivity. No test-label threshold is optimized.

The twelve inputs, in model order, are:

1. Total event count
2. Positive event count
3. Negative event count
4. Positive event fraction
5. Mean consecutive event interval
6. Standard deviation of consecutive intervals
7. Maximum consecutive interval
8. Mean horizontal coordinate
9. Horizontal-coordinate standard deviation
10. Mean vertical coordinate
11. Vertical-coordinate standard deviation
12. Polarity-switch count in stored event order

The four joint-masking families are abundance/polarity (1–4), timestamp intervals (5–7), spatial distribution (8–11), and polarity switching (12). Joint-family Shapley values come from the corresponding sixteen group coalitions. They are not generally equal to sums of individual-feature values because grouping changes the coalition game.

## Intervals, panel mapping and case selection

Summaries use 5,000 class-stratified **source-family cluster bootstrap** replicates. When a family is selected, all its recording aliases are retained together. The intervals are conditional on the fitted models and donor background; neither checkpoints nor coalitions are treated as independent animals. This bootstrap does not remove any source overlap in the original training design or incorporate variation from training again or choosing a different donor background.

| Current panel | Analysis |
|---|---|
| Fig. 2g | Class-specific signed mean SHAP with source-family intervals |
| Fig. 2h | Raw-descriptor Spearman correlations; feature codes match Fig. 2g |
| Supplementary a | Mean absolute individual-feature SHAP |
| Supplementary b | Individual-recording SHAP distribution, coloured by training-donor percentile |
| Supplementary c | Mean absolute exact joint-family SHAP with intervals |
| Supplementary d | Joint-family SHAP versus the sum of individual values |
| Supplementary e,f | Healthy and MCAO local additive decompositions |

The current text places the supplementary panels in Supplementary Fig. 6. An earlier deck labelled the same panel set Supplementary Fig. 19. Set `--supplementary-figure` to match the manuscript; this only changes source filenames and panel labels, not numerical results. Fig. 2g,h and Supplementary b,c reflect the requested panel exchange. Numerical source CSVs and the feature-code key are exported alongside `plot_data.json`.

Local examples are correctly classified recordings nearest their respective within-class median predicted probability, with lexicographic ID tie breaking. All descriptor contributions are shown in the global importance order. Different local horizontal ranges are permitted and must remain labelled.

## Interpretation boundaries

Counts and polarities describe **video-derived grayscale changes**, not blood-flow direction. Intervals depend on assigned frame timestamps and can be zero for shared timestamps; small mean intervals do not imply physiological temporal resolution. Coordinates describe localization and spread within the imaged field, not calibrated vessel density or flow speed. Polarity switches depend on stored event ordering and do not measure cardiac frequency. These inputs differ from the RF comparator's speed/frequency/amplitude proxies.

Interventional masking can produce combinations outside the observed descriptor dependencies. Joint masking retains dependencies within each family but not necessarily across families. SHAP values explain model computation, not disease-specific physiology or causation. Positional importance may also reflect alignment or acquisition effects. The analysis does not establish a new source-disjoint or external validation result.

Exact enumeration follows the standard finite-background interventional convention: [SHAP ExactExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html). The algorithm is implemented directly here and tested against additive functions, pairwise interactions and explicit full-network calculations.
