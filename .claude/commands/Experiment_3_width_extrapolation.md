---
description: Test lower-to-higher width extrapolation for warmstart context selection
---

# Experiment 3 — Width Extrapolation

Goal: answer the supervisor's key transfer question: which lower-architecture contexts should be fed into iFBO to predict bigger architectures?

## Hypothesis

Lower-scale context curves can help iFBO predict larger target architectures, but performance depends on scale gap and partner HPs.

## Analysis

1. Restrict non-oracle context partners to lower target scale:
   - `partner_target_N < query_target_N`
2. Compare lower-context policies:
   - `prefix_only`
   - `context_ifbo`
   - `all_lower_only`
   - `nearest_lower_only`
   - `lower_final_pt_only`
   - `lower_geom_sparse_k8`
   - `lower_shrink_lt_1`
   - `lower_high_tkpm_cold_start`
   - `random_lower_k`
   - `lower_start_final_pt`
   - `all_context_oracle` — explicitly labeled oracle control
3. Predict larger target futures.
4. Evaluate rank quality, regret, and curve shape error.

Interpolation/bracketing policies live in Ex2 width interpolation. Do not include them in lower-to-higher transfer claims.

## What to observe

If lower-to-higher extrapolation works:

- lower-context policies beat prefix-only
- nearest/all-lower contexts preserve high-scale HP ranking
- regret is lower than prefix-only
- performance degrades smoothly as scale gap grows

If it fails:

- rank ρ collapses for large scale gaps
- regret approaches prefix-only
- curve-shape error grows sharply for largest targets

## Metrics

Primary:

- Spearman ρ
- transfer regret
- top-k hit rate
- pairwise agreement

Secondary:

- curve RMSE
- final-loss MAE/RMSE
- coverage90

## Derived fields

Use real columns and explicit aliases after joins:

- `query_target_N` = held-out target trajectory `target_N`
- `partner_target_N` = context trajectory `target_N`
- `partner_base_N` = context trajectory `base_N`
- `scale_gap = query_target_N / partner_target_N`
- `base_gap = query_target_N / partner_base_N`

## Output files

- `experiments/ex3_width_extrapolation/results/predictions.parquet`
- `experiments/ex3_width_extrapolation/results/_partial/{policy}__obs{tag}.parquet`
- `experiments/ex3_width_extrapolation/rank_metrics.csv`
- `experiments/ex3_width_extrapolation/regret_metrics.csv`
- `experiments/ex3_width_extrapolation/plots/_summary/scale_gap_summary.png`
- `experiments/ex3_width_extrapolation/README.md`

## Code layout

Shared policy-agnostic machinery:

- `experiments/_lib/common.py` — defaults, seed, device, invariants
- `experiments/_lib/data.py` — load processed, PAWS filter, HP norm, cache
- `experiments/_lib/predict_one.py` — single-query FT-PFN prediction
- `experiments/_lib/runner.py` — CLI, checkpointing, execution loop
- `experiments/_lib/metrics.py` — rank/regret/curve metrics
- `experiments/_lib/plot.py` — supervisor-style plots

Ex3-specific policy registry:

- `experiments/ex3_width_extrapolation/policies.py`

Per-policy scripts:

- `experiments/ex3_width_extrapolation/policy_scripts/prefix_only.py`
- `experiments/ex3_width_extrapolation/policy_scripts/context_ifbo.py`
- `experiments/ex3_width_extrapolation/policy_scripts/all_lower_only.py`
- `experiments/ex3_width_extrapolation/policy_scripts/nearest_lower_only.py`
- `experiments/ex3_width_extrapolation/policy_scripts/lower_final_pt_only.py`
- `experiments/ex3_width_extrapolation/policy_scripts/lower_geom_sparse_k8.py`
- `experiments/ex3_width_extrapolation/policy_scripts/lower_shrink_lt_1.py`
- `experiments/ex3_width_extrapolation/policy_scripts/lower_high_tkpm_cold_start.py`
- `experiments/ex3_width_extrapolation/policy_scripts/random_lower_k.py`
- `experiments/ex3_width_extrapolation/policy_scripts/lower_start_final_pt.py`
- `experiments/ex3_width_extrapolation/policy_scripts/all_context_oracle.py`

Compatibility wrapper:

- `experiments/ex3_width_extrapolation/predict_width_extrapolation.py`

## Workflow commands

Use venv: `/home/baile/master_project/automl-ifbo-scale/.venv/bin/python`

### Run single policy

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex3_width_extrapolation/policy_scripts/prefix_only.py \
  --device cuda --output-dir experiments/ex3_width_extrapolation
```

### Run all policies

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex3_width_extrapolation/predict_width_extrapolation.py \
  --tier all --device cuda
```

### Run by tier

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex3_width_extrapolation/predict_width_extrapolation.py \
  --tier 1 --device cuda
```

### Smoke test

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex3_width_extrapolation/policy_scripts/prefix_only.py \
  --device cpu --max-targets-per-scale 1 --obs-fracs 0.10 \
  --output-dir experiments/ex3_width_extrapolation/smoke
```

### Compute metrics

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex3_width_extrapolation/compute_width_extrapolation_metrics.py \
  --predictions experiments/ex3_width_extrapolation/results/predictions.parquet \
  --output-dir experiments/ex3_width_extrapolation \
  --plots-dir experiments/ex3_width_extrapolation/plots/_summary
```

### Plot predictions

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex3_width_extrapolation/plot_supervisor_style_predictions.py \
  --predictions experiments/ex3_width_extrapolation/results/predictions.parquet \
  --output-dir experiments/ex3_width_extrapolation/plots \
  --diagnostics experiments/ex3_width_extrapolation/results/uncertainty_diagnostics.csv
```

## Done when

- every non-oracle context policy uses only lower partner scales
- results are stratified by scale gap and obs_frac
- regret/top-k metrics are reported alongside Spearman ρ
