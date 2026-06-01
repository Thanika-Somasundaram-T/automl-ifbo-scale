---
description: Test width interpolation with nearest lower and higher context curves
---

# Experiment 2 — Width Interpolation

Goal: test whether iFBO improves when predicting a middle target width using context curves from both smaller and larger widths.

## Hypothesis

For target architectures with available lower and higher widths, iFBO should predict better when given nearest bracketing context than when given only the target prefix.

## Analysis

1. Choose target scales that have at least one lower and one higher context scale.
2. Build context regimes:
   - `bracket_nearest` — nearest lower + nearest higher target scales; primary interpolation policy.
   - `prefix_only` — target prefix only; no partner curves.
   - `all_context_oracle` — all non-query trajectories; explicitly labeled oracle baseline.
3. Keep target prefix fixed by `obs_frac`.
4. Predict future losses and HP rankings.
5. Compare interpolation performance across regimes.

Lower-only / higher-only transfer policies are extrapolation policies owned by Ex3, not Ex2.

## What to observe

If interpolation works:

- `bracket_nearest` beats `prefix_only`
- interpolation is strongest for middle scales with tight lower/upper brackets
- rank ρ improves and regret decreases

If not:

- adding bracketing context does not improve over prefix-only
- predictions collapse toward one side of the scale range

## Metrics

Primary:

- Spearman ρ
- transfer regret
- top-k overlap / top-k hit rate

Secondary:

- curve RMSE
- final-loss MAE
- coverage90

## Context selection rules

Use explicit aliases after joins:

- query architecture: `query_target_N` of held-out target trajectory
- lower context: `partner_target_N < query_target_N`
- higher context: `partner_target_N > query_target_N`

`bracket_nearest` skips targets that lack either side.

## Output files

- `experiments/ex2_width_interpolation/results/predictions.parquet`
- `experiments/ex2_width_interpolation/results/_partial/{policy}__obs{tag}.parquet`
- `experiments/ex2_width_interpolation/results/rank_metrics.csv`
- `experiments/ex2_width_interpolation/results/regret_metrics.csv`
- `experiments/ex2_width_interpolation/plots/_summary/*.png`
- `experiments/ex2_width_interpolation/plots/obs_<NNN>/supervisor_styles/<policy>/*.png`
- `experiments/ex2_width_interpolation/README.md`

## Workflow commands

Use venv: `/home/baile/master_project/automl-ifbo-scale/.venv/bin/python`

### Run primary interpolation policy

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex2_width_interpolation/policy_scripts/bracket_nearest.py \
  --device cuda --output-dir experiments/ex2_width_interpolation
```

### Run all Ex2 policies

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex2_width_interpolation/predict_width_interpolation.py \
  --tier all --device cuda
```

### Smoke test

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex2_width_interpolation/policy_scripts/bracket_nearest.py \
  --device cpu --max-targets-per-scale 1 --obs-fracs 0.10 \
  --output-dir experiments/ex2_width_interpolation/smoke
```

### Compute metrics

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex2_width_interpolation/compute_width_interpolation_metrics.py \
  --predictions experiments/ex2_width_interpolation/results/predictions.parquet \
  --output-dir experiments/ex2_width_interpolation/results \
  --plots-dir experiments/ex2_width_interpolation/plots/_summary
```

### Plot predictions

```bash
/home/baile/master_project/automl-ifbo-scale/.venv/bin/python \
  experiments/ex2_width_interpolation/plot_supervisor_style_predictions.py \
  --predictions experiments/ex2_width_interpolation/results/predictions.parquet \
  --output-dir experiments/ex2_width_interpolation/plots \
  --diagnostics experiments/ex2_width_interpolation/results/uncertainty_diagnostics.csv
```

## Done when

- `bracket_nearest`, `prefix_only`, and `all_context_oracle` are compared on the same eligible middle targets
- target scales without both sides are excluded or documented
- results are reported by obs_frac and target_N
