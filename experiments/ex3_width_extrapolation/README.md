# Ex3 — Width Extrapolation (All-Lower-Only)

## Question

Can FT-PFN use learning curves from strictly lower-architecture PAWS trajectories to predict bigger architectures? Specifically: given all available lower-scale context curves plus a short observed prefix of the target, how well does the model predict the target's full validation loss curve?

## Policy

- `all_lower_only`: for each query trajectory, use all PAWS trajectories with `partner_target_N < query_target_N` as context, plus the target's observed prefix.

No oracle or same-scale context is included. This is a strict lower-to-higher transfer test.

## Scope

- PAWS trajectories only (`method == "paws"`).
- No additional `G < 10` or `target_N < 6e8` filter — all PAWS curves are included.
- `obs_frac` values: `0.05, 0.10, 0.20, 0.50`.
- Query targets: `77M, 134M, 287M, 610M`.
- Normalization: LOO log-loss range over all non-query PAWS trajectories.
- FT-PFN model version `0.0.1`, resampled curves at `k=256` hybrid grid.

## Context sizes

Lower-curve counts available per query target:

| query_target_N | n_lower_available | context_size (with prefix) |
|---|---|---|
| 77M | 15 | 16 |
| 134M | 39 | 40 |
| 287M | 66 | 67 |
| 610M | 101 | 102 |

## Result summary

Final p05–p95 band width at `obs_frac=0.10`:

| query_target_N | n_trajectories | mean_band_width | median_band_width |
|---|---|---|---|
| 77M | 24 | 1.785 | 1.867 |
| 134M | 27 | 1.088 | 0.996 |
| 287M | 35 | 0.793 | 0.819 |
| 610M | 8 | 0.550 | 0.331 |

610M has the tightest bands despite being the largest architecture. This is because it has the most lower-scale context (101 curves) and the nearest-scale 287M anchors provide strong shape evidence.

## Interpretation

- More lower-scale context generally tightens uncertainty.
- 610M benefits most because it has the richest lower-context pool (all 4 lower scales).
- 77M has only 32M as lower context (15 curves), which is far in scale, leading to wide bands.
- 134M is tighter than 77M because it has both 32M and 77M lower curves, and 77M is a close neighbor.
- The tight 610M bands motivated follow-up ablations to understand the mechanism.

## Follow-up ablations

- **Ex3.1A** (nearest-scale ablation): nearest 287M context alone explains most of 610M tightening; far-scale-only context is much wider.
- **Ex3.1E** (duplicate-context ablation): duplicating 4 Ex0.8-style curves to 101 entries does not reproduce all-lower tightness. Real PAWS-wide diversity matters, not raw context count.
- **Ex4B** (2HP lower-target ablation): tests which specific 287M partner configurations improve 610M prediction beyond a matched anchor.

## Normalization details

```python
log_min, log_max = loo_log_loss_range(cache, target_id)
# LOO = leave-one-out: excludes the query trajectory, uses all other PAWS curves
```

Performance normalization:

```python
perf = 1 - clip((log(loss) - log_min) / (log_max - log_min), 0, 1)
```

## Outputs

- `predictions.parquet` — 38,634 rows across all obs_fracs and targets
- `plots/all_lower_only/` — cascade curve plots per descriptor slice

## Key columns in predictions.parquet

- `policy`, `trajectory_id`, `query_target_N`, `obs_frac`
- `context_size`, `n_lower_available`, `n_lower_selected`
- `nearest_partner_target_N`, `min_scale_gap`, `max_scale_gap`
- `partner_trajectory_ids`, `partner_target_Ns`, `partner_base_Ns`
- `normalization_scope`, `loss_log_min`, `loss_log_max`
- `x_idx`, `x_norm`, `y_true`, `pred_p05`, `pred_p50`, `pred_p95`

## Caveats

- Current broad PAWS filter gives 101 lower curves for 610M. Old supervisor demos used `G < 10` and `target_N < 6e8` which gives 86 lower curves. Results are not directly comparable to old filtered counts.
- Tight bands are only meaningful if NLL and coverage remain healthy. 610M coverage90 at obs=0.10 is imperfect (~0.4–0.6 depending on trajectory), indicating some positive bias.
- This experiment establishes that lower-to-higher transfer works but does not identify which specific lower-curve properties drive it. That is the role of Ex3.1A, Ex3.1E, and Ex4B.
