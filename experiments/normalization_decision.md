# Normalization Decision Record

## Problem

FT-PFN operates in a normalized `[0, 1]` performance space. Raw validation loss must be mapped into this space before feeding context curves and query prefixes to the model. The mapping requires choosing a `(log_min, log_max)` range that defines the coordinate system.

The choice of normalization range is **not** a context selection decision — it is a coordinate system decision. But a bad choice can make predictions systematically biased or uninterpretable.

## Options considered

### Option 1: Per-curve normalization

```text
For each curve independently:
  log_min = log(min(curve_loss))
  log_max = log(max(curve_loss))
```

**Problem:** Every curve maps to the full `[0, 1]` range regardless of absolute loss scale. A 32M curve with loss range `[2.0, 10.0]` and a 610M curve with loss range `[1.5, 4.0]` both get mapped to `[0, 1]`. The model cannot distinguish scale differences, which is the entire point of lower-to-higher transfer.

**Status:** Rejected early. Never implemented in the experiment pipeline.

### Option 2: Lower-partner-only normalization (pre-LOO, broken)

```text
For each query target:
  log_min = log(min(all lower partner losses))
  log_max = log(max(all lower partner losses))
```

**Problem:** Lower architectures systematically have higher losses than the query target. The normalization floor `exp(log_min)` sits above the target's actual future loss values. This causes:

- `loss_to_perf` clips the target's observed prefix to `perf ≈ 1.0`
- The model cannot predict below the floor
- 100% one-directional systematic overprediction
- coverage90 drops to ~18% (should be ~90%)
- `pred_p50` systematically > `y_true`

**Evidence:** Pre-LOO backup in `experiments/_backups/ex0_8_to_ex0_12_pre_loo_20260525_031706/`

Pre-LOO Ex3 headline metrics (biased):

| Scale | Spearman ρ (t=1.0) |
|---|---|
| 77M | 0.552 |
| 134M | 0.843 |
| 287M | 0.777 |
| 610M | 0.167 |

Note: Spearman ρ was partially preserved because ranking is invariant to monotone bias, but absolute predictions were unusable.

**Status:** Rejected. Was the default in Ex0.8–Ex0.12 and Ex3 before 2026-05-25.

### Option 3: Lower-partners-plus-prefix normalization (intermediate fix)

```text
For each query target:
  losses = concat(all lower partner losses, target observed prefix losses)
  log_min = log(min(losses))
  log_max = log(max(losses))
```

**Problem:** Anchors the floor to the target's actual loss scale, which fixes the clipping issue. But the range changes depending on `obs_frac` (more observed prefix → different range) and on which partners are selected (policy-dependent range). This makes cross-policy and cross-obs_frac comparisons inconsistent.

**Evidence:** Mentioned in `experiments/ex3_width_extrapolation/_backup_pre_normfix/BACKUP_NOTES.md` as the first attempted fix.

**Status:** Rejected as final solution. Used briefly as intermediate fix before LOO was adopted.

### Option 4: LOO over all non-query PAWS trajectories (adopted)

```text
For each query target:
  log_min = log(min(all non-query trajectory losses))
  log_max = log(max(all non-query trajectory losses))
```

Excludes only the query target itself. Includes all other PAWS trajectories regardless of whether they are in the context or not.

**Why this works:**

1. The range covers the full loss scale of the dataset, including both high-loss small architectures and low-loss large architectures.
2. The target's actual loss values fall within the range (not clipped).
3. The range is independent of context policy — same normalization whether you use `all_lower_only`, `nearest_scale_only`, or `prefix_only`.
4. The range is independent of `obs_frac` — same normalization at 5% or 50% observation.
5. LOO exclusion prevents the query target from anchoring its own normalization (avoids information leakage about the target's full curve).

**Evidence:** After applying LOO normalization:

Ex0.8 `all_lower_only` obs=0.20 band diagnostics:

| target_N | n_lower_partners | band_width_final |
|---|---|---|
| 77M | 1 | 1.706 |
| 134M | 2 | 0.164 |
| 287M | 3 | 2.025 |
| 610M | 4 | 1.077 |

Ex3 `all_lower_only` obs=0.10 (PAWS-wide, 101 lower curves for 610M):

| target_N | mean_band_width | median_band_width |
|---|---|---|
| 77M | 1.785 | 1.867 |
| 134M | 1.088 | 0.996 |
| 287M | 0.793 | 0.819 |
| 610M | 0.550 | 0.331 |

Coverage90 improved from ~18% (pre-LOO) to ~40–60% for 610M (still imperfect due to positive bias, but no longer catastrophically broken).

**Status:** Adopted. Canonical implementation in `demo_analysis/_common/normalization.py`.

## Final formula

```python
# LOO range
log_min, log_max = loo_log_loss_range(cache, target_id)
# Excludes target_id, uses all other PAWS trajectories' raw y values

# Loss → performance (FT-PFN input space)
perf = 1 - clip((log(loss) - log_min) / (log_max - log_min), 0, 1)

# Performance → loss (prediction inversion)
log_loss = log_min + (1 - perf) * (log_max - log_min)
loss = exp(log_loss)
```

## Special case: prefix-only baseline

For `prefix_only` policy (no lower-scale context at all), the normalization uses only the observed prefix:

```python
log_min, log_max = prefix_log_loss_range(y_raw, observed_mask)
```

This is intentionally different from LOO because there is no source context to normalize against. The prefix-only baseline measures what FT-PFN can do from the target curve alone, in its own coordinate system.

## Key principle

**Normalization is a coordinate system, not context information.**

- Context tokens are policy-specific (lower-only, nearest-only, etc.).
- Normalization range is policy-independent (LOO over all non-query PAWS).
- This separation ensures that context selection experiments compare policies on the same scale.

## Scope sensitivity

The LOO range depends on which trajectories are in the cache:

- Current Ex3 broad PAWS filter: 109 trajectories (all `method == "paws"`).
- Old supervisor filter: `G < 10` and `target_N < 6e8` gives fewer trajectories.

If the filter changes, the LOO range changes slightly. Results are not bit-exactly comparable across different filter scopes, but the difference is small because the loss range is dominated by the smallest/largest architectures which are always included.

## Caveats

1. LOO normalization does not guarantee perfect calibration. 610M coverage90 is ~40–60%, not 90%. The model still has positive bias for 610M predictions. But the bias is now a model limitation, not a normalization artifact.
2. The LOO range includes trajectories that are not in the context. This is intentional — normalization defines the coordinate system, not what the model "sees."
3. For corruption experiments (H1b value regimes), the LOO range must be computed on original uncorrupted values. See `demo_analysis/_common/normalization.py` contract.

## Timeline

| Date | Event |
|---|---|
| Pre 2026-05-25 | Ex0.8–Ex0.12 and Ex3 used lower-partner-only normalization |
| 2026-05-25 | Discovered systematic overprediction; coverage90 ~18% |
| 2026-05-25 | Applied LOO fix; backed up pre-LOO outputs |
| 2026-05-25+ | All subsequent experiments use LOO via shared module |

## References

- Canonical implementation: `demo_analysis/_common/normalization.py`
- Pre-LOO backup: `experiments/_backups/ex0_8_to_ex0_12_pre_loo_20260525_031706/`
- Ex3 pre-normfix backup: `experiments/ex3_width_extrapolation/_backup_pre_normfix/`
- Current Ex3 predict: `experiments/ex3_width_extrapolation/predict_one.py` lines 47–51
