# Ex0.6 Prefix-Only Band Artifact — Explanation

## What you see

In `prefix_only_supervisor_style.png` at `obs_frac=0.10`, the 77M and 134M curves have a visible "stain" at the start of the prediction region:

- **77M**: band starts at width 0.77, then grows normally (visible bump at boundary)
- **134M**: band starts at width 4.92 (wider than final width 4.05!), then shrinks before growing — very visible artifact

32M, 287M, and 610M start tight (~0.08–0.09) and grow monotonically — no artifact.

## Why it happens

Ex0.6 `prefix_only` uses **prefix-only normalization**:

```python
log_min, log_max = log_loss_range(y_raw[observed_mask])
normalization_scope = "observed_prefix"
```

This computes the coordinate system from only the first 10% of the target curve. For 77M and 134M, their prefix loss values produce a narrow/awkward normalization range. When `perf_to_loss` inverts the model's quantiles back to loss space, the narrow range amplifies the band at certain loss levels.

## Why other experiments don't have this

Ex0.7+ and Ex3/Ex4B use **LOO normalization**:

```python
log_min, log_max = loo_log_loss_range(cache, target_id)
normalization_scope = "loo_all_non_query_paws"
```

LOO provides a wide, stable range covering the full loss landscape of all non-query curves. The quantile inversion is well-behaved everywhere.

## Evidence

```text
target_N  start_band  end_band  ratio(start/end)
32M       0.093       3.21      0.03  ← normal
77M       0.766       4.08      0.19  ← bump artifact
134M      4.919       4.05      1.21  ← starts WIDER than end (inverted)
287M      0.094       4.43      0.02  ← normal
610M      0.075       4.85      0.02  ← normal
```

## Fix

Apply LOO normalization to `prefix_only` as well. This is consistent with our normalization principle:

> "Normalization is a coordinate system, not context information."

The model still only sees the prefix as context — it just operates in a shared coordinate system, making NLL comparisons fair across policies.

## Status

- `obs0_10_prefix_norm_backup/` — original plots with prefix-only normalization (this artifact)
- `obs0_10/` — regenerated with LOO normalization (artifact fixed)
