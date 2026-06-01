# Ex0.6 — Supervisor Prefix-Only FT-PFN Figure

## Purpose

Paper-style prefix-only FT-PFN extrapolation figure on same PAWS HP slice as Ex0.5.
This is not lower-context comparison; no partner curves enter context.

## Slice

- `method == "paws"`
- `base_N = 14562560`
- `shrink = 0.4`
- `tkpm = 20.0`
- `obs_fracs = [0.05, 0.1, 0.2, 0.5, 0.9]`

## Outputs

- `prefix_only_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/prefix_only_supervisor_style.png` — ground truth plus prefix-only predictions
- `plots/obs_<NNN>/supervisor_styles/*.png` — one prediction panel per target curve
- `plots/obs_<NNN>/cascade_styles/*.png` — stacked prediction panels for all predicted targets

## Band diagnostics

|    target_N |   band_width_start |   band_width_final |   band_width_ratio |
|------------:|-------------------:|-------------------:|-------------------:|
| 3.22708e+07 |          0.0769932 |          0.076425  |           1.00743  |
| 7.71246e+07 |          0.0721362 |          0.0755232 |           0.955154 |
| 1.34561e+08 |          0.0704857 |          0.0730155 |           0.965353 |
| 2.87183e+08 |          0.0679334 |          0.0697782 |           0.973563 |
| 6.10488e+08 |          2.03428   |          2.13327   |           0.953597 |

## Interpretation

Inspect whether p05-p95 bands inflate immediately after observed prefix and whether this varies with target scale.
