# Ex0.8 — all_lower_only FT-PFN Figure

## Purpose

Supervisor-style lower-to-higher transfer sanity check on the same PAWS HP slice as Ex0.6/Ex0.7.
Each predicted target uses only strictly lower-scale full curves plus its own observed prefix.

## Slice

- `method == "paws"`
- `base_N = 14562560`
- `shrink = 0.4`
- `tkpm = 20.0`
- `obs_frac = 0.9`

## Policy

- `all_lower_only`: partner curves must satisfy `partner_target_N < query_target_N`.
- targets with zero lower partners are skipped; 32M is skipped on this slice.
- normalization uses LOO log-loss range over all non-query PAWS slice trajectories.
- skipped targets: [32270848]
- predicted targets: [77124608, 134561280, 287183360, 610488320]

## Outputs

- `all_lower_only_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/all_lower_only_supervisor_style.png` — ground truth plus lower-only predictions
- `plots/obs_<NNN>/supervisor_styles/*.png` — one prediction panel per predicted target curve
- `plots/obs_<NNN>/cascade_styles/*.png` — stacked prediction panels for all predicted targets

## Band diagnostics

|    target_N |   n_lower_partners |   band_width_start |   band_width_final |   band_width_ratio |
|------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 7.71246e+07 |                  1 |          0.0679216 |          0.0703008 |           0.966157 |
| 1.34561e+08 |                  2 |          0.0649532 |          0.0667383 |           0.973252 |
| 2.87183e+08 |                  3 |          0.057522  |          0.0592125 |           0.971449 |
| 6.10488e+08 |                  4 |          0.0894863 |          0.226351  |           0.395343 |

## Interpretation

Inspect whether uncertainty tightens as lower-partner count increases and compare against Ex0.6 prefix-only and Ex0.7 sibling-context reference.
