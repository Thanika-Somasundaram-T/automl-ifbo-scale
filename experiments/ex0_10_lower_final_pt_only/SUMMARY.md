# Ex0.10 — lower_final_pt_only FT-PFN Figure

## Purpose

Supervisor-style lower-to-higher transfer sanity check on the same PAWS HP slice as Ex0.6-0.9.
Each predicted target uses only final points from strictly lower-scale full curves plus its own observed prefix.

## Slice

- `method == "paws"`
- `base_N = 14562560`
- `shrink = 0.4`
- `tkpm = 20.0`
- `obs_frac = 0.9`

## Policy

- `lower_final_pt_only`: partner curves must satisfy `partner_target_N < query_target_N`.
- each lower partner contributes exactly one point: final row after sorting by `x_norm`.
- targets with zero lower partners are skipped; 32M is skipped on this slice.
- normalization uses LOO log-loss range over all non-query PAWS slice trajectories.
- skipped targets: [32270848]
- predicted targets: [77124608, 134561280, 287183360, 610488320]

## Outputs

- `lower_final_pt_only_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/lower_final_pt_only_supervisor_style.png` — ground truth plus lower-final-point predictions
- `plots/obs_<NNN>/cascade_styles/lower_final_pt_only_cascade_curves.png` — aligned cascade-style multi-target panel

## Band diagnostics

|    target_N |   n_lower_partners |   band_width_start |   band_width_final |   band_width_ratio |
|------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 7.71246e+07 |                  1 |          0.0677764 |          0.0702552 |           0.964717 |
| 1.34561e+08 |                  2 |          0.0637504 |          0.0646989 |           0.98534  |
| 2.87183e+08 |                  3 |          0.067655  |          0.0696028 |           0.972014 |
| 6.10488e+08 |                  4 |          2.21896   |          2.25793   |           0.982738 |

## Interpretation

Compare against Ex0.8 all_lower_only to test whether full lower-curve shape matters beyond final loss endpoints.
