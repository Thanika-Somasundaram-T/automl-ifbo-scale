# Ex0.12 — lower_start_final_pt FT-PFN Figure

## Purpose

Supervisor-style lower-to-higher transfer sanity check on the same PAWS HP slice as Ex0.6-0.11.
Each predicted target uses start and final points from strictly lower-scale curves plus its own observed prefix.

## Slice

- `method == "paws"`
- `base_N = 14562560`
- `shrink = 0.4`
- `tkpm = 20.0`
- `obs_frac = 0.9`

## Policy

- `lower_start_final_pt`: partner curves must satisfy `partner_target_N < query_target_N`.
- each lower partner contributes exactly two points: first and final rows after sorting by `x_norm`.
- targets with zero lower partners are skipped; 32M is skipped on this slice.
- normalization uses LOO log-loss range over all non-query PAWS slice trajectories.
- skipped targets: [32270848]
- predicted targets: [77124608, 134561280, 287183360, 610488320]

## Outputs

- `lower_start_final_pt_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/lower_start_final_pt_supervisor_style.png` — ground truth plus start+final lower-curve predictions
- `plots/obs_<NNN>/cascade_styles/lower_start_final_pt_cascade_curves.png` — aligned cascade-style multi-target panel

## Band diagnostics

|    target_N |   n_lower_partners |   points_per_partner |   band_width_start |   band_width_final |   band_width_ratio |
|------------:|-------------------:|---------------------:|-------------------:|-------------------:|-------------------:|
| 7.71246e+07 |                  1 |                    2 |          0.0694596 |          0.0735414 |           0.944497 |
| 1.34561e+08 |                  2 |                    2 |          0.0642011 |          0.0649055 |           0.989147 |
| 2.87183e+08 |                  3 |                    2 |          0.0660697 |          0.0677503 |           0.975194 |
| 6.10488e+08 |                  4 |                    2 |          2.06797   |          2.11998   |           0.975466 |

## Interpretation

Compare against Ex0.10 lower_final_pt_only and Ex0.11 lower_geom_sparse_k8 to test whether endpoints capture enough lower-curve shape.
