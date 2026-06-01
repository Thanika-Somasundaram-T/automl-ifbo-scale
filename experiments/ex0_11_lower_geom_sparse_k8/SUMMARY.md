# Ex0.11 — lower_geom_sparse_k8 FT-PFN Figure

## Purpose

Supervisor-style lower-to-higher transfer sanity check on the same PAWS HP slice as Ex0.6-0.10.
Each predicted target uses 8 geometrically-spaced points from strictly lower-scale curves plus its own observed prefix.

## Slice

- `method == "paws"`
- `base_N = 14562560`
- `shrink = 0.4`
- `tkpm = 20.0`
- `obs_frac = 0.9`

## Policy

- `lower_geom_sparse_k8`: partner curves must satisfy `partner_target_N < query_target_N`.
- each lower partner contributes at most 8 geometric points biased toward early x values.
- targets with zero lower partners are skipped; 32M is skipped on this slice.
- normalization uses LOO log-loss range over all non-query PAWS slice trajectories.
- skipped targets: [32270848]
- predicted targets: [77124608, 134561280, 287183360, 610488320]

## Outputs

- `lower_geom_sparse_k8_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/lower_geom_sparse_k8_supervisor_style.png` — ground truth plus sparse lower-curve predictions
- `plots/obs_<NNN>/cascade_styles/lower_geom_sparse_k8_cascade_curves.png` — aligned cascade-style multi-target panel

## Band diagnostics

|    target_N |   n_lower_partners |   points_per_partner |   band_width_start |   band_width_final |   band_width_ratio |
|------------:|-------------------:|---------------------:|-------------------:|-------------------:|-------------------:|
| 7.71246e+07 |                  1 |                    8 |          0.0666307 |          0.0689474 |           0.966399 |
| 1.34561e+08 |                  2 |                    8 |          0.0649525 |          0.0663871 |           0.978392 |
| 2.87183e+08 |                  3 |                    8 |          0.0623317 |          0.0641549 |           0.971581 |
| 6.10488e+08 |                  4 |                    8 |          2.14478   |          2.21403   |           0.968722 |

## Interpretation

Compare against Ex0.8 all_lower_only and Ex0.10 lower_final_pt_only to test whether sparse curve shape is enough.
