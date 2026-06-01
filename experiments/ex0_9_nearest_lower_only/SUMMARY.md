# Ex0.9 — nearest_lower_only FT-PFN Figure

## Purpose

Supervisor-style lower-to-higher transfer sanity check on the same PAWS HP slice as Ex0.6-0.8.
Each predicted target uses only the nearest lower-scale full curve plus its own observed prefix.

## Slice

- `method == "paws"`
- `base_N = 14562560`
- `shrink = 0.4`
- `tkpm = 20.0`
- `obs_frac = 0.9`

## Policy

- `nearest_lower_only`: partner curve is the largest `partner_target_N < query_target_N`.
- targets with zero lower partners are skipped; 32M is skipped on this slice.
- normalization uses LOO log-loss range over all non-query PAWS slice trajectories.
- skipped targets: [32270848]
- predicted targets: [77124608, 134561280, 287183360, 610488320]

## Outputs

- `nearest_lower_only_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/nearest_lower_only_supervisor_style.png` — ground truth plus nearest-lower predictions
- `plots/obs_<NNN>/supervisor_styles/*.png` — one prediction panel per predicted target curve
- `plots/obs_<NNN>/cascade_styles/*.png` — stacked prediction panels for all predicted targets

## Band diagnostics

|    target_N |   nearest_partner_target_N |   band_width_start |   band_width_final |   band_width_ratio |
|------------:|---------------------------:|-------------------:|-------------------:|-------------------:|
| 7.71246e+07 |                3.22708e+07 |          0.0679216 |          0.0703008 |           0.966157 |
| 1.34561e+08 |                7.71246e+07 |          0.0664172 |          0.0686562 |           0.967388 |
| 2.87183e+08 |                1.34561e+08 |          0.062575  |          0.064549  |           0.969418 |
| 6.10488e+08 |                2.87183e+08 |          0.376425  |          0.673811  |           0.558651 |

## Interpretation

Compare against Ex0.8 all_lower_only to test whether nearest scale is sufficient or whether additional lower scales help.
