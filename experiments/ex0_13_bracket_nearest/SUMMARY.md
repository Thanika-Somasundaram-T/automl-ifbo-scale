# Ex0.13 — bracket_nearest FT-PFN Figure

## Purpose

Toy interpolation sanity check on the same PAWS HP slice as Ex0.6/Ex0.7/Ex0.8.
Each predicted target uses the nearest lower-scale full curve and nearest upper-scale full curve plus its own observed prefix.

## Slice

- `method == "paws"`
- `base_N = 14562560`
- `shrink = 0.4`
- `tkpm = 20.0`
- `obs_frac = 0.9`

## Policy

- `bracket_nearest`: context = nearest scale below ∪ nearest scale above.
- edge targets 32M/610M are skipped because they have no bracket on this 5-curve slice.
- predicts 77M/134M/287M on this 5-curve slice.
- normalization uses LOO log-loss range over all non-query PAWS slice trajectories.
- skipped targets: [32270848, 610488320]
- predicted targets: [77124608, 134561280, 287183360]

## Outputs

- `bracket_nearest_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/bracket_nearest_supervisor_style.png` — ground truth plus bracket-nearest predictions
- `plots/obs_<NNN>/supervisor_styles/*.png` — one prediction panel per predicted target curve
- `plots/obs_<NNN>/cascade_styles/*.png` — stacked prediction panels for all predicted targets

## Band diagnostics

|    target_N |   n_partners |   nearest_below |   nearest_above |   band_width_start |   band_width_final |   band_width_ratio |   obs_frac |
|------------:|-------------:|----------------:|----------------:|-------------------:|-------------------:|-------------------:|-----------:|
| 7.71246e+07 |            2 |     3.22708e+07 |     1.34561e+08 |          0.0785949 |          0.718351  |          0.10941   |       0.05 |
| 1.34561e+08 |            2 |     7.71246e+07 |     2.87183e+08 |          0.0909785 |          0.985587  |          0.092309  |       0.05 |
| 2.87183e+08 |            2 |     1.34561e+08 |     6.10488e+08 |          0.0742138 |          2.59167   |          0.0286355 |       0.05 |
| 7.71246e+07 |            2 |     3.22708e+07 |     1.34561e+08 |          0.0755126 |          0.49069   |          0.153891  |       0.1  |
| 1.34561e+08 |            2 |     7.71246e+07 |     2.87183e+08 |          0.0754468 |          0.397503  |          0.189802  |       0.1  |
| 2.87183e+08 |            2 |     1.34561e+08 |     6.10488e+08 |          0.0710764 |          2.31378   |          0.0307188 |       0.1  |
| 7.71246e+07 |            2 |     3.22708e+07 |     1.34561e+08 |          0.0703351 |          0.259739  |          0.270791  |       0.2  |
| 1.34561e+08 |            2 |     7.71246e+07 |     2.87183e+08 |          0.0676492 |          0.20131   |          0.336045  |       0.2  |
| 2.87183e+08 |            2 |     1.34561e+08 |     6.10488e+08 |          0.0668613 |          1.84322   |          0.0362742 |       0.2  |
| 7.71246e+07 |            2 |     3.22708e+07 |     1.34561e+08 |          0.0681041 |          0.122486  |          0.556017  |       0.5  |
| 1.34561e+08 |            2 |     7.71246e+07 |     2.87183e+08 |          0.067588  |          0.107259  |          0.630136  |       0.5  |
| 2.87183e+08 |            2 |     1.34561e+08 |     6.10488e+08 |          0.0614186 |          0.343248  |          0.178934  |       0.5  |
| 7.71246e+07 |            2 |     3.22708e+07 |     1.34561e+08 |          0.0667037 |          0.0699676 |          0.953351  |       0.9  |
| 1.34561e+08 |            2 |     7.71246e+07 |     2.87183e+08 |          0.0643381 |          0.0661305 |          0.972896  |       0.9  |
| 2.87183e+08 |            2 |     1.34561e+08 |     6.10488e+08 |          0.0634453 |          0.0651315 |          0.974112  |       0.9  |

## Interpretation

Inspect whether nearest lower/upper bracket context behaves like a sane width-interpolation counterpart to Ex0.8 lower-only extrapolation.
