# Ex0.5 — Demo Notebook Reproduction (New Dataset)

## Purpose

Reproduce key plot patterns from `demo_analysis/demo.ipynb` on the new flattened
PAWS dataset. Lets supervisor visually compare old vs new dataset with identical
HP filters, with no preprocessing in between.

## Source

- **Raw parquet**: `demo_analysis/warmstart_runs_flattened.parquet`
- Filter: `method == "paws"`
- PAWS rows (all flops): 1,870,699
- PAWS rows with non-null `Validation Loss`: 187,235

This script reads the **raw** flattened parquet directly, faithful to
`demo.ipynb` which also reads from a single raw parquet. It does NOT use the
derived `validation_points_with_trajectory.parquet` from Ex0 processing.

## HP space

- `base_N`: [14562560, 32270848, 77124608]
- `target_N`: [32270848, 77124608, 134561280, 287183360, 610488320]
- `shrinking`: [0.2, 0.4, 0.6, 0.8, 1.0]
- `tkpm`: [10.0, 20.0, 30.0]

## Plot files

- `plot_single_panel_shrink04_tkpm20.png` — matches demo cell 12 (one line per target scale, base_N=BASE_SCALES[0], shrink=0.4, tkpm=20)
- `plot_single_panel_shrink04_tkpm30.png` — matches demo cell 13 (same but tkpm=30)
- `plot_multi_panel_all_hps.png` — matches demo cell 14 (one panel per target scale, all HP combos)

## Y-axis

- Current: `[1.5, 3.5]`
- New PAWS dataset raw range: 1.61 to 4.55
- If clipping observed at top, rerun with `--ylim 1.5 4.5`

## X-axis

- `flops` normalized to `[0, 1]` per curve via `flops / max(flops)`

## Done when

Supervisor can compare old (`demo.ipynb`) vs new (this folder) dataset
side-by-side using identical HP filters and faithful raw inputs.
