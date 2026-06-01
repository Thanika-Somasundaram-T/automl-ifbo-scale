"""Experiment 0.5 — Demo notebook reproduction on new flattened dataset.

Reproduces the key ground-truth plot patterns from `demo_analysis/demo.ipynb`
on the new PAWS-only dataset, so the supervisor can visually compare
old vs new dataset with the same HP filters.

Outputs:
  - plot_single_panel_shrink04_tkpm20.png   (matches demo cell 12)
  - plot_single_panel_shrink04_tkpm30.png   (matches demo cell 13)
  - plot_multi_panel_all_hps.png            (matches demo cell 14)
  - SUMMARY.md
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_RAW_PARQUET = Path("demo_analysis/warmstart_runs_flattened.parquet")
DEFAULT_OUTPUT_DIR = Path("experiments/ex0_5_demo_reproduction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ex0.5: demo notebook reproduction.")
    parser.add_argument("--raw-parquet", type=Path, default=DEFAULT_RAW_PARQUET,
                        help="Raw flattened parquet (faithful demo.ipynb input).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ylim", type=float, nargs=2, default=[1.5, 3.5])
    return parser.parse_args()


def plot_curve(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str | None = None,
    xcol: str = "flops",
    ycol: str = "Validation Loss",
    normalize_x: bool = True,
) -> None:
    sub = df.loc[df[ycol].dropna().index].copy()
    if sub.empty:
        return
    if normalize_x:
        max_x = sub[xcol].max()
        if max_x > 0:
            sub[xcol] = sub[xcol] / max_x
    if label is None:
        ax.scatter(sub[xcol], sub[ycol], marker="o", alpha=0.3, s=2)
    else:
        ax.scatter(sub[xcol], sub[ycol], marker="o", label=label, alpha=0.3, s=2)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True)


def plot_single_panel(
    paws_df: pd.DataFrame,
    base_n: int,
    target_scales: list[int],
    shrinking: float,
    tkpm: float,
    ylim: tuple[float, float],
    output_path: Path,
) -> None:
    sub = paws_df.loc[
        (paws_df.base_N == base_n)
        & (paws_df.target_N.isin(target_scales))
        & (paws_df.shrinking == shrinking)
        & (paws_df.tkpm == tkpm)
    ]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    for tn in target_scales:
        td = sub.loc[sub.target_N == tn]
        if td.empty:
            continue
        plot_curve(ax, td, label=f"Target scale: {tn / 1e6:.2f}M", normalize_x=True)
    ax.set_ylim(*ylim)
    ax.set_title(
        f"PAWS, base_N={base_n / 1e6:.2f}M, shrink={shrinking}, tkpm={tkpm}",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_panel(
    paws_df: pd.DataFrame,
    base_scales: list[int],
    target_scales: list[int],
    shrink_hps: list[float],
    tkpm_hps: list[float],
    ylim: tuple[float, float],
    output_path: Path,
) -> None:
    panel_targets = target_scales[1:] if len(target_scales) > 1 else target_scales
    fig, axes = plt.subplots(
        1, len(panel_targets),
        figsize=(len(panel_targets) * 4, 3.5),
        squeeze=False,
    )
    for i, tn in enumerate(panel_targets):
        td = paws_df.loc[paws_df.target_N == tn]
        ax = axes[0, i]
        for base_n, tkpm, shrinking in itertools.product(base_scales, tkpm_hps, shrink_hps):
            if td.empty or base_n >= tn:
                continue
            sd = td.loc[
                (td.base_N == base_n)
                & (td.tkpm == tkpm)
                & (td.shrinking == shrinking)
            ]
            if sd.empty:
                continue
            plot_curve(ax, sd, normalize_x=True)
        ax.set_title(f"Target scale: {tn / 1e6:.2f}M", fontsize=10)
        ax.set_ylim(*ylim)
    fig.suptitle("PAWS validation curves by target scale (all HP combos)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    output_dir: Path,
    raw_parquet: Path,
    paws_df: pd.DataFrame,
    base_scales: list[int],
    target_scales: list[int],
    shrink_hps: list[float],
    tkpm_hps: list[float],
    ylim: tuple[float, float],
) -> None:
    n_paws_rows = len(paws_df)
    n_paws_val = int(paws_df["Validation Loss"].notna().sum())
    notes = f"""# Ex0.5 — Demo Notebook Reproduction (New Dataset)

## Purpose

Reproduce key plot patterns from `demo_analysis/demo.ipynb` on the new flattened
PAWS dataset. Lets supervisor visually compare old vs new dataset with identical
HP filters, with no preprocessing in between.

## Source

- **Raw parquet**: `{raw_parquet}`
- Filter: `method == "paws"`
- PAWS rows (all flops): {n_paws_rows:,}
- PAWS rows with non-null `Validation Loss`: {n_paws_val:,}

This script reads the **raw** flattened parquet directly, faithful to
`demo.ipynb` which also reads from a single raw parquet. It does NOT use the
derived `validation_points_with_trajectory.parquet` from Ex0 processing.

## HP space

- `base_N`: {base_scales}
- `target_N`: {target_scales}
- `shrinking`: {shrink_hps}
- `tkpm`: {tkpm_hps}

## Plot files

- `plot_single_panel_shrink04_tkpm20.png` — matches demo cell 12 (one line per target scale, base_N=BASE_SCALES[0], shrink=0.4, tkpm=20)
- `plot_single_panel_shrink04_tkpm30.png` — matches demo cell 13 (same but tkpm=30)
- `plot_multi_panel_all_hps.png` — matches demo cell 14 (one panel per target scale, all HP combos)

## Y-axis

- Current: `[{ylim[0]}, {ylim[1]}]`
- New PAWS dataset raw range: 1.61 to 4.55
- If clipping observed at top, rerun with `--ylim 1.5 4.5`

## X-axis

- `flops` normalized to `[0, 1]` per curve via `flops / max(flops)`

## Done when

Supervisor can compare old (`demo.ipynb`) vs new (this folder) dataset
side-by-side using identical HP filters and faithful raw inputs.
"""
    (output_dir / "SUMMARY.md").write_text(notes)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.raw_parquet.exists():
        raise FileNotFoundError(f"Missing {args.raw_parquet}")

    df = pd.read_parquet(args.raw_parquet)
    paws_df = df[df.method == "paws"].copy()
    print(f"Raw rows: {len(df):,}")
    print(f"PAWS rows (raw, all flops): {len(paws_df):,}")
    print(f"PAWS rows with non-null Validation Loss: {paws_df['Validation Loss'].notna().sum():,}")

    base_scales = sorted(paws_df.base_N.unique())
    target_scales = sorted(paws_df.target_N.unique())
    shrink_hps = sorted(paws_df.shrinking.unique())
    tkpm_hps = sorted(paws_df.tkpm.unique())
    print(f"base_N: {base_scales}")
    print(f"target_N: {target_scales}")
    print(f"shrinking: {shrink_hps}")
    print(f"tkpm: {tkpm_hps}")

    ylim = tuple(args.ylim)

    print("\nPlotting single-panel cell 12 reproduction (shrink=0.4, tkpm=20)...")
    plot_single_panel(
        paws_df,
        base_n=base_scales[0],
        target_scales=target_scales,
        shrinking=0.4,
        tkpm=20.0,
        ylim=ylim,
        output_path=args.output_dir / "plot_single_panel_shrink04_tkpm20.png",
    )

    print("Plotting single-panel cell 13 reproduction (shrink=0.4, tkpm=30)...")
    plot_single_panel(
        paws_df,
        base_n=base_scales[0],
        target_scales=target_scales,
        shrinking=0.4,
        tkpm=30.0,
        ylim=ylim,
        output_path=args.output_dir / "plot_single_panel_shrink04_tkpm30.png",
    )

    print("Plotting multi-panel cell 14 reproduction (all HP combos)...")
    plot_multi_panel(
        paws_df,
        base_scales=base_scales,
        target_scales=target_scales,
        shrink_hps=shrink_hps,
        tkpm_hps=tkpm_hps,
        ylim=ylim,
        output_path=args.output_dir / "plot_multi_panel_all_hps.png",
    )

    write_notes(args.output_dir, args.raw_parquet, paws_df, base_scales, target_scales, shrink_hps, tkpm_hps, ylim)
    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
