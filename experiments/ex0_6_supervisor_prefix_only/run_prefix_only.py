"""Experiment 0.6 — supervisor-style FT-PFN prefix-only figure."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ifbo import Curve
from ifbo.surrogate import FTPFN
from utils import get_device

from demo_analysis._common.normalization import loss_to_perf, perf_to_loss
from experiments._lib.shards import merge_shards, shard_tag

DEFAULT_RAW_PARQUET = Path("demo_analysis/warmstart_runs_flattened.parquet")
DEFAULT_OUTPUT_DIR = Path("experiments/ex0_6_supervisor_prefix_only")
DEFAULT_MODEL_PATH = Path(".model")
DEFAULT_MODEL_VERSION = "0.0.1"
DEFAULT_BASE_N = 14562560
DEFAULT_SHRINK = 0.4
DEFAULT_TKPM = 20.0
DEFAULT_OBS_FRACS = [0.05, 0.10, 0.20, 0.50, 0.90]
DEFAULT_YLIM = (1.5, 4.5)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ex0.6: prefix-only FT-PFN supervisor figure.")
    parser.add_argument("--raw-parquet", type=Path, default=DEFAULT_RAW_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--base-n", type=int, default=DEFAULT_BASE_N)
    parser.add_argument("--shrink", type=float, default=DEFAULT_SHRINK)
    parser.add_argument("--tkpm", type=float, default=DEFAULT_TKPM)
    parser.add_argument("--obs-fracs", type=float, nargs="+", default=DEFAULT_OBS_FRACS)
    parser.add_argument("--ylim", type=float, nargs=2, default=list(DEFAULT_YLIM))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return get_device()
    return torch.device(name)


def obs_dir_name(obs_frac: float) -> str:
    return f"obs_{int(round(obs_frac * 100)):03d}"


def load_slice(raw_parquet: Path, base_n: int, shrink: float, tkpm: float) -> pd.DataFrame:
    if not raw_parquet.exists():
        raise FileNotFoundError(f"Missing raw parquet: {raw_parquet}")
    raw = pd.read_parquet(raw_parquet)
    if "shrink" not in raw.columns and "shrinking" in raw.columns:
        raw = raw.assign(shrink=raw["shrinking"])
    required = {"method", "base_N", "target_N", "shrink", "tkpm", "flops", "Validation Loss"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw parquet missing columns: {missing}")
    sliced = raw.loc[
        (raw["method"] == "paws")
        & (raw["base_N"] == base_n)
        & np.isclose(raw["shrink"], shrink)
        & np.isclose(raw["tkpm"], tkpm)
        & raw["Validation Loss"].notna()
    ].copy()
    if sliced.empty:
        raise ValueError(f"No PAWS rows for base_N={base_n}, shrink={shrink}, tkpm={tkpm}")
    return sliced


def validate_one_curve_per_target(curves: pd.DataFrame) -> None:
    key_cols = ["seed", "warmstart_type", "g_width", "g_N"]
    for target_n, group in curves.groupby("target_N", sort=True):
        duplicate_flops = group["flops"].duplicated().sum()
        non_unique_keys = {
            col: int(group[col].nunique(dropna=False))
            for col in key_cols
            if col in group.columns and int(group[col].nunique(dropna=False)) != 1
        }
        if duplicate_flops or non_unique_keys:
            raise ValueError(
                f"Expected one curve for target_N={int(target_n)}; "
                f"duplicate_flops={int(duplicate_flops)}, non_unique_keys={non_unique_keys}"
            )


def minmax(values: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(values)), index=values.index), {"min": lo, "max": hi}
    return (values - lo) / (hi - lo), {"min": lo, "max": hi}


def add_hp_columns(curves: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    summary = curves.groupby("target_N", as_index=False).agg(
        base_N=("base_N", "first"),
        shrink=("shrink", "first"),
        tkpm=("tkpm", "first"),
    )
    summary["G"] = summary["target_N"] / summary["base_N"]
    summary["target_N_hp"] = np.log10(summary["target_N"])
    summary["G_hp"] = np.log10(summary["G"])
    summary["shrink_hp"] = summary["shrink"]
    summary["tkpm_hp"] = summary["tkpm"]

    ranges: dict[str, dict[str, float]] = {}
    for src, dst in [
        ("target_N_hp", "target_N_norm"),
        ("G_hp", "G_norm"),
        ("shrink_hp", "shrink_norm"),
        ("tkpm_hp", "tkpm_norm"),
    ]:
        summary[dst], ranges[dst] = minmax(summary[src].astype(float))
    return curves.merge(summary, on=["target_N", "base_N", "shrink", "tkpm"], how="left"), ranges


def build_curve(row: pd.Series, obs_frac: float, log_min: float, log_max: float) -> tuple[Curve, torch.Tensor, torch.Tensor, torch.Tensor]:
    curve_df = row.sort_values("x_norm")
    t = torch.tensor(curve_df["x_norm"].to_numpy(dtype=np.float32), dtype=torch.float32)
    y_raw = torch.tensor(curve_df["Validation Loss"].to_numpy(dtype=np.float32), dtype=torch.float32)
    hp_values = curve_df[["target_N_norm", "G_norm", "shrink_norm", "tkpm_norm"]].iloc[0].to_numpy(dtype=np.float32)
    hp = torch.tensor(hp_values, dtype=torch.float32).clamp(0.0, 1.0)
    observed_mask = t <= obs_frac
    if int(observed_mask.sum()) == 0:
        raise ValueError(f"No observed prefix for target_N={int(curve_df['target_N'].iloc[0])}")
    observed_y = loss_to_perf(y_raw[observed_mask], log_min, log_max)
    return Curve(hyperparameters=hp, t=t[observed_mask], y=observed_y), t, y_raw, hp


def predict_prefix_only(model: FTPFN, curves: pd.DataFrame, obs_frac: float) -> pd.DataFrame:
    rows = []
    for target_n, curve_df in curves.groupby("target_N", sort=True):
        # LOO normalization: use all curves EXCEPT this target
        other_losses = curves[curves["target_N"] != target_n]["Validation Loss"].to_numpy(dtype=float)
        log_values = np.log(np.clip(other_losses, EPS, None))
        log_min = float(np.min(log_values))
        log_max = float(np.max(log_values))
        if math.isclose(log_min, log_max):
            log_max = log_min + EPS

        context_curve, t, y_raw, hp = build_curve(curve_df, obs_frac, log_min, log_max)
        query_mask = t > obs_frac
        if int(query_mask.sum()) == 0:
            raise ValueError(f"No query points for target_N={int(target_n)}")
        query_t = t[query_mask]
        prediction = model.predict(context=[context_curve], query=[Curve(hyperparameters=hp, t=query_t)])[0]
        perf_q05 = prediction.quantile(0.05).detach().cpu().numpy()
        perf_q50 = prediction.quantile(0.5).detach().cpu().numpy()
        perf_q95 = prediction.quantile(0.95).detach().cpu().numpy()
        loss_q05 = perf_to_loss(perf_q05, log_min, log_max)
        loss_q50 = perf_to_loss(perf_q50, log_min, log_max)
        loss_q95 = perf_to_loss(perf_q95, log_min, log_max)
        pred_p05 = np.minimum(loss_q05, loss_q95)
        pred_p95 = np.maximum(loss_q05, loss_q95)
        future_df = curve_df.sort_values("x_norm").loc[query_mask.detach().cpu().numpy()].copy()
        rows.append(pd.DataFrame({
            "policy": "prefix_only",
            "target_N": int(target_n),
            "base_N": int(future_df["base_N"].iloc[0]),
            "G": float(future_df["G"].iloc[0]),
            "shrink": float(future_df["shrink"].iloc[0]),
            "tkpm": float(future_df["tkpm"].iloc[0]),
            "obs_frac": float(obs_frac),
            "x_norm": future_df["x_norm"].to_numpy(dtype=float),
            "flops": future_df["flops"].to_numpy(dtype=float),
            "y_true": future_df["Validation Loss"].to_numpy(dtype=float),
            "pred_p05": pred_p05,
            "pred_p50": loss_q50,
            "pred_p95": pred_p95,
            "loss_log_min": log_min,
            "loss_log_max": log_max,
        }))
    return pd.concat(rows, ignore_index=True)


def uncertainty_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for target_n, group in predictions.groupby("target_N", sort=True):
        ordered = group.sort_values("x_norm")
        band_width = ordered["pred_p95"].to_numpy(dtype=float) - ordered["pred_p05"].to_numpy(dtype=float)
        start = float(band_width[0])
        final = float(band_width[-1])
        records.append({
            "policy": "prefix_only",
            "target_N": int(target_n),
            "base_N": int(ordered["base_N"].iloc[0]),
            "G": float(ordered["G"].iloc[0]),
            "shrink": float(ordered["shrink"].iloc[0]),
            "tkpm": float(ordered["tkpm"].iloc[0]),
            "obs_frac": float(ordered["obs_frac"].iloc[0]),
            "band_width_start": start,
            "band_width_final": final,
            "band_width_ratio": start / max(final, EPS),
        })
    return pd.DataFrame(records)


def plot_supervisor(curves: pd.DataFrame, predictions: pd.DataFrame, obs_frac: float, ylim: tuple[float, float], output_path: Path) -> None:
    targets = sorted(curves["target_N"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    legend_handles = {}
    target_handles = {}
    for i, target_n in enumerate(targets):
        color = colors[i % len(colors)]
        curve = curves.loc[curves["target_N"] == target_n].sort_values("x_norm")
        observed = curve.loc[curve["x_norm"] <= obs_frac]
        future = curve.loc[curve["x_norm"] > obs_frac]
        label = f"{target_n / 1e6:.0f}M"
        axes[0].plot(curve["x_norm"], curve["Validation Loss"], color=color, linewidth=1.5, label=label)
        axes[0].scatter(observed["x_norm"], observed["Validation Loss"], color=color, s=10, alpha=0.8)
        (obs_line,) = axes[1].plot(observed["x_norm"], observed["Validation Loss"], color=color, linewidth=1.8)
        (truth_line,) = axes[1].plot(future["x_norm"], future["Validation Loss"], color="#111827", linewidth=1.0, alpha=0.45)

        pred = predictions.loc[predictions["target_N"] == target_n].sort_values("x_norm")
        x = pred["x_norm"].to_numpy(dtype=float)
        (pred_line,) = axes[1].plot(x, pred["pred_p50"].to_numpy(dtype=float), color=color, linestyle="--", linewidth=1.6)
        band = axes[1].fill_between(
            x,
            pred["pred_p05"].to_numpy(dtype=float),
            pred["pred_p95"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
        )
        target_handles[label] = obs_line
        legend_handles.setdefault("future truth", truth_line)

    axes[0].set_title("Ground-truth PAWS curves + observed dots")
    axes[1].set_title("FT-PFN prefix-only predictions + p05-p95 bands")
    for ax in axes:
        ax.axvline(obs_frac, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*ylim)
        ax.set_xlabel("flops / max(flops)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation Loss")
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    all_handles = list(target_handles.values()) + list(legend_handles.values())
    all_labels = [f"{k} (solid/dashed)" for k in target_handles] + list(legend_handles.keys())
    axes[1].legend(
        all_handles,
        all_labels,
        fontsize=7,
        loc="upper right",
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.4,
        handlelength=1.6,
    )
    fig.suptitle("Ex0.6 prefix-only FT-PFN — base_N=14.56M, shrink=0.4, tkpm=20")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_supervisor_curves(
    curves: pd.DataFrame,
    predictions: pd.DataFrame,
    obs_frac: float,
    ylim: tuple[float, float],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    targets = sorted(int(target_n) for target_n in predictions["target_N"].unique())

    for target_n in targets:
        curve = curves.loc[curves["target_N"] == target_n].sort_values("x_norm")
        pred = predictions.loc[predictions["target_N"] == target_n].sort_values("x_norm")
        observed = curve.loc[curve["x_norm"] <= obs_frac]
        future = curve.loc[curve["x_norm"] > obs_frac]

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(
            observed["x_norm"],
            observed["Validation Loss"],
            color="#f97316",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            label="observed prefix",
        )
        ax.plot(
            future["x_norm"],
            future["Validation Loss"],
            color="#111827",
            linewidth=1.4,
            label="future truth",
        )
        x = pred["x_norm"].to_numpy(dtype=float)
        ax.plot(
            x,
            pred["pred_p50"].to_numpy(dtype=float),
            color="#2563eb",
            linestyle="--",
            linewidth=1.8,
            label="FT-PFN p50",
        )
        ax.fill_between(
            x,
            pred["pred_p05"].to_numpy(dtype=float),
            pred["pred_p95"].to_numpy(dtype=float),
            color="#2563eb",
            alpha=0.18,
            label="p05-p95 band",
        )
        ax.axvline(obs_frac, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*ylim)
        ax.set_xlabel("flops / max(flops)")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"Ex0.6 prefix_only — target_N={target_n}, obs={obs_frac:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        fig.tight_layout()

        output_path = output_dir / f"target{target_n}.png"
        fig.savefig(output_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(output_path))

    return outputs



def plot_cascade_curves(
    curves: pd.DataFrame,
    predictions: pd.DataFrame,
    obs_frac: float,
    ylim: tuple[float, float],
    output_path: Path,
) -> None:
    targets = sorted(int(target_n) for target_n in predictions["target_N"].unique())
    fig, axes = plt.subplots(len(targets), 1, figsize=(8, max(3.0 * len(targets), 3.5)), squeeze=False)

    for i, target_n in enumerate(targets):
        ax = axes[i, 0]
        curve = curves.loc[curves["target_N"] == target_n].sort_values("x_norm")
        pred = predictions.loc[predictions["target_N"] == target_n].sort_values("x_norm")
        observed = curve.loc[curve["x_norm"] <= obs_frac]
        future = curve.loc[curve["x_norm"] > obs_frac]

        ax.plot(observed["x_norm"], observed["Validation Loss"], color="#f97316", linewidth=1.8, label="observed prefix")
        ax.plot(future["x_norm"], future["Validation Loss"], color="#111827", linewidth=1.5, label="future truth")
        x = pred["x_norm"].to_numpy(dtype=float)
        ax.plot(x, pred["pred_p50"].to_numpy(dtype=float), color="#2563eb", linestyle="--", linewidth=1.4, alpha=0.9, label="prediction p50")
        ax.fill_between(
            x,
            pred["pred_p05"].to_numpy(dtype=float),
            pred["pred_p95"].to_numpy(dtype=float),
            color="#60a5fa",
            alpha=0.22,
            label="p05-p95 band",
        )
        ax.axvline(obs_frac, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.set_title(f"target_N={int(target_n / 1e6)}M, obs={obs_frac:.2f}", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*ylim)
        ax.set_xlabel("flops / max(flops)")
        ax.set_ylabel("Validation Loss")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="best")

    fig.suptitle("Ex0.6 prefix_only", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_notes(output_dir: Path, predictions: pd.DataFrame, diagnostics: pd.DataFrame, args: argparse.Namespace) -> None:
    summary = diagnostics[["target_N", "band_width_start", "band_width_final", "band_width_ratio"]]
    notes = f"""# Ex0.6 — Supervisor Prefix-Only FT-PFN Figure

## Purpose

Paper-style prefix-only FT-PFN extrapolation figure on same PAWS HP slice as Ex0.5.
This is not lower-context comparison; no partner curves enter context.

## Slice

- `method == "paws"`
- `base_N = {args.base_n}`
- `shrink = {args.shrink}`
- `tkpm = {args.tkpm}`
- `obs_fracs = {args.obs_fracs}`

## Outputs

- `prefix_only_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/prefix_only_supervisor_style.png` — ground truth plus prefix-only predictions
- `plots/obs_<NNN>/supervisor_styles/*.png` — one prediction panel per target curve
- `plots/obs_<NNN>/cascade_styles/*.png` — stacked prediction panels for all predicted targets

## Band diagnostics

{summary.to_markdown(index=False)}

## Interpretation

Inspect whether p05-p95 bands inflate immediately after observed prefix and whether this varies with target scale.
"""
    (output_dir / "SUMMARY.md").write_text(notes)


def write_summary(output_dir: Path, args: argparse.Namespace, curves: pd.DataFrame, predictions: pd.DataFrame, hp_ranges: dict[str, dict[str, float]]) -> None:
    summary = {
        "experiment": "ex0_6_supervisor_prefix_only",
        "raw_parquet": str(args.raw_parquet),
        "model_path": str(args.model_path),
        "model_version": args.model_version,
        "policy": "prefix_only",
        "hp_slice": {
            "method": "paws",
            "base_N": args.base_n,
            "shrink": args.shrink,
            "tkpm": args.tkpm,
            "target_N": [int(x) for x in sorted(curves["target_N"].unique())],
        },
        "obs_fracs": args.obs_fracs,
        "ylim": [float(args.ylim[0]), float(args.ylim[1])],
        "hp_ranges": hp_ranges,
        "n_raw_validation_rows": int(len(curves)),
        "n_prediction_rows": int(len(predictions)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = results_dir / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    curves = load_slice(args.raw_parquet, args.base_n, args.shrink, args.tkpm)
    validate_one_curve_per_target(curves)
    curves = curves.sort_values(["target_N", "flops"]).copy()
    curves["x_norm"] = curves.groupby("target_N")["flops"].transform(lambda x: x / x.max())
    curves, hp_ranges = add_hp_columns(curves)

    device = resolve_device(args.device)
    model = FTPFN(target_path=args.model_path, version=args.model_version, device=device)

    last_predictions = None
    last_diagnostics = None
    for obs_frac in args.obs_fracs:
        predictions = predict_prefix_only(model, curves, obs_frac)
        diagnostics = uncertainty_diagnostics(predictions)
        obs_dir = obs_dir_name(obs_frac)
        shard = partial_dir / f"prefix_only__obs{shard_tag(obs_frac)}.parquet"
        predictions.to_parquet(shard, index=False)
        diagnostics.to_csv(results_dir / "uncertainty_diagnostics.csv", index=False)
        write_summary(results_dir, args, curves, predictions, hp_ranges)
        plot_supervisor(
            curves,
            predictions,
            obs_frac,
            (float(args.ylim[0]), float(args.ylim[1])),
            args.output_dir / "plots" / obs_dir / "supervisor_styles" / "prefix_only_supervisor_style.png",
        )
        plot_supervisor_curves(
            curves,
            predictions,
            obs_frac,
            (float(args.ylim[0]), float(args.ylim[1])),
            args.output_dir / "plots" / obs_dir / "supervisor_styles",
        )
        plot_cascade_curves(
            curves,
            predictions,
            obs_frac,
            (float(args.ylim[0]), float(args.ylim[1])),
            args.output_dir / "plots" / obs_dir / "cascade_styles" / "prefix_only_cascade_curves.png",
        )
        last_predictions = predictions
        last_diagnostics = diagnostics
        print(f"[checkpoint] {shard.name} ({len(predictions):,} rows)")

    merge_shards(partial_dir, results_dir, "prefix_only_predictions.parquet")
    if last_predictions is not None and last_diagnostics is not None:
        write_notes(args.output_dir, last_predictions, last_diagnostics, args)

    print(f"Raw validation rows: {len(curves):,}")
    print(f"Obs fracs: {args.obs_fracs}")
    print(f"Targets: {sorted(int(x) for x in curves['target_N'].unique())}")
    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
