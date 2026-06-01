"""Experiment 0.7 — context_ifbo supervisor-style figure.

Same PAWS HP slice as Ex0.6 (base_N=14.56M, shrink=0.4, tkpm=20),
but instead of prefix-only, each target gets the other 4 sibling
scale curves as full context + its own observed prefix.

This is the within-slice iFBO regime: shows what FT-PFN can do
when it sees full learning curves at neighboring scales.
"""

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

from demo_analysis._common.normalization import log_loss_range, loss_to_perf, perf_to_loss
from experiments._lib.shards import merge_shards, shard_tag

DEFAULT_RAW_PARQUET = Path("demo_analysis/warmstart_runs_flattened.parquet")
DEFAULT_OUTPUT_DIR = Path("experiments/ex0_7_context_ifbo")
DEFAULT_MODEL_PATH = Path(".model")
DEFAULT_MODEL_VERSION = "0.0.1"
DEFAULT_BASE_N = 14562560
DEFAULT_SHRINK = 0.4
DEFAULT_TKPM = 20.0
DEFAULT_OBS_FRACS = [0.05, 0.10, 0.20, 0.50, 0.90]
DEFAULT_YLIM = (1.5, 4.5)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ex0.7: context_ifbo supervisor figure.")
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


def build_curves_by_target(sliced: pd.DataFrame) -> dict[int, pd.DataFrame]:
    curves = {}
    for target_n, group in sliced.groupby("target_N", sort=True):
        curve = group.sort_values("flops").copy()
        curve["x_norm"] = curve["flops"] / curve["flops"].max()
        curves[int(target_n)] = curve
    return curves


def minmax(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        return np.zeros_like(values, dtype=np.float32), {"min": lo, "max": hi}
    return ((values - lo) / (hi - lo)).astype(np.float32), {"min": lo, "max": hi}


def compute_hp_vectors(curves: dict[int, pd.DataFrame], base_n: int, shrink: float, tkpm: float) -> dict[int, torch.Tensor]:
    targets = sorted(curves.keys())
    target_arr = np.array(targets, dtype=np.float64)
    g_arr = target_arr / base_n

    log_target = np.log10(target_arr)
    log_g = np.log10(g_arr)
    shrink_arr = np.full(len(targets), shrink, dtype=np.float64)
    tkpm_arr = np.full(len(targets), tkpm, dtype=np.float64)

    target_norm, _ = minmax(log_target)
    g_norm, _ = minmax(log_g)
    shrink_norm, _ = minmax(shrink_arr)
    tkpm_norm, _ = minmax(tkpm_arr)

    hp_map = {}
    for i, tn in enumerate(targets):
        hp = torch.tensor(
            [float(target_norm[i]), float(g_norm[i]), float(shrink_norm[i]), float(tkpm_norm[i])],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        hp_map[tn] = hp
    return hp_map


def compute_shared_log_range(curves: dict[int, pd.DataFrame], exclude_target: int) -> tuple[float, float]:
    all_losses = []
    for tn, df in curves.items():
        if tn == exclude_target:
            continue
        all_losses.append(df["Validation Loss"].to_numpy(dtype=np.float64))
    combined = np.concatenate(all_losses)
    return log_loss_range(combined)


def predict_context_ifbo(
    model: FTPFN, curves: dict[int, pd.DataFrame], obs_frac: float, hp_map: dict[int, torch.Tensor]
) -> pd.DataFrame:
    targets = sorted(curves.keys())
    rows = []

    for target_n in targets:
        target_df = curves[target_n]
        t = torch.tensor(target_df["x_norm"].to_numpy(dtype=np.float32), dtype=torch.float32)
        y_raw = torch.tensor(target_df["Validation Loss"].to_numpy(dtype=np.float32), dtype=torch.float32)

        log_min, log_max = compute_shared_log_range(curves, target_n)

        observed_mask = t <= obs_frac
        query_mask = t > obs_frac
        if int(observed_mask.sum()) == 0 or int(query_mask.sum()) == 0:
            continue

        hp = hp_map[target_n]

        context: list[Curve] = []
        for partner_n in targets:
            if partner_n == target_n:
                continue
            partner_df = curves[partner_n]
            pt = torch.tensor(partner_df["x_norm"].to_numpy(dtype=np.float32), dtype=torch.float32)
            py_raw = torch.tensor(partner_df["Validation Loss"].to_numpy(dtype=np.float32), dtype=torch.float32)
            partner_perf = loss_to_perf(py_raw, log_min, log_max)
            partner_hp = hp_map[partner_n]
            context.append(Curve(hyperparameters=partner_hp, t=pt, y=partner_perf))

        observed_perf = loss_to_perf(y_raw[observed_mask], log_min, log_max)
        context.append(Curve(hyperparameters=hp, t=t[observed_mask], y=observed_perf))

        query_t = t[query_mask]
        prediction = model.predict(context=context, query=[Curve(hyperparameters=hp, t=query_t)])[0]

        perf_q05 = prediction.quantile(0.05).detach().cpu().numpy()
        perf_q50 = prediction.quantile(0.5).detach().cpu().numpy()
        perf_q95 = prediction.quantile(0.95).detach().cpu().numpy()
        loss_q05 = perf_to_loss(perf_q05, log_min, log_max)
        loss_q50 = perf_to_loss(perf_q50, log_min, log_max)
        loss_q95 = perf_to_loss(perf_q95, log_min, log_max)
        pred_p05 = np.minimum(loss_q05, loss_q95)
        pred_p95 = np.maximum(loss_q05, loss_q95)

        future_df = target_df.loc[query_mask.detach().cpu().numpy()].copy()
        rows.append(pd.DataFrame({
            "policy": "context_ifbo",
            "target_N": int(target_n),
            "obs_frac": float(obs_frac),
            "context_size": len(context),
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


def plot_supervisor(
    curves: dict[int, pd.DataFrame],
    predictions: pd.DataFrame,
    obs_frac: float,
    ylim: tuple[float, float],
    output_path: Path,
) -> None:
    targets = sorted(curves.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    legend_handles = {}
    target_handles = {}
    for i, target_n in enumerate(targets):
        color = colors[i % len(colors)]
        curve = curves[target_n]
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
            color=color, alpha=0.18,
        )
        target_handles[f"{label} (solid/dashed)"] = obs_line
        legend_handles.setdefault("future truth (thin black)", truth_line)
        legend_handles.setdefault("FT-PFN p50 (dashed color)", pred_line)
        legend_handles.setdefault("p05-p95 uncertainty band", band)

    axes[0].set_title("Ground-truth PAWS curves + observed dots")
    axes[1].set_title("context_ifbo predictions + p05-p95 bands")
    for ax in axes:
        ax.axvline(obs_frac, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*ylim)
        ax.set_xlabel("flops / max(flops)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation Loss")
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    axes[1].legend(
        [*target_handles.values(), *legend_handles.values()],
        [*target_handles.keys(), *legend_handles.keys()],
        fontsize=7,
        loc="upper right",
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.4,
        handlelength=1.6,
    )
    fig.suptitle("Ex0.7 context_ifbo FT-PFN — base_N=14.56M, shrink=0.4, tkpm=20")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_separate_curves(
    curves: dict[int, pd.DataFrame],
    predictions: pd.DataFrame,
    obs_frac: float,
    ylim: tuple[float, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for target_n in sorted(int(target_n) for target_n in predictions["target_N"].unique()):
        curve = curves[target_n].sort_values("x_norm")
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
        ax.set_title(f"Ex0.7 context_ifbo — target_N={target_n}, obs={obs_frac:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(output_dir / f"target{target_n}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)



def plot_cascade_curves(
    curves: dict[int, pd.DataFrame],
    predictions: pd.DataFrame,
    obs_frac: float,
    ylim: tuple[float, float],
    output_path: Path,
) -> None:
    targets = sorted(int(target_n) for target_n in predictions["target_N"].unique())
    fig, axes = plt.subplots(len(targets), 1, figsize=(8, max(3.0 * len(targets), 3.5)), squeeze=False)

    for i, target_n in enumerate(targets):
        ax = axes[i, 0]
        curve = curves[target_n].sort_values("x_norm")
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

    fig.suptitle("Ex0.7 context_ifbo", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = results_dir / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    sliced = load_slice(args.raw_parquet, args.base_n, args.shrink, args.tkpm)
    curves = build_curves_by_target(sliced)
    hp_map = compute_hp_vectors(curves, args.base_n, args.shrink, args.tkpm)

    device = resolve_device(args.device)
    model = FTPFN(target_path=args.model_path, version=args.model_version, device=device)

    last_predictions = None
    last_diagnostics = None
    last_predicted_targets: list[int] = []
    last_skipped_targets: list[int] = []
    for obs_frac in args.obs_fracs:
        predictions = predict_context_ifbo(model, curves, obs_frac, hp_map)
        diagnostics = None
        obs_dir = obs_dir_name(obs_frac)
        predicted_targets = sorted(int(target_n) for target_n in predictions["target_N"].unique())
        skipped_targets = [target_n for target_n in sorted(curves.keys()) if target_n not in predicted_targets]

        shard = partial_dir / f"context_ifbo__obs{shard_tag(obs_frac)}.parquet"
        predictions.to_parquet(shard, index=False)
        plot_supervisor(
            curves,
            predictions,
            obs_frac,
            (float(args.ylim[0]), float(args.ylim[1])),
            args.output_dir / "plots" / obs_dir / "supervisor_styles" / "context_ifbo_supervisor_style.png"
        )
        plot_separate_curves(
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
            args.output_dir / "plots" / obs_dir / "cascade_styles" / "context_ifbo_cascade_curves.png",
        )
        last_predictions = predictions
        last_diagnostics = diagnostics
        last_predicted_targets = predicted_targets
        last_skipped_targets = skipped_targets
        print(f"[checkpoint] {shard.name} ({len(predictions):,} rows)")

    merge_shards(partial_dir, results_dir, "context_ifbo_predictions.parquet")

    print(f"Targets: {sorted(curves.keys())}")
    print(f"Predicted targets: {last_predicted_targets}")
    print(f"Skipped targets: {last_skipped_targets}")
    print(f"Obs fracs: {args.obs_fracs}")
    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
