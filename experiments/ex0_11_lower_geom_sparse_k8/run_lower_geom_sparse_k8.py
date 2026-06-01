"""Experiment 0.11 — lower_geom_sparse_k8 supervisor-style figure.

Same PAWS HP slice as Ex0.6-0.10, but each query target gets every
strictly lower-scale curve subsampled to 8 geometric points plus its own
observed prefix.
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
DEFAULT_OUTPUT_DIR = Path("experiments/ex0_11_lower_geom_sparse_k8")
DEFAULT_MODEL_PATH = Path(".model")
DEFAULT_MODEL_VERSION = "0.0.1"
DEFAULT_BASE_N = 14562560
DEFAULT_SHRINK = 0.4
DEFAULT_TKPM = 20.0
DEFAULT_OBS_FRACS = [0.05, 0.10, 0.20, 0.50, 0.90]
DEFAULT_YLIM = (1.5, 4.5)
SPARSE_K = 8
SKIPPED_TARGETS = {32270848}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ex0.11: lower_geom_sparse_k8 supervisor figure.")
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


def geom_sparse_indices(n_points: int, k: int = SPARSE_K) -> np.ndarray:
    if n_points <= k:
        return np.arange(n_points)
    log_idx = np.geomspace(1, n_points, num=k)
    return np.unique(np.clip(np.round(log_idx).astype(int) - 1, 0, n_points - 1))


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

    target_norm, _ = minmax(np.log10(target_arr))
    g_norm, _ = minmax(np.log10(g_arr))
    shrink_norm, _ = minmax(np.full(len(targets), shrink, dtype=np.float64))
    tkpm_norm, _ = minmax(np.full(len(targets), tkpm, dtype=np.float64))

    hp_map = {}
    for i, target_n in enumerate(targets):
        hp_map[target_n] = torch.tensor(
            [float(target_norm[i]), float(g_norm[i]), float(shrink_norm[i]), float(tkpm_norm[i])],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
    return hp_map


def compute_loo_log_range(curves: dict[int, pd.DataFrame], exclude_target: int) -> tuple[float, float]:
    losses = [
        curve["Validation Loss"].to_numpy(dtype=np.float64)
        for target_n, curve in curves.items()
        if target_n != exclude_target
    ]
    return log_loss_range(np.concatenate(losses))


def predict_lower_geom_sparse_k8(
    model: FTPFN,
    curves: dict[int, pd.DataFrame],
    obs_frac: float,
    hp_map: dict[int, torch.Tensor],
) -> pd.DataFrame:
    targets = sorted(curves.keys())
    rows = []

    for target_n in targets:
        if target_n in SKIPPED_TARGETS:
            continue
        lower_targets = [partner_n for partner_n in targets if partner_n < target_n]
        if not lower_targets:
            continue

        target_df = curves[target_n]
        t = torch.tensor(target_df["x_norm"].to_numpy(dtype=np.float32), dtype=torch.float32)
        y_raw = torch.tensor(target_df["Validation Loss"].to_numpy(dtype=np.float32), dtype=torch.float32)
        observed_mask = t <= obs_frac
        query_mask = t > obs_frac
        if int(observed_mask.sum()) == 0 or int(query_mask.sum()) == 0:
            continue

        log_min, log_max = compute_loo_log_range(curves, target_n)
        hp = hp_map[target_n]
        context: list[Curve] = []
        selected_counts = []

        for partner_n in lower_targets:
            partner_df = curves[partner_n].sort_values("x_norm")
            idx = geom_sparse_indices(len(partner_df), SPARSE_K)
            pt = torch.tensor(partner_df["x_norm"].to_numpy(dtype=np.float32)[idx], dtype=torch.float32)
            py_raw = torch.tensor(partner_df["Validation Loss"].to_numpy(dtype=np.float32)[idx], dtype=torch.float32)
            partner_perf = loss_to_perf(py_raw, log_min, log_max)
            context.append(Curve(hyperparameters=hp_map[partner_n], t=pt, y=partner_perf))
            selected_counts.append(int(len(idx)))

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
            "policy": "lower_geom_sparse_k8",
            "target_N": int(target_n),
            "obs_frac": float(obs_frac),
            "context_size": len(context),
            "n_lower_partners": len(lower_targets),
            "points_per_partner": max(selected_counts),
            "x_norm": future_df["x_norm"].to_numpy(dtype=float),
            "flops": future_df["flops"].to_numpy(dtype=float),
            "y_true": future_df["Validation Loss"].to_numpy(dtype=float),
            "pred_p05": pred_p05,
            "pred_p50": loss_q50,
            "pred_p95": pred_p95,
            "loss_log_min": log_min,
            "loss_log_max": log_max,
        }))

    if not rows:
        raise RuntimeError("No targets had lower partners and query points")
    return pd.concat(rows, ignore_index=True)


def uncertainty_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for target_n, group in predictions.groupby("target_N", sort=True):
        ordered = group.sort_values("x_norm")
        band = ordered["pred_p95"].to_numpy(dtype=float) - ordered["pred_p05"].to_numpy(dtype=float)
        records.append({
            "target_N": int(target_n),
            "n_lower_partners": int(ordered["n_lower_partners"].iloc[0]),
            "points_per_partner": int(ordered["points_per_partner"].iloc[0]),
            "band_width_start": float(band[0]),
            "band_width_final": float(band[-1]),
            "band_width_ratio": float(band[0] / max(band[-1], EPS)),
        })
    return pd.DataFrame(records)


def plot_supervisor(
    curves: dict[int, pd.DataFrame],
    predictions: pd.DataFrame,
    obs_frac: float,
    ylim: tuple[float, float],
    output_path: Path,
    *,
    base_n: int,
    shrink: float,
    tkpm: float,
) -> None:
    targets = sorted(curves.keys())
    predicted_targets = set(predictions["target_N"].unique())
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

        if target_n not in predicted_targets:
            continue

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
        target_handles[f"{label} ({int(pred['n_lower_partners'].iloc[0])}×k8)"] = obs_line
        legend_handles.setdefault("future truth (thin black)", truth_line)
        legend_handles.setdefault("FT-PFN p50 (dashed color)", pred_line)
        legend_handles.setdefault("p05-p95 uncertainty band", band)

    axes[0].set_title("Ground-truth PAWS curves + observed dots")
    axes[1].set_title("lower_geom_sparse_k8 predictions + p05-p95 bands")
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
    fig.suptitle(
        f"Ex0.11 lower_geom_sparse_k8 FT-PFN — base_N={base_n / 1e6:.2f}M, "
        f"shrink={shrink:g}, tkpm={tkpm:g}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
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

    fig.suptitle("Ex0.11 lower_geom_sparse_k8", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    output_dir: Path,
    targets: list[int],
    skipped_targets: list[int],
    diagnostics: pd.DataFrame,
    *,
    base_n: int,
    shrink: float,
    tkpm: float,
    obs_frac: float,
) -> None:
    diagnostics_md = diagnostics.to_markdown(index=False)
    output_dir.joinpath("SUMMARY.md").write_text(
        f"""# Ex0.11 — lower_geom_sparse_k8 FT-PFN Figure

## Purpose

Supervisor-style lower-to-higher transfer sanity check on the same PAWS HP slice as Ex0.6-0.10.
Each predicted target uses 8 geometrically-spaced points from strictly lower-scale curves plus its own observed prefix.

## Slice

- `method == "paws"`
- `base_N = {base_n}`
- `shrink = {shrink}`
- `tkpm = {tkpm}`
- `obs_frac = {obs_frac}`

## Policy

- `lower_geom_sparse_k8`: partner curves must satisfy `partner_target_N < query_target_N`.
- each lower partner contributes at most 8 geometric points biased toward early x values.
- targets with zero lower partners are skipped; 32M is skipped on this slice.
- normalization uses LOO log-loss range over all non-query PAWS slice trajectories.
- skipped targets: {skipped_targets}
- predicted targets: {targets}

## Outputs

- `lower_geom_sparse_k8_predictions.parquet` — full prediction rows
- `uncertainty_diagnostics.csv` — band width diagnostics
- `summary.json` — run metadata/config
- `plots/obs_<NNN>/supervisor_styles/lower_geom_sparse_k8_supervisor_style.png` — ground truth plus sparse lower-curve predictions
- `plots/obs_<NNN>/cascade_styles/lower_geom_sparse_k8_cascade_curves.png` — aligned cascade-style multi-target panel

## Band diagnostics

{diagnostics_md}

## Interpretation

Compare against Ex0.8 all_lower_only and Ex0.10 lower_final_pt_only to test whether sparse curve shape is enough.
""",
        encoding="utf-8",
    )


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
        predictions = predict_lower_geom_sparse_k8(model, curves, obs_frac, hp_map)
        diagnostics = uncertainty_diagnostics(predictions)
        obs_dir = obs_dir_name(obs_frac)
        predicted_targets = sorted(int(target_n) for target_n in predictions["target_N"].unique())
        skipped_targets = [target_n for target_n in sorted(curves.keys()) if target_n not in predicted_targets]
        if not SKIPPED_TARGETS.issubset(skipped_targets):
            raise RuntimeError(f"Configured skipped targets were predicted: {sorted(SKIPPED_TARGETS - set(skipped_targets))}")

        shard = partial_dir / f"lower_geom_sparse_k8__obs{shard_tag(obs_frac)}.parquet"
        predictions.to_parquet(shard, index=False)
        diagnostics.to_csv(results_dir / "uncertainty_diagnostics.csv", index=False)
        plot_supervisor(
            curves,
            predictions,
            obs_frac,
            (float(args.ylim[0]), float(args.ylim[1])),
            args.output_dir / "plots" / obs_dir / "supervisor_styles" / "lower_geom_sparse_k8_supervisor_style.png",
            base_n=args.base_n,
            shrink=args.shrink,
            tkpm=args.tkpm,
        )
        plot_cascade_curves(
            curves,
            predictions,
            obs_frac,
            (float(args.ylim[0]), float(args.ylim[1])),
            args.output_dir / "plots" / obs_dir / "cascade_styles" / "lower_geom_sparse_k8_cascade_curves.png",
        )
        last_predictions = predictions
        last_diagnostics = diagnostics
        last_predicted_targets = predicted_targets
        last_skipped_targets = skipped_targets
        print(f"[checkpoint] {shard.name} ({len(predictions):,} rows)")

    merge_shards(partial_dir, results_dir, "lower_geom_sparse_k8_predictions.parquet")

    summary = {
        "experiment": "ex0_11_lower_geom_sparse_k8",
        "raw_parquet": str(args.raw_parquet),
        "model_path": str(args.model_path),
        "model_version": args.model_version,
        "policy": "lower_geom_sparse_k8",
        "hp_slice": {
            "method": "paws",
            "base_N": args.base_n,
            "shrink": args.shrink,
            "tkpm": args.tkpm,
            "target_N": sorted(curves.keys()),
        },
        "obs_fracs": args.obs_fracs,
        "ylim": [float(args.ylim[0]), float(args.ylim[1])],
        "predicted_targets": last_predicted_targets,
        "skipped_targets": last_skipped_targets,
        "n_prediction_rows": int(len(pd.read_parquet(results_dir / "lower_geom_sparse_k8_predictions.parquet"))),
        "seed": args.seed,
        "device": str(device),
        "normalization_policy": "LOO log-loss range over all non-query PAWS slice trajectories",
        "context_size_by_target": {
            str(int(target_n)): int(size)
            for target_n, size in last_predictions.groupby("target_N")["context_size"].first().items()
        },
        "n_lower_partners_by_target": {
            str(int(target_n)): int(n)
            for target_n, n in last_predictions.groupby("target_N")["n_lower_partners"].first().items()
        },
        "points_per_partner_by_target": {
            str(int(target_n)): int(n)
            for target_n, n in last_predictions.groupby("target_N")["points_per_partner"].first().items()
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if last_diagnostics is not None:
        write_notes(
            args.output_dir,
            last_predicted_targets,
            last_skipped_targets,
            last_diagnostics,
            base_n=args.base_n,
            shrink=args.shrink,
            tkpm=args.tkpm,
            obs_frac=args.obs_fracs[-1],
        )

    print(f"Targets: {sorted(curves.keys())}")
    print(f"Predicted targets: {last_predicted_targets}")
    print(f"Skipped targets: {last_skipped_targets}")
    print(f"Obs fracs: {args.obs_fracs}")
    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
