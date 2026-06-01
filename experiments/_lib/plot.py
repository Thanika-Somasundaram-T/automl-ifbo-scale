"""Shared supervisor-style prediction plots and uncertainty diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_PREDICTIONS = Path("experiments/width_transfer/results/predictions.parquet")
DEFAULT_RAW_PARQUET = Path("demo_analysis/warmstart_runs_flattened.parquet")
DEFAULT_OUTPUT_DIR = Path("experiments/width_transfer/plots")
DEFAULT_DIAGNOSTICS = Path("experiments/width_transfer/results/uncertainty_diagnostics.csv")
DEFAULT_POLICIES: list[str] = []
DEFAULT_BASE_N = 14562560
DEFAULT_SHRINK = 0.4
DEFAULT_TKPM = 20.0
DEFAULT_YLIM = (1.5, 4.5)
EPS = 1e-12


POLICY_TITLES: dict[str, str] = {}


def parse_args(
    description: str = "Supervisor-style prediction plots.",
    default_predictions: Path = DEFAULT_PREDICTIONS,
    default_raw_parquet: Path = DEFAULT_RAW_PARQUET,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    default_diagnostics: Path = DEFAULT_DIAGNOSTICS,
    default_policies: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--predictions", type=Path, default=default_predictions)
    parser.add_argument("--raw-parquet", type=Path, default=default_raw_parquet)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--diagnostics", type=Path, default=default_diagnostics)
    parser.add_argument("--policies", nargs="+", default=default_policies or DEFAULT_POLICIES)
    parser.add_argument("--base-n", type=int, default=DEFAULT_BASE_N)
    parser.add_argument("--shrink", type=float, default=DEFAULT_SHRINK)
    parser.add_argument("--tkpm", type=float, default=DEFAULT_TKPM)
    parser.add_argument("--obs-fracs", type=float, nargs="+", default=None)
    parser.add_argument("--ylim", type=float, nargs=2, default=list(DEFAULT_YLIM))
    return parser.parse_args()


def load_raw_slice(raw_parquet: Path, base_n: int, shrink: float, tkpm: float) -> pd.DataFrame:
    if not raw_parquet.exists():
        raise FileNotFoundError(
            f"Missing raw parquet: {raw_parquet}\n"
            "The raw warmstart dataset is untracked by git (not committed) and "
            "the default path is relative to the repo root. Run from the repo "
            "root, or pass --raw-parquet with an absolute path. See DATA.md."
        )
    raw = pd.read_parquet(raw_parquet)
    required = {"method", "base_N", "target_N", "shrinking", "tkpm", "flops", "Validation Loss"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw parquet missing columns: {missing}")
    sliced = raw.loc[
        (raw["method"] == "paws")
        & (raw["base_N"] == base_n)
        & np.isclose(raw["shrinking"], shrink)
        & np.isclose(raw["tkpm"], tkpm)
        & raw["Validation Loss"].notna()
    ].copy()
    if sliced.empty:
        raise ValueError(
            f"No raw PAWS rows for base_N={base_n}, shrink={shrink}, tkpm={tkpm}"
        )
    return sliced


def load_predictions(predictions_path: Path, policies: list[str], shrink: float, tkpm: float) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing predictions: {predictions_path}. Run predict_width_extrapolation.py first."
        )
    predictions = pd.read_parquet(predictions_path)
    required = {
        "policy",
        "trajectory_id",
        "query_target_N",
        "base_N",
        "target_N",
        "shrink",
        "tkpm",
        "obs_frac",
        "x_norm",
        "y_true",
        "pred_p05",
        "pred_p50",
        "pred_p95",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")
    filtered = predictions.loc[
        predictions["policy"].isin(policies)
        & np.isclose(predictions["shrink"], shrink)
        & np.isclose(predictions["tkpm"], tkpm)
    ].copy()
    if filtered.empty:
        raise ValueError(f"No predictions for policies={policies}, shrink={shrink}, tkpm={tkpm}")
    return filtered


def resolve_base_n(predictions: pd.DataFrame, requested_base_n: int) -> int:
    available = set(int(x) for x in predictions["base_N"].unique())
    if requested_base_n not in available:
        raise ValueError(f"base_N={requested_base_n} not present in predictions; available={sorted(available)}")
    return int(requested_base_n)


def raw_curve(raw_slice: pd.DataFrame, target_n: int) -> pd.DataFrame:
    curve = raw_slice.loc[raw_slice["target_N"] == target_n].sort_values("flops").copy()
    if curve.empty:
        return curve
    max_flops = float(curve["flops"].max())
    curve["x_norm"] = curve["flops"] / max_flops if max_flops > 0 else 0.0
    return curve


def uncertainty_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = ["policy", "trajectory_id", "obs_frac"]
    for (policy, trajectory_id, obs_frac), group in predictions.groupby(group_cols, sort=True):
        ordered = group.sort_values("x_norm")
        band_width = ordered["pred_p95"].to_numpy(dtype=float) - ordered["pred_p05"].to_numpy(dtype=float)
        future_mask = ordered["x_norm"].to_numpy(dtype=float) > float(obs_frac)
        if not future_mask.any():
            raise ValueError(f"No future prediction points for trajectory_id={trajectory_id}, obs_frac={obs_frac}")
        start = float(band_width[future_mask][0])
        final = float(band_width[-1])
        records.append({
            "policy": policy,
            "trajectory_id": int(trajectory_id),
            "obs_frac": float(obs_frac),
            "query_target_N": int(ordered["query_target_N"].iloc[0]),
            "base_N": int(ordered["base_N"].iloc[0]),
            "shrink": float(ordered["shrink"].iloc[0]),
            "tkpm": float(ordered["tkpm"].iloc[0]),
            "context_size": int(ordered["context_size"].iloc[0]) if "context_size" in ordered else np.nan,
            "loss_log_min": float(ordered["loss_log_min"].iloc[0]) if "loss_log_min" in ordered else np.nan,
            "loss_log_max": float(ordered["loss_log_max"].iloc[0]) if "loss_log_max" in ordered else np.nan,
            "band_width_start": start,
            "band_width_final": final,
            "band_width_ratio": start / max(final, EPS),
        })
    return pd.DataFrame(records)


def plot_policy_obs(
    predictions: pd.DataFrame,
    raw_slice: pd.DataFrame,
    policy: str,
    obs_frac: float,
    ylim: tuple[float, float],
    output_path: Path,
    policy_titles: dict[str, str] | None = None,
) -> None:
    sub = predictions.loc[(predictions["policy"] == policy) & np.isclose(predictions["obs_frac"], obs_frac)]
    if sub.empty:
        return
    targets = sorted(sub["query_target_N"].unique())
    fig, axes = plt.subplots(len(targets), 1, figsize=(8, max(3.0 * len(targets), 3.5)), squeeze=False)

    for i, target_n in enumerate(targets):
        ax = axes[i, 0]
        target_pred = sub.loc[sub["query_target_N"] == target_n]
        curve = raw_curve(raw_slice, int(target_n))
        observed = pd.DataFrame()
        if not curve.empty:
            observed = curve.loc[curve["x_norm"] <= obs_frac].sort_values("x_norm")
            future_truth = curve.loc[curve["x_norm"] > obs_frac].sort_values("x_norm")
            if not observed.empty:
                ax.plot(observed["x_norm"], observed["Validation Loss"], color="#f97316", linewidth=1.8, label="observed prefix")
            if not observed.empty and not future_truth.empty:
                truth_x = np.r_[observed["x_norm"].iloc[-1], future_truth["x_norm"].to_numpy(dtype=float)]
                truth_y = np.r_[observed["Validation Loss"].iloc[-1], future_truth["Validation Loss"].to_numpy(dtype=float)]
                ax.plot(truth_x, truth_y, color="#111827", linewidth=1.5, label="future truth")

        for _, traj_pred in target_pred.groupby("trajectory_id", sort=True):
            ordered = traj_pred.sort_values("x_norm")
            future_pred = ordered.loc[ordered["x_norm"] > obs_frac]
            if future_pred.empty:
                raise ValueError(f"No future prediction points for target_N={target_n}, obs_frac={obs_frac}")
            x = future_pred["x_norm"].to_numpy(dtype=float)
            p05 = future_pred["pred_p05"].to_numpy(dtype=float)
            p50 = future_pred["pred_p50"].to_numpy(dtype=float)
            p95 = future_pred["pred_p95"].to_numpy(dtype=float)
            if not observed.empty:
                last_x = float(observed["x_norm"].iloc[-1])
                last_y = float(observed["Validation Loss"].iloc[-1])
                band_x = np.r_[last_x, x]
                band_p05 = np.r_[last_y, p05]
                band_p95 = np.r_[last_y, p95]
                ax.plot(
                    [last_x, float(x[0])],
                    [last_y, float(p50[0])],
                    color="#2563eb",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.65,
                    label="visual seam connector",
                )
            else:
                band_x = x
                band_p05 = p05
                band_p95 = p95
            ax.plot(x, p50, color="#2563eb", linestyle="--", linewidth=1.4, alpha=0.9, label="prediction p50")
            ax.fill_between(band_x, band_p05, band_p95, color="#60a5fa", alpha=0.22, label="p05-p95 band (anchored at observed seam)")

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

    titles = policy_titles or POLICY_TITLES
    fig.suptitle(titles.get(policy, policy), fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(
    description: str = "Supervisor-style prediction plots.",
    default_predictions: Path = DEFAULT_PREDICTIONS,
    default_raw_parquet: Path = DEFAULT_RAW_PARQUET,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    default_diagnostics: Path = DEFAULT_DIAGNOSTICS,
    default_policies: list[str] | None = None,
    policy_titles: dict[str, str] | None = None,
) -> None:
    args = parse_args(
        description,
        default_predictions,
        default_raw_parquet,
        default_output_dir,
        default_diagnostics,
        default_policies,
    )
    predictions = load_predictions(args.predictions, args.policies, args.shrink, args.tkpm)
    base_n = resolve_base_n(predictions, args.base_n)
    predictions = predictions.loc[predictions["base_N"] == base_n].copy()
    if predictions.empty:
        raise ValueError(f"No predictions for base_N={base_n}, shrink={args.shrink}, tkpm={args.tkpm}")

    raw_slice = load_raw_slice(args.raw_parquet, base_n, args.shrink, args.tkpm)
    obs_fracs = args.obs_fracs if args.obs_fracs is not None else sorted(predictions["obs_frac"].unique())
    ylim = (float(args.ylim[0]), float(args.ylim[1]))

    diagnostics = uncertainty_diagnostics(predictions)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.diagnostics, index=False)
    print(f"Wrote diagnostics: {args.diagnostics} ({len(diagnostics)} rows)")

    for policy in args.policies:
        for obs_frac in obs_fracs:
            obs_dir = f"obs_{int(round(float(obs_frac) * 100)):03d}"
            output_path = (
                args.output_dir
                / obs_dir
                / "supervisor_styles"
                / policy
                / f"base{base_n}_shrink{args.shrink:g}_tkpm{args.tkpm:g}.png"
            )
            plot_policy_obs(predictions, raw_slice, policy, float(obs_frac), ylim, output_path, policy_titles)
            if output_path.exists():
                print(f"Wrote plot: {output_path}")

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
