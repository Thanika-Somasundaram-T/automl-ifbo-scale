"""Shared rank, regret, and curve metrics for width-transfer experiments."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

DEFAULT_OUTPUT_DIR = Path("experiments/width_transfer_metrics")
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "results" / "predictions.parquet"
TARGET_T_VALUES = tuple(round(x, 2) for x in np.linspace(0.1, 1.0, 10))


def parse_args(
    description: str = "Width-transfer metrics.",
    default_predictions: Path = DEFAULT_PREDICTIONS,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    default_plots_dir: Path | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--predictions", type=Path, default=default_predictions)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--plots-dir", type=Path, default=default_plots_dir)
    parser.add_argument("--t-values", type=float, nargs="+", default=list(TARGET_T_VALUES))
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    return parser.parse_args()


def rank_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan, np.nan, np.nan, np.nan
    rho, p_rho = sp_stats.spearmanr(a, b)
    tau, p_tau = sp_stats.kendalltau(a, b)
    return float(rho), float(p_rho), float(tau), float(p_tau)


# ─────────────────────────────────────────────────────────────────────────────
# Horizon picking
# ─────────────────────────────────────────────────────────────────────────────

def points_at_t(predictions: pd.DataFrame, t_values: list[float]) -> pd.DataFrame:
    rows = []
    group_cols = ["policy", "obs_frac", "trajectory_id"]
    for _, group in predictions.groupby(group_cols, sort=True):
        x = group["x_norm"].to_numpy(dtype=float)
        obs_frac = float(group["obs_frac"].iloc[0])
        for t in t_values:
            if t <= obs_frac:
                continue
            idx = int(np.argmin(np.abs(x - float(t))))
            picked = group.iloc[idx].copy()
            picked["t_target"] = float(t)
            rows.append(picked)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1: Spearman rho per (policy, query_target_N, obs_frac, t_target)
# ─────────────────────────────────────────────────────────────────────────────

def per_scale_rank_metrics(points: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = ["policy", "obs_frac", "t_target", "query_target_N"]
    for (policy, obs_frac, t_target, qtn), group in points.groupby(group_cols, sort=True):
        if len(group) < 3:
            continue
        true_loss = group["y_true"].to_numpy(dtype=float)
        pred_loss = group["pred_p50"].to_numpy(dtype=float)
        rho, p_rho, tau, p_tau = rank_correlation(true_loss, pred_loss)
        records.append({
            "policy": policy,
            "obs_frac": float(obs_frac),
            "t_target": float(t_target),
            "query_target_N": int(qtn),
            "n_configs": int(len(group)),
            "spearman_rho": rho,
            "kendall_tau": tau,
            "spearman_p": p_rho,
            "kendall_p": p_tau,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2: Transfer regret and top-k overlap
# ─────────────────────────────────────────────────────────────────────────────

def regret_and_topk(points: pd.DataFrame, top_k_values: list[int]) -> pd.DataFrame:
    records = []
    group_cols = ["policy", "obs_frac", "t_target", "query_target_N"]
    for (policy, obs_frac, t_target, qtn), group in points.groupby(group_cols, sort=True):
        if len(group) < 2:
            continue
        true_loss = group["y_true"].to_numpy(dtype=float)
        pred_loss = group["pred_p50"].to_numpy(dtype=float)
        traj_ids = group["trajectory_id"].to_numpy()

        best_true_idx = int(np.argmin(true_loss))
        best_pred_idx = int(np.argmin(pred_loss))
        oracle_loss = true_loss[best_true_idx]
        selected_loss = true_loss[best_pred_idx]
        regret = float(selected_loss - oracle_loss)
        normalized_regret = regret / (float(np.max(true_loss)) - oracle_loss) if np.max(true_loss) > oracle_loss else 0.0

        true_ranking = np.argsort(true_loss)
        pred_ranking = np.argsort(pred_loss)

        row = {
            "policy": policy,
            "obs_frac": float(obs_frac),
            "t_target": float(t_target),
            "query_target_N": int(qtn),
            "n_configs": int(len(group)),
            "regret": regret,
            "normalized_regret": normalized_regret,
            "oracle_loss": oracle_loss,
            "selected_loss": selected_loss,
        }

        for k in top_k_values:
            if k > len(group):
                continue
            true_top_k = set(traj_ids[true_ranking[:k]])
            pred_top_k = set(traj_ids[pred_ranking[:k]])
            overlap = len(true_top_k & pred_top_k)
            row[f"top{k}_overlap"] = overlap
            row[f"top{k}_hit_rate"] = overlap / k
            row[f"top{k}_best_in_pred"] = int(traj_ids[best_true_idx] in pred_top_k)

        records.append(row)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3: Pairwise agreement across scale pairs
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_scale_agreement(points: pd.DataFrame) -> pd.DataFrame:
    records = []
    hp_key = ("shrink", "tkpm")
    group_cols = ["policy", "obs_frac", "t_target"]
    for (policy, obs_frac, t_target), group in points.groupby(group_cols, sort=True):
        scales = sorted(group["query_target_N"].unique())
        if len(scales) < 2:
            continue
        per_scale = {}
        for n in scales:
            sub = group[group["query_target_N"] == n]
            per_scale[n] = sub.groupby(list(hp_key), sort=False)[["y_true", "pred_p50"]].mean()
        for n_a, n_b in combinations(scales, 2):
            shared = per_scale[n_a].index.intersection(per_scale[n_b].index)
            if len(shared) < 3:
                continue
            pa = per_scale[n_a].loc[shared, "pred_p50"].to_numpy(dtype=float)
            pb = per_scale[n_b].loc[shared, "pred_p50"].to_numpy(dtype=float)
            ya = per_scale[n_a].loc[shared, "y_true"].to_numpy(dtype=float)
            yb = per_scale[n_b].loc[shared, "y_true"].to_numpy(dtype=float)
            pred_rho, _, pred_tau, _ = rank_correlation(pa, pb)
            true_rho, _, true_tau, _ = rank_correlation(ya, yb)
            records.append({
                "policy": policy,
                "obs_frac": float(obs_frac),
                "t_target": float(t_target),
                "query_target_N_a": int(n_a),
                "query_target_N_b": int(n_b),
                "scale_gap": float(n_b) / float(n_a),
                "n_shared_configs": int(len(shared)),
                "pred_pairwise_rho": pred_rho,
                "pred_pairwise_tau": pred_tau,
                "true_pairwise_rho": true_rho,
                "true_pairwise_tau": true_tau,
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4: Curve-level error metrics
# ─────────────────────────────────────────────────────────────────────────────

def curve_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["policy", "trajectory_id", "obs_frac"]
    for (policy, tid, obs_frac), group in predictions.groupby(group_cols, sort=True):
        y_true = group["y_true"].to_numpy(dtype=float)
        pred = group["pred_p50"].to_numpy(dtype=float)
        err = pred - y_true
        rows.append({
            "policy": policy,
            "trajectory_id": int(tid),
            "obs_frac": float(obs_frac),
            "query_target_N": int(group["query_target_N"].iloc[0]),
            "base_N": int(group["base_N"].iloc[0]),
            "shrink": float(group["shrink"].iloc[0]),
            "tkpm": float(group["tkpm"].iloc[0]),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "coverage_90": float(np.mean(
                (y_true >= group["pred_p05"].to_numpy()) &
                (y_true <= group["pred_p95"].to_numpy())
            )),
            "final_abs_error": float(abs(pred[-1] - y_true[-1])),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Scale gap summary
# ─────────────────────────────────────────────────────────────────────────────

def scale_gap_summary(rank_df: pd.DataFrame) -> pd.DataFrame:
    if rank_df.empty:
        return pd.DataFrame()
    t_final = rank_df[rank_df["t_target"] == 1.0]
    if t_final.empty:
        t_final = rank_df[rank_df["t_target"] == rank_df["t_target"].max()]
    return (
        t_final.groupby(["policy", "query_target_N"], as_index=False)
        .agg(
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
            n_obs_fracs=("obs_frac", "nunique"),
            n_configs=("n_configs", "first"),
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_POLICY_COLOR = "#444"
PALETTE = (
    "#6b7280",
    "#2563eb",
    "#16a34a",
    "#f59e0b",
    "#9333ea",
    "#dc2626",
    "#000000",
    "#0891b2",
)


def policy_color(policy: str, policy_colors: dict[str, str] | None = None) -> str:
    if policy_colors and policy in policy_colors:
        return policy_colors[policy]
    return PALETTE[abs(hash(policy)) % len(PALETTE)] if policy else DEFAULT_POLICY_COLOR


def plot_rank_by_scale(rank_df: pd.DataFrame, output_dir: Path, policy_colors: dict[str, str] | None = None) -> None:
    if rank_df.empty:
        return
    t_final = rank_df[rank_df["t_target"] == 1.0]
    if t_final.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, grp in t_final.groupby("policy"):
        agg = grp.groupby("query_target_N", as_index=False)["spearman_rho"].mean()
        agg = agg.sort_values("query_target_N")
        labels = [f"{int(n/1e6)}M" for n in agg["query_target_N"]]
        ax.plot(labels, agg["spearman_rho"], marker="o",
                color=policy_color(str(policy), policy_colors), label=str(policy), linewidth=1.5)
    ax.set_xlabel("Query target scale")
    ax.set_ylabel("Mean Spearman ρ (t=1.0)")
    ax.set_title("Rank correlation by query scale and context policy")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(-0.3, 1.05)
    ax.axhline(0, color="#999", linestyle=":", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rank_by_scale.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_regret_by_policy(regret_df: pd.DataFrame, output_dir: Path, policy_colors: dict[str, str] | None = None) -> None:
    if regret_df.empty:
        return
    t_final = regret_df[regret_df["t_target"] == 1.0]
    if t_final.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    policies = sorted(t_final["policy"].unique())
    agg = t_final.groupby("policy", as_index=False)["normalized_regret"].mean()
    agg = agg.set_index("policy").loc[policies]
    colors = [policy_color(str(p), policy_colors) for p in policies]
    ax.barh(range(len(policies)), agg["normalized_regret"], color=colors)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies, fontsize=9)
    ax.set_xlabel("Mean normalized regret (t=1.0)")
    ax.set_title("Transfer regret by context policy (lower is better)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "regret_by_policy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_scale_gap_heatmap(gap_df: pd.DataFrame, output_dir: Path) -> None:
    if gap_df.empty:
        return
    pivot = gap_df.pivot_table(
        index="policy", columns="query_target_N", values="mean_spearman_rho"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(n/1e6)}M" for n in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#666")
            else:
                color = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel("Query target scale")
    ax.set_ylabel("Context policy")
    ax.set_title("Mean Spearman ρ at t=1.0 by policy × query scale")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(output_dir / "scale_gap_summary.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_rank_by_obs(rank_df: pd.DataFrame, output_dir: Path, policy_colors: dict[str, str] | None = None) -> None:
    if rank_df.empty:
        return
    t_final = rank_df[rank_df["t_target"] == 1.0]
    if t_final.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, grp in t_final.groupby("policy"):
        agg = grp.groupby("obs_frac", as_index=False)["spearman_rho"].mean().sort_values("obs_frac")
        ax.plot(agg["obs_frac"], agg["spearman_rho"], marker="o",
                color=policy_color(str(policy), policy_colors), label=str(policy), linewidth=1.5)
    ax.set_xlabel("Observation fraction")
    ax.set_ylabel("Mean Spearman ρ (t=1.0)")
    ax.set_title("Rank correlation by obs_frac and context policy")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.3, 1.05)
    ax.axhline(0, color="#999", linestyle=":", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rank_by_obs_frac.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    description: str = "Width-transfer metrics.",
    default_predictions: Path = DEFAULT_PREDICTIONS,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    default_plots_dir: Path | None = None,
    policy_colors: dict[str, str] | None = None,
) -> None:
    args = parse_args(description, default_predictions, default_output_dir, default_plots_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.plots_dir if args.plots_dir is not None else args.output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Predictions not found: {args.predictions}. "
            "Run predict_width_extrapolation.py first."
        )

    predictions = pd.read_parquet(args.predictions)
    n_traj = predictions["trajectory_id"].nunique()
    print(f"Loaded {len(predictions):,} prediction rows over {n_traj} trajectories.")
    print(f"Policies: {sorted(predictions['policy'].unique())}")

    points = points_at_t(predictions, args.t_values)
    print(f"Evaluation points after horizon picking: {len(points):,}")

    print("\nComputing per-scale rank metrics...")
    rank_df = per_scale_rank_metrics(points)
    rank_df.to_csv(args.output_dir / "rank_metrics.csv", index=False)
    print(f"  Wrote rank_metrics.csv ({len(rank_df)} rows)")

    print("Computing regret and top-k metrics...")
    regret_df = regret_and_topk(points, args.top_k)
    regret_df.to_csv(args.output_dir / "regret_metrics.csv", index=False)
    print(f"  Wrote regret_metrics.csv ({len(regret_df)} rows)")

    print("Computing pairwise scale agreement...")
    pairwise_df = pairwise_scale_agreement(points)
    pairwise_df.to_csv(args.output_dir / "pairwise_metrics.csv", index=False)
    print(f"  Wrote pairwise_metrics.csv ({len(pairwise_df)} rows)")

    print("Computing curve-level metrics...")
    curve_df = curve_metrics(predictions)
    curve_df.to_csv(args.output_dir / "curve_metrics.csv", index=False)
    print(f"  Wrote curve_metrics.csv ({len(curve_df)} rows)")

    print("Computing scale gap summary...")
    gap_df = scale_gap_summary(rank_df)
    gap_df.to_csv(args.output_dir / "scale_gap_summary.csv", index=False)
    print(f"  Wrote scale_gap_summary.csv ({len(gap_df)} rows)")

    print("\nGenerating plots...")
    plot_rank_by_scale(rank_df, plots_dir, policy_colors)
    plot_regret_by_policy(regret_df, plots_dir, policy_colors)
    plot_scale_gap_heatmap(gap_df, plots_dir)
    plot_rank_by_obs(rank_df, plots_dir, policy_colors)

    print("\n=== Headline: mean Spearman ρ at t=1.0 ===")
    if not gap_df.empty:
        headline = gap_df.pivot_table(index="policy", columns="query_target_N", values="mean_spearman_rho")
        headline.columns = [f"{int(n/1e6)}M" for n in headline.columns]
        print(headline.to_string())

    print("\n=== Headline: mean normalized regret at t=1.0 ===")
    if not regret_df.empty:
        t_final_regret = regret_df[regret_df["t_target"] == 1.0]
        if not t_final_regret.empty:
            print(t_final_regret.groupby("policy")["normalized_regret"].mean().to_string())

    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
