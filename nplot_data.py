import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from train.predict import normalize_log_loss_curve

# ── paths ─────────────────────────────────────────────────────
EXPERIMENTS_PATH = "experiments.json"
PREDICTIONS_PATH = "june12/a1_predicting_135_feeding_all135/base14562560_target134561280_obs0.json"
OUTPUT_DIR       = "june12/a1_predicting_135_feeding_all135/900"
# ─────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(EXPERIMENTS_PATH, "r") as f:
    experiments = json.load(f)

with open(PREDICTIONS_PATH, "r") as f:
    predictions = json.load(f)

# ── global min/max (real loss space) ──────────────────────────
all_val_losses = []
for run in experiments.values():
    all_val_losses.extend(run["curve"]["val_loss"])

global_min = min(all_val_losses)
global_max = max(all_val_losses)

print(f"Global min: {global_min:.4f}  Global max: {global_max:.4f}")

# ── compute GLOBAL normalized y-range (NEW) ───────────────────
all_norm_losses = []

for run in experiments.values():
    losses = np.array(run["curve"]["val_loss"])
    norm_losses = normalize_log_loss_curve(
        losses,
        global_min,
        global_max
    )
    all_norm_losses.extend(norm_losses)

global_y_min = float(np.min(all_norm_losses))
global_y_max = float(np.max(all_norm_losses))

y_margin = (global_y_max - global_y_min) * 0.05

print(f"Global normalized y-range: {global_y_min:.4f} → {global_y_max:.4f}")


def denormalize(y_norm, global_min, global_max):
    log_min = np.log(global_min + 1e-8)
    log_max = np.log(global_max + 1e-8)
    norm    = 1.0 - y_norm
    log_val = norm * (log_max - log_min) + log_min
    return float(np.exp(log_val) - 1e-8)


for run_key, pred in predictions.items():

    obs_frac = pred["observe_fraction"]
    base_N   = pred["base_N"]
    target_N = pred["target_N"]
    shrink   = pred["shrink"]
    tkpm     = pred["tkpm"]
    max_lr   = pred["max_lr"]

    gt           = experiments[run_key]["curve"]
    gt_flops     = np.array(gt["flops"])
    gt_tokens    = np.array(gt["tokens"])
    gt_loss_real = np.array(gt["val_loss"])
    gt_loss      = normalize_log_loss_curve(gt_loss_real, global_min, global_max)

    # ── normalized time axis [0, 1] ───────────────────────────
    gt_t      = gt_flops / (gt_flops[-1] + 1e-8)
    n_observe = int(len(gt_t) * obs_frac)

    obs_t       = gt_t[:n_observe]
    obs_loss    = gt_loss[:n_observe]
    future_t    = gt_t[n_observe:]
    future_loss = gt_loss[n_observe:]

    cutoff_t        = gt_t[n_observe] if n_observe < len(gt_t) else gt_t[-1]
    has_obs         = n_observe > 0
    loss_at_cutoff  = float(obs_loss[-1])  if has_obs else float(gt_loss[0])
    loss_drop_obs   = float(obs_loss[0]) - float(obs_loss[-1]) if has_obs else 0.0
    loss_drop_total = float(gt_loss[0])  - float(gt_loss[-1])
    pct_drop_obs    = loss_drop_obs / (loss_drop_total + 1e-8) * 100
    obs_idx         = max(0, n_observe - 1)

    median = np.array(pred["median"])
    q05    = np.array(pred["q05"])
    q95    = np.array(pred["q95"])

    t_pred = np.linspace(obs_frac, 1.0, len(median))

    # ── metrics ───────────────────────────────────────────────
    true_final      = float(gt_loss[-1])
    pred_final      = float(median[-1])
    true_final_real = float(gt_loss_real[-1])
    pred_final_real = denormalize(pred_final, global_min, global_max)

    abs_err_norm    = abs(pred_final - true_final)
    abs_err_real    = abs(pred_final_real - true_final_real)
    rel_err         = abs_err_real / (true_final_real + 1e-8) * 100

    ci_width        = float(q95[-1]) - float(q05[-1])
    true_in_ci      = float(q05[-1]) <= true_final <= float(q95[-1])

    # ── layout ────────────────────────────────────────────────
    fig    = plt.figure(figsize=(16, 7), facecolor="#f9f9f9")
    gs     = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    gs.update(wspace=0.05)

    ax     = fig.add_subplot(gs[0])
    axinfo = fig.add_subplot(gs[1])

    ax.set_facecolor("#f9f9f9")
    axinfo.set_facecolor("#f4f4f4")
    axinfo.axis("off")

    # ── main plot ─────────────────────────────────────────────
    if has_obs:
        ax.axvspan(gt_t[0], obs_t[-1],
                   color="#DBEAFE", alpha=0.4, label="_nolegend_")

        ax.plot(obs_t, obs_loss,
                color="#2563EB", linewidth=2.5,
                label=f"Ground truth — observed ({int(obs_frac*100)}%)")

    ax.plot(future_t, future_loss,
            color="#2563EB", linewidth=2.5,
            alpha=0.35 if has_obs else 1.0,
            label="Ground truth — unobserved" if has_obs else "Ground truth")

    ax.plot(t_pred, median,
            color="#DC2626", linewidth=2.5,
            label="IFBO median prediction")

    # ── CI (UPDATED: no dotted borders) ───────────────────────
    ax.fill_between(
        t_pred,
        q05,
        q95,
        color="#DC2626",
        alpha=0.25,
        edgecolor="none",
        label="IFBO 90% CI"
    )

    ax.axvline(cutoff_t, color="#6B7280",
               linestyle="--", linewidth=1.8,
               label=f"Observation cutoff ({int(obs_frac*100)}%)")

    # ── annotations ───────────────────────────────────────────
    ax.annotate(
        f"Start\n{gt_loss[0]:.3f}",
        xy=(gt_t[0], gt_loss[0]),
        xytext=(30, -20),
        textcoords="offset points",
        fontsize=8,
        color="#2563EB",
        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.0)
    )

    ax.annotate(
        f"True final\n{true_final:.4f}\n({true_final_real:.4f} real)",
        xy=(gt_t[-1], true_final),
        xytext=(-120, -35),
        textcoords="offset points",
        fontsize=8.5,
        color="#2563EB",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2)
    )

    ax.annotate(
        f"Pred final\n{pred_final:.4f}\n({pred_final_real:.4f} real)",
        xy=(t_pred[-1], pred_final),
        xytext=(-120, 20),
        textcoords="offset points",
        fontsize=8.5,
        color="#DC2626",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2)
    )

    # ── axes (FIXED Y) ────────────────────────────────────────
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.set_xlim(0, 1.05)

    ax.set_ylim(
        global_y_min - y_margin,
        global_y_max + y_margin
    )

    ax.set_xlabel("Normalized Compute (FLOPs / total FLOPs)", fontsize=11)
    ax.set_ylabel("Normalized Validation Loss", fontsize=11)

    ax.set_title(
        f"IFBO Scaling Law Prediction  ·  "
        f"{base_N/1e6:.0f}M → {target_N/1e6:.0f}M params  ·  "
        f"shrink={shrink}  ·  tkpm={int(tkpm)}×  ·  "
        f"obs={int(obs_frac*100)}%",
        fontsize=12,
        fontweight="bold",
        pad=12
    )

    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── info panel (unchanged) ────────────────────────────────
    axinfo.text(0.05, 0.97, "MODEL / METRICS PANEL",
                fontsize=9, fontweight="bold")

    axinfo.text(0.05, 0.90,
                f"Rel error: {rel_err:.2f}%\n"
                f"CI width: {ci_width:.4f}\n"
                f"CI contains truth: {true_in_ci}",
                fontsize=8)

    # ── save ──────────────────────────────────────────────────
    filename = (
        f"base{base_N/1e6:.0f}M"
        f"_target{target_N/1e6:.0f}M"
        f"_shrink{shrink}"
        f"_tkpm{int(tkpm)}"
        f"_lr{max_lr}"
        f"_obs{int(obs_frac*100)}pct"
        f"_relerr{rel_err:.1f}pct"
        f".png"
    )

    fname = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved → {fname}")
    print(f"Pred final: {pred_final:.4f} | True final: {true_final:.4f}")
    print(f"Rel error: {rel_err:.2f}% | CI hit: {true_in_ci}\n")