import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from train.predict import normalize_log_loss_curve

# ── paths — change these only ────────────────────────────────
EXPERIMENTS_PATH = "experiments.json"
PREDICTIONS_PATH = "ablation_1_target_full_curves_per_scale/base77124608_target134561280_obs0.json"
OUTPUT_DIR       = "ablation_1_target_full_curves_per_scale/base77124608"
# ─────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(EXPERIMENTS_PATH, "r") as f:
    experiments = json.load(f)

with open(PREDICTIONS_PATH, "r") as f:
    predictions = json.load(f)


def denormalize(y_norm, global_min, global_max):
    print(global_min, global_max, "global min max")
    log_min = np.log(global_min + 1e-8)
    log_max = np.log(global_max + 1e-8)
    log_val = y_norm * (log_max - log_min) + log_min
    return np.exp(log_val) - 1e-8

all_val_losses = []
for run in experiments.values():
    all_val_losses.extend(run["curve"]["val_loss"])
global_min = min(all_val_losses)
global_max = max(all_val_losses)

for run_key, pred in predictions.items():

    obs_frac = pred["observe_fraction"]
    base_N   = pred["base_N"]
    target_N = pred["target_N"]
    shrink   = pred["shrink"]
    tkpm     = pred["tkpm"]
    max_lr   = pred["max_lr"]

    gt         = experiments[run_key]["curve"]
    gt_flops   = np.array(gt["flops"])
    gt_loss    = normalize_log_loss_curve(np.array(gt["val_loss"]), global_min, global_max)
    gt_tokens  = np.array(gt["tokens"])

    n_observe    = max(2, int(len(gt_flops) * obs_frac))
    obs_flops    = gt_flops[:n_observe]
    obs_loss     = gt_loss[:n_observe]
    future_flops = gt_flops[n_observe:]
    future_loss  = gt_loss[n_observe:]

    t_pred     = np.linspace(obs_frac, 1.0, len(pred["median"]))
    pred_flops = t_pred * gt_flops[-1]

    # median = denormalize(np.array(pred["median"]), global_min, global_max)
    # q05    = denormalize(np.array(pred["q05"]),    global_min, global_max)
    # q95    = denormalize(np.array(pred["q95"]),    global_min, global_max)
    median = np.array(pred["median"]),
    q05    = np.array(pred["q05"]),
    q95    = np.array(pred["q95"]),

    true_final  = gt_loss[-1]
    pred_final  = median[-1]
    abs_err     = abs(pred_final - true_final)
    rel_err     = abs_err / true_final * 100
    ci_width    = q95[-1] - q05[-1]
    true_in_ci  = q05[-1] <= true_final <= q95[-1]

    # loss drop stats
    loss_at_cutoff  = obs_loss[-1]
    loss_drop_obs   = obs_loss[0] - obs_loss[-1]
    loss_drop_total = gt_loss[0] - gt_loss[-1]
    pct_drop_obs    = loss_drop_obs / loss_drop_total * 100

    # ── layout: main plot left, info panel right ─────────────
    fig = plt.figure(figsize=(16, 7), facecolor="#f9f9f9")
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    gs.update(wspace=0.05)

    ax   = fig.add_subplot(gs[0])
    axinfo = fig.add_subplot(gs[1])
    ax.set_facecolor("#f9f9f9")
    axinfo.set_facecolor("#f4f4f4")
    axinfo.axis("off")

    # ── main plot ────────────────────────────────────────────

    # shaded observed region
    ax.axvspan(gt_flops[0], obs_flops[-1],
               color="#DBEAFE", alpha=0.4, label="_nolegend_")

    # observed ground truth
    ax.plot(obs_flops, obs_loss,
            color="#2563EB", linewidth=2.5,
            label=f"Ground truth — observed ({int(obs_frac*100)}%)")

    # unobserved ground truth
    ax.plot(future_flops, future_loss,
            color="#2563EB", linewidth=2.5, alpha=0.35,
            label="Ground truth — unobserved (held out)")

    # IFBO median
    ax.plot(pred_flops, median,
            color="#DC2626", linewidth=2.5,
            label="IFBO median prediction")

    # uncertainty band
    ax.fill_between(pred_flops, q05, q95,
                    color="#DC2626", alpha=0.12,
                    label="IFBO 90% CI")

    # cutoff line
    cutoff_flop = gt_flops[min(n_observe, len(gt_flops) - 1)]
    ax.axvline(cutoff_flop, color="#6B7280",
               linestyle="--", linewidth=1.8,
               label=f"Observation cutoff ({int(obs_frac*100)}%)")

    # ── annotations ──────────────────────────────────────────

    # annotate initial loss
    ax.annotate(f"Start\n{gt_loss[0]:.3f}",
                xy=(gt_flops[0], gt_loss[0]),
                xytext=(30, 10), textcoords="offset points",
                fontsize=8, color="#2563EB",
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.0))

    # annotate cutoff loss
    ax.annotate(f"Cutoff\n{loss_at_cutoff:.3f}",
                xy=(cutoff_flop, loss_at_cutoff),
                xytext=(20, -30), textcoords="offset points",
                fontsize=8, color="#6B7280",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.0))

    # annotate true final
    ax.annotate(f"True final\n{true_final:.4f}",
                xy=(gt_flops[-1], true_final),
                xytext=(-90, 20), textcoords="offset points",
                fontsize=8.5, color="#2563EB", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2))

    # annotate predicted final
    ax.annotate(f"Pred final\n{pred_final:.4f}",
                xy=(pred_flops[-1], pred_final),
                xytext=(-90, -30), textcoords="offset points",
                fontsize=8.5, color="#DC2626", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2))

    # annotate CI at end
    ax.annotate("",
                xy=(pred_flops[-1], q95[-1]),
                xytext=(pred_flops[-1], q05[-1]),
                arrowprops=dict(arrowstyle="<->", color="#DC2626",
                                lw=1.0, alpha=0.6))
    ax.text(pred_flops[-1] * 1.001, (q05[-1] + q95[-1]) / 2,
            f"CI\n{ci_width:.4f}", fontsize=7.5,
            color="#DC2626", va="center")

    # ── axes formatting ───────────────────────────────────────
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e15:.1f}P"))
    ax.set_xlabel("Compute (FLOPs)", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.set_title(
        f"IFBO Scaling Law Prediction  ·  "
        f"{base_N/1e6:.0f}M → {target_N/1e6:.0f}M params  ·  "
        f"shrink={shrink}  ·  tkpm={int(tkpm)}×  ·  "
        f"obs={int(obs_frac*100)}%",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(fontsize=9, loc="upper right",
              framealpha=0.9, edgecolor="#D1D5DB")
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── info panel ────────────────────────────────────────────
    sections = [
        ("MODEL SCALE", [
            ("Base model",     f"{base_N/1e6:.2f}M params"),
            ("Target model",   f"{target_N/1e6:.2f}M params"),
            ("Scale factor",   f"{target_N/base_N:.1f}×"),
        ]),
        ("TRAINING CONFIG", [
            ("Shrink factor",  str(shrink)),
            ("Tokens / param", f"{int(tkpm)}×"),
            ("Max LR",         str(max_lr)),
            ("Total tokens",   f"{gt_tokens[-1]/1e9:.2f}B"),
            ("Total FLOPs",    f"{gt_flops[-1]/1e15:.2f} PFLOPs"),
            ("Checkpoints",    str(len(gt_loss))),
        ]),
        ("OBSERVATION", [
            ("Obs fraction",   f"{int(obs_frac*100)}%"),
            ("Obs FLOPs",      f"{obs_flops[-1]/1e15:.2f} PFLOPs"),
            ("Obs checkpoints",f"{n_observe}"),
            ("Loss at start",  f"{gt_loss[0]:.4f}"),
            ("Loss at cutoff", f"{loss_at_cutoff:.4f}"),
            ("Drop observed",  f"{loss_drop_obs:.4f} ({pct_drop_obs:.0f}%)"),
        ]),
        ("PREDICTION", [
            ("True final loss",f"{true_final:.4f}"),
            ("Pred final loss",f"{pred_final:.4f}"),
            ("Abs error",      f"{abs_err:.4f}"),
            ("Rel error",      f"{rel_err:.2f}%"),
            ("90% CI width",   f"{ci_width:.4f}"),
            ("CI q05",         f"{q05[-1]:.4f}"),
            ("CI q95",         f"{q95[-1]:.4f}"),
            ("True in CI",     "✓ yes" if true_in_ci else "✗ no"),
        ]),
    ]

    y = 0.97
    for section_title, rows in sections:
        axinfo.text(0.05, y, section_title,
                    transform=axinfo.transAxes,
                    fontsize=8, fontweight="bold",
                    color="#374151", va="top")
        y -= 0.04
        for label, value in rows:
            color = "#DC2626" if section_title == "PREDICTION" and "error" in label.lower() \
                    else ("#16A34A" if true_in_ci and label == "True in CI" \
                    else "#DC2626" if not true_in_ci and label == "True in CI" \
                    else "#111827")
            axinfo.text(0.05, y, f"{label}:",
                        transform=axinfo.transAxes,
                        fontsize=8, color="#6B7280", va="top")
            axinfo.text(0.98, y, value,
                        transform=axinfo.transAxes,
                        fontsize=8, color=color,
                        va="top", ha="right", fontweight="bold")
            y -= 0.038
        y -= 0.02   # gap between sections
        axinfo.plot([0.02, 0.98], [y + 0.01, y + 0.01],
                    transform=axinfo.transAxes,
                    color="#D1D5DB", linewidth=0.5,
                    clip_on=False)
        y -= 0.01

    plt.tight_layout()

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