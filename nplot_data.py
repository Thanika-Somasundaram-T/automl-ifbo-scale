import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from train.predict import normalize_log_loss_curve

# ── paths ─────────────────────────────────────────────────────
EXPERIMENTS_PATH = "experiments.json"
PREDICTIONS_PATH = "a5_predicting_135_only_feeding_135/base14562560_target134561280_obs0.json"
OUTPUT_DIR       = "a5_predicting_135_only_feeding_135/plot"
# ─────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(EXPERIMENTS_PATH, "r") as f:
    experiments = json.load(f)

with open(PREDICTIONS_PATH, "r") as f:
    predictions = json.load(f)

# ── global min/max ────────────────────────────────────────────
all_val_losses = []
for run in experiments.values():
    all_val_losses.extend(run["curve"]["val_loss"])
global_min = min(all_val_losses)
global_max = max(all_val_losses)
print(f"Global min: {global_min:.4f}  Global max: {global_max:.4f}")


def denormalize(y_norm, global_min, global_max):
    log_min = np.log(global_min + 1e-8)
    log_max = np.log(global_max + 1e-8)
    log_val = y_norm * (log_max - log_min) + log_min
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
    n_observe = int(len(gt_t) * obs_frac)   # 0 is valid now

    obs_t       = gt_t[:n_observe]           # empty array if obs_frac=0
    obs_loss    = gt_loss[:n_observe]        # empty array if obs_frac=0
    future_t    = gt_t[n_observe:]           # full array if obs_frac=0
    future_loss = gt_loss[n_observe:]        # full array if obs_frac=0

    # safe cutoff — if no observation, cutoff is at t=0
    cutoff_t        = gt_t[n_observe] if n_observe < len(gt_t) else gt_t[-1]
    has_obs         = n_observe > 0
    loss_at_cutoff  = float(obs_loss[-1])  if has_obs else float(gt_loss[0])
    loss_drop_obs   = float(obs_loss[0]) - float(obs_loss[-1]) if has_obs else 0.0
    loss_drop_total = float(gt_loss[0])  - float(gt_loss[-1])
    pct_drop_obs    = loss_drop_obs / (loss_drop_total + 1e-8) * 100

    # safe index for info panel
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

    # shaded observed region — skip if no observation
    if has_obs:
        ax.axvspan(gt_t[0], obs_t[-1],
                   color="#DBEAFE", alpha=0.4, label="_nolegend_")
        ax.plot(obs_t, obs_loss,
                color="#2563EB", linewidth=2.5,
                label=f"Ground truth — observed ({int(obs_frac*100)}%)")

    # full ground truth shown as unobserved
    ax.plot(future_t, future_loss,
            color="#2563EB", linewidth=2.5,
            alpha=0.35 if has_obs else 1.0,
            label="Ground truth — unobserved (held out)" if has_obs
                  else "Ground truth (fully unobserved)")

    ax.plot(t_pred, median,
            color="#DC2626", linewidth=2.5,
            label="IFBO median prediction")

    ax.fill_between(t_pred, q05, q95,
                    color="#DC2626", alpha=0.12,
                    label="IFBO 90% CI")

    ax.axvline(cutoff_t, color="#6B7280",
               linestyle="--", linewidth=1.8,
               label=f"Observation cutoff ({int(obs_frac*100)}%)")

    # ── annotations ───────────────────────────────────────────
    ax.annotate(
        f"Start\n{gt_loss[0]:.3f} norm\n({gt_loss_real[0]:.3f} real)",
        xy=(gt_t[0], gt_loss[0]),
        xytext=(30, 10), textcoords="offset points",
        fontsize=8, color="#2563EB",
        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.0))

    if has_obs:
        ax.annotate(
            f"Cutoff\n{loss_at_cutoff:.3f} norm",
            xy=(cutoff_t, loss_at_cutoff),
            xytext=(20, -40), textcoords="offset points",
            fontsize=8, color="#6B7280",
            arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.0))

    ax.annotate(
        f"True final\n{true_final:.4f} norm\n({true_final_real:.4f} real)",
        xy=(gt_t[-1], true_final),
        xytext=(-120, 20), textcoords="offset points",
        fontsize=8.5, color="#2563EB", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2))

    ax.annotate(
        f"Pred final\n{pred_final:.4f} norm\n({pred_final_real:.4f} real)",
        xy=(t_pred[-1], pred_final),
        xytext=(-120, -40), textcoords="offset points",
        fontsize=8.5, color="#DC2626", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2))

    ax.annotate("",
                xy=(t_pred[-1], float(q95[-1])),
                xytext=(t_pred[-1], float(q05[-1])),
                arrowprops=dict(arrowstyle="<->", color="#DC2626",
                                lw=1.0, alpha=0.6))
    ax.text(t_pred[-1] * 1.001,
            (float(q05[-1]) + float(q95[-1])) / 2,
            f"CI\n{ci_width:.4f}", fontsize=7.5,
            color="#DC2626", va="center")

    # ── axes ──────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Normalized Compute (FLOPs / total FLOPs)", fontsize=11)
    ax.set_ylabel("Normalized Validation Loss", fontsize=11)
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
            ("Base model",        f"{base_N/1e6:.2f}M params"),
            ("Target model",      f"{target_N/1e6:.2f}M params"),
            ("Scale factor",      f"{target_N/base_N:.1f}×"),
        ]),
        ("TRAINING CONFIG", [
            ("Shrink factor",     str(shrink)),
            ("Tokens / param",    f"{int(tkpm)}×"),
            ("Max LR",            str(max_lr)),
            ("Total tokens",      f"{gt_tokens[-1]/1e9:.2f}B"),
            ("Total FLOPs",       f"{gt_flops[-1]/1e15:.2f} PFLOPs"),
            ("Checkpoints",       str(len(gt_loss))),
        ]),
        ("OBSERVATION", [
            ("Obs fraction",      f"{int(obs_frac*100)}%"),
            ("Obs FLOPs",         f"{gt_flops[obs_idx]/1e15:.2f} PFLOPs" if has_obs else "none"),
            ("Obs checkpoints",   f"{n_observe}"),
            ("Loss at start",     f"{gt_loss_real[0]:.4f} ({gt_loss[0]:.3f} norm)"),
            ("Loss at cutoff",    f"{gt_loss_real[obs_idx]:.4f} ({loss_at_cutoff:.3f} norm)" if has_obs else "n/a"),
            ("Drop observed",     f"{loss_drop_obs:.4f} ({pct_drop_obs:.0f}%)" if has_obs else "n/a"),
        ]),
        ("PREDICTION", [
            ("True final (real & norm)", f"{true_final_real:.4f} / {true_final:.4f}"),
            ("Pred final (real & norm)", f"{pred_final_real:.4f} / {pred_final:.4f}"),
            ("Abs err (real & norm)",    f"{abs_err_real:.4f} / {abs_err_norm:.4f}"),
            ("Rel error (real)",         f"{rel_err:.2f}%"),
            ("90% CI width",             f"{ci_width:.4f}"),
            ("CI q05",                   f"{float(q05[-1]):.4f}"),
            ("CI q95",                   f"{float(q95[-1]):.4f}"),
            ("True in CI",               "✓ yes" if true_in_ci else "✗ no"),
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
            is_err     = section_title == "PREDICTION" and "error" in label.lower()
            is_true_ci = label == "True in CI"
            color = (
                "#DC2626" if is_err
                else "#16A34A" if is_true_ci and true_in_ci
                else "#DC2626" if is_true_ci and not true_in_ci
                else "#111827"
            )
            axinfo.text(0.05, y, f"{label}:",
                        transform=axinfo.transAxes,
                        fontsize=8, color="#6B7280", va="top")
            axinfo.text(0.98, y, value,
                        transform=axinfo.transAxes,
                        fontsize=8, color=color,
                        va="top", ha="right", fontweight="bold")
            y -= 0.038
        y -= 0.02
        axinfo.plot([0.02, 0.98], [y + 0.01, y + 0.01],
                    transform=axinfo.transAxes,
                    color="#D1D5DB", linewidth=0.5,
                    clip_on=False)
        y -= 0.01

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
    print(f"  True final (norm) : {true_final:.4f}")
    print(f"  Pred final (norm) : {pred_final:.4f}")
    print(f"  True final (real) : {true_final_real:.4f}")
    print(f"  Pred final (real) : {pred_final_real:.4f}")
    print(f"  Abs err (real)    : {abs_err_real:.4f}")
    print(f"  Rel error (real)  : {rel_err:.2f}%")
    print(f"  True in CI        : {'yes' if true_in_ci else 'no'}")
    print(f"  Saved → {fname}\n")