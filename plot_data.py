import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from utils import compute_loo, load_data, normalize_log_loss_curve, subsample_curve

# ── paths ─────────────────────────────────────────────────────
EXPERIMENTS_PATH = "all_curves.json"

# ── context curves to overlay ─────────────────────────────────
# Set to None to disable, or provide a list/range of run keys.
# Examples:
#   CONTEXT_KEYS = None                          # no context overlay
#   CONTEXT_KEYS = ["run_0", "run_1", "run_2"]  # explicit keys
#   CONTEXT_KEYS = list(experiments.keys())[:10] # first 10 runs  ← set after loading
CONTEXT_KEYS = None   # ← edit this after loading experiments if using a range
# ─────────────────────────────────────────────────────────────

import os
import json
import glob

EXPERIMENTS_PATH = "all_curves.json"
BASE_DIR = "june/baseline/134561280"
TARGET_N_THRESHOLD = 134561280

with open(EXPERIMENTS_PATH, "r") as f:
    experiments = json.load(f)




prediction_files = glob.glob(
    os.path.join(BASE_DIR, "**", "target*_obs*.json"),
    recursive=True,
)
print(f"Found {len(prediction_files)} prediction files")

for pred_path in prediction_files:

    # example:
    # june/t610/32270848/target610488320_obs90.json

    context = int(os.path.basename(os.path.dirname(pred_path)))
    # 32270848
    
    # CONTEXT_KEYS = [
    # k for k, v in experiments.items()
    # if v["hyperparameters"]["target_N"] == context
    # ]

    filename = os.path.splitext(os.path.basename(pred_path))[0]
    # target610488320_obs90

    with open(pred_path, "r") as f:
        predictions = json.load(f)

    print(f"Processing {pred_path}")
    df = load_data(EXPERIMENTS_PATH)


    def denormalize(y_norm: np.ndarray) -> np.ndarray:
        """Invert normalize_log_loss_curve: norm → real loss space."""
        log_min, log_max = compute_loo(df=df, target_N=None, buffer=0.05)
        y_norm  = np.clip(y_norm, 0.0, 1.0)
        log_val = y_norm * (log_max - log_min) + log_min
        return np.exp(log_val) - 1e-8


    # ── context curve style ───────────────────────────────────────
    CONTEXT_COLOR   = "#CBE360"   # green — distinct from blue (GT) and red (pred)
    CONTEXT_ALPHA   = 0.09
    CONTEXT_LW      = 1.2


    for run_key, pred in predictions.items():

        obs_frac = pred["observe_fraction"]
        obs_folder = str(int(obs_frac * 100))
        obs_folder = str(int(obs_frac * 100))  # 0, 20, 50, 90

        output_dir = os.path.join(
            BASE_DIR,
            str(context),
            obs_folder,
        )
        
        os.makedirs(output_dir, exist_ok=True)
        base_N   = pred["base_N"]
        target_N = pred["target_N"]
        shrink   = pred["shrink"]
        tkpm     = pred["tkpm"]
        max_lr   = pred["max_lr"]

        gt           = experiments[run_key]["curve"]
        gt_flops     = np.array(gt["flops"])
        gt_tokens    = np.array(gt["tokens"])
        gt_loss_real = np.array(gt["val_loss"])

        gt_loss_norm = normalize_log_loss_curve(gt_loss_real, df=df, target_N=None)

        gt_t      = gt_flops / (gt_flops[-1] + 1e-8)
        n_observe = int(len(gt_t) * obs_frac)

        obs_t        = gt_t[:n_observe]
        obs_loss     = gt_loss_real[:n_observe]
        future_t     = gt_t[n_observe:]
        future_loss  = gt_loss_real[n_observe:]

        cutoff_t        = gt_t[n_observe] if n_observe < len(gt_t) else gt_t[-1]
        has_obs         = n_observe > 0
        loss_at_cutoff  = float(obs_loss[-1])         if has_obs else float(gt_loss_real[0])
        loss_drop_obs   = float(obs_loss[0]) - float(obs_loss[-1]) if has_obs else 0.0
        loss_drop_total = float(gt_loss_real[0]) - float(gt_loss_real[-1])
        pct_drop_obs    = loss_drop_obs / (loss_drop_total + 1e-8) * 100
        obs_idx         = max(0, n_observe - 1)

        median_norm = np.array(pred["median"])
        q05_norm    = np.array(pred["q05"])
        q95_norm    = np.array(pred["q95"])

        median = denormalize(median_norm)
        q05    = denormalize(q05_norm)
        q95    = denormalize(q95_norm)

        t_pred = np.linspace(obs_frac, 1.0, len(median))

        print(f"\n[DEBUG] {run_key}")
        print(f"  median[-1]     : {median[-1]:.6f}")
        print(f"  q05[-1] (real) : {q05[-1]:.6f}  ← should be LOWER than median")
        print(f"  q95[-1] (real) : {q95[-1]:.6f}  ← should be HIGHER than median")
        print(f"  CI width       : {q95[-1] - q05[-1]:.6f}")

        true_final  = float(gt_loss_real[-1])
        pred_final  = float(median[-1])
        abs_err     = abs(pred_final - true_final)
        rel_err     = abs_err / (true_final + 1e-8) * 100
        ci_width    = float(q95[-1]) - float(q05[-1])
        true_in_ci  = float(q05[-1]) <= true_final <= float(q95[-1])

        # ── layout ────────────────────────────────────────────────
        fig    = plt.figure(figsize=(16, 7), facecolor="#f9f9f9")
        gs     = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
        gs.update(wspace=0.05)
        ax     = fig.add_subplot(gs[0])
        axinfo = fig.add_subplot(gs[1])
        ax.set_facecolor("#f9f9f9")
        axinfo.set_facecolor("#f4f4f4")
        axinfo.axis("off")

        # ── context curves overlay ────────────────────────────────
        context_plotted = 0
        # if CONTEXT_KEYS:
        #     for i, ctx_key in enumerate(CONTEXT_KEYS):
        #         if ctx_key == run_key:
        #             continue                          # skip the target run itself
        #         if ctx_key not in experiments:
        #             continue
        #         ctx       = experiments[ctx_key]["curve"]
        #         ctx_flops = np.array(ctx["flops"])
        #         ctx_loss  = np.array(ctx["val_loss"])
        #         ctx_t     = ctx_flops / (ctx_flops[-1] + 1e-8)
        #         label     = "Context curves" if context_plotted == 0 else "_nolegend_"
        #         ax.plot(ctx_t, ctx_loss,
        #                 color=CONTEXT_COLOR,
        #                 alpha=CONTEXT_ALPHA,
        #                 linewidth=CONTEXT_LW,
        #                 label=label,
        #                 zorder=1)                     # draw behind everything else
        #         context_plotted += 1

        # ── main plot ─────────────────────────────────────────────
        if has_obs:
            # ax.axvspan(gt_t[0], obs_t[-1],
            #            color="#DBEAFE", alpha=0.4, label="_nolegend_")
            ax.plot(obs_t, obs_loss,
                    color="#000000", linewidth=2.5, zorder=3, alpha=0.5 if has_obs else 1.0,
                    label=f"Ground truth — observed ({int(obs_frac*100)}%)")

        ax.plot(future_t, future_loss,
                color="#000000", linewidth=2.5, zorder=3,
                alpha=1.0,
                label="Ground truth — unobserved (held out)" if has_obs
                    else "Ground truth (fully unobserved)")

        ax.plot(t_pred, median,
                color="#2563EB", linewidth=2.5, zorder=4,
                label="IFBO median prediction")

        ax.fill_between(t_pred, q05, q95,
                        color="#2563EB", alpha=0.12, zorder=2,
                        label="IFBO 90% CI")

        ax.axvline(cutoff_t, color="#6B7280",
                linestyle="--", linewidth=1.0, zorder=5,
                label=f"Observation cutoff ({int(obs_frac*100)}%)")

        # ── annotations ───────────────────────────────────────────
        ax.annotate(
            f"Start\n{gt_loss_real[0]:.4f}",
            xy=(gt_t[0], gt_loss_real[0]),
            xytext=(30, 10), textcoords="offset points",
            fontsize=8, color="#000000",
            arrowprops=dict(arrowstyle="->", color="#000000", lw=1.0))

        if has_obs:
            ax.annotate(
                f"Cutoff\n{loss_at_cutoff:.4f}",
                xy=(cutoff_t, loss_at_cutoff),
                xytext=(20, -40), textcoords="offset points",
                fontsize=8, color="#6B7280",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.0))
        
        idx_80 = int(0.8 * (len(gt_t) - 1))

        ax.annotate(
            f"Real Loss at 0.8\n{gt_loss_real[idx_80]:.4f}",
            xy=(gt_t[idx_80], gt_loss_real[idx_80]),
            xytext=(-80, 80),
            textcoords="offset points",
            fontsize=8,
            color="#000000",
            arrowprops=dict(arrowstyle="->", color="#000000", lw=1.0)
        )
        idx_80 = np.argmin(np.abs(t_pred - 0.8))
        ax.annotate(
            f"Pred Loss\n{median[idx_80]:.4f}",
            xy=(t_pred[idx_80], median[idx_80]),
            xytext=(-100, 50),
            textcoords="offset points",
            fontsize=8,
            color="#2563EB",
            arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.0)
        )

        ax.annotate(
            f"True final\n{true_final:.4f}",
            xy=(gt_t[-1], true_final),
            xytext=(-30, 50), textcoords="offset points",
            fontsize=8.5, color="#000000", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#000000", lw=1.2))

        ax.annotate(
            f"Pred final\n{pred_final:.4f}",
            xy=(t_pred[-1], pred_final),
            xytext=(-100, 50), textcoords="offset points",
            fontsize=8.5, color="#2563EB", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2))

        ax.annotate("",
                    xy=(t_pred[-1], float(q95[-1])),
                    xytext=(t_pred[-1], float(q05[-1])),
                    arrowprops=dict(arrowstyle="<->", color="#000000",
                                    lw=1.0, alpha=0.6))
        ax.text(t_pred[-1] * 1.001,
                (float(q05[-1]) + float(q95[-1])) / 2,
                f"CI\n{ci_width:.4f}", fontsize=7.5,
                color="#000000", va="center")

        # ── axes ──────────────────────────────────────────────────
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Normalized Compute (FLOPs / total FLOPs)", fontsize=11)
        ax.set_ylabel("Validation Loss", fontsize=11)
        ax.set_title(
            f"IFBO Scaling Law Prediction  ·  "
            f"{base_N/1e6:.0f}M → {target_N/1e6:.0f}M params  ·  "
            f"shrink={shrink}  ·  tkpm={int(tkpm)}×  ·  "
            f"obs={int(obs_frac*100)}%"
            + (f"  ·  {context_plotted} context curves" if context_plotted else ""),
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
                ("Checkpoints",       str(len(gt_loss_real))),
            ]),
            ("OBSERVATION", [
                ("Obs fraction",      f"{int(obs_frac*100)}%"),
                ("Obs FLOPs",         f"{gt_flops[obs_idx]/1e15:.2f} PFLOPs" if has_obs else "none"),
                ("Obs checkpoints",   f"{n_observe}"),
                ("Loss at start",     f"{gt_loss_real[0]:.4f}"),
                ("Loss at cutoff",    f"{loss_at_cutoff:.4f}" if has_obs else "n/a"),
                ("Drop observed",     f"{loss_drop_obs:.4f} ({pct_drop_obs:.0f}%)" if has_obs else "n/a"),
            ]),
            ("PREDICTION", [
                ("True final",        f"{true_final:.4f}"),
                ("Pred final",        f"{pred_final:.4f}"),
                ("Abs error",         f"{abs_err:.4f}"),
                ("Rel error",         f"{rel_err:.2f}%"),
                ("90% CI width",      f"{ci_width:.4f}"),
                ("CI q05",            f"{float(q05[-1]):.4f}"),
                ("CI q95",            f"{float(q95[-1]):.4f}"),
                ("True in CI",        "✓ yes" if true_in_ci else "✗ no"),
            ]),
            *([("CONTEXT", [("Context curves", f"{context} : {context_plotted}")])] if context_plotted else []),
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
                is_context = section_title == "CONTEXT"
                color = (
                    "#DC2626" if is_err
                    else "#16A34A" if (is_true_ci and true_in_ci) or is_context
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
                        color="#D1D5DB", linewidth=0.5, clip_on=False)
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
        fname = os.path.join(output_dir, filename)
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  True final        : {true_final:.4f}")
        print(f"  Pred final        : {pred_final:.4f}")
        print(f"  Abs error         : {abs_err:.4f}")
        print(f"  Rel error         : {rel_err:.2f}%")
        print(f"  True in CI        : {'yes' if true_in_ci else 'no'}")
        print(f"  Saved → {fname}\n")