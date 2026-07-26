"""
Plot val_loss curves for specific (lr, weight_decay) configs.

Usage:
    python plot_val_loss.py                      # uses DEFAULT_JSON_PATH below
    python plot_val_loss.py path/to/results.json  # or pass a path explicitly

Expects a JSON file shaped like:
{
    "some_run_name": {
        "lr": 0.001,
        "num_layers": 4,
        "hidden_dim": 64,
        "weight_decay": 0.0,
        "lr_schedule": "cosine",
        "train_loss_curve": [...],
        "val_loss_curve": [...],
        "best_val_loss": 0.331...
    },
    ...
}

It will find, for each requested (lr, weight_decay) pair, the run(s) that match
and plot their val_loss_curve with the color/label you specify below.
"""

import sys
import json
import math
import matplotlib.pyplot as plt

# ---- Configure the (lr, weight_decay, color, label) combos you want plotted ----
# Edit this list to match whatever runs you care about.
TARGETS = [
    {"lr": 0.00003,    "wd": 0.0, "color": "skyblue",    "label": "lr=0.00003, wd=0.0"},
    {"lr": 0.0001,   "wd": 0.0,    "color": "blue",  "label": "lr=0.0001, wd=0.0"},
    {"lr": 0.00003,   "wd": 0.01, "color": "green",   "label": "lr=0.00003, wd=0.01"},
    {"lr": 0.00001,  "wd": 0.0,  "color": "red",  "label": "lr=0.00001, wd=0.0"},
    {"lr": 0.003,  "wd": 0.0,    "color": "brown",   "label": "lr=0.003, wd=0.0"},
]

REL_TOL = 1e-6  # tolerance for float matching lr / weight_decay

DEFAULT_JSON_PATH = "./results_***/results_metrics.json"


def close(a, b, tol=REL_TOL):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def find_matches(data, lr, wd):
    matches = []
    for run_name, cfg in data.items():
        try:
            run_lr = cfg["lr"]
            run_wd = cfg["weight_decay"]
        except (KeyError, TypeError):
            continue
        if close(run_lr, lr) and close(run_wd, wd):
            matches.append((run_name, cfg))
    return matches


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON_PATH
    data = load_results(json_path)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    plotted_any = False
    for target in TARGETS:
        matches = find_matches(data, target["lr"], target["wd"])

        if not matches:
            print(f"[warning] no run found for lr={target['lr']}, wd={target['wd']}")
            continue

        if len(matches) > 1:
            print(
                f"[info] {len(matches)} runs matched lr={target['lr']}, wd={target['wd']}: "
                f"{[m[0] for m in matches]} -- plotting all of them."
            )

        for run_name, cfg in matches:
            val_curve = cfg.get("val_loss_curve", [])
            if not val_curve:
                print(f"[warning] run '{run_name}' has an empty val_loss_curve, skipping.")
                continue

            epochs = range(1, 101)
            label = target["label"]
            if len(matches) > 1:
                label = f"{label} ({run_name})"

            ax.plot(
                epochs,
                val_curve[:100],
                color=target["color"],
                linewidth=1.6,      # same as first script
                label=label,
            )
            plotted_any = True

    if not plotted_any:
        print("Nothing was plotted -- check that your JSON contains the requested configs.")
        sys.exit(1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_ylim(0.2, 1.0)

    # Match styling of first script
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Uncomment if desired
    # ax.legend(frameon=False, fontsize=9, loc="best")

    out_path = "ranking.svg"

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=600,              # ignored for SVG but harmless
        bbox_inches="tight",
        pad_inches=0.02,
    )

    print(f"Saved plot to {out_path}")

    plt.show()
    
    
if __name__ == "__main__":
    main()