"""
Plot val_loss curves for every run found in a results JSON file.

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

Every run with a non-empty val_loss_curve gets plotted, each with its own
color (auto-assigned from a colormap) and a legend entry built from its
lr / weight_decay.
"""

import sys
import json
import matplotlib.pyplot as plt

DEFAULT_JSON_PATH = "./results_***/results_hd128.json"


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def make_label(run_name, cfg):
    lr = cfg.get("lr")
    wd = cfg.get("weight_decay")
    if lr is not None and wd is not None:
        return f"lr={lr}, wd={wd}"
    return run_name


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON_PATH
    data = load_results(json_path)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Only keep runs that actually have a val_loss_curve to plot.
    runs = []
    for run_name, cfg in data.items():
        val_curve = cfg.get("val_loss_curve", [])
        if val_curve:
            runs.append((run_name, cfg, val_curve))
        else:
            print(f"[warning] run '{run_name}' has an empty val_loss_curve, skipping.")

    if not runs:
        print("Nothing to plot -- no runs with a val_loss_curve were found.")
        sys.exit(1)

    colors = plt.colormaps["tab20"].resampled(len(runs))

    for i, (run_name, cfg, val_curve) in enumerate(runs):
        epochs = range(1, len(val_curve) + 1)
        ax.plot(
            epochs,
            val_curve,
            color=colors(i),
            label=make_label(run_name, cfg),
            linewidth=1.8,
        )

    ax.xlabel("Epoch")
    ax.ylabel("Validation Loss")
    ax.ylim(0.2, 1.0)
    # plt.title("Validation Loss — All Runs")
    # plt.legend(fontsize=8, loc="upper right", ncol=1 if len(runs) <= 15 else 2)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    out_path = "val_loss_comparison.png"
    
    
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=600,              # ignored for SVG but harmless
        bbox_inches="tight",
        pad_inches=0.02,
    )


    plt.show()


if __name__ == "__main__":
    main()