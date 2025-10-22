import json
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import parse_key

def plot_results(results_path="./results/results_summary.json", save_dir="./results/plots"):
    if not os.path.exists(results_path):
        print(f"❌ No results file found at {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs(save_dir, exist_ok=True)

    for key, entries in data.items():
        num_layers, use_context = key.split("_")

        # Parse hyperparams and values
        lrs, hds, wds, vals = [], [], [], []
        for subkey, acc in entries.items():
            lr, hd, wd = parse_key(subkey)
            lrs.append(lr)
            hds.append(hd)
            wds.append(wd)
            vals.append(acc)

        lrs, hds, wds, vals = np.array(lrs), np.array(hds), np.array(wds), np.array(vals)

        # --- 4D scatter plot ---
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            lrs,
            vals,
            c=hds,
            s=(np.array(wds) * 90000 + 50),  # scale size to make effect visible
            cmap="viridis",
            alpha=0.8,
            edgecolors="k",
        )

        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Validation Accuracy")
        plt.title(f"num_layers={num_layers}, use_context={use_context}")

        # --- Colorbar for hidden dim ---
        cbar = plt.colorbar(scatter)
        cbar.set_label("Hidden Dim (hd)")

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        # --- Save ---
        save_path = os.path.join(save_dir, f"{num_layers}_{use_context}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved plot for {num_layers}_{use_context} at {save_path}")


if __name__ == "__main__":
    plot_results()
