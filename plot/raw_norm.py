import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import fixed_range_normalize, min_max_normalize

# --------------------------------------------------------
# Min–max normalization function
# --------------------------------------------------------


# --------------------------------------------------------
# Blue shades for hidden_dim
# --------------------------------------------------------
HD_COLORS = {
    4:    "#a8c5ff",   # light blue
    8:    "#76a9ff",   # medium light
    12:   "#3b82f6",   # blue (Tailwind style)
    16:   "#1e40af",   # deep blue
    24:   "#1e3a8a",   # darker navy
    32:   "#1e3a70",   # slightly darker navy
    64:   "#1e2d5c",   # deep navy
    128:  "#0f1a3d"    # very dark navy
}



# --------------------------------------------------------
# Main plotting
# --------------------------------------------------------
def plot_curves(json_path="./results/results_metrics.json", save_dir="./normalized_curves"):
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} runs")

    # -------------------------------
    # Prepare figures
    # -------------------------------
    plt_raw = plt.figure(figsize=(12, 6))
    ax_raw = plt_raw.add_subplot(111)
    ax_raw.set_title("Raw Validation Accuracy Curves (Blue Shades per Hidden Dim)")
    ax_raw.set_xlabel("Epoch")
    ax_raw.set_ylabel("Validation Accuracy")
    ax_raw.grid(True)

    plt_norm = plt.figure(figsize=(12, 6))
    ax_norm = plt_norm.add_subplot(111)
    ax_norm.set_title("Normalized Validation Accuracy Curves (Blue Shades per Hidden Dim)")
    ax_norm.set_xlabel("Epoch")
    ax_norm.set_ylabel("Norm Accuracy (0–1)")
    ax_norm.grid(True)

    # -------------------------------
    # Plot runs
    # -------------------------------
    for run_key, run_data in results.items():

        y_raw = run_data.get("val_acc_curve")
        if not y_raw:
            continue

        hidden_dim = run_data.get("hidden_dim", None)
        color = HD_COLORS.get(hidden_dim, "#60a5fa")   # fallback blue

        epochs = np.arange(1, len(y_raw) + 1)
        y_norm = min_max_normalize(y_raw)

        label = f"hd={hidden_dim}, lr={run_data['lr']}"

        # RAW plot
        ax_raw.plot(epochs, y_raw, color=color, linewidth=2, label=label)

        # NORMALIZED plot
        ax_norm.plot(epochs, y_norm, color=color, linewidth=2, label=label)

    # -------------------------------
    # Save figures
    # -------------------------------
    ax_raw.legend()
    ax_norm.legend()

    raw_path = os.path.join(save_dir, "all_curves_raw_min_max.png")
    norm_path = os.path.join(save_dir, "all_curves_norm_min_max.png")

    plt_raw.tight_layout()
    plt_norm.tight_layout()

    plt_raw.savefig(raw_path)
    plt_norm.savefig(norm_path)

    print("✔ Saved:")
    print(" →", raw_path)
    print(" →", norm_path)


if __name__ == "__main__":
    plot_curves()
