import json
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import normalize_log_loss_curve, normalize_log_loss_curve_per_curve


# ----------------------------
# Load data
# ----------------------------

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


# ----------------------------
# Group by width
# ----------------------------

def group_by_width(data):
    grouped = {}

    for key, value in data.items():
        width = value["hidden_dim"]

        if width not in grouped:
            grouped[width] = []

        grouped[width].append((key, value["val_loss_curve"]))

    return grouped


# ----------------------------
# Group by key across widths
# ----------------------------

def group_by_key_across_widths(data):
    grouped = {}

    for key, value in data.items():
        width = value["hidden_dim"]

        base_key = key.replace(f"_hd{width}", "")

        if base_key not in grouped:
            grouped[base_key] = []

        grouped[base_key].append((width, np.array(value["val_loss_curve"])))

    return grouped


# ----------------------------
# Plot all curves for one width
# ----------------------------

def plot_width_curves(width, curves, keys, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, len(curves)))

    for i in range(len(curves)):
        curve = np.array(curves[i])

        global_norm = normalize_log_loss_curve(curve)
        per_curve_norm = normalize_log_loss_curve_per_curve(curve)

        color = colors[i]

        plt.plot(global_norm, color=color, linestyle="solid", alpha=0.9)
        plt.plot(per_curve_norm, color=color, linestyle="dashed", alpha=0.9)

    plt.title(f"Width {width} - Global (solid) vs Per-curve (dashed)")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Performance")
    plt.grid()

    plt.savefig(os.path.join(save_dir, f"width_{width}_all_curves.png"))
    plt.close()


# ----------------------------
# Plot per key across widths (FIXED)
# ----------------------------

def plot_key_across_widths(grouped_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for key, items in grouped_data.items():

        # Sort by width
        items = sorted(items, key=lambda x: x[0])
        widths = [w for w, _ in items]
        curves = [c for _, c in items]

        plt.figure(figsize=(10, 6))

        # one color per width
        colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

        for i in range(len(curves)):
            curve = np.array(curves[i])
            color = colors[i]

            global_norm = normalize_log_loss_curve(curve)
            per_curve_norm = normalize_log_loss_curve_per_curve(curve)

            # SAME COLOR → SAME WIDTH
            plt.plot(
                global_norm,
                color=color,
                linestyle="solid",
                linewidth=2,
                label=f"w={widths[i]}"
            )

            plt.plot(
                per_curve_norm,
                color=color,
                linestyle="dashed",
                alpha=0.9
            )

        plt.title(f"{key} across widths")
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Performance")
        plt.legend(title="Width")
        plt.grid()

        safe_key = key.replace("/", "_")
        plt.savefig(os.path.join(save_dir, f"{safe_key}.png"))
        plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    data_path = "./results/results_metrics.json"
    save_root = "plots_scale"

    data = load_data(data_path)

    # -------- Width-wise plots --------
    grouped = group_by_width(data)

    for width, items in grouped.items():
        keys = [k for k, _ in items]
        curves = [v for _, v in items]

        print(f"Processing width {width} with {len(curves)} curves")

        plot_width_curves(
            width,
            curves,
            keys,
            save_dir=os.path.join(save_root, f"width_{width}", "combined")
        )

    # -------- Key-wise across widths --------
    print("Processing per-key across widths...")

    grouped_keys = group_by_key_across_widths(data)

    plot_key_across_widths(
        grouped_keys,
        save_dir=os.path.join(save_root, "per_key_across_widths")
    )


if __name__ == "__main__":
    main()