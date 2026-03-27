import matplotlib.pyplot as plt
import numpy as np
import json

# Load your JSON data
with open("residual_nll_mse_table.json") as f:
    data = json.load(f)

widths = [64, 32, 24, 16, 8]  # Example widths
epochs = ["ep0", "ep5", "ep20", "ep50", "ep90"]

for width in widths:
    configs_x = [f"{width}_config_{i}_obs_target" for i in range(1, 13)]
    configs_y = [f"config_{i}" for i in range(1, 13)]
    for epoch in epochs:
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.axis('off')

        n_rows = 13  # 12 configs + 1 baseline row
        n_cols = 12
        cell_width = 1 / n_cols
        subcell_width = cell_width / 2

        # Prepare table values (rows x cols x 2 metrics)
        table_values = np.zeros((n_rows, n_cols, 2))

        # --- Fill baseline row first ---
        baseline_key = f"baseline_{epoch}"
        for j, col_key in enumerate(configs_y):
            if baseline_key in data and col_key in data[baseline_key]:
                metrics = data[baseline_key][col_key]
                table_values[0, j, 0] = metrics['mse']
                table_values[0, j, 1] = metrics['nll']

        # --- Fill config rows ---
        for i, row_key in enumerate(configs_x):
            key_full = f"{row_key}_{epoch}"
            if key_full in data:
                for j, col_key in enumerate(configs_y):
                    metrics = data[key_full][col_key]
                    table_values[i+1, j, 0] = metrics['mse']  # shift by 1 for baseline row
                    table_values[i+1, j, 1] = metrics['nll']

        # Find min and max per column (ignore baseline row for coloring)
        min_mse = np.min(table_values[1:, :, 0], axis=0)
        max_mse = np.max(table_values[1:, :, 0], axis=0)
        min_nll = np.min(table_values[1:, :, 1], axis=0)
        max_nll = np.max(table_values[1:, :, 1], axis=0)

        # Draw cells
        for i in range(n_rows):
            for j in range(n_cols):
                # MSE mini-cell
                color_mse = 'lightgreen' if (i>0 and table_values[i,j,0]==min_mse[j]) else \
                            ('salmon' if (i>0 and table_values[i,j,0]==max_mse[j]) else 'white')
                rect = plt.Rectangle((j*cell_width , 1-(i+1)/n_rows ),
                                    subcell_width, 1/n_rows,
                                    facecolor=color_mse, edgecolor='black')
                ax.add_patch(rect)
                ax.text(j*cell_width + subcell_width/2, 1-(i+0.5)/n_rows,
                        f"{table_values[i,j,0]:.3e}", ha='center', va='center', fontsize=6)

                # NLL mini-cell
                color_nll = 'lightgreen' if (i>0 and table_values[i,j,1]==min_nll[j]) else \
                            ('salmon' if (i>0 and table_values[i,j,1]==max_nll[j]) else 'white')
                rect = plt.Rectangle((j*cell_width + subcell_width , 1-(i+1)/n_rows ),
                                    subcell_width, 1/n_rows,
                                    facecolor=color_nll, edgecolor='black')
                ax.add_patch(rect)
                ax.text(j*cell_width + 3*subcell_width/2, 1-(i+0.5)/n_rows,
                        f"{table_values[i,j,1]:.2f}", ha='center', va='center', fontsize=6)

        # Column labels (1-12)
        for j in range(n_cols):
            ax.text((j+0.5)*cell_width, 1+0.05, str(j+1), ha='center', va='bottom', fontsize=8)

            # Sub-column titles
            ax.text(j*cell_width + subcell_width/2, 1+0.02, "MSE", ha='center', va='bottom', fontsize=6)
            ax.text(j*cell_width + 3*subcell_width/2, 1+0.02, "NLL", ha='center', va='bottom', fontsize=6)

        # Row labels (Baseline + 1-12)
        ax.text(-0.01, 1-(0.5)/n_rows, "Baseline", ha='right', va='center', fontsize=8)
        for i in range(1, n_rows):
            ax.text(-0.01, 1-(i+0.5)/n_rows, str(i), ha='right', va='center', fontsize=8)

        ax.set_title(f"Width={width}, Obs={epoch} | MSE/NLL Table (min green, max red)", 
                    fontsize=12, pad=35)

        # Save each plot as PNG
        plt.savefig(f"norm_table_hd{width}_{epoch}.png", bbox_inches='tight', dpi=300)
        plt.close()