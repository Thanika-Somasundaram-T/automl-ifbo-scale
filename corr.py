import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --- Load results ---
with open("./results/results_loss_curves.json", "r") as f:
    results = json.load(f)

epochs = 60  # maximum training epochs
hyperparam = input("Enter hyperparameter to plot (e.g., num_layers, hidden_dim, lr, weight_decay, lr_schedule): ")
epoch_interval = 5  # compute rank correlation every 5 epochs

# --- Group results by hyperparameter value ---
groups = {}
for key, data in results.items():
    val = data[hyperparam]
    if val not in groups:
        groups[val] = []
    groups[val].append(data)

# --- Compute Spearman rank correlation over epochs ---
correlations = {}
for val, models in groups.items():
    final_losses = np.array([m["best_val_loss"] for m in models])
    final_ranks = np.argsort(np.argsort(final_losses))  # final performance ranking
    rhos = []

    for epoch in range(0, epochs, epoch_interval):
        epoch_vals = [m["val_curve"][epoch] if len(m["val_curve"]) > epoch else m["val_curve"][-1] for m in models]
        epoch_ranks = np.argsort(np.argsort(epoch_vals))
        rho, _ = spearmanr(epoch_ranks, final_ranks)
        rhos.append(rho)

    correlations[val] = rhos

# --- Handle categorical mapping for color ---
vals_sorted = list(correlations.keys())
try:
    # Try numeric
    vals_numeric = np.array(vals_sorted, dtype=float)
except ValueError:
    # Map categorical to numeric
    vals_numeric = np.arange(len(vals_sorted))
    val_to_num = dict(zip(vals_sorted, vals_numeric))
    vals_numeric = np.array([val_to_num[v] for v in vals_sorted])

# --- Color mapping (use darker oranges) ---
norm = Normalize(vmin=min(vals_numeric), vmax=max(vals_numeric))
def darken(val):
    return 0.4 + 0.6 * norm(val)  # scale to 0.4-1 for visible color
cmap = plt.cm.Blues
colors = {val: cmap(darken(vals_numeric[i])) for i, val in enumerate(vals_sorted)}

# --- Plot ---
plt.figure(figsize=(10, 6))
x_epochs = list(range(0, epochs, epoch_interval))
for val, rhos in correlations.items():
    plt.plot(x_epochs, rhos, label=f"{hyperparam}={val}", color=colors[val], linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Spearman rank correlation (ρ)")
plt.title(f"Rank correlation (within-group) for hyperparameter '{hyperparam}'")
plt.legend(title=hyperparam)
plt.ylim(-0.5, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Save figure ---
plt.savefig(f"rank_correlation_{hyperparam}.png", dpi=300)
plt.show()
