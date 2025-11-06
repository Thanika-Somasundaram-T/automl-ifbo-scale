import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.colors import Normalize

# --- Load results ---
with open("./results/results_loss_curves.json", "r") as f:
    results = json.load(f)

epochs = 60  # maximum training epochs
epoch_interval = 5  # compute rank correlation every N epochs

# --- Ask user for hyperparameter to compare ---
hyperparam = input("Enter hyperparameter to compare (num_layers, hidden_dim, lr, weight_decay, lr_schedule): ")

# --- Collect unique hyperparameter values ---
hp_values = sorted(list({data[hyperparam] for data in results.values()}))

# --- Assign colors (darker blues for higher values or later in the list) ---
norm = Normalize(vmin=0, vmax=len(hp_values) - 1)
cmap = plt.cm.Blues
colors = {val: cmap(0.4 + 0.6 * norm(i)) for i, val in enumerate(hp_values)}  # darken to 0.4–1

# --- Prepare final losses and ranks across all models ---
all_models = list(results.values())
final_losses = np.array([m["val_curve"][59] for m in all_models])
final_ranks = np.argsort(np.argsort(final_losses))

# --- Compute Spearman correlation per hyperparameter value ---
correlations = {val: [] for val in hp_values}

for epoch in range(0, epochs, epoch_interval):
    epoch_vals = np.array([
        m["val_curve"][epoch] if len(m["val_curve"]) > epoch else m["val_curve"][59]
        for m in all_models
    ])
    epoch_ranks = np.argsort(np.argsort(epoch_vals))

    for val in hp_values:
        # Indices of models with this hyperparameter value
        indices = [i for i, m in enumerate(all_models) if m[hyperparam] == val]
        # Spearman correlation of these models' epoch ranks vs final ranks
        rho, _ = spearmanr(epoch_ranks[indices], final_ranks[indices])
        correlations[val].append(rho)

# --- Plot ---
plt.figure(figsize=(10, 6))
x_epochs = list(range(0, epochs, epoch_interval))
for val, rhos in correlations.items():
    plt.plot(x_epochs, rhos, label=f"{hyperparam}={val}", color=colors[val], linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Spearman rank correlation (ρ)")
plt.title(f"Rank correlation vs best performance across hyperparameter '{hyperparam}'")
plt.legend(title=hyperparam)
plt.ylim(-1, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Save figure ---
plt.savefig(f"rank_correlation_all_{hyperparam}.png", dpi=300)
plt.show()
