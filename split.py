import json
import os

# Paths
input_path = "./results/results_metrics.json"  # existing results
output_path = "./results/512.json"  # filtered output

# Load the full results
if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} not found!")

with open(input_path, "r") as f:
    all_results = json.load(f)

# Filter for num_layers=4 and hidden_dim=32
filtered_results = {}
for key, run_data in all_results.items():
    if run_data.get("num_layers") == 4 and run_data.get("hidden_dim") == 512:
        filtered_results[key] = run_data

# Save filtered results
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
with open(output_path, "w") as f:
    json.dump(filtered_results, f, indent=4)

print(f"Saved {len(filtered_results)} 4-layer, hd=32 entries to {output_path}")
