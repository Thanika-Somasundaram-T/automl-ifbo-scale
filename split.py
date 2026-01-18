"""
Split experiment results by architecture configuration.

This script loads a JSON file containing experiment metrics and
splits runs by hidden dimension for a fixed number of layers.
Each split is saved as a separate JSON file.

Intended to be run as a standalone utility script.
"""

import json
import os


# Input path
input_path = "./results/results_metrics.json"

# Output directory
output_dir = "./results/"

# Split configuration
NUM_LAYERS = 4
HIDDEN_DIMS = [4, 8, 16, 18, 24, 32, 64, 128]


def filter_results(results, num_layers, hidden_dim):
    """
    Filter experiment results by model architecture.

    Args:
        results : dict
            Dictionary containing all experiment runs.
        num_layers : int
            Number of layers to filter for.
        hidden_dim : int
            Hidden dimension to filter for.

    Returns:
        dict
            Filtered experiment results.
    """
    return {
        key: run_data
        for key, run_data in results.items()
        if run_data.get("num_layers") == num_layers
        and run_data.get("hidden_dim") == hidden_dim
    }


# Load full results file
if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} not found!")

with open(input_path, "r") as f:
    all_results = json.load(f)


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# Split and save per hidden dimension
for hidden_dim in HIDDEN_DIMS:
    filtered_results = filter_results(
        all_results,
        num_layers=NUM_LAYERS,
        hidden_dim=hidden_dim,
    )

    output_path = os.path.join(
        output_dir, f"results_hd{hidden_dim}.json"
    )

    with open(output_path, "w") as f:
        json.dump(filtered_results, f, indent=4)

    print(
        f"Saved {len(filtered_results)} entries "
        f"(num_layers={NUM_LAYERS}, hidden_dim={hidden_dim}) "
        f"to {output_path}"
    )
