"""
Same ablation as run_diverse_lower_k.py, restricted to the 4-scale ladder
{24, 32, 64, 128} (mirroring the real-world data's scale count), instead of
all 7 MLP scales. Both the context pool AND the target/query configs are
restricted to these scales -- so e.g. all_lower for target=64 only pulls
from {24,32}, not {4,8,16,24,32}.

24 is the smallest scale in this ladder, so (like hidden_dim=4 in the full
7-scale run) it has no lower scale to draw context from and is excluded as
a target scale when computing the final table (see compute_diverse_table_ladder.py).

Run from the repo root:
    python run_diverse_lower_k_ladder.py
"""
import json

from run_diverse_lower_k import RESULTS_JSON, run_batch

SCALE_LADDER = {24, 32, 64, 128}
SAVE_DIR = "./results_diverse_lower_k_ladder"


def main():
    with open(RESULTS_JSON) as f:
        all_results = json.load(f)

    ladder_results = {
        run_key: run_data
        for run_key, run_data in all_results.items()
        if run_data.get("hidden_dim") in SCALE_LADDER
    }
    print(f"Restricted to {len(ladder_results)} runs across scales {sorted(SCALE_LADDER)}")

    run_batch(ladder_results, SAVE_DIR)


if __name__ == "__main__":
    main()
