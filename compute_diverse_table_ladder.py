"""
Same table as compute_diverse_table.py, but for the restricted 4-scale
ladder {24, 32, 64, 128} produced by run_diverse_lower_k_ladder.py.

24 is the smallest scale in this ladder -- it has no lower scale within the
ladder to draw context from, so (mirroring how hidden_dim=4 is excluded in
the full 7-scale table) it's excluded as a target scale here too.

Run from the repo root, after run_diverse_lower_k_ladder.py has produced
predictions:
    python compute_diverse_table_ladder.py
"""
from compute_diverse_table import compute_table
from run_diverse_lower_k_ladder import SAVE_DIR


def main():
    compute_table(SAVE_DIR, excluded_target_scales={24})


if __name__ == "__main__":
    main()
