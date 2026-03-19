"""
CSV generation: raw metrics export and rank pivot tables.
"""
import csv
import os

import numpy as np

from config import (
    BASE_SCALES, OBS_EPOCHS, SOURCE_CONFIGS,
    sorted_target_hps, short_target_label, source_hp_label,
)


def generate_csv_tables(all_rows, output_dir):
    """Raw CSV with all metrics + rank per target HP."""
    os.makedirs(output_dir, exist_ok=True)

    # Full data CSV
    csv_path = os.path.join(output_dir, "all_metrics.csv")
    fieldnames = ["scale", "source_config", "source_hp", "epoch",
                   "target_hp", "target_hp_short", "nll", "mse"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"✅ Saved: {csv_path}")

    # Rank pivot: for each (scale, epoch, target_hp), rank source HPs by NLL
    target_hps = sorted_target_hps(all_rows)
    pivot_rows = []

    for scale in BASE_SCALES:
        for epoch in OBS_EPOCHS:
            for tgt_hp in target_hps:
                relevant = [(r["source_config"], r["nll"])
                            for r in all_rows
                            if r["scale"] == scale and r["epoch"] == epoch
                            and r["target_hp"] == tgt_hp
                            and r["scale"] != "baseline"
                            and not np.isnan(r["nll"])]
                relevant.sort(key=lambda x: x[1])
                for rank, (src_cfg, nll_val) in enumerate(relevant, 1):
                    pivot_rows.append({
                        "scale": scale,
                        "epoch": epoch,
                        "target_hp": short_target_label(tgt_hp),
                        "source_config": src_cfg,
                        "source_hp": source_hp_label(src_cfg),
                        "nll": round(nll_val, 4),
                        "rank": rank,
                    })

    pivot_path = os.path.join(output_dir, "rank_source_per_target.csv")
    with open(pivot_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scale", "epoch", "target_hp", "source_config", "source_hp", "nll", "rank",
        ])
        writer.writeheader()
        for r in pivot_rows:
            writer.writerow(r)
    print(f"✅ Saved: {pivot_path}")
