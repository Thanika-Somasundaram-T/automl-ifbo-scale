"""
HTML generation: cross-ranking source→target tables.
Supports both NLL and MSE metrics.
"""
import html as html_mod
import os

import numpy as np

from config import (
    BASE_SCALES, OBS_EPOCHS, SOURCE_CONFIGS,
    sorted_target_hps, short_target_label, source_hp_label, clean_scale,
)


def generate_cross_ranking_html(all_rows, output_dir, metric="nll"):
    """
    HTML with 12×12 rank heatmap matrices.
    One table per (scale, epoch): rows=source HPs, cols=target HPs.
    Cell = rank of that (source, target) metric within its column (per target HP).
    """
    metric_label = metric.upper()
    target_hps = sorted_target_hps(all_rows)
    target_labels = [short_target_label(hp) for hp in target_hps]

    sections = []
    table_idx = 0

    for scale in BASE_SCALES:
        for epoch in OBS_EPOCHS:
            table_idx += 1

            # Build metric matrix
            metric_matrix = {}
            for r in all_rows:
                if r["scale"] != scale or r["epoch"] != epoch or r["scale"] == "baseline":
                    continue
                metric_matrix[(r["source_config"], r["target_hp"])] = r[metric]

            if not metric_matrix:
                continue

            # Rank within each target HP column (rank 1 = lowest = best)
            rank_matrix = {}
            for tgt_hp in target_hps:
                col_vals = [(src, metric_matrix.get((src, tgt_hp), np.nan))
                            for src in SOURCE_CONFIGS]
                col_vals.sort(key=lambda x: x[1] if not np.isnan(x[1]) else float("inf"))
                for rank, (src, _) in enumerate(col_vals, 1):
                    rank_matrix[(src, tgt_hp)] = rank

            max_rank = 12

            def rank_to_style(rank_val):
                t = (rank_val - 1) / (max_rank - 1) if max_rank > 1 else 0
                r = int(60 + 180 * t)
                g = int(180 - 120 * t)
                b = int(60 - 30 * t)
                bg = f"rgb({r},{g},{b})"
                txt = "#ffffff" if t > 0.4 else "#1d2430"
                return f"background:{bg};color:{txt};"

            header = "".join(
                f"<th style='font-size:10px'>{html_mod.escape(tl)}</th>"
                for tl in target_labels
            )
            body = []
            for src_cfg in SOURCE_CONFIGS:
                cells = []
                for tgt_hp in target_hps:
                    rank = rank_matrix.get((src_cfg, tgt_hp), "")
                    style = rank_to_style(rank) if rank != "" else ""
                    cells.append(f"<td style='{style};text-align:center;font-weight:700'>{rank}</td>")
                body.append(
                    f"<tr><th style='text-align:left;font-size:10px'>"
                    f"{html_mod.escape(source_hp_label(src_cfg))}</th>"
                    f"{''.join(cells)}</tr>"
                )

            section = (
                f"<h2>Table {table_idx}: Scale={html_mod.escape(clean_scale(scale))}, T={epoch}</h2>"
                f"<p style='font-size:12px;color:#666'>Rank 1 (green) = best source HP for that target. "
                f"Rank 12 (red) = worst.</p>"
                f"<table><thead><tr>"
                f"<th style='min-width:150px'>Source HP \\ Target HP</th>"
                f"{header}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>"
            )
            sections.append(section)

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Source→Target HP Cross-Ranking ({metric_label})</title>"
        "<style>"
        "body{font-family:'Segoe UI',Arial,sans-serif;margin:20px;background:#f8f9fa;color:#1d2430;}"
        "h1{margin:0 0 8px 0;} h2{margin:28px 0 8px 0;font-size:16px;}"
        "table{border-collapse:collapse;margin-bottom:20px;}"
        "th,td{border:1px solid #d5dbe3;padding:4px 6px;font-size:11px;}"
        "th{background:#e9eef5;}"
        "tbody tr:hover{outline:2px solid #1E88E5;}"
        "</style></head><body>"
        f"<h1>Source→Target HP Cross-Ranking ({metric_label})</h1>"
        f"<p>For each (scale, T): which source HP produces the best prediction "
        f"for each target HP? Rank 1 = lowest {metric_label} (best). "
        f"Green cells = good, red = bad.</p>"
        f"{''.join(sections)}"
        "</body></html>"
    )

    out = os.path.join(output_dir, f"cross_ranking_source_target.html")
    os.makedirs(output_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"✅ Saved: {out}")
