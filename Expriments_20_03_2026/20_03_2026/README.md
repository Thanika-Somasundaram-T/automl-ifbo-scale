# IFBO Scale Prediction Analysis

Analysis of FT-PFN predictions across different source hyperparameter contexts and base scales.

## Objective

Evaluate how well the model predicts target HP learning curves (hd=128) when conditioned on different source HPs, across base scales (hd: 8, 16, 24, 32, 64) and observation epochs (T = 0, 5, 10, 20, 50, 90).

## Project Structure

```
├── run_analysis.py       # Entry point — runs full pipeline
├── config.py             # Paths, constants, HP mappings
├── metrics.py            # NLL/MSE calculations, normalization, data loading
├── plot_boxplots.py      # Cross-scale + per-source-HP boxplots
├── plot_heatmaps.py      # 12×12 source→target NLL heatmaps
├── generate_html.py      # Cross-ranking HTML tables (NLL & MSE)
├── generate_csv.py       # CSV export + rank pivot
├── data/                 # Self-contained input data
│   ├── predictions/      # FT-PFN prediction JSONs
│   └── results_metrics.json  # Ground truth val_loss curves
├── outputs/              # Normalized mode results
│   ├── nll/              # Boxplots, heatmaps, HTML ranking
│   ├── mse/              # HTML ranking only
│   ├── all_metrics.csv
│   └── rank_source_per_target.csv
└── outputs_raw/          # Raw loss mode results
    ├── nll/              # Boxplots (log-scale), heatmaps, HTML ranking
    ├── mse/              # HTML ranking only
    ├── all_metrics.csv
    └── rank_source_per_target.csv
```

## Evaluation Modes

### NORMALIZED (all-future avg)
- **Space**: Normalized `[0, 1]` performance (higher = better)
- **Metric**: NLL and MSE averaged across the full predicted trajectory
- **Plots**: Linear scale, universal y-axis bounds across epochs

### RAW LOSS (final-point only)
- **Space**: Raw validation loss (lower = better)
- **Metric**: NLL and MSE at the final timestamp (epoch 100) only
- **Plots**: Log-scale y-axis for boxplots; linear heatmaps with universal bounds

## Usage

```bash
python run_analysis.py
```

## Generated Outputs (per mode)

| Output | NLL | MSE |
|--------|-----|-----|
| Cross-scale boxplots | ✅ | — |
| Source HP boxplots (per scale) | ✅ | — |
| 12×12 heatmaps (per scale × epoch) | ✅ | — |
| Cross-ranking HTML table | ✅ | ✅ |
| CSV (all_metrics + rank pivot) | ✅ (combined) | ✅ (combined) |
