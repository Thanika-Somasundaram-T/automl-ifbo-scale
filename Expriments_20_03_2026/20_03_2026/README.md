# IFBO Scale Prediction Analysis

This folder contains the consolidated analysis script (`supervisor_analysis.py`) for evaluating the performance of Foundation Transformer for Parameter Prediction (FT-PFN).

## Objective
The primary goal is to analyze and compare the impact of providing different hyperparameter learning curves as context. We evaluate how well the model predicts the learning curves of *target* hyperparameters when conditioned on the learning curve of a *source* hyperparameter at various base scales (hidden dimensions: 8, 16, 24, 32, 64).

## Evaluation Modes
The analysis evaluates the predictions across multiple observation epochs (T = 0, 5, 10, 20, 50, 90) using two distinct methodologies:

1. **NORMALIZED (all-future avg)**
   - **Space**: Predictions and ground truth are compared in the normalized `[0, 1]` performance space (where higher is better).
   - **Metric**: The Negative Log-Likelihood (NLL) and Mean Squared Error (MSE) are averaged across the entire predicted trajectory curve.
   - **Visualization**: All plots use linear scaling with universal bounds. Output is saved to `outputs/`.

2. **RAW LOSS (final-point only)**
   - **Space**: Predictions are unnormalized back into the raw validation loss space (`val_loss`, lower is better) and compared against the raw ground truth loss.
   - **Metric**: The NLL and MSE calculations look *only* at the final timestamp (epoch 100), rather than averaging over the curve.
   - **Visualization**: Boxplots use a **log-scale** y-axis to accommodate the massive NLL spread between T=0 (~1e10) and T=90 (~1). Heatmaps use linear scaling with universal color bounds. Output is saved to `outputs_raw/`.

## Generated Outputs
For both modes, running the script generates the following artifacts:

- **Cross-scale Boxplots:** `cross_scale_nll_aggregated.png` compares NLL distributions across all base scale hidden dimensions and observation epochs.
- **Source HP Boxplots:** `source_hp_nll_scale_{X}.png` visualizes the distribution of target prediction NLLs for each of the 12 specific source hyperparameters used as context.
- **NLL Heatmaps:** `heatmap_nll_scale_{X}_T{Y}.png` visualizes the 12x12 matrix of absolute NLL values when predicting target HPs from source HPs. Uses universal color mapping across all matrices for consistent visual contrast.
- **CSV Data:** `all_metrics.csv` containing raw results, and `rank_source_per_target.csv` for rankings.
- **Cross-ranking HTML Table:** `cross_ranking_source_target.html` interactively displays the relative rank (1 to 12) of each source HP when predicting a specific target HP. Green cells (Rank 1) indicate the best source HP context for that given target column.
