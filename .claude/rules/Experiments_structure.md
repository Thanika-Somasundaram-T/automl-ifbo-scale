# Experiments Folder Structure

Canonical layout for everything under `experiments/`. Mandatory for **curve**
experiments; defines the exception for **ablation** experiments. When adding or
moving plots, conform to this — do not invent new top-level plot dirs.

---

## 1. Experiment types

| Type | What it produces | Examples |
|------|------------------|----------|
| **Curve** | Per-`obs_frac` learning-curve prediction plots (uncertainty bands over a horizon) | `ex0_5`…`ex0_12`, `ex2_width_interpolation`, `ex3_width_extrapolation`, `ex3_1_nearest_scale_ablation_A` |
| **Ablation** | Heatmaps / by-scale / by-G summaries — *not* per-obs prediction curves | `ex4_2hp_lower_target_context_ablation`, `ex4d_multi_scale_partner_ablation`, `ex3_1_E_duplicate_context_ablation` |

If an experiment emits prediction curves, it is a **curve** experiment and §2
applies. If it only emits heatmaps/aggregate summaries, it is an **ablation**
experiment and §3 applies.

`experiments/_lib/` is the shared policy-agnostic machinery package for reusable
curve experiment code (data loading, prediction loops, metrics, plotting). It
must not own experiment-specific policy registries or policy names. Keep policy
semantics in the experiment package (`ex2_width_interpolation/policies.py`,
`ex3_width_extrapolation/policies.py`, etc.) and pass those registries into
`_lib` functions.

Ex2 owns width interpolation/bracketing policies such as `bracket_nearest`.
Ex3 is lower-to-higher width extrapolation only; do not add interpolation
policies to Ex3 policy registries or Ex3 transfer claims.

---

## 2. Curve experiments (mandatory layout)

```text
Experiment_*/
├── plots/
│   ├── _summary/                 # cross-obs / aggregate plots (see §4)
│   ├── obs_010/
│   │   ├── cascade_styles/       # cascade-stacked multi-target curves
│   │   └── supervisor_styles/    # supervisor-style single/grid prediction plots
│   ├── obs_020/
│   │   ├── cascade_styles/
│   │   └── supervisor_styles/
│   ├── obs_050/
│   │   └── ...
│   └── obs_090/
│       └── ...
└── _backups/                     # see §5
```

### Rules

- **`plots/` is the only plot root.** No `plots_supervisor_style/`,
  `plots_seam_artifact_backup/`, `cascade_curves/`, `supervisor_curves/`, or
  bare `obs0_10/` at the experiment root. Everything lives under `plots/`.
- **obs dirs are `obs_<NNN>`** where `NNN = round(obs_frac × 100)`, zero-padded
  to **3 digits**: `obs_005`, `obs_010`, `obs_020`, `obs_050`, `obs_090`. This
  sorts lexicographically in obs order.
- **Each `obs_<NNN>/` contains exactly two style dirs**: `cascade_styles/` and
  `supervisor_styles/`. A style dir may be empty if that style was not rendered,
  but the dir name is fixed.
- **The obs level lives in the directory, not the filename.** Inside
  `obs_010/supervisor_styles/`, name files by what varies *within* that obs level
  (policy, target, base/shrink/tkpm) — e.g.
  `all_lower_only_base14562560_shrink0.4_tkpm20.png`. Do **not** repeat `obs0_10`
  in the filename.

---

## 3. Ablation experiments (exception)

Ablation experiments keep their own layout because heatmaps do not map onto the
cascade/supervisor split. Standardize them as:

```text
Experiment_*/
├── plots/
│   ├── _summary/                 # cross-scale / cross-G aggregate plots
│   ├── <scale>/                  # e.g. 32M/ 77M/ 134M/ 287M/  (by-scale heatmaps)
│   └── cross_scale/              # plots spanning all scales
└── _backups/
```

- Per-scale heatmaps go in `plots/<scale>/` (e.g. `plots/287M/`).
- Loose heatmaps at the experiment root must be moved under `plots/`.
- If an ablation also has an obs dimension, encode obs **in the filename**
  (`..._obs0_10.png`) rather than creating `obs_<NNN>/` dirs — the
  cascade/supervisor structure does not apply here.

---

## 4. Aggregate / cross-obs plots → `plots/_summary/`

Any plot that is **not** specific to a single obs level lives in `plots/_summary/`
(leading underscore sorts it above the `obs_*` dirs):

- ranking-vs-obs sweeps (`rank_by_obs_frac.png`, `rank_by_scale.png`)
- regret / band-width summaries (`regret_by_policy.png`, `scale_gap_summary.png`)
- curve-shape overlays (`curve_shape_overlay.png`)
- any bar/box comparison aggregating across obs or policy

Do not leave these at the experiment root.

---

## 5. Backups → per-experiment `_backups/` with a NOTE

Each experiment keeps its **own** `_backups/` dir. Do not pool backups in a
global `experiments/_backups/`. Every backup subdir must carry a short `NOTE.md`
explaining *why* it was kept, so a future reader knows whether it is safe to
delete.

```text
Experiment_*/
└── _backups/
    ├── pre_normfix/
    │   ├── NOTE.md               # "snapshot before LOO normalization fix (2026-05-29); …"
    │   └── … old plots …
    └── seam_artifact/
        ├── NOTE.md               # "obs-boundary seam gap before the ribbon fix; …"
        └── … old plots …
```

`NOTE.md` minimum content:

```markdown
# Backup: <short name>
**Date:** YYYY-MM-DD
**Why kept:** <what state this captures and what changed after>
**Safe to delete when:** <condition, e.g. "after the fix is accepted in the writeup">
```

Backups are untracked by git — deletion is **not** recoverable. Never delete a
backup without a NOTE confirming it is obsolete.

---

## 6. Computed data → per-experiment `results/`

Prediction/run code writes its **computed data outputs** to `Experiment_*/results/`,
not the experiment root:

```text
Experiment_*/
├── results/                      # written by run_*/predict_* code
│   ├── <name>_predictions.parquet
│   ├── uncertainty_diagnostics.csv
│   ├── summary.json              # or final_metrics.csv / audit_counts.json (ablations)
│   └── _partial/                 # shard dir, if used
├── plots/                        # §2–§4
└── _backups/                     # §5
```

- The `predictions.parquet` (and per-experiment metrics/summary) the run script
  computes is the input the matching `plot_*.py` reads. Both sides point at
  `results/`.
- Code, docs, and rank/regret summary CSVs that are *inputs to `_summary/` plots*
  may stay at the experiment root if a script expects them there — but new
  prediction outputs go to `results/`.

### ⚠️ Cross-experiment references (do not break)

Some experiments read **another** experiment's prediction parquet as a reference.
If you move a parquet, update every downstream constant in the same change:

| Producer | `results/` path | Consumers (constant) |
|----------|-----------------|----------------------|
| `ex3_width_extrapolation` | `results/predictions.parquet` | `ex3_1_nearest_scale_ablation_A` · `ex3_1_E_duplicate_context_ablation` · `ex4_2hp_lower_target_context_ablation` (all `REFERENCE_PREDICTIONS`) |
| `ex4_2hp_lower_target_context_ablation` | `results/predictions.parquet` | `ex4d_multi_scale_partner_ablation` (`EX4B_OUTPUT_DIR / "results" / "predictions.parquet"`) |

A missed reference fails silently at the next downstream run — grep for the old
root path after any move.

Only **code** and **docs** (`README.md`, decision notes) reliably stay at the
experiment root.

### Per-experiment summary doc

Each experiment's run script writes a human-readable summary at the experiment
**root**, named **`SUMMARY.md`** (not `NOTES.md`). It describes the slice,
policy, normalization, and what the outputs mean.

- Filename is fixed: `SUMMARY.md`. Do not use `NOTES.md` — that name is retired.
- This is distinct from `_backups/<name>/NOTE.md` (§5), which documents *why a
  backup was kept*. Summary = what the experiment does; NOTE = why a snapshot
  exists.
- If a run script emits the summary, the `write_text("SUMMARY.md")` target lives
  in the script — renaming the file without editing the script is clobbered on
  the next run.

---

## 7. Checklist before committing plot/data changes

- [ ] All plots live under `plots/` (no sibling `plots_*` dirs)
- [ ] Curve experiment: obs dirs are `obs_<NNN>` (3-digit) with `cascade_styles/` + `supervisor_styles/`
- [ ] obs level is in the directory, not duplicated in filenames
- [ ] Cross-obs / aggregate plots are in `plots/_summary/`
- [ ] Computed data outputs (parquet/diagnostics/summary) write to `results/`
- [ ] Cross-experiment `REFERENCE_PREDICTIONS` / `EX4B_OUTPUT_DIR` reads updated to `results/`
- [ ] Per-experiment summary is `SUMMARY.md` at root (not `NOTES.md`); the `write_text` target in the run script matches
- [ ] Stale snapshots moved to `_backups/<name>/` with a `NOTE.md`
- [ ] No `__pycache__`, `*_test`, or scratch dirs left in `plots/`
