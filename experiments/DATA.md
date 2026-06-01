# Data Dependencies

The experiment code under `experiments/` depends on raw datasets that are
**untracked by git** (not committed — see repo `.gitignore` conventions; the
whole `experiments/` tree and the large `demo_analysis/*.parquet` files are
intentionally outside version control).

## Required raw datasets

| File | Size | Used by |
|------|------|---------|
| `demo_analysis/warmstart_runs_flattened.parquet` | ~32 MB | `experiments/_lib/plot.py`, all `predict_*`/`run_*` scripts (curve experiments) |
| `demo_analysis/warmstart_runs_g5.parquet` | ~26 MB | secondary / G5 slice |

## How paths resolve

Scripts default to a **repo-root-relative** path:

```python
DEFAULT_RAW_PARQUET = Path("demo_analysis/warmstart_runs_flattened.parquet")
```

This resolves correctly **only when you run from the repository root**. From any
other working directory (including a git worktree that does not have the
untracked file), pass an absolute path explicitly:

```bash
python experiments/ex3_width_extrapolation/plot_supervisor_style_predictions.py \
  --raw-parquet /abs/path/to/demo_analysis/warmstart_runs_flattened.parquet \
  ...
```

## Worktree caveat

Git worktrees do **not** receive untracked files. If you run prediction/plot
code inside a worktree under `.claude/worktrees/`, the raw parquet will be
missing — pass `--raw-parquet` pointing at the main checkout, or run from the
main tree. This is expected, not a bug.

## Do not

- **Do not** commit the raw `.parquet` datasets (≈58 MB combined) into git — it
  bloats history and fights the existing untracked-data convention. Use a shared
  path or data store instead.
