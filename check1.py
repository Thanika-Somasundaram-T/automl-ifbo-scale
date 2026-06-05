from IPython.display import display
import json
import math
import pandas as pd

with open("experiments.json", "r") as f:
    data = json.load(f)

# print(f"Total runs: {len(data)}\n")

# issues = []
# summary = []

# for run_key, run in data.items():
#     hp    = run["hyperparameters"]
#     curve = run["curve"]
    
#     tokens     = curve["tokens"]
#     flops      = curve["flops"]
#     train_loss = curve["train_loss"]
#     val_loss   = curve["val_loss"]
    
#     run_issues = []
    
#     # length consistency
#     lengths = {"tokens": len(tokens), "flops": len(flops), 
#                 "train_loss": len(train_loss), "val_loss": len(val_loss)}
#     if len(set(lengths.values())) > 1:
#         run_issues.append(f"length mismatch: {lengths}")
    
#     # monotonically increasing tokens
#     if not all(t2 > t1 for t1, t2 in zip(tokens, tokens[1:])):
#         run_issues.append("tokens not monotonically increasing")
    
#     # monotonically increasing flops
#     if not all(f2 > f1 for f1, f2 in zip(flops, flops[1:])):
#         run_issues.append("flops not monotonically increasing")
    
#     # NaNs
#     for field, values in curve.items():
#         nan_count = sum(1 for v in values if isinstance(v, float) and math.isnan(v))
#         if nan_count > 0:
#             run_issues.append(f"{nan_count} NaNs in {field}")
    
#     # val loss actually decreasing overall (first vs last)
#     if val_loss[-1] >= val_loss[0]:
#         run_issues.append(f"val loss did not decrease: {val_loss[0]:.3f} → {val_loss[-1]:.3f}")
    
#     # reasonable loss range
#     if val_loss[-1] > 10 or val_loss[-1] < 0.5:
#         run_issues.append(f"suspicious final val loss: {val_loss[-1]:.4f}")
    
#     if run_issues:
#         issues.append((run_key, run_issues))
    
#     summary.append({
#         "run":           run_key,
#         "base_N":        hp["base_N"],
#         "target_N":      hp["target_N"],
#         "tkpm":          hp["tkpm"],
#         "shrink":        hp.get("shrink", "?"),
#         "max_lr":        hp.get("max_lr", "?"),
#         "n_embd":        hp.get("n_embd", "?"),
#         "n_checkpoints": len(val_loss),
#         "first_val":     round(val_loss[0],  4),
#         "final_val":     round(val_loss[-1], 4),
#         "total_tokens":  tokens[-1],
#         "total_flops":   flops[-1],
#         "issues":        len(run_issues)
#     })

# summary_df = pd.DataFrame(summary)

# print("── Curve lengths ───────────────────────────────────────")
# print(summary_df["n_checkpoints"].describe().to_string())

# print("\n── Final val loss by (base_N, target_N, tkpm) ──────────")
# display(
#     summary_df.pivot_table(
#         index=["base_N", "target_N"],
#         columns="tkpm",
#         values="final_val",
#         aggfunc="first"
#     ).round(4)
# )

# print("\n── All runs overview ────────────────────────────────────")
# display(
#     summary_df[[
#         "base_N", "target_N", "tkpm", "shrink", "max_lr",
#         "n_embd", "n_checkpoints", "first_val", "final_val", "issues"
#     ]].sort_values(["base_N", "target_N", "tkpm"]).reset_index(drop=True)
# )

# print("\n── Issues found ─────────────────────────────────────────")
# if issues:
#     for run_key, run_issues in issues:
#         print(f"\n  {run_key}")
#         for issue in run_issues:
#             print(f"    ✗ {issue}")
# else:
#     print("  All 54 runs passed ✓")
    
    
# # parse all hyperparameters from the run keys
# import re

# def parse_key(key):
#     parts = {}
#     # extract all numeric values from key
#     patterns = {
#         "base_N":  r"base(\d+)",
#         "target_N": r"target(\d+)",
#         "lr":      r"lr([\d.]+)",
#         "shrink":  r"shrink([\d.]+)",
#         "tkpm":    r"tkpm([\d.]+)",
#         "emb":     r"emb(\d+)",
#         "head":    r"head(\d+)",
#         "gw":      r"gw([\d.]+)",
#         "gn":      r"gn([\d.]+)",
#     }
#     for name, pattern in patterns.items():
#         match = re.search(pattern, key)
#         if match:
#             parts[name] = float(match.group(1))
#     return parts

# # build dataframe from keys
# key_df = pd.DataFrame([
#     {"run_key": k, **parse_key(k)} 
#     for k in data.keys()
# ])

# print("Unique values per key-parsed hyperparameter:")
# for col in ["base_N", "target_N", "lr", "shrink", "tkpm", "emb", "head", "gw", "gn"]:
#     vals = sorted(key_df[col].unique())
#     print(f"  {col:10s}: {vals}")

# print("\n── head unique values ──")
# print(key_df.groupby("target_N")["head"].unique())

# print("\n── gw (growth width) unique values ──")
# print(key_df.groupby(["base_N","target_N"])["gw"].unique())

# print("\n── gn (growth N) unique values ──")
# print(key_df.groupby(["base_N","target_N"])["gn"].unique())

# print("\n── are gw and gn fixed per (base_N, target_N) or do they vary? ──")
# varies = (
#     key_df.groupby(["base_N","target_N","shrink"])[["gw","gn"]]
#     .nunique()
# )
# print(varies)


import numpy as np
import json

def check_flops_sorted(path):
    with open(path, "r") as f:
        data = json.load(f)

    bad_runs = []

    for run_key, run_data in data.items():
        flops = np.array(run_data["curve"]["flops"], dtype=np.float64)
        tokens = np.array(run_data["curve"]["tokens"], dtype=np.float64)

        print(f"{run_key}: {np.allclose(flops / flops[-1], tokens / tokens[-1])}")
        

        if len(flops) < 2:
            continue

        if not np.all(np.diff(flops) >= 0):
            bad_runs.append(run_key)

    print(f"Total runs: {len(data)}")
    print(f"Unsorted runs: {len(bad_runs)}")

    if bad_runs:
        print("\nExample bad runs:")
        print(bad_runs[:10])

    return bad_runs

check_flops_sorted("./experiments.json")