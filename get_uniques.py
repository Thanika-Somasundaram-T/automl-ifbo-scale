from IPython.display import display
import json
import math
import pandas as pd

with open("all_curves.json", "r") as f:
    data = json.load(f)

print(f"Total runs: {len(data)}\n")

unique_list = {}

unique_points = {}


for run_key, run in data.items():
    hp    = run["hyperparameters"]
    curve = run["curve"]
    
    val_loss   = curve["val_loss"]
    flops = curve["flops"]
    
    unique_list[len(val_loss)] = {
        "count": unique_list.get(len(val_loss), {"count": 0})["count"] + 1,
        "tkpm": unique_list.get(len(val_loss), {"tkpm": []})["tkpm"] + [hp["tkpm"]] if len(val_loss) in unique_list else [hp["tkpm"]],
        "target_N": unique_list.get(len(val_loss), {"target_N": []})["target_N"] + [hp["target_N"]] if len(val_loss) in unique_list else [hp["target_N"]],
    }
    # for flops in curve["flops"]:
    #     unique_points[round(float(flops), 2)] = unique_points.get(round(float(flops), 2), 0) + 1
    t_flops = flops / (flops[-1] + 1e-8)

# print("Unique lengths of val_loss:", unique_list)
print("Unique FLOPs values:", list(unique_points.values()))

