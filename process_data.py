import pandas as pd
import json


DATA_PATH = "./warmstarting_data.csv"
OUT_PATH = "./experiments.json"


def make_key(row):
    return (
        f"base{int(row['base_N'])}_"
        f"target{int(row['target_N'])}_"
        f"lr{row['max_lr']}_"
        f"shrink{row['shrink']}_"
        f"tkpm{row['tkpm']}_"
        f"emb{int(row['n_embd'])}_"
        f"head{int(row['n_head'])}_"
        f"gw{row['g_width']}_"
        f"gn{row['g_N']}"
    )


def convert_csv_to_json():

    df = pd.read_csv(DATA_PATH)

    # clean column names (safe)
    df.columns = df.columns.str.strip()

    print(f"Loaded: {len(df)} rows")
    print("Columns:", df.columns.tolist())

    experiments = {}

    group_cols = [
        "base_N", "target_N", "max_lr", "shrink",
        "tkpm", "n_embd", "n_head", "g_width", "g_N"
    ]

    for _, g in df.groupby(group_cols):

        g = g.sort_values("tokens")
        row0 = g.iloc[0]

        key = make_key(row0)

        experiments[key] = {
            "hyperparameters": {
                "base_N": int(row0["base_N"]),
                "target_N": int(row0["target_N"]),
                "max_lr": float(row0["max_lr"]),
                "shrink": float(row0["shrink"]),
                "tkpm": float(row0["tkpm"]),
                "n_embd": int(row0["n_embd"]),
                "n_head": int(row0["n_head"]),
                "g_width": float(row0["g_width"]),
                "g_N": float(row0["g_N"]),
            },
            "curve": {
                "tokens": g["tokens"].tolist(),
                "flops": g["flops"].tolist(),
                "train_loss": g["Train Loss"].tolist(),
                "val_loss": g["Validation Loss"].tolist(),
            }
        }

    with open(OUT_PATH, "w") as f:
        json.dump(experiments, f, indent=2)

    print(f"\nSaved → {OUT_PATH}")
    print(f"Total runs → {len(experiments)}")


if __name__ == "__main__":
    convert_csv_to_json()