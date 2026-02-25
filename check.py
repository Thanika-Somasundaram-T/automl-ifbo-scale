import os
import json

# ============================================
# CONFIG
# ============================================
RESULTS_DIR = "./results"
PREFIX = "ifbo_pred_"

# ============================================
# FIND ALL PREDICTION FILES
# ============================================
all_files = os.listdir(RESULTS_DIR)
pred_files = [
    f for f in all_files
    if f.startswith(PREFIX) and f.endswith(".json")
]

pred_files = sorted(pred_files)

print(f"\n🔍 Found {len(pred_files)} prediction files.\n")

if len(pred_files) == 0:
    print("❌ No prediction files found.")
    exit()

# ============================================
# CHECK FILES
# ============================================
missing = []
empty = []
corrupt = []

for filename in pred_files:
    path = os.path.join(RESULTS_DIR, filename)

    try:
        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"⚠️  NOT A DICT: {filename}")
            continue

        key_count = len(data)

        if key_count == 0:
            print(f"🟡 EMPTY (0 keys): {filename}")
            empty.append(filename)
        else:
            print(f"✅ {filename} → {key_count} keys")

    except Exception as e:
        print(f"💥 CORRUPT: {filename} ({e})")
        corrupt.append(filename)

# ============================================
# SUMMARY
# ============================================
print("\n==============================")
print(f"Total prediction files : {len(pred_files)}")
print(f"Empty files            : {len(empty)}")
print(f"Corrupt files          : {len(corrupt)}")
print("==============================\n")