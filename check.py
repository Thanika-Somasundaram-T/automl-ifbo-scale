import os
import json

# ============================================
# CONFIG
# ============================================
RESULTS_DIR = "./results"
PREFIX = "ifbo_pred_"
MIN_KEYS = 12  # Minimum expected keys per file

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
less_than_min = []

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
        elif key_count < MIN_KEYS:
            print(f"⚠️  LESS THAN {MIN_KEYS} KEYS ({key_count} keys): {filename}")
            less_than_min.append((filename, key_count))
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
print(f"Files < {MIN_KEYS} keys   : {len(less_than_min)}")
print(f"Corrupt files          : {len(corrupt)}")
print("==============================\n")

if less_than_min:
    print("📌 Files with less than 12 keys:")
    for fname, count in less_than_min:
        print(f"  - {fname} → {count} keys")