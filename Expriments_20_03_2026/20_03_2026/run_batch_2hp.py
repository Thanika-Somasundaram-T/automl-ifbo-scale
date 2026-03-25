"""
Batch runner for 2-HP combination predictions.

For each target HP config:
  - Anchor = same (lr, wd) from a lower scale
  - Partner = each of the other 11 (lr, wd) combos from the same lower scale
  → 11 pairs per target × 12 targets × N epochs × M scales

Usage:
    python run_batch_2hp.py
    python run_batch_2hp.py --scale 64 --epochs 0 5 10 20 50 90
"""
import os
import sys
import time
import argparse
import itertools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict_2hp import predict_2hp

# ─────────────────────────────────────────────
# All 12 HP configs (lr, wd)
# ─────────────────────────────────────────────
HP_CONFIGS = [
    (1e-5, 0.0),   (1e-5, 0.01),
    (3e-5, 0.0),   (3e-5, 0.01),
    (1e-4, 0.0),   (1e-4, 0.01),
    (3e-4, 0.0),   (3e-4, 0.01),
    (1e-3, 0.0),   (1e-3, 0.01),
    (3e-3, 0.0),   (3e-3, 0.01),
]

DEFAULT_EPOCHS = [0, 5, 10, 20, 50, 90]
DEFAULT_SCALES = [64]  # Start with largest scale; extend to [8, 16, 24, 32, 64]

SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "predictions")


def run_all(scales, epochs_list):
    # Build all (target_hp, partner_hp) pairs where target != partner
    pairs = []
    for target_hp in HP_CONFIGS:
        for partner_hp in HP_CONFIGS:
            if target_hp != partner_hp:
                pairs.append((target_hp, partner_hp))

    combos = list(itertools.product(scales, epochs_list, pairs))
    total = len(combos)

    print(f"\n{'='*60}")
    print(f"2-HP Combination Batch Runner")
    print(f"{'='*60}")
    print(f"Scales:    {scales}")
    print(f"Epochs:    {epochs_list}")
    print(f"HP pairs:  {len(pairs)} (12 targets × 11 partners)")
    print(f"Total runs: {total}")
    print(f"Save dir:  {SAVE_DIR}")
    print(f"{'='*60}\n")

    success = 0
    errors = 0

    for i, (scale, epoch, (target_hp, partner_hp)) in enumerate(combos, 1):
        target_lr, target_wd = target_hp
        partner_lr, partner_wd = partner_hp

        print(f"\n[{i}/{total}] Scale={scale}, T={epoch} | "
              f"target=(lr={target_lr},wd={target_wd}) + partner=(lr={partner_lr},wd={partner_wd})")

        t0 = time.time()
        try:
            result = predict_2hp(
                target_lr=target_lr,
                target_wd=target_wd,
                partner_lr=partner_lr,
                partner_wd=partner_wd,
                context_scale=scale,
                epochs=epoch,
                save_dir=SAVE_DIR,
            )
            elapsed = time.time() - t0
            print(f"   ✅ {elapsed:.1f}s (pred={result:.4f})")
            success += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"   ❌ {elapsed:.1f}s: {e}")
            errors += 1

    print(f"\n{'='*60}")
    print(f"Complete: {success} succeeded, {errors} failed out of {total}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-HP combination batch predictions")
    parser.add_argument("--scale", type=int, nargs="+", default=DEFAULT_SCALES,
                        help="Context scales to use (default: [64])")
    parser.add_argument("--epochs", type=int, nargs="+", default=DEFAULT_EPOCHS,
                        help="Observation epochs (default: [0,5,10,20,50,90])")
    args = parser.parse_args()

    run_all(args.scale, args.epochs)
