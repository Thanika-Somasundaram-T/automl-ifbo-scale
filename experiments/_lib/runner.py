"""Shared experiment runner: CLI, checkpointing, policy execution loop."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from ifbo.surrogate import FTPFN

from experiments._lib.common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_VERSION,
    DEFAULT_OBS_FRACS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RESAMPLED_K,
    check_invariants,
    resolve_device,
    set_seed,
)
from experiments._lib.data import add_normalized_hps, build_curve_cache, filter_paws, load_processed
from experiments._lib.predict_one import predict_one
from experiments._lib.shards import (
    finalize_shard,
    flush_inprogress,
    inprogress_path,
    load_done_trajectory_ids,
    merge_shards,
    shard_tag,
)


def build_parser(
    description: str = "Width-transfer predictions.",
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--obs-fracs", type=float, nargs="+", default=DEFAULT_OBS_FRACS)
    parser.add_argument("--resampled-k", type=int, default=DEFAULT_RESAMPLED_K)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-k", type=int, default=None,
                        help="Fixed context size for policies that sample a random subset.")
    parser.add_argument("--max-targets-per-scale", type=int, default=None,
                        help="Limit targets per query_target_N for smoke testing.")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Flush in-progress shard + log every N trajectories (mid-shard crash safety).")
    return parser


def run_policies(policies: list[str], args: argparse.Namespace, registry: ModuleType) -> None:
    unknown = [policy for policy in policies if policy not in registry.POLICIES_ALL]
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}. Must be one of {registry.POLICIES_ALL}")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Policies: {policies}")

    summary, curves = load_processed(args.processed_dir, args.resampled_k)
    filtered = filter_paws(summary)
    filtered, _ = add_normalized_hps(filtered)

    print(f"\nTarget scales: {sorted(filtered['target_N'].unique())}")
    print("Trajectories per scale:")
    for n, count in filtered.groupby("target_N").size().items():
        print(f"  {n:>12,}: {count}")

    cache = build_curve_cache(filtered, curves)

    if args.max_targets_per_scale is not None:
        limited_ids = []
        for _, group in filtered.groupby("target_N"):
            limited_ids.extend(group["trajectory_id"].head(args.max_targets_per_scale).tolist())
        target_ids = sorted(limited_ids)
        print(f"Smoke mode: {len(target_ids)} targets ({args.max_targets_per_scale}/scale)")
    else:
        target_ids = sorted(cache)

    device = resolve_device(args.device)
    model = FTPFN(target_path=args.model_path, version=args.model_version, device=device)

    partial_dir = results_dir / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    checkpoint_every = max(1, int(getattr(args, "checkpoint_every", 5)))
    shard_paths: list[Path] = []
    for policy in policies:
        for obs_frac in args.obs_fracs:
            tag = shard_tag(obs_frac)
            shard = partial_dir / f"{policy}__obs{tag}.parquet"
            shard_paths.append(shard)
            if shard.exists():
                print(f"[resume] skipping {policy} obs={obs_frac:.2f}")
                continue

            # Mid-shard resume: reload any trajectories already persisted before a crash.
            inprogress = inprogress_path(partial_dir, policy, tag)
            done_ids = load_done_trajectory_ids(inprogress)
            frames = [pd.read_parquet(inprogress)] if done_ids else []
            if done_ids:
                print(f"[resume] {policy} obs={obs_frac:.2f}: {len(done_ids)} trajectories already done, continuing", flush=True)

            skipped = 0
            total = len(target_ids)
            for i, target_id in enumerate(target_ids, start=1):
                if target_id in done_ids:
                    continue
                target_row = cache[target_id]["row"]
                query_target_n = float(target_row["target_N"])

                if policy in registry.LOWER_ONLY_POLICIES:
                    has_lower = any(
                        float(cache[tid]["row"]["target_N"]) < query_target_n
                        for tid in cache if tid != target_id
                    )
                    if not has_lower:
                        skipped += 1
                        continue

                partner_ids = registry.select_context_partners(
                    cache, target_id, target_row, policy, args.random_k, rng
                )

                if registry.requires_partners(policy) and not partner_ids:
                    skipped += 1
                    continue

                frames.append(predict_one(model, cache, target_id, obs_frac, policy, partner_ids, registry))

                # Progress log + crash-safe intra-shard checkpoint.
                if i % checkpoint_every == 0:
                    flush_inprogress(frames, inprogress)
                    print(f"  [{policy} obs={obs_frac:.2f}] {i}/{total} trajectories", flush=True)

            if skipped:
                print(f"  [{policy} obs={obs_frac:.2f}] skipped {skipped} targets (no eligible context)")

            if frames:
                # Final flush, then atomically promote in-progress -> final shard.
                flush_inprogress(frames, inprogress)
                finalize_shard(inprogress, shard)
                n_rows = sum(len(f) for f in frames)
                print(f"[checkpoint] {shard.name} ({n_rows} rows)")
            else:
                print(f"[warning] {policy} obs={obs_frac:.2f}: no predictions generated")

    existing_shards = [p for p in shard_paths if p.exists()]
    if not existing_shards:
        print("No predictions generated. Check policy/data compatibility.")
        return

    merge_shards(partial_dir, results_dir, "predictions.parquet")
    predictions = pd.read_parquet(results_dir / "predictions.parquet")

    print(f"\nSaved {len(predictions):,} prediction rows to {results_dir / 'predictions.parquet'}")
    print(predictions.groupby(["policy", "obs_frac"]).size().unstack(fill_value=0).to_string())

    check_invariants(predictions, registry.LOWER_ONLY_POLICIES)
    print("\nInvariant checks passed.")


def main_for_policy(
    policy_name: str,
    registry: ModuleType,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    description: str = "Width-transfer predictions.",
) -> None:
    if policy_name not in registry.POLICIES_ALL:
        raise ValueError(f"Unknown policy: {policy_name}. Must be one of {registry.POLICIES_ALL}")
    parser = build_parser(description, default_output_dir)
    args = parser.parse_args()
    run_policies([policy_name], args, registry)
