#!/usr/bin/env python3
"""
Script 02: Run Separator

Runs vocal separation on all tracks in manifest and caches results.
Uses Music-Source-Separation-Training or torchaudio Demucs as fallback.

Usage:
    python scripts/02_run_separator.py [--config configs/trackA_config.yaml]

Input:
    metadata/trackA_manifest.csv

Output:
    cache/separated/{model_name}/{track_id}/vocals.wav
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.separation import (
    separate_track,
    get_cache_path,
    is_cached,
    get_separation_stats,
)


def load_manifest(path: Path) -> pd.DataFrame:
    """Load manifest CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path)


def check_mss_availability(mss_root: Path) -> bool:
    """Check if MSS Training is available."""
    if not mss_root.exists():
        return False

    inference_script = mss_root / "inference.py"
    if not inference_script.exists():
        return False

    return True


def check_torchaudio_demucs() -> bool:
    """Check if torchaudio Demucs is available."""
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

        return True
    except ImportError:
        return False


def run_separation_batch(
    manifest_df: pd.DataFrame,
    config,
    force_rerun: bool = False,
    max_tracks: Optional[int] = None,
    skip_existing: bool = True,
) -> dict:
    """Run separation on all tracks in manifest.

    Args:
        manifest_df: Manifest DataFrame
        config: TrackAConfig
        force_rerun: Force re-separation even if cached
        max_tracks: Maximum tracks to process (for testing)
        skip_existing: Skip tracks that are already cached

    Returns:
        Dictionary with results summary
    """
    cache_root = Path(config.paths.cache_root)
    model_name = config.separator.name

    # Check MSS availability
    mss_root = Path(config.paths.mss_training_root)

    mss_available = check_mss_availability(mss_root)
    torchaudio_available = check_torchaudio_demucs()

    print(f"\nSeparation backend status:")
    print(f"  MSS Training: {'✓ Available' if mss_available else '✗ Not available'}")
    print(
        f"  torchaudio Demucs: {'✓ Available' if torchaudio_available else '✗ Not available'}"
    )

    if not mss_available and not torchaudio_available:
        print("\nError: No separation backend available!")
        print("Please either:")
        print("  1. Set up Music-Source-Separation-Training")
        print("  2. Install torchaudio with Demucs support")
        return {"success": 0, "failed": 0, "skipped": 0, "error": "No backend"}

    # Get config paths
    config_path = None
    if hasattr(config.separator, "config_path") and config.separator.config_path:
        config_path = Path(config.separator.config_path)
    elif (
        hasattr(config.separator, "mss_config_path")
        and config.separator.mss_config_path
    ):
        config_path = Path(config.separator.mss_config_path)

    checkpoint_path = None
    if (
        hasattr(config.separator, "checkpoint_path")
        and config.separator.checkpoint_path
    ):
        checkpoint_path = Path(config.separator.checkpoint_path)

    device_ids = [0]
    if hasattr(config.separator, "device_ids"):
        device_ids = config.separator.device_ids

    # Process tracks
    tracks = manifest_df.to_dict("records")
    if max_tracks:
        tracks = tracks[:max_tracks]

    results = {"success": 0, "failed": 0, "skipped": 0, "tracks": []}

    start_time = time.time()

    for track in tqdm(tracks, desc="Separating tracks"):
        track_id = track["track_id"]
        mixture_path = Path(track["mixture_path"])

        # Check if already cached
        if skip_existing and not force_rerun:
            if is_cached(cache_root, model_name, track_id, "vocals"):
                results["skipped"] += 1
                results["tracks"].append(
                    {"track_id": track_id, "status": "skipped", "reason": "cached"}
                )
                continue

        # Run separation
        try:
            vocal_path = separate_track(
                mixture_path=mixture_path,
                cache_root=cache_root,
                track_id=track_id,
                model_name=model_name,
                mss_root=mss_root if mss_available else None,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device_ids=device_ids,
                force_rerun=force_rerun,
                use_torchaudio_fallback=torchaudio_available,
            )

            if vocal_path and vocal_path.exists():
                results["success"] += 1
                results["tracks"].append(
                    {
                        "track_id": track_id,
                        "status": "success",
                        "output_path": str(vocal_path),
                    }
                )
            else:
                results["failed"] += 1
                results["tracks"].append(
                    {
                        "track_id": track_id,
                        "status": "failed",
                        "reason": "output not found",
                    }
                )

        except Exception as e:
            results["failed"] += 1
            results["tracks"].append(
                {"track_id": track_id, "status": "failed", "reason": str(e)}
            )
            print(f"\n  Error processing {track_id}: {e}")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["tracks_per_min"] = len(tracks) / (elapsed / 60) if elapsed > 0 else 0

    return results


def print_separation_summary(results: dict, config):
    """Print summary of separation results."""
    print(f"\n{'='*50}")
    print("Separation Summary")
    print(f"{'='*50}")

    total = results["success"] + results["failed"] + results["skipped"]
    print(f"Total tracks: {total}")
    print(f"  ✓ Success: {results['success']}")
    print(f"  ✗ Failed: {results['failed']}")
    print(f"  → Skipped (cached): {results['skipped']}")

    if results.get("elapsed_sec"):
        elapsed = results["elapsed_sec"]
        print(
            f"\nTime: {elapsed/60:.1f} minutes ({results.get('tracks_per_min', 0):.1f} tracks/min)"
        )

    # Cache stats
    cache_root = Path(config.paths.cache_root)
    model_name = config.separator.name
    stats = get_separation_stats(cache_root, model_name)

    print(f"\nCache status:")
    print(f"  Tracks cached: {stats['n_tracks']}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Location: {stats.get('cache_path', 'N/A')}")

    # Failed tracks
    failed_tracks = [t for t in results.get("tracks", []) if t["status"] == "failed"]
    if failed_tracks:
        print(f"\nFailed tracks ({len(failed_tracks)}):")
        for t in failed_tracks[:5]:  # Show first 5
            print(f"  - {t['track_id']}: {t.get('reason', 'unknown')}")
        if len(failed_tracks) > 5:
            print(f"  ... and {len(failed_tracks) - 5} more")

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Run vocal separation on all tracks")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trackA_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV (default: from config)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-separation even if cached"
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Maximum tracks to process (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running separation",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Script 02: Run Separator")
    print("=" * 50)

    # Load config
    print(f"\nLoading config from {args.config}")
    config = load_config(args.config)

    # Ensure directories exist
    config.paths.ensure_dirs()

    # Load manifest
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(config.paths.metadata_root) / "trackA_manifest.csv"
    )
    print(f"Loading manifest from {manifest_path}")
    manifest_df = load_manifest(manifest_path)
    print(f"Loaded {len(manifest_df)} tracks")

    # Print config info
    print(f"\nSeparation settings:")
    print(f"  Model: {config.separator.name}")
    if hasattr(config.separator, "model_type"):
        print(f"  Model type: {config.separator.model_type}")
    print(f"  Cache: {config.paths.cache_root}/separated/{config.separator.name}/")

    if args.dry_run:
        print("\nDry run - checking what would be done:")

        cache_root = Path(config.paths.cache_root)
        model_name = config.separator.name

        cached = 0
        to_process = 0

        for _, row in manifest_df.iterrows():
            if is_cached(cache_root, model_name, row["track_id"], "vocals"):
                cached += 1
            else:
                to_process += 1

        print(f"  Already cached: {cached}")
        print(f"  To process: {to_process}")

        if args.max_tracks:
            print(f"  (Limited to {args.max_tracks} tracks)")

        return

    # Run separation
    print(f"\nRunning separation...")
    if args.max_tracks:
        print(f"  (Limited to {args.max_tracks} tracks for testing)")

    results = run_separation_batch(
        manifest_df,
        config,
        force_rerun=args.force,
        max_tracks=args.max_tracks,
        skip_existing=not args.force,
    )

    # Print summary
    print_separation_summary(results, config)

    if results["failed"] == 0:
        print("✓ Separation complete!")
    else:
        print(f"⚠ Separation complete with {results['failed']} failures")


if __name__ == "__main__":
    main()
