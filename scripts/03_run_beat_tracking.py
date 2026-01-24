#!/usr/bin/env python3
"""
Script 03: Run Beat Tracking

Runs beat tracking on all tracks in manifest and caches results.
Uses librosa as the backend (essentia/madmom not available on Windows).

Usage:
    python scripts/03_run_beat_tracking.py [--config configs/trackA_config.yaml]
    
Input:
    metadata/trackA_manifest.csv
    
Output:
    cache/beat_tracking/{backend}/{track_id}/beats.json
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
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.beat_tracking import (
    run_beat_tracking,
    is_beat_cached,
    get_beat_tracking_stats,
    load_beat_result,
    get_beat_cache_path
)


def load_manifest(path: Path) -> pd.DataFrame:
    """Load manifest CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path)


def run_beat_tracking_batch(
    manifest_df: pd.DataFrame,
    config,
    force_rerun: bool = False,
    max_tracks: Optional[int] = None,
    skip_existing: bool = True
) -> dict:
    """Run beat tracking on all tracks in manifest.
    
    Args:
        manifest_df: Manifest DataFrame
        config: TrackAConfig
        force_rerun: Force re-computation even if cached
        max_tracks: Maximum tracks to process (for testing)
        skip_existing: Skip tracks that are already cached
        
    Returns:
        Dictionary with results summary
    """
    cache_root = Path(config.paths.cache_root)
    backend = config.beat_tracking.backend
    sr = config.audio.sample_rate
    min_beats = config.beat_tracking.min_beats_per_track
    
    print(f"\nBeat tracking settings:")
    print(f"  Backend: {backend}")
    print(f"  Min beats per track: {min_beats}")
    print(f"  Sample rate: {sr}")
    
    # Process tracks
    tracks = manifest_df.to_dict('records')
    if max_tracks:
        tracks = tracks[:max_tracks]
    
    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "available": 0,
        "unavailable": 0,
        "tracks": []
    }
    
    start_time = time.time()
    
    for track in tqdm(tracks, desc="Beat tracking"):
        track_id = track['track_id']
        mixture_path = Path(track['mixture_path'])
        
        # Check if already cached
        if skip_existing and not force_rerun:
            if is_beat_cached(cache_root, backend, track_id):
                # Load to check availability
                cache_path = get_beat_cache_path(cache_root, backend, track_id)
                cached_result = load_beat_result(cache_path)
                
                results["skipped"] += 1
                if cached_result and cached_result.track_available:
                    results["available"] += 1
                else:
                    results["unavailable"] += 1
                
                results["tracks"].append({
                    "track_id": track_id,
                    "status": "skipped",
                    "reason": "cached",
                    "available": cached_result.track_available if cached_result else None
                })
                continue
        
        # Run beat tracking
        try:
            result = run_beat_tracking(
                mixture_path=mixture_path,
                track_id=track_id,
                cache_root=cache_root,
                backend=backend,
                sr=sr,
                min_beats_per_track=min_beats,
                force_rerun=force_rerun
            )
            
            if result:
                results["success"] += 1
                
                if result.track_available:
                    results["available"] += 1
                else:
                    results["unavailable"] += 1
                
                results["tracks"].append({
                    "track_id": track_id,
                    "status": "success",
                    "n_beats": result.n_beats,
                    "bpm": result.bpm,
                    "available": result.track_available,
                    "unavailable_reason": result.unavailable_reason
                })
            else:
                results["failed"] += 1
                results["tracks"].append({
                    "track_id": track_id,
                    "status": "failed",
                    "reason": "no result"
                })
                
        except Exception as e:
            results["failed"] += 1
            results["tracks"].append({
                "track_id": track_id,
                "status": "failed",
                "reason": str(e)
            })
            print(f"\n  Error processing {track_id}: {e}")
    
    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["tracks_per_min"] = len(tracks) / (elapsed / 60) if elapsed > 0 else 0
    
    return results


def print_beat_tracking_summary(results: dict, config):
    """Print summary of beat tracking results."""
    print(f"\n{'='*50}")
    print("Beat Tracking Summary")
    print(f"{'='*50}")
    
    total = results["success"] + results["failed"] + results["skipped"]
    print(f"Total tracks: {total}")
    print(f"  ✓ Success: {results['success']}")
    print(f"  ✗ Failed: {results['failed']}")
    print(f"  → Skipped (cached): {results['skipped']}")
    
    print(f"\nAvailability:")
    total_processed = results["available"] + results["unavailable"]
    if total_processed > 0:
        avail_pct = results["available"] / total_processed * 100
        print(f"  Available: {results['available']} ({avail_pct:.1f}%)")
        print(f"  Unavailable: {results['unavailable']} ({100-avail_pct:.1f}%)")
    
    if results.get("elapsed_sec"):
        elapsed = results["elapsed_sec"]
        print(f"\nTime: {elapsed:.1f} seconds ({results.get('tracks_per_min', 0):.1f} tracks/min)")
    
    # Cache stats
    cache_root = Path(config.paths.cache_root)
    backend = config.beat_tracking.backend
    stats = get_beat_tracking_stats(cache_root, backend)
    
    print(f"\nCache status:")
    print(f"  Tracks cached: {stats['n_tracks']}")
    print(f"  Coverage ratio: {stats['coverage_ratio']*100:.1f}%")
    print(f"  Location: {stats.get('cache_path', 'N/A')}")
    
    # Unavailable reasons
    if stats.get('unavailable_reasons'):
        print(f"\nUnavailable reasons:")
        for reason, count in stats['unavailable_reasons'].items():
            print(f"  - {reason}: {count}")
    
    # Sample results
    print(f"\nSample results (first 5):")
    for track_info in results.get("tracks", [])[:5]:
        track_id = track_info["track_id"]
        status = track_info["status"]
        
        if status == "success":
            n_beats = track_info.get("n_beats", "?")
            bpm = track_info.get("bpm", 0)
            avail = "✓" if track_info.get("available") else "✗"
            print(f"  {track_id}: {n_beats} beats, {bpm:.1f} BPM [{avail}]")
        elif status == "skipped":
            avail = "✓" if track_info.get("available") else "✗"
            print(f"  {track_id}: (cached) [{avail}]")
        else:
            reason = track_info.get("reason", "unknown")
            print(f"  {track_id}: FAILED - {reason}")
    
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Run beat tracking on all tracks")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trackA_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV (default: from config)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-computation even if cached"
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Maximum tracks to process (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running"
    )
    args = parser.parse_args()
    
    print("="*50)
    print("Script 03: Run Beat Tracking")
    print("="*50)
    
    # Load config
    print(f"\nLoading config from {args.config}")
    config = load_config(args.config)
    
    # Ensure directories exist
    config.paths.ensure_dirs()
    
    # Check backend
    backend = config.beat_tracking.backend
    if backend in ["essentia", "madmom"]:
        print(f"\n⚠ Warning: {backend} is not available on Windows.")
        print("  Falling back to librosa.")
        # Note: The actual fallback happens in tracker.py
    
    # Load manifest
    manifest_path = Path(args.manifest) if args.manifest else Path(config.paths.metadata_root) / "trackA_manifest.csv"
    print(f"Loading manifest from {manifest_path}")
    manifest_df = load_manifest(manifest_path)
    print(f"Loaded {len(manifest_df)} tracks")
    
    if args.dry_run:
        print("\nDry run - checking what would be done:")
        
        cache_root = Path(config.paths.cache_root)
        
        cached = 0
        to_process = 0
        
        for _, row in manifest_df.iterrows():
            if is_beat_cached(cache_root, backend, row['track_id']):
                cached += 1
            else:
                to_process += 1
        
        print(f"  Already cached: {cached}")
        print(f"  To process: {to_process}")
        
        if args.max_tracks:
            print(f"  (Limited to {args.max_tracks} tracks)")
        
        return
    
    # Run beat tracking
    print(f"\nRunning beat tracking...")
    if args.max_tracks:
        print(f"  (Limited to {args.max_tracks} tracks for testing)")
    
    results = run_beat_tracking_batch(
        manifest_df,
        config,
        force_rerun=args.force,
        max_tracks=args.max_tracks,
        skip_existing=not args.force
    )
    
    # Print summary
    print_beat_tracking_summary(results, config)
    
    if results["failed"] == 0:
        print("✓ Beat tracking complete!")
    else:
        print(f"⚠ Beat tracking complete with {results['failed']} failures")


if __name__ == "__main__":
    main()
