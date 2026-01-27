#!/usr/bin/env python3
"""
Script 04: Compute Difficulty Metrics

Computes SI-SDR and SDR for each segment by comparing
estimated (separated) vocals with ground truth vocals.

Usage:
    python scripts/04_compute_difficulty.py --config configs/trackA_config.yaml
    python scripts/04_compute_difficulty.py --max-tracks 5  # Test with 5 tracks
    python scripts/04_compute_difficulty.py --dry-run       # Preview only

Input:
    - metadata/trackA_manifest.csv (track info)
    - tables/segments_trackA.parquet (segment index)
    - cache/separated/htdemucs/{track_id}/vocals.wav (separated vocals)
    - Dataset/musdb18hq/.../vocals.wav (ground truth)

Output:
    - tables/difficulty_trackA.parquet (segment-level SI-SDR, SDR)
    - results/trackA/difficulty_stats.json (summary statistics)
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.metrics import DifficultyComputer, compute_difficulty_stats
from src.data_types import DifficultyResult


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load track manifest."""
    return pd.read_csv(manifest_path)


def load_segments(segments_path: Path) -> pd.DataFrame:
    """Load segment index."""
    return pd.read_parquet(segments_path)


def compute_difficulty_batch(
    manifest_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    computer: DifficultyComputer,
    max_tracks: Optional[int] = None,
    skip_silent: bool = True,
    verbose: bool = True
) -> List[DifficultyResult]:
    """Compute difficulty for all segments across all tracks.
    
    Args:
        manifest_df: Track manifest DataFrame
        segments_df: Segment index DataFrame
        computer: DifficultyComputer instance
        max_tracks: Limit number of tracks (for testing)
        skip_silent: Skip silent segments
        verbose: Print progress info
        
    Returns:
        List of DifficultyResult for all segments
    """
    all_results = []
    
    # Get unique tracks
    tracks = manifest_df.to_dict('records')
    if max_tracks:
        tracks = tracks[:max_tracks]
    
    # Process each track
    for track in tqdm(tracks, desc="Computing difficulty", disable=not verbose):
        track_id = track['track_id']
        gt_vocal_path = track['gt_vocal_path']
        
        # Get segments for this track
        track_segments = segments_df[segments_df['track_id'] == track_id]
        
        # Optionally skip silent segments
        if skip_silent and 'is_silent' in track_segments.columns:
            track_segments = track_segments[~track_segments['is_silent']]
        
        if len(track_segments) == 0:
            continue
        
        # Convert to list of dicts
        segments_list = track_segments.to_dict('records')
        
        # Compute difficulty for this track
        results, stats = computer.compute_for_track(
            track_id=track_id,
            gt_vocal_path=gt_vocal_path,
            segments=segments_list
        )
        
        all_results.extend(results)
    
    return all_results


def results_to_dataframe(results: List[DifficultyResult]) -> pd.DataFrame:
    """Convert list of DifficultyResult to DataFrame.
    
    Args:
        results: List of DifficultyResult
        
    Returns:
        DataFrame with columns: seg_id, sisdr_vocal, sdr_vocal, valid, error_msg
    """
    records = [r.to_dict() for r in results]
    return pd.DataFrame(records)


def print_difficulty_summary(df: pd.DataFrame, stats: Dict):
    """Print summary of difficulty computation.
    
    Args:
        df: Difficulty DataFrame
        stats: Statistics dictionary
    """
    print("=" * 50)
    print("Difficulty Computation Summary")
    print("=" * 50)
    
    print(f"\nTotal segments: {len(df)}")
    print(f"  Valid: {stats['n_valid']} ({stats['n_valid']/len(df)*100:.1f}%)")
    print(f"  Invalid: {len(df) - stats['n_valid']} ({(len(df)-stats['n_valid'])/len(df)*100:.1f}%)")
    
    if stats['sisdr']['mean'] is not None:
        print(f"\nSI-SDR (dB) - Primary Metric:")
        print(f"  Mean: {stats['sisdr']['mean']:.2f}")
        print(f"  Std:  {stats['sisdr']['std']:.2f}")
        print(f"  Range: [{stats['sisdr']['min']:.2f}, {stats['sisdr']['max']:.2f}]")
        print(f"  Median: {stats['sisdr']['median']:.2f}")
        print(f"  IQR: [{stats['sisdr']['p25']:.2f}, {stats['sisdr']['p75']:.2f}]")
        
        print(f"\nSDR (dB) - Reference Metric:")
        print(f"  Mean: {stats['sdr']['mean']:.2f}")
        print(f"  Std:  {stats['sdr']['std']:.2f}")
        print(f"  Median: {stats['sdr']['median']:.2f}")
    else:
        print("\nNo valid results to compute statistics.")
    
    print("=" * 50)


def save_difficulty_results(
    df: pd.DataFrame,
    output_path: Path,
    stats: Dict,
    stats_path: Path
):
    """Save difficulty results to parquet and stats to JSON.
    
    Args:
        df: Difficulty DataFrame
        output_path: Path for parquet file
        stats: Statistics dictionary
        stats_path: Path for JSON stats file
    """
    # Save parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved difficulty results to {output_path}")
    
    # Save stats
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute difficulty metrics (SI-SDR, SDR) for each segment"
    )
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
        "--segments",
        type=str,
        default=None,
        help="Path to segments parquet (default: from config)"
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit number of tracks (for testing)"
    )
    parser.add_argument(
        "--include-silent",
        action="store_true",
        help="Include silent segments (default: skip)"
    )
    parser.add_argument(
        "--no-sdr",
        action="store_true",
        help="Skip SDR computation (only compute SI-SDR)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without computing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files"
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("Script 04: Compute Difficulty Metrics")
    print("=" * 50)
    
    # Load config
    print(f"\nLoading config from {args.config}")
    config = load_config(args.config)
    
    # Determine paths
    manifest_path = Path(args.manifest) if args.manifest else Path(config.paths.metadata_root) / "trackA_manifest.csv"
    segments_path = Path(args.segments) if args.segments else Path(config.paths.tables_root) / "segments_trackA.parquet"
    output_path = Path(config.paths.tables_root) / "difficulty_trackA.parquet"
    stats_path = Path(config.paths.results_root) / "trackA" / "difficulty_stats.json"
    
    # Check if output exists
    if output_path.exists() and not args.force and not args.dry_run:
        print(f"\nOutput file already exists: {output_path}")
        print("Use --force to overwrite")
        return
    
    # Load data
    print(f"\nLoading manifest from {manifest_path}")
    manifest_df = load_manifest(manifest_path)
    print(f"Loaded {len(manifest_df)} tracks")
    
    print(f"Loading segments from {segments_path}")
    segments_df = load_segments(segments_path)
    print(f"Loaded {len(segments_df)} segments")
    
    # Print settings
    print(f"\nDifficulty settings:")
    print(f"  Separator model: {config.separator.name}")
    print(f"  Cache root: {config.paths.cache_root}")
    print(f"  Compute SDR: {not args.no_sdr}")
    print(f"  Skip silent: {not args.include_silent}")
    print(f"  Alignment: {config.separator.output_alignment}")
    
    if args.max_tracks:
        print(f"  Limited to {args.max_tracks} tracks (testing)")
    
    # Dry run
    if args.dry_run:
        print("\n[DRY RUN] Would compute difficulty for:")
        
        n_tracks = min(args.max_tracks, len(manifest_df)) if args.max_tracks else len(manifest_df)
        
        # Count segments per track
        track_ids = manifest_df['track_id'].iloc[:n_tracks].tolist()
        track_segments = segments_df[segments_df['track_id'].isin(track_ids)]
        
        if not args.include_silent and 'is_silent' in track_segments.columns:
            track_segments = track_segments[~track_segments['is_silent']]
        
        print(f"  Tracks: {n_tracks}")
        print(f"  Segments: {len(track_segments)}")
        
        # Check cache status
        cache_root = Path(config.paths.cache_root)
        cached_count = 0
        for track_id in track_ids:
            vocal_path = cache_root / "separated" / config.separator.name / track_id / "vocals.wav"
            if vocal_path.exists():
                cached_count += 1
        
        print(f"  Cached vocals: {cached_count}/{n_tracks}")
        
        if cached_count < n_tracks:
            print(f"\n  ⚠ Warning: {n_tracks - cached_count} tracks missing separated vocals")
            print("  Run 02_run_separator.py first")
        
        print(f"\nOutput would be saved to:")
        print(f"  {output_path}")
        print(f"  {stats_path}")
        return
    
    # Create computer
    computer = DifficultyComputer(
        cache_root=Path(config.paths.cache_root),
        model_name=config.separator.name,
        sr=config.audio.sample_rate,
        compute_sdr=not args.no_sdr,
        alignment_policy=config.separator.output_alignment
    )
    
    # Compute difficulty
    print(f"\nComputing difficulty metrics...")
    start_time = time.time()
    
    results = compute_difficulty_batch(
        manifest_df=manifest_df,
        segments_df=segments_df,
        computer=computer,
        max_tracks=args.max_tracks,
        skip_silent=not args.include_silent,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # Convert to DataFrame
    difficulty_df = results_to_dataframe(results)
    
    # Compute statistics
    stats = compute_difficulty_stats(results)
    stats['computation_time_sec'] = elapsed
    stats['config'] = {
        'separator_model': config.separator.name,
        'sample_rate': config.audio.sample_rate,
        'alignment_policy': config.separator.output_alignment,
        'compute_sdr': not args.no_sdr,
        'skip_silent': not args.include_silent
    }
    
    # Print summary
    print_difficulty_summary(difficulty_df, stats)
    print(f"\nComputation time: {elapsed:.1f} seconds ({len(results)/elapsed:.1f} segments/sec)")
    
    # Save results
    save_difficulty_results(difficulty_df, output_path, stats, stats_path)
    
    # Show sample results
    print(f"\nSample results (first 10 valid):")
    valid_df = difficulty_df[difficulty_df['valid']].head(10)
    for _, row in valid_df.iterrows():
        print(f"  {row['seg_id']}: SI-SDR={row['sisdr_vocal']:.2f} dB, SDR={row['sdr_vocal']:.2f} dB")
    
    print(f"\n✓ Difficulty computation complete!")


if __name__ == "__main__":
    main()
