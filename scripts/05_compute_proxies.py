#!/usr/bin/env python3
"""
Script 05: Compute Artifact Proxies

Computes artifact proxy measures from separated vocals (WITHOUT GT).
These proxies are validated against SI-SDR in Track A, then used
in Track B to estimate difficulty without ground truth.

Proxies:
    - leakage_lowfreq: Low-frequency energy ratio (drum/bass leakage)
    - instability: Temporal spectral flux (separation artifacts)
    - transient_leakage: Percussive energy ratio (drum leakage)

Usage:
    python scripts/05_compute_proxies.py --config configs/trackA_config.yaml
    python scripts/05_compute_proxies.py --max-tracks 5  # Test
    python scripts/05_compute_proxies.py --dry-run

Input:
    - metadata/trackA_manifest.csv
    - tables/segments_trackA.parquet
    - cache/separated/htdemucs/{track_id}/vocals.wav

Output:
    - tables/proxies_trackA.parquet
    - results/trackA/proxy_stats.json
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.metrics import ProxyComputer, compute_proxy_stats
from src.data_types import ProxyResult


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load track manifest."""
    return pd.read_csv(manifest_path)


def load_segments(segments_path: Path) -> pd.DataFrame:
    """Load segment index."""
    return pd.read_parquet(segments_path)


def compute_proxies_batch(
    manifest_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    computer: ProxyComputer,
    max_tracks: Optional[int] = None,
    skip_silent: bool = True,
    verbose: bool = True
) -> List[ProxyResult]:
    """Compute proxies for all segments across all tracks.
    
    Args:
        manifest_df: Track manifest DataFrame
        segments_df: Segment index DataFrame
        computer: ProxyComputer instance
        max_tracks: Limit number of tracks (for testing)
        skip_silent: Skip silent segments
        verbose: Print progress
        
    Returns:
        List of ProxyResult for all segments
    """
    all_results = []
    
    # Get unique tracks
    tracks = manifest_df.to_dict('records')
    if max_tracks:
        tracks = tracks[:max_tracks]
    
    for track in tqdm(tracks, desc="Computing proxies", disable=not verbose):
        track_id = track['track_id']
        
        # Get segments for this track
        track_segments = segments_df[segments_df['track_id'] == track_id]
        
        # Skip silent segments
        if skip_silent and 'is_silent' in track_segments.columns:
            track_segments = track_segments[~track_segments['is_silent']]
        
        if len(track_segments) == 0:
            continue
        
        segments_list = track_segments.to_dict('records')
        
        # Compute proxies
        results, stats = computer.compute_for_track(
            track_id=track_id,
            segments=segments_list
        )
        
        all_results.extend(results)
    
    return all_results


def results_to_dataframe(results: List[ProxyResult]) -> pd.DataFrame:
    """Convert list of ProxyResult to DataFrame."""
    records = [r.to_dict() for r in results]
    return pd.DataFrame(records)


def print_proxy_summary(df: pd.DataFrame, stats: Dict):
    """Print summary of proxy computation."""
    print("=" * 50)
    print("Proxy Computation Summary")
    print("=" * 50)
    
    print(f"\nTotal segments: {len(df)}")
    print(f"  Valid: {stats['n_valid']} ({stats['n_valid']/len(df)*100:.1f}%)")
    print(f"  Invalid: {len(df) - stats['n_valid']}")
    
    if stats['leakage_lowfreq']['mean'] is not None:
        print(f"\nLow-Frequency Leakage (50-150 Hz ratio):")
        print(f"  Mean: {stats['leakage_lowfreq']['mean']:.4f}")
        print(f"  Std:  {stats['leakage_lowfreq']['std']:.4f}")
        print(f"  Range: [{stats['leakage_lowfreq']['min']:.4f}, {stats['leakage_lowfreq']['max']:.4f}]")
        
        print(f"\nTemporal Instability (spectral flux):")
        print(f"  Mean: {stats['instability']['mean']:.4f}")
        print(f"  Std:  {stats['instability']['std']:.4f}")
        print(f"  Range: [{stats['instability']['min']:.4f}, {stats['instability']['max']:.4f}]")
        
        print(f"\nTransient Leakage (percussive ratio):")
        print(f"  Mean: {stats['transient_leakage']['mean']:.4f}")
        print(f"  Std:  {stats['transient_leakage']['std']:.4f}")
        print(f"  Range: [{stats['transient_leakage']['min']:.4f}, {stats['transient_leakage']['max']:.4f}]")
    
    print("=" * 50)


def save_proxy_results(
    df: pd.DataFrame,
    output_path: Path,
    stats: Dict,
    stats_path: Path
):
    """Save proxy results to parquet and stats to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved proxy results to {output_path}")
    
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute artifact proxy measures from separated vocals"
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
        help="Path to manifest CSV"
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=None,
        help="Path to segments parquet"
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
        help="Include silent segments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("Script 05: Compute Artifact Proxies")
    print("=" * 50)
    
    # Load config
    print(f"\nLoading config from {args.config}")
    config = load_config(args.config)
    
    # Paths
    manifest_path = Path(args.manifest) if args.manifest else Path(config.paths.metadata_root) / "trackA_manifest.csv"
    segments_path = Path(args.segments) if args.segments else Path(config.paths.tables_root) / "segments_trackA.parquet"
    output_path = Path(config.paths.tables_root) / "proxies_trackA.parquet"
    stats_path = Path(config.paths.results_root) / "trackA" / "proxy_stats.json"
    
    # Check existing
    if output_path.exists() and not args.force and not args.dry_run:
        print(f"\nOutput exists: {output_path}")
        print("Use --force to overwrite")
        return
    
    # Load data
    print(f"\nLoading manifest from {manifest_path}")
    manifest_df = load_manifest(manifest_path)
    print(f"Loaded {len(manifest_df)} tracks")
    
    print(f"Loading segments from {segments_path}")
    segments_df = load_segments(segments_path)
    print(f"Loaded {len(segments_df)} segments")
    
    # Settings
    print(f"\nProxy settings:")
    print(f"  Separator model: {config.separator.name}")
    print(f"  Cache root: {config.paths.cache_root}")
    print(f"  Low-freq range: 50-150 Hz")
    print(f"  Skip silent: {not args.include_silent}")
    
    if args.max_tracks:
        print(f"  Limited to {args.max_tracks} tracks")
    
    # Dry run
    if args.dry_run:
        print("\n[DRY RUN] Would compute proxies for:")
        
        n_tracks = min(args.max_tracks, len(manifest_df)) if args.max_tracks else len(manifest_df)
        track_ids = manifest_df['track_id'].iloc[:n_tracks].tolist()
        track_segments = segments_df[segments_df['track_id'].isin(track_ids)]
        
        if not args.include_silent and 'is_silent' in track_segments.columns:
            track_segments = track_segments[~track_segments['is_silent']]
        
        print(f"  Tracks: {n_tracks}")
        print(f"  Segments: {len(track_segments)}")
        
        # Check cache
        cache_root = Path(config.paths.cache_root)
        cached = sum(1 for tid in track_ids 
                     if (cache_root / "separated" / config.separator.name / tid / "vocals.wav").exists())
        print(f"  Cached vocals: {cached}/{n_tracks}")
        
        print(f"\nOutput:")
        print(f"  {output_path}")
        print(f"  {stats_path}")
        return
    
    # Create computer
    computer = ProxyComputer(
        cache_root=Path(config.paths.cache_root),
        model_name=config.separator.name,
        sr=config.audio.sample_rate,
        low_freq_range=(50.0, 150.0)
    )
    
    # Compute
    print(f"\nComputing proxies...")
    start_time = time.time()
    
    results = compute_proxies_batch(
        manifest_df=manifest_df,
        segments_df=segments_df,
        computer=computer,
        max_tracks=args.max_tracks,
        skip_silent=not args.include_silent,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # Convert to DataFrame
    proxy_df = results_to_dataframe(results)
    
    # Compute stats
    stats = compute_proxy_stats(results)
    stats['computation_time_sec'] = elapsed
    stats['config'] = {
        'separator_model': config.separator.name,
        'sample_rate': config.audio.sample_rate,
        'low_freq_range': [50.0, 150.0],
        'skip_silent': not args.include_silent
    }
    
    # Summary
    print_proxy_summary(proxy_df, stats)
    print(f"\nTime: {elapsed:.1f}s ({len(results)/elapsed:.1f} segments/sec)")
    
    # Save
    save_proxy_results(proxy_df, output_path, stats, stats_path)
    
    # Sample
    print(f"\nSample results (first 10 valid):")
    valid_df = proxy_df[proxy_df['valid']].head(10)
    for _, row in valid_df.iterrows():
        print(f"  {row['seg_id']}: lowfreq={row['leakage_lowfreq']:.4f}, "
              f"instab={row['instability']:.2f}, trans={row['transient_leakage']:.4f}")
    
    print(f"\n✓ Proxy computation complete!")


if __name__ == "__main__":
    main()
