#!/usr/bin/env python3
"""
Script 00: Build Track Manifest

Creates trackA_manifest.csv from MUSDB18 dataset.
This is the first step in the Track A pipeline.

Usage:
    python scripts/00_build_manifest.py [--config configs/trackA_config.yaml]

Output:
    metadata/trackA_manifest.csv
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from typing import List, Optional

from src.config import load_config, TrackAConfig
from src.data_types import TrackInfo
from src.audio_utils import get_audio_duration


def discover_musdb18_tracks(
    musdb_root: Path, subset: str = "test", sr: int = 44100
) -> List[TrackInfo]:
    """Discover tracks in MUSDB18 dataset.

    MUSDB18 structure:
        musdb_root/
        ├── train/
        │   ├── track_name/
        │   │   ├── mixture.wav
        │   │   ├── vocals.wav
        │   │   ├── drums.wav
        │   │   ├── bass.wav
        │   │   └── other.wav
        │   └── ...
        └── test/
            └── ...

    Args:
        musdb_root: Path to MUSDB18 root
        subset: "train" or "test"
        sr: Sample rate

    Returns:
        List of TrackInfo objects
    """
    subset_dir = musdb_root / subset

    if not subset_dir.exists():
        raise FileNotFoundError(f"MUSDB18 subset not found: {subset_dir}")

    tracks = []
    track_dirs = sorted([d for d in subset_dir.iterdir() if d.is_dir()])

    print(f"Found {len(track_dirs)} tracks in {subset_dir}")

    for i, track_dir in enumerate(track_dirs):
        track_id = track_dir.name

        # Check for required files
        mixture_path = track_dir / "mixture.wav"
        vocals_path = track_dir / "vocals.wav"

        if not mixture_path.exists():
            print(f"  Warning: Missing mixture.wav in {track_dir}, skipping")
            continue
        if not vocals_path.exists():
            print(f"  Warning: Missing vocals.wav in {track_dir}, skipping")
            continue

        # Get duration
        try:
            duration_sec, duration_samples = get_audio_duration(mixture_path, sr=sr)
        except Exception as e:
            print(f"  Warning: Failed to get duration for {track_id}: {e}")
            continue

        # Collect other stem paths
        other_paths = {}
        for stem in ["drums", "bass", "other"]:
            stem_path = track_dir / f"{stem}.wav"
            if stem_path.exists():
                other_paths[stem] = str(stem_path)

        track_info = TrackInfo(
            track_id=track_id,
            mixture_path=str(mixture_path),
            gt_vocal_path=str(vocals_path),
            duration_sec=duration_sec,
            duration_samples=duration_samples,
            fold_id=0,  # Will be assigned later
            gt_other_paths=other_paths if other_paths else None,
        )
        tracks.append(track_info)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(track_dirs)} tracks")

    print(f"Successfully loaded {len(tracks)} tracks")
    return tracks


def discover_musdb18_hq_tracks(
    musdb_root: Path, subset: str = "test", sr: int = 44100
) -> List[TrackInfo]:
    """Discover tracks in MUSDB18-HQ dataset (stem format).

    MUSDB18-HQ uses .stem.mp4 files or separate wav files.
    This function handles the wav directory structure.

    Args:
        musdb_root: Path to MUSDB18-HQ root
        subset: "train" or "test"
        sr: Sample rate

    Returns:
        List of TrackInfo objects
    """
    # Try standard wav structure first
    return discover_musdb18_tracks(musdb_root, subset, sr)


def assign_fold_ids(
    tracks: List[TrackInfo], n_folds: int = 5, seed: int = 42
) -> List[TrackInfo]:
    """Assign fold IDs to tracks for cross-validation.

    Uses fixed seed for reproducibility.

    Args:
        tracks: List of tracks
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Tracks with fold_id assigned
    """
    np.random.seed(seed)
    n_tracks = len(tracks)

    # Shuffle and assign folds
    indices = np.random.permutation(n_tracks)

    for i, idx in enumerate(indices):
        tracks[idx].fold_id = i % n_folds

    return tracks


def tracks_to_dataframe(tracks: List[TrackInfo]) -> pd.DataFrame:
    """Convert list of TrackInfo to DataFrame.

    Args:
        tracks: List of TrackInfo objects

    Returns:
        DataFrame with manifest columns
    """
    records = []
    for track in tracks:
        record = {
            "track_id": track.track_id,
            "mixture_path": track.mixture_path,
            "gt_vocal_path": track.gt_vocal_path,
            "duration_sec": track.duration_sec,
            "duration_samples": track.duration_samples,
            "fold_id": track.fold_id,
        }
        # Add other stem paths if present
        if track.gt_other_paths:
            for stem, path in track.gt_other_paths.items():
                record[f"gt_{stem}_path"] = path
        records.append(record)

    return pd.DataFrame(records)


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    """Save manifest to CSV.

    Args:
        df: Manifest DataFrame
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved manifest to {path}")


def print_manifest_summary(df: pd.DataFrame):
    """Print summary of manifest.

    Args:
        df: Manifest DataFrame
    """
    print(f"\n{'='*50}")
    print("Manifest Summary")
    print(f"{'='*50}")
    print(f"Total tracks: {len(df)}")
    print(f"Total duration: {df['duration_sec'].sum() / 60:.1f} minutes")
    print(f"Average track length: {df['duration_sec'].mean():.1f}s")
    print(
        f"Duration range: [{df['duration_sec'].min():.1f}, {df['duration_sec'].max():.1f}]s"
    )

    print(f"\nFold distribution:")
    for fold_id in sorted(df["fold_id"].unique()):
        fold_tracks = len(df[df["fold_id"] == fold_id])
        fold_duration = df[df["fold_id"] == fold_id]["duration_sec"].sum() / 60
        print(f"  Fold {fold_id}: {fold_tracks} tracks ({fold_duration:.1f} min)")

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Build Track A manifest from MUSDB18")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trackA_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=["train", "test"],
        help="MUSDB18 subset to use",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for fold assignment"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be done without saving"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Script 00: Build Track Manifest")
    print("=" * 50)

    # Load config
    print(f"\nLoading config from {args.config}")
    config = load_config(args.config)

    # Check MUSDB path
    musdb_root = Path(config.paths.musdb_root)
    if not musdb_root.exists():
        print(f"\nError: MUSDB18 root not found: {musdb_root}")
        print("Please update 'paths.musdb_root' in your config file.")
        print("\nTo download MUSDB18:")
        print("  pip install musdb")
        print('  python -c "import musdb; musdb.DB(download=True)"')
        sys.exit(1)

    print(f"MUSDB18 root: {musdb_root}")
    print(f"Subset: {args.subset}")

    # Discover tracks
    print(f"\nDiscovering tracks...")
    tracks = discover_musdb18_tracks(
        musdb_root, subset=args.subset, sr=config.audio.sample_rate
    )

    if not tracks:
        print("Error: No tracks found!")
        sys.exit(1)

    # Assign fold IDs
    print(f"\nAssigning fold IDs (n_folds={args.n_folds}, seed={args.seed})...")
    tracks = assign_fold_ids(tracks, n_folds=args.n_folds, seed=args.seed)

    # Convert to DataFrame
    manifest_df = tracks_to_dataframe(tracks)

    # Print summary
    print_manifest_summary(manifest_df)

    if args.dry_run:
        print("Dry run - not saving manifest")
        print("\nFirst 5 tracks:")
        print(manifest_df.head())
        return

    # Save manifest
    output_path = Path(config.paths.metadata_root) / "trackA_manifest.csv"
    save_manifest(manifest_df, output_path)

    print("✓ Manifest build complete!")


if __name__ == "__main__":
    main()
