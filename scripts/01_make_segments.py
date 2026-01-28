#!/usr/bin/env python3
"""
Script 01: Make Segments

Creates segment index from manifest and computes GT vocal RMS
for silence detection.

Usage:
    python scripts/01_make_segments.py [--config configs/trackA_config.yaml]

Input:
    metadata/trackA_manifest.csv

Output:
    tables/segments_trackA.parquet
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import load_config, TrackAConfig
from src.data_types import TrackInfo, SilenceCalibrationResult
from src.segmentation import (
    create_segment_index,
    compute_segment_rms,
    calibrate_and_flag_silence,
    get_track_vocal_paths_from_manifest,
    save_segments,
    print_segment_summary,
)


def load_manifest(path: Path) -> pd.DataFrame:
    """Load manifest CSV.

    Args:
        path: Path to manifest CSV

    Returns:
        Manifest DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    return pd.read_csv(path)


def manifest_to_track_infos(manifest_df: pd.DataFrame) -> list:
    """Convert manifest DataFrame to list of TrackInfo objects.

    Args:
        manifest_df: Manifest DataFrame

    Returns:
        List of TrackInfo objects
    """
    tracks = []

    for _, row in manifest_df.iterrows():
        # Collect other stem paths
        other_paths = {}
        for col in manifest_df.columns:
            if (
                col.startswith("gt_")
                and col.endswith("_path")
                and col != "gt_vocal_path"
            ):
                stem = col.replace("gt_", "").replace("_path", "")
                if pd.notna(row[col]):
                    other_paths[stem] = row[col]

        track = TrackInfo(
            track_id=row["track_id"],
            mixture_path=row["mixture_path"],
            gt_vocal_path=row["gt_vocal_path"],
            duration_sec=row["duration_sec"],
            duration_samples=int(row["duration_samples"]),
            fold_id=int(row["fold_id"]),
            gt_other_paths=other_paths if other_paths else None,
        )
        tracks.append(track)

    return tracks


def save_calibration_result(result: SilenceCalibrationResult, path: Path) -> None:
    """Save silence calibration result to JSON.

    Args:
        result: Calibration result
        path: Output path
    """
    import numpy as np

    def convert_numpy(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    data = convert_numpy(result.to_dict())
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved calibration result to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create segment index with silence detection"
    )
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
        "--skip-rms",
        action="store_true",
        help="Skip RMS computation (use dummy values for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be done without saving"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    print("=" * 50)
    print("Script 01: Make Segments")
    print("=" * 50)

    # Load config
    print(f"\nLoading config from {args.config}")
    config = load_config(args.config)

    # Create directories
    config.paths.ensure_dirs()

    # Check existing file
    output_path = Path(config.paths.tables_root) / "segments_trackA.parquet"
    if output_path.exists() and not args.force and not args.dry_run:
        print(f"\nOutput exists: {output_path}")
        print("Use --force to overwrite")
        return

    # Load manifest
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(config.paths.metadata_root) / "trackA_manifest.csv"
    )
    print(f"Loading manifest from {manifest_path}")
    manifest_df = load_manifest(manifest_path)
    print(f"Loaded {len(manifest_df)} tracks")

    # Convert to TrackInfo objects
    tracks = manifest_to_track_infos(manifest_df)

    # Create segment index
    print(f"\nCreating segment index...")
    print(f"  Window: {config.segmentation.window_sec}s")
    print(f"  Hop: {config.segmentation.hop_sec}s")

    segments_df = create_segment_index(
        tracks,
        window_sec=config.segmentation.window_sec,
        hop_sec=config.segmentation.hop_sec,
        sr=config.audio.sample_rate,
    )

    print(f"Created {len(segments_df)} segments from {len(tracks)} tracks")

    # Compute RMS for each segment
    if args.skip_rms:
        print("\nSkipping RMS computation (using dummy values)")
        # Generate plausible dummy RMS values for testing
        np.random.seed(42)
        n_seg = len(segments_df)
        # Mix of silent and normal segments
        n_silent = int(n_seg * 0.05)  # ~5% silent
        rms_values = np.concatenate(
            [
                np.random.normal(-65, 5, n_silent),  # Silent
                np.random.normal(-25, 12, n_seg - n_silent),  # Normal
            ]
        )
        np.random.shuffle(rms_values)
        segments_df["gt_vocal_rms_db"] = rms_values[:n_seg]
    else:
        print("\nComputing GT vocal RMS for each segment...")
        track_to_vocal = get_track_vocal_paths_from_manifest(manifest_df)
        segments_df = compute_segment_rms(
            segments_df, track_to_vocal, sr=config.audio.sample_rate, verbose=True
        )

    # Calibrate silence threshold and flag silent segments
    print("\nCalibrating silence threshold...")
    segments_df, calibration = calibrate_and_flag_silence(
        segments_df,
        initial_threshold_db=config.silence.initial_threshold_db,
        method=config.silence.calibration_method,
        percentile_cutoff=config.silence.percentile_cutoff,
    )

    # Print summary
    print_segment_summary(segments_df, calibration)

    if args.dry_run:
        print("Dry run - not saving outputs")
        print("\nFirst 10 segments:")
        print(segments_df.head(10))
        return

    # Save outputs
    output_path = Path(config.paths.tables_root) / "segments_trackA.parquet"
    save_segments(segments_df, output_path)
    print(f"Saved segment index to {output_path}")

    # Save calibration result
    calibration_path = (
        Path(config.paths.results_root) / "trackA" / "silence_calibration.json"
    )
    save_calibration_result(calibration, calibration_path)

    print("\n✓ Segment index creation complete!")

    # Quick stats for verification
    print("\nQuick verification:")
    print(f"  Segment ID format: {segments_df['seg_id'].iloc[0]}")
    print(
        f"  Time range: [{segments_df['start_sec'].min():.2f}, {segments_df['end_sec'].max():.2f}]s"
    )
    print(
        f"  RMS range: [{segments_df['gt_vocal_rms_db'].min():.1f}, {segments_df['gt_vocal_rms_db'].max():.1f}] dB"
    )


if __name__ == "__main__":
    main()
