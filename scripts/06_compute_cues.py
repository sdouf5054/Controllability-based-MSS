#!/usr/bin/env python3
"""
Script 06: Compute Cues

Computes the four controllability cues for each segment:
    - r(t): Repetition (short-term similarity, past 12s)
    - s(t): Structure (long-range similarity, >= 30s)
    - b(t): Beat synchrony (onset-beat alignment)
    - p(t): Processing (spectral texture)

Usage:
    python scripts/06_compute_cues.py --config configs/trackA_config.yaml
    python scripts/06_compute_cues.py --max-tracks 5  # Test
    python scripts/06_compute_cues.py --dry-run

Input:
    - metadata/trackA_manifest.csv
    - tables/segments_trackA.parquet
    - cache/separated/htdemucs/{track_id}/vocals.wav
    - cache/beat_tracking/librosa/{track_id}/beats.json

Output:
    - tables/cues_trackA.parquet
    - results/trackA/cue_stats.json
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.audio_utils import load_audio
from src.data_types import CueResult, BeatResult
from src.cues import (
    EmbeddingComputer,
    compute_similarity_matrix,
    RepetitionComputer,
    StructureComputer,
    BeatSyncComputer,
    ProcessingComputer,
)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load track manifest."""
    return pd.read_csv(manifest_path)


def load_segments(segments_path: Path) -> pd.DataFrame:
    """Load segment index."""
    return pd.read_parquet(segments_path)


def load_beat_result(
    cache_root: Path, backend: str, track_id: str
) -> Optional[BeatResult]:
    """Load beat tracking result for a track."""
    beat_path = cache_root / "beat_tracking" / backend / track_id / "beats.json"
    if not beat_path.exists():
        return None
    return BeatResult.load_json(beat_path)


def compute_cues_for_track(
    track_id: str,
    segments: List[dict],
    vocal_path: Path,
    beat_result: Optional[BeatResult],
    config,
    sr: int = 44100,
) -> Tuple[List[CueResult], Dict]:
    """Compute all four cues for a track.

    Args:
        track_id: Track identifier
        segments: List of segment dicts for this track
        vocal_path: Path to separated vocal audio
        beat_result: Beat tracking result (or None)
        config: Configuration object
        sr: Sample rate

    Returns:
        Tuple of (list of CueResult, stats dict)
    """
    stats = {
        "track_id": track_id,
        "n_segments": len(segments),
        "beat_available": beat_result is not None and beat_result.track_available,
    }

    # Load vocal audio
    if not vocal_path.exists():
        # Return empty results
        results = [
            CueResult(
                seg_id=seg["seg_id"],
                r=float("nan"),
                b=None,
                s=float("nan"),
                p=float("nan"),
                beat_available=False,
            )
            for seg in segments
        ]
        stats["error"] = "vocal_not_found"
        return results, stats

    vocal_audio, _ = load_audio(vocal_path, sr=sr, mono=True)

    # Get beat times
    beat_times = np.array(beat_result.beat_times) if beat_result else np.array([])
    track_beat_available = beat_result.track_available if beat_result else False

    # Initialize computers
    similarity_config = config.cues.similarity

    embedding_computer = EmbeddingComputer(
        sr=sr,
        n_mels=similarity_config.n_mels,
        n_fft=similarity_config.n_fft,
        hop_length=similarity_config.hop_length,
        fmin=similarity_config.fmin,
        fmax=similarity_config.fmax,
        pooling=similarity_config.pooling,
        normalize=similarity_config.normalize,
    )

    repetition_computer = RepetitionComputer(
        max_lookback_sec=config.cues.repetition.max_lookback_sec,
        aggregation=config.cues.repetition.aggregation,
        top_k=config.cues.repetition.top_k,
    )

    structure_computer = StructureComputer(
        min_distance_sec=config.cues.structure.min_distance_sec,
        aggregation=config.cues.structure.aggregation,
        top_k=config.cues.structure.top_k,
    )

    beat_sync_computer = BeatSyncComputer(
        sr=sr,
        tolerance_sec=config.cues.beat_sync.tolerance_sec,
        min_beats=config.beat_tracking.min_beats_per_segment,
        method="energy_ratio",
        energy_window_sec=0.05,
    )

    processing_computer = ProcessingComputer(sr=sr)

    # Set track data
    embedding_computer.set_track_audio(vocal_audio)
    beat_sync_computer.set_track_data(vocal_audio, beat_times)
    processing_computer.set_track_audio(vocal_audio)

    # Compute embeddings for all segments
    embeddings = embedding_computer.compute_all_embeddings(segments)
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Compute r(t) and s(t)
    r_values, r_counts = repetition_computer.compute(similarity_matrix, segments)
    s_values, s_counts = structure_computer.compute(similarity_matrix, segments)

    # Compute b(t) and p(t) for each segment
    results = []
    b_available_count = 0

    for i, seg in enumerate(segments):
        # b(t)
        if track_beat_available:
            b_val, b_avail, b_info = beat_sync_computer.compute_segment(
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                start_sample=seg["start_sample"],
                end_sample=seg["end_sample"],
            )
            if b_avail:
                b_available_count += 1
        else:
            b_val = None
            b_avail = False

        # p(t)
        p_val, p_info = processing_computer.compute_segment(
            start_sample=seg["start_sample"], end_sample=seg["end_sample"]
        )

        result = CueResult(
            seg_id=seg["seg_id"],
            r=float(r_values[i]),
            b=b_val,
            s=float(s_values[i]),
            p=float(p_val),
            beat_available=b_avail,
            r_candidate_count=int(r_counts[i]),
            s_candidate_count=int(s_counts[i]),
        )
        results.append(result)

    stats["b_available_count"] = b_available_count
    stats["b_coverage"] = b_available_count / len(segments) if segments else 0

    return results, stats


def results_to_dataframe(results: List[CueResult]) -> pd.DataFrame:
    """Convert CueResult list to DataFrame."""
    records = [r.to_dict() for r in results]
    return pd.DataFrame(records)


def compute_cue_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistics for cue results."""
    stats: Dict[str, Any] = {
        "n_total": len(df),
        "n_beat_available": int(df["beat_available"].sum()),
    }

    # Stats for each cue
    for cue in ["r", "s", "p"]:
        values = df[cue].dropna()
        if len(values) > 0:
            stats[cue] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "p5": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
            }
        else:
            stats[cue] = {"mean": None}

    # b(t) separately (may have None values)
    b_values = df[df["beat_available"]]["b"].dropna()
    if len(b_values) > 0:
        stats["b"] = {
            "mean": float(np.mean(b_values)),
            "std": float(np.std(b_values)),
            "min": float(np.min(b_values)),
            "max": float(np.max(b_values)),
            "median": float(np.median(b_values)),
            "coverage": len(b_values) / len(df),
        }
    else:
        stats["b"] = {"mean": None, "coverage": 0}

    return stats


def print_cue_summary(df: pd.DataFrame, stats: Dict):
    """Print summary of cue computation."""
    print("=" * 50)
    print("Cue Computation Summary")
    print("=" * 50)

    print(f"\nTotal segments: {stats['n_total']}")
    print(
        f"Beat available: {stats['n_beat_available']} ({stats['n_beat_available']/stats['n_total']*100:.1f}%)"
    )

    for cue_name, cue_label in [
        ("r", "r(t) Repetition"),
        ("s", "s(t) Structure"),
        ("b", "b(t) Beat Sync"),
        ("p", "p(t) Processing"),
    ]:
        print(f"\n{cue_label}:")
        cue_stats = stats.get(cue_name, {})
        if cue_stats.get("mean") is not None:
            print(f"  Mean: {cue_stats['mean']:.4f}")
            print(f"  Std:  {cue_stats['std']:.4f}")
            print(f"  Range: [{cue_stats['min']:.4f}, {cue_stats['max']:.4f}]")
            if "coverage" in cue_stats:
                print(f"  Coverage: {cue_stats['coverage']:.1%}")
        else:
            print("  No valid values")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Compute controllability cues (r, s, b, p) for each segment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trackA_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--manifest", type=str, default=None, help="Path to manifest CSV"
    )
    parser.add_argument(
        "--segments", type=str, default=None, help="Path to segments parquet"
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit number of tracks (for testing)",
    )
    parser.add_argument(
        "--include-silent", action="store_true", help="Include silent segments"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    print("=" * 50)
    print("Script 06: Compute Cues")
    print("=" * 50)

    # Load config
    print(f"\nLoading config from {args.config}")
    config = load_config(args.config)

    # Paths
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(config.paths.metadata_root) / "trackA_manifest.csv"
    )
    segments_path = (
        Path(args.segments)
        if args.segments
        else Path(config.paths.tables_root) / "segments_trackA.parquet"
    )
    output_path = Path(config.paths.tables_root) / "cues_trackA.parquet"
    stats_path = Path(config.paths.results_root) / "trackA" / "cue_stats.json"

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
    print(f"\nCue settings:")
    print(f"  r(t) max lookback: {config.cues.repetition.max_lookback_sec}s")
    print(f"  s(t) min distance: {config.cues.structure.min_distance_sec}s")
    print(f"  b(t) tolerance: {config.cues.beat_sync.tolerance_sec * 1000:.0f}ms")
    print(f"  Separator model: {config.separator.name}")
    print(f"  Beat backend: {config.beat_tracking.backend}")

    if args.max_tracks:
        print(f"  Limited to {args.max_tracks} tracks")

    # Dry run
    if args.dry_run:
        print("\n[DRY RUN] Would compute cues for:")

        n_tracks = (
            min(args.max_tracks, len(manifest_df))
            if args.max_tracks
            else len(manifest_df)
        )
        track_ids = manifest_df["track_id"].iloc[:n_tracks].tolist()
        track_segments = segments_df[segments_df["track_id"].isin(track_ids)]

        if not args.include_silent and "is_silent" in track_segments.columns:
            track_segments = track_segments[~track_segments["is_silent"]]

        print(f"  Tracks: {n_tracks}")
        print(f"  Segments: {len(track_segments)}")

        # Check cache
        cache_root = Path(config.paths.cache_root)
        vocals_cached = sum(
            1
            for tid in track_ids
            if (
                cache_root / "separated" / config.separator.name / tid / "vocals.wav"
            ).exists()
        )
        beats_cached = sum(
            1
            for tid in track_ids
            if (
                cache_root
                / "beat_tracking"
                / config.beat_tracking.backend
                / tid
                / "beats.json"
            ).exists()
        )

        print(f"  Vocals cached: {vocals_cached}/{n_tracks}")
        print(f"  Beats cached: {beats_cached}/{n_tracks}")

        print(f"\nOutput:")
        print(f"  {output_path}")
        print(f"  {stats_path}")
        return

    # Process tracks
    print(f"\nComputing cues...")
    start_time = time.time()

    all_results = []
    tracks = manifest_df.to_dict("records")
    if args.max_tracks:
        tracks = tracks[: args.max_tracks]

    cache_root = Path(config.paths.cache_root)

    for track in tqdm(tracks, desc="Processing tracks"):
        track_id = track["track_id"]

        # Get segments for this track
        track_segments = segments_df[segments_df["track_id"] == track_id]

        if not args.include_silent and "is_silent" in track_segments.columns:
            track_segments = track_segments[~track_segments["is_silent"]]

        if len(track_segments) == 0:
            continue

        segments_list = track_segments.to_dict("records")

        # Get vocal path
        vocal_path = (
            cache_root / "separated" / config.separator.name / track_id / "vocals.wav"
        )

        # Load beat result
        beat_result = load_beat_result(
            cache_root, config.beat_tracking.backend, track_id
        )

        # Compute cues
        results, track_stats = compute_cues_for_track(
            track_id=track_id,
            segments=segments_list,
            vocal_path=vocal_path,
            beat_result=beat_result,
            config=config,
            sr=config.audio.sample_rate,
        )

        all_results.extend(results)

    elapsed = time.time() - start_time

    # Convert to DataFrame
    cues_df = results_to_dataframe(all_results)

    # Compute stats
    stats = compute_cue_stats(cues_df)
    stats["computation_time_sec"] = elapsed
    stats["config"] = {
        "r_max_lookback_sec": config.cues.repetition.max_lookback_sec,
        "s_min_distance_sec": config.cues.structure.min_distance_sec,
        "b_tolerance_sec": config.cues.beat_sync.tolerance_sec,
        "separator_model": config.separator.name,
        "beat_backend": config.beat_tracking.backend,
    }

    # Summary
    print_cue_summary(cues_df, stats)
    print(f"\nTime: {elapsed:.1f}s ({len(all_results)/elapsed:.1f} segments/sec)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cues_df.to_parquet(output_path, index=False)
    print(f"\nSaved cues to {output_path}")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # Sample
    print(f"\nSample results (first 10):")
    for _, row in cues_df.head(10).iterrows():
        b_str = (
            f"{row['b']:.3f}"
            if row["beat_available"] and row["b"] is not None
            else "N/A"
        )
        print(
            f"  {row['seg_id']}: r={row['r']:.3f}, s={row['s']:.3f}, b={b_str}, p={row['p']:.3f}"
        )

    print(f"\n✓ Cue computation complete!")


if __name__ == "__main__":
    main()
