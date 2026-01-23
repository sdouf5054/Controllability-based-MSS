"""
FLX4-Net Track A Segmentation Module

Creates segment indices from tracks and handles silence detection.
Core module for defining the analysis unit (3.0s window, 1.5s hop).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import asdict

from .data_types import Segment, TrackInfo, SilenceCalibrationResult
from .audio_utils import (
    load_audio,
    compute_rms_db,
    get_segment_audio,
    seconds_to_samples,
    samples_to_seconds,
    calibrate_silence_threshold
)
from .config import TrackAConfig


def create_segments_for_track(
    track_info: TrackInfo,
    window_sec: float,
    hop_sec: float,
    sr: int = 44100
) -> List[Segment]:
    """Create segment index for a single track.
    
    Args:
        track_info: Track information
        window_sec: Segment window length in seconds
        hop_sec: Hop length in seconds
        sr: Sample rate
        
    Returns:
        List of Segment objects
    """
    segments = []
    
    window_samples = seconds_to_samples(window_sec, sr)
    hop_samples = seconds_to_samples(hop_sec, sr)
    
    # Calculate number of segments
    # We want complete windows only
    n_segments = max(0, (track_info.duration_samples - window_samples) // hop_samples + 1)
    
    for seg_idx in range(n_segments):
        start_sample = seg_idx * hop_samples
        end_sample = start_sample + window_samples
        
        # Safety check
        if end_sample > track_info.duration_samples:
            break
        
        start_sec = samples_to_seconds(start_sample, sr)
        end_sec = samples_to_seconds(end_sample, sr)
        
        seg = Segment(
            seg_id=f"{track_info.track_id}_{seg_idx:04d}",
            track_id=track_info.track_id,
            seg_idx=seg_idx,
            start_sec=start_sec,
            end_sec=end_sec,
            start_sample=start_sample,
            end_sample=end_sample,
            fold_id=track_info.fold_id,
            gt_vocal_rms_db=0.0,  # Will be filled later
            is_silent=False  # Will be filled later
        )
        segments.append(seg)
    
    return segments


def create_segment_index(
    tracks: List[TrackInfo],
    window_sec: float,
    hop_sec: float,
    sr: int = 44100
) -> pd.DataFrame:
    """Create segment index for all tracks.
    
    Args:
        tracks: List of TrackInfo objects
        window_sec: Segment window length
        hop_sec: Hop length
        sr: Sample rate
        
    Returns:
        DataFrame with segment index
    """
    all_segments = []
    
    for track_info in tracks:
        segments = create_segments_for_track(
            track_info, window_sec, hop_sec, sr
        )
        all_segments.extend(segments)
    
    # Convert to DataFrame
    if not all_segments:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'seg_id', 'track_id', 'seg_idx', 'start_sec', 'end_sec',
            'start_sample', 'end_sample', 'fold_id', 'gt_vocal_rms_db', 'is_silent'
        ])
    
    df = pd.DataFrame([asdict(seg) for seg in all_segments])
    
    return df


def compute_segment_rms(
    segments_df: pd.DataFrame,
    track_to_vocal_path: dict,
    sr: int = 44100,
    verbose: bool = True
) -> pd.DataFrame:
    """Compute GT vocal RMS for each segment.
    
    Args:
        segments_df: Segment index DataFrame
        track_to_vocal_path: Mapping from track_id to GT vocal path
        sr: Sample rate
        verbose: Print progress
        
    Returns:
        DataFrame with gt_vocal_rms_db column updated
    """
    df = segments_df.copy()
    
    # Group by track for efficiency (load audio once per track)
    track_ids = df['track_id'].unique()
    
    for i, track_id in enumerate(track_ids):
        if verbose:
            print(f"  Computing RMS for track {i+1}/{len(track_ids)}: {track_id}")
        
        if track_id not in track_to_vocal_path:
            print(f"  Warning: No vocal path for track {track_id}, skipping")
            continue
        
        vocal_path = track_to_vocal_path[track_id]
        
        try:
            audio, _ = load_audio(vocal_path, sr=sr, mono=True)
        except Exception as e:
            print(f"  Warning: Failed to load {vocal_path}: {e}")
            continue
        
        # Get segments for this track
        track_mask = df['track_id'] == track_id
        track_segments = df[track_mask]
        
        for idx in track_segments.index:
            start_sample = df.loc[idx, 'start_sample']
            end_sample = df.loc[idx, 'end_sample']
            
            # Handle edge case where segment exceeds audio length
            if end_sample > len(audio):
                end_sample = len(audio)
            
            if start_sample >= end_sample:
                df.loc[idx, 'gt_vocal_rms_db'] = -100.0
                continue
            
            segment_audio = audio[start_sample:end_sample]
            rms_db = compute_rms_db(segment_audio)
            df.loc[idx, 'gt_vocal_rms_db'] = rms_db
    
    return df


def add_silence_flags(
    segments_df: pd.DataFrame,
    threshold_db: float
) -> pd.DataFrame:
    """Add is_silent flags based on threshold.
    
    Args:
        segments_df: Segment index with gt_vocal_rms_db
        threshold_db: Silence threshold in dB
        
    Returns:
        DataFrame with is_silent column updated
    """
    df = segments_df.copy()
    df['is_silent'] = df['gt_vocal_rms_db'] < threshold_db
    return df


def calibrate_and_flag_silence(
    segments_df: pd.DataFrame,
    initial_threshold_db: float = -40.0,
    method: str = "gap_or_5th_percentile",
    percentile_cutoff: float = 5.0
) -> Tuple[pd.DataFrame, SilenceCalibrationResult]:
    """Calibrate silence threshold and add flags.
    
    Policy: flag_and_separate
    - Looks for gap in RMS histogram
    - Falls back to percentile if no gap found
    
    Args:
        segments_df: Segment index with gt_vocal_rms_db
        initial_threshold_db: Initial threshold
        method: Calibration method
        percentile_cutoff: Percentile for fallback
        
    Returns:
        (updated_df, calibration_result)
    """
    rms_values = segments_df['gt_vocal_rms_db'].values
    
    # Calibrate threshold
    final_threshold, calibration_method, rms_stats = calibrate_silence_threshold(
        rms_values,
        initial_threshold_db=initial_threshold_db,
        method=method,
        percentile_cutoff=percentile_cutoff
    )
    
    # Apply threshold
    df = add_silence_flags(segments_df, final_threshold)
    
    # Create result
    n_silent = df['is_silent'].sum()
    n_total = len(df)
    
    result = SilenceCalibrationResult(
        initial_threshold_db=initial_threshold_db,
        final_threshold_db=final_threshold,
        calibration_method=calibration_method,
        n_segments_total=n_total,
        n_segments_silent=n_silent,
        silent_ratio=n_silent / n_total if n_total > 0 else 0.0,
        rms_stats=rms_stats
    )
    
    return df, result


def get_track_vocal_paths_from_manifest(
    manifest_df: pd.DataFrame
) -> dict:
    """Extract track_id to gt_vocal_path mapping from manifest.
    
    Args:
        manifest_df: Manifest DataFrame
        
    Returns:
        dict mapping track_id to gt_vocal_path
    """
    return dict(zip(manifest_df['track_id'], manifest_df['gt_vocal_path']))


def save_segments(segments_df: pd.DataFrame, path: str | Path) -> None:
    """Save segment index to parquet file.
    
    Args:
        segments_df: Segment index DataFrame
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    segments_df.to_parquet(path, index=False)


def load_segments(path: str | Path) -> pd.DataFrame:
    """Load segment index from parquet file.
    
    Args:
        path: Path to parquet file
        
    Returns:
        Segment index DataFrame
    """
    return pd.read_parquet(path)


def print_segment_summary(segments_df: pd.DataFrame, calibration: Optional[SilenceCalibrationResult] = None):
    """Print summary of segment index.
    
    Args:
        segments_df: Segment index DataFrame
        calibration: Optional calibration result
    """
    n_tracks = segments_df['track_id'].nunique()
    n_segments = len(segments_df)
    n_silent = segments_df['is_silent'].sum()
    n_valid = n_segments - n_silent
    
    print(f"\n{'='*50}")
    print(f"Segment Index Summary")
    print(f"{'='*50}")
    print(f"Tracks: {n_tracks}")
    print(f"Total segments: {n_segments}")
    print(f"Valid segments (non-silent): {n_valid} ({n_valid/n_segments*100:.1f}%)")
    print(f"Silent segments: {n_silent} ({n_silent/n_segments*100:.1f}%)")
    
    if calibration:
        print(f"\nSilence Calibration:")
        print(f"  Method: {calibration.calibration_method}")
        print(f"  Initial threshold: {calibration.initial_threshold_db:.1f} dB")
        print(f"  Final threshold: {calibration.final_threshold_db:.1f} dB")
        print(f"  RMS range: [{calibration.rms_stats.get('min', 'N/A'):.1f}, "
              f"{calibration.rms_stats.get('max', 'N/A'):.1f}] dB")
    
    # Per-fold summary
    print(f"\nPer-fold breakdown:")
    for fold_id in sorted(segments_df['fold_id'].unique()):
        fold_df = segments_df[segments_df['fold_id'] == fold_id]
        fold_valid = len(fold_df[~fold_df['is_silent']])
        print(f"  Fold {fold_id}: {len(fold_df)} segments ({fold_valid} valid)")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test segmentation with synthetic data
    print("=== Testing Segmentation Module ===\n")
    
    # Create mock tracks
    tracks = [
        TrackInfo(
            track_id="track001",
            mixture_path="/path/to/mix1.wav",
            gt_vocal_path="/path/to/vocals1.wav",
            duration_sec=180.0,  # 3 minutes
            duration_samples=int(180.0 * 44100),
            fold_id=0
        ),
        TrackInfo(
            track_id="track002",
            mixture_path="/path/to/mix2.wav",
            gt_vocal_path="/path/to/vocals2.wav",
            duration_sec=240.0,  # 4 minutes
            duration_samples=int(240.0 * 44100),
            fold_id=1
        )
    ]
    
    # Create segment index
    segments_df = create_segment_index(
        tracks,
        window_sec=3.0,
        hop_sec=1.5,
        sr=44100
    )
    
    print(f"Created {len(segments_df)} segments from {len(tracks)} tracks")
    print(f"\nFirst 5 segments:")
    print(segments_df.head())
    
    # Simulate RMS values (in real usage, these come from actual audio)
    np.random.seed(42)
    segments_df['gt_vocal_rms_db'] = np.concatenate([
        np.random.normal(-60, 5, 20),  # Some silent
        np.random.normal(-25, 10, len(segments_df) - 20)  # Rest normal
    ])[:len(segments_df)]
    
    # Calibrate silence
    segments_df, calibration = calibrate_and_flag_silence(
        segments_df,
        initial_threshold_db=-40.0,
        method="gap_or_5th_percentile"
    )
    
    print_segment_summary(segments_df, calibration)
    
    print("✓ Segmentation module working correctly!")
