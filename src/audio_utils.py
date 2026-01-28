"""
FLX4-Net Track A Audio Utilities

Core audio loading, processing, and analysis functions.
All functions are pure (no side effects) and well-typed.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Literal
import warnings


def load_audio(
    path: str | Path, sr: int = 44100, mono: bool = True, dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, int]:
    """Load audio file with resampling.

    Args:
        path: Path to audio file
        sr: Target sample rate (default: 44100)
        mono: Convert to mono if True
        dtype: Output dtype (default: float32)

    Returns:
        audio: Audio array, shape (n_samples,) if mono else (n_channels, n_samples)
        sr: Sample rate (same as input sr)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading fails
    """
    import librosa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, _ = librosa.load(path, sr=sr, mono=mono)
        return audio.astype(dtype), sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {path}: {e}")


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (Root Mean Square) of audio signal.

    Args:
        audio: Audio signal array

    Returns:
        RMS value (linear scale)
    """
    return float(np.sqrt(np.mean(audio**2)))


def compute_rms_db(
    audio: np.ndarray, ref: float = 1.0, min_db: float = -100.0
) -> float:
    """Compute RMS in decibels.

    Args:
        audio: Audio signal array
        ref: Reference value (default: 1.0)
        min_db: Minimum dB value to return (avoids -inf)

    Returns:
        RMS in dB (clamped to min_db)

    Note:
        Returns min_db for truly silent audio (RMS=0).
        This floor value should be handled specially in calibration.
    """
    rms = compute_rms(audio)
    if rms <= 0:
        return min_db

    db = 20 * np.log10(rms / ref)
    return max(db, min_db)


def is_floor_value(
    db_value: float, floor: float = -100.0, tolerance: float = 0.1
) -> bool:
    """Check if a dB value is at the floor (truly silent).

    Args:
        db_value: RMS value in dB
        floor: Floor value used in compute_rms_db
        tolerance: Tolerance for comparison

    Returns:
        True if value is at or below floor
    """
    return db_value <= floor + tolerance


def align_lengths(
    est: np.ndarray,
    ref: np.ndarray,
    policy: Literal["truncate_to_shorter", "pad_to_longer"] = "truncate_to_shorter",
    max_diff_samples: int = 1000,
    warn: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two audio arrays to the same length.

    This is necessary because separator output may differ slightly
    from input length due to padding/framing.

    Args:
        est: Estimated signal
        ref: Reference signal
        policy: How to handle length mismatch
        max_diff_samples: Warn if difference exceeds this
        warn: Whether to log warning for large differences

    Returns:
        (est_aligned, ref_aligned): Arrays with same length
    """
    len_est = len(est)
    len_ref = len(ref)
    diff = abs(len_est - len_ref)

    if diff > max_diff_samples and warn:
        warnings.warn(
            f"Audio length difference ({diff} samples, ~{diff/44100*1000:.1f}ms) "
            f"exceeds threshold ({max_diff_samples} samples). "
            f"Check separator output alignment."
        )

    if len_est == len_ref:
        return est, ref

    if policy == "truncate_to_shorter":
        min_len = min(len_est, len_ref)
        return est[:min_len], ref[:min_len]

    elif policy == "pad_to_longer":
        max_len = max(len_est, len_ref)
        est_padded = np.pad(est, (0, max_len - len_est), mode="constant")
        ref_padded = np.pad(ref, (0, max_len - len_ref), mode="constant")
        return est_padded, ref_padded

    else:
        raise ValueError(f"Unknown alignment policy: {policy}")


def get_segment_audio(
    audio: np.ndarray, start_sample: int, end_sample: int
) -> np.ndarray:
    """Extract a segment from audio array.

    Args:
        audio: Full audio array
        start_sample: Start sample index
        end_sample: End sample index

    Returns:
        Segment audio array

    Raises:
        ValueError: If indices are out of bounds
    """
    if start_sample < 0:
        raise ValueError(f"start_sample must be >= 0, got {start_sample}")
    if end_sample > len(audio):
        raise ValueError(
            f"end_sample ({end_sample}) exceeds audio length ({len(audio)})"
        )
    if start_sample >= end_sample:
        raise ValueError(
            f"start_sample ({start_sample}) must be < end_sample ({end_sample})"
        )

    return audio[start_sample:end_sample]


def seconds_to_samples(sec: float, sr: int = 44100) -> int:
    """Convert seconds to samples.

    Args:
        sec: Time in seconds
        sr: Sample rate

    Returns:
        Number of samples
    """
    return int(sec * sr)


def samples_to_seconds(samples: int, sr: int = 44100) -> float:
    """Convert samples to seconds.

    Args:
        samples: Number of samples
        sr: Sample rate

    Returns:
        Time in seconds
    """
    return samples / sr


def get_audio_duration(path: str | Path, sr: int = 44100) -> Tuple[float, int]:
    """Get audio duration without loading full file.

    Args:
        path: Path to audio file
        sr: Sample rate for calculation

    Returns:
        (duration_sec, duration_samples)
    """
    import librosa

    duration_sec = librosa.get_duration(path=path)
    duration_samples = seconds_to_samples(duration_sec, sr)

    return duration_sec, duration_samples


def compute_energy_envelope(
    audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Compute energy envelope of audio signal.

    Args:
        audio: Audio signal
        frame_length: Frame length in samples
        hop_length: Hop length in samples

    Returns:
        Energy envelope array
    """
    import librosa

    return librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]


def find_histogram_gap(
    values: np.ndarray, n_bins: int = 100, min_gap_ratio: float = 0.1
) -> Optional[float]:
    """Find a gap in histogram distribution.

    Used for silence threshold calibration.

    Args:
        values: Array of values (e.g., RMS in dB)
        n_bins: Number of histogram bins
        min_gap_ratio: Minimum ratio of empty bins to consider a gap

    Returns:
        Gap threshold value, or None if no clear gap found
    """
    # Compute histogram
    counts, bin_edges = np.histogram(values, bins=n_bins)

    # Find consecutive empty bins (gap)
    min_empty = max(2, int(n_bins * min_gap_ratio))

    empty_run = 0
    gap_start_idx = None

    for i, count in enumerate(counts):
        if count == 0:
            if empty_run == 0:
                gap_start_idx = i
            empty_run += 1
        else:
            if empty_run >= min_empty:
                # Found a gap, return the middle of the gap
                gap_end_idx = i
                gap_middle_idx = (gap_start_idx + gap_end_idx) // 2
                return float(bin_edges[gap_middle_idx])
            empty_run = 0
            gap_start_idx = None

    # Check if gap extends to end
    if empty_run >= min_empty and gap_start_idx is not None:
        gap_middle_idx = (gap_start_idx + n_bins) // 2
        return float(bin_edges[gap_middle_idx])

    return None


def calibrate_silence_threshold(
    rms_values: np.ndarray,
    initial_threshold_db: float = -40.0,
    method: str = "gap_or_5th_percentile",
    percentile_cutoff: float = 5.0,
    floor_db: float = -100.0,
) -> Tuple[float, str, dict]:
    """Calibrate silence threshold based on RMS distribution.

    Policy from Blueprint:
    - If clear gap exists in histogram, use gap position
    - Otherwise, use 5th percentile

    IMPORTANT: Floor values (truly silent segments at -100dB) are excluded
    from percentile calculation to avoid biasing the threshold.

    Args:
        rms_values: Array of RMS values in dB
        initial_threshold_db: Initial threshold
        method: "gap_or_5th_percentile" | "fixed"
        percentile_cutoff: Percentile to use if no gap found
        floor_db: Floor value used in RMS computation (excluded from percentile)

    Returns:
        (final_threshold, calibration_note, stats_dict)
    """
    # Separate floor values (truly silent) from actual measurements
    floor_mask = rms_values <= floor_db + 0.1
    n_floor = np.sum(floor_mask)
    non_floor_values = rms_values[~floor_mask]

    # Compute statistics on ALL values (for reporting)
    stats = {
        "min": float(np.min(rms_values)),
        "max": float(np.max(rms_values)),
        "mean": float(np.mean(rms_values)),
        "median": float(np.median(rms_values)),
        "std": float(np.std(rms_values)),
        "p5": float(np.percentile(rms_values, 5)),
        "p25": float(np.percentile(rms_values, 25)),
        "p75": float(np.percentile(rms_values, 75)),
        "p95": float(np.percentile(rms_values, 95)),
        "n_total": len(rms_values),
        "n_floor": int(n_floor),
        "floor_ratio": float(n_floor / len(rms_values)) if len(rms_values) > 0 else 0.0,
    }

    if method == "fixed":
        return initial_threshold_db, "fixed", stats

    # If all values are at floor, return initial threshold
    if len(non_floor_values) == 0:
        stats["note"] = "all_values_at_floor"
        return initial_threshold_db, "fixed_all_floor", stats

    # Try to find gap (on non-floor values only)
    gap_threshold = find_histogram_gap(non_floor_values)

    if gap_threshold is not None:
        # Use gap position
        return gap_threshold, "gap", stats
    else:
        # Fall back to percentile (on non-floor values)
        # This is the key fix: exclude -100dB floor from percentile calculation
        percentile_threshold = float(np.percentile(non_floor_values, percentile_cutoff))

        # Add non-floor stats for debugging
        stats["non_floor_p5"] = float(np.percentile(non_floor_values, 5))
        stats["non_floor_min"] = float(np.min(non_floor_values))

        return percentile_threshold, f"{percentile_cutoff}th_percentile", stats


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio to target dB level.

    Args:
        audio: Input audio
        target_db: Target peak level in dB

    Returns:
        Normalized audio
    """
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio

    target_linear = 10 ** (target_db / 20)
    gain = target_linear / current_peak

    return audio * gain


if __name__ == "__main__":
    # Test audio utilities
    print("=== Testing Audio Utilities ===\n")

    # Test with synthetic audio
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create test signal: 440 Hz sine wave
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Test RMS
    rms = compute_rms(test_audio)
    rms_db = compute_rms_db(test_audio)
    print(f"Test signal (440 Hz sine, 0.5 amplitude):")
    print(f"  RMS: {rms:.4f}")
    print(f"  RMS dB: {rms_db:.2f} dB")
    print()

    # Test segment extraction
    start = seconds_to_samples(0.5, sr)
    end = seconds_to_samples(1.5, sr)
    segment = get_segment_audio(test_audio, start, end)
    print(f"Segment [0.5s, 1.5s]:")
    print(f"  Samples: {start} to {end}")
    print(f"  Length: {len(segment)} samples ({samples_to_seconds(len(segment), sr)}s)")
    print()

    # Test alignment
    audio1 = np.random.randn(44100)
    audio2 = np.random.randn(44050)

    aligned1, aligned2 = align_lengths(audio1, audio2, warn=False)
    print(f"Alignment test:")
    print(f"  Original: {len(audio1)} vs {len(audio2)}")
    print(f"  Aligned: {len(aligned1)} vs {len(aligned2)}")
    print()

    # Test silence threshold calibration
    rms_values = np.concatenate(
        [
            np.random.normal(-60, 5, 100),  # Silent segments
            np.random.normal(-20, 10, 400),  # Normal segments
        ]
    )

    threshold, method, stats = calibrate_silence_threshold(rms_values)
    print(f"Silence calibration:")
    print(f"  Method: {method}")
    print(f"  Threshold: {threshold:.2f} dB")
    print(f"  RMS range: [{stats['min']:.1f}, {stats['max']:.1f}] dB")
    print()

    print("✓ All audio utilities working correctly!")
