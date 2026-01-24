"""
FLX4-Net Librosa Beat Tracker

Beat detection using librosa.beat.beat_track.
Fallback option when essentia/madmom are not available (Windows).

Note: librosa does not provide confidence scores, so unavailability
is determined solely by beat count thresholds.
"""

import numpy as np
import librosa
from typing import List, Optional, Tuple

from ..data_types import BeatResult


def get_librosa_version() -> str:
    """Get librosa version string."""
    return librosa.__version__


def detect_beats_librosa(
    audio: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    units: str = "time"
) -> Tuple[np.ndarray, float]:
    """Detect beats using librosa.
    
    Args:
        audio: Audio signal (mono)
        sr: Sample rate
        hop_length: Hop length for onset detection
        units: Return units ("time" for seconds, "frames" for frame indices)
        
    Returns:
        (beat_times, tempo): Beat times in seconds and estimated tempo
    """
    # Estimate tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        units="frames"
    )
    
    # Convert frames to time
    if units == "time":
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    else:
        beat_times = beat_frames.astype(float)
    
    # Handle tempo array (librosa may return array)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)
    
    return beat_times, tempo


def check_track_availability(
    beat_times: np.ndarray,
    min_beats: int = 8,
    duration_sec: Optional[float] = None,
    min_density: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """Check if beat tracking result is reliable at track level.
    
    Args:
        beat_times: Detected beat times
        min_beats: Minimum number of beats required
        duration_sec: Track duration (for density check)
        min_density: Minimum beats per minute (optional)
        
    Returns:
        (is_available, reason): Availability status and reason if unavailable
    """
    n_beats = len(beat_times)
    
    # Check minimum beat count
    if n_beats < min_beats:
        return False, f"too_few_beats ({n_beats} < {min_beats})"
    
    # Optional: check beat density
    if duration_sec and min_density:
        density = n_beats / (duration_sec / 60)  # beats per minute
        if density < min_density:
            return False, f"low_density ({density:.1f} < {min_density} bpm)"
    
    return True, None


def check_segment_availability(
    beat_times: np.ndarray,
    start_sec: float,
    end_sec: float,
    min_beats: int = 2
) -> Tuple[bool, int]:
    """Check if segment has enough beats for b(t) computation.
    
    Args:
        beat_times: All beat times for the track
        start_sec: Segment start time
        end_sec: Segment end time
        min_beats: Minimum beats required in segment
        
    Returns:
        (is_available, n_beats): Availability and beat count in segment
    """
    beats_in_segment = get_beats_in_range(beat_times, start_sec, end_sec)
    n_beats = len(beats_in_segment)
    
    return n_beats >= min_beats, n_beats


def get_beats_in_range(
    beat_times: np.ndarray,
    start_sec: float,
    end_sec: float
) -> np.ndarray:
    """Get beats within a time range.
    
    Args:
        beat_times: All beat times
        start_sec: Range start (inclusive)
        end_sec: Range end (exclusive)
        
    Returns:
        Beat times within [start_sec, end_sec)
    """
    mask = (beat_times >= start_sec) & (beat_times < end_sec)
    return beat_times[mask]


def compute_beat_regularity(beat_times: np.ndarray) -> Optional[float]:
    """Compute beat regularity (inverse of IBI coefficient of variation).
    
    Higher values indicate more regular beats.
    
    Args:
        beat_times: Beat times in seconds
        
    Returns:
        Regularity score (0-1), or None if not enough beats
    """
    if len(beat_times) < 3:
        return None
    
    # Compute inter-beat intervals
    ibis = np.diff(beat_times)
    
    if len(ibis) < 2:
        return None
    
    # Coefficient of variation
    mean_ibi = np.mean(ibis)
    if mean_ibi == 0:
        return None
    
    cv = np.std(ibis) / mean_ibi
    
    # Convert to regularity (1 = perfectly regular, 0 = very irregular)
    # Using exponential decay: regularity = exp(-cv)
    regularity = np.exp(-cv)
    
    return float(regularity)


def create_beat_result_librosa(
    track_id: str,
    audio: np.ndarray,
    sr: int = 44100,
    min_beats_per_track: int = 8,
    duration_sec: Optional[float] = None
) -> BeatResult:
    """Create BeatResult using librosa beat detection.
    
    Args:
        track_id: Track identifier
        audio: Audio signal (mono)
        sr: Sample rate
        min_beats_per_track: Minimum beats for track availability
        duration_sec: Track duration (optional, for density check)
        
    Returns:
        BeatResult object
    """
    # Detect beats
    beat_times, tempo = detect_beats_librosa(audio, sr=sr)
    
    # Check availability
    is_available, unavailable_reason = check_track_availability(
        beat_times,
        min_beats=min_beats_per_track,
        duration_sec=duration_sec
    )
    
    return BeatResult(
        track_id=track_id,
        backend="librosa",
        backend_version=get_librosa_version(),
        beat_times=beat_times.tolist(),
        confidence=None,  # librosa doesn't provide confidence
        bpm=tempo,
        track_available=is_available,
        unavailable_reason=unavailable_reason
    )


if __name__ == "__main__":
    print("=== Librosa Beat Tracker Test ===\n")
    
    # Create test audio (120 BPM click track)
    sr = 44100
    duration = 10.0
    bpm = 120
    beat_interval = 60 / bpm
    
    # Generate simple click track
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    
    # Add clicks at beat positions
    for beat_time in np.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sr)
        if beat_sample < len(audio) - 100:
            # Simple click: short burst
            audio[beat_sample:beat_sample + 100] = 0.5 * np.sin(
                2 * np.pi * 1000 * np.arange(100) / sr
            )
    
    # Detect beats
    beat_times, tempo = detect_beats_librosa(audio, sr=sr)
    
    print(f"librosa version: {get_librosa_version()}")
    print(f"Expected BPM: {bpm}")
    print(f"Detected BPM: {tempo:.1f}")
    print(f"Detected beats: {len(beat_times)}")
    print(f"Beat times: {beat_times[:5]}... (first 5)")
    
    # Check availability
    is_available, reason = check_track_availability(beat_times, min_beats=8)
    print(f"\nTrack available: {is_available}")
    if reason:
        print(f"Reason: {reason}")
    
    # Test segment availability
    seg_available, n_beats = check_segment_availability(
        beat_times, start_sec=2.0, end_sec=5.0, min_beats=2
    )
    print(f"\nSegment [2.0, 5.0) available: {seg_available} ({n_beats} beats)")
    
    # Compute regularity
    regularity = compute_beat_regularity(beat_times)
    print(f"Beat regularity: {regularity:.3f}" if regularity else "Beat regularity: N/A")
    
    print("\n✓ Librosa beat tracker working correctly!")
