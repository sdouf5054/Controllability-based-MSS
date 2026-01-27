"""
FLX4-Net Track A Beat Synchrony Cue b(t)

Measures how well vocal onsets align with detected beats.
High synchrony suggests the vocal follows the rhythm closely.

Blueprint Policy:
    - Tolerance: ±70ms for onset-beat alignment
    - Unavailable if segment has < 2 beats
    - Uses librosa onset detection on separated vocal

Interpretation:
    - High b(t): Vocals are rhythmically aligned with beats
                 → Easier to beatmatch/control
    - Low b(t):  Vocals are off-beat or rhythmically free
                 → Harder to predict/control
    - NaN:       Not enough beats in segment (< 2)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


def detect_vocal_onsets(
    audio: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    backtrack: bool = True
) -> np.ndarray:
    """Detect onsets in vocal audio.
    
    Args:
        audio: Vocal audio signal (1D)
        sr: Sample rate
        hop_length: Hop length for onset detection
        backtrack: Whether to backtrack to nearest local minimum
        
    Returns:
        Array of onset times in seconds
    """
    import librosa
    
    # Check for silent audio
    if np.max(np.abs(audio)) < 1e-6:
        return np.array([])
    
    # Compute onset strength
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=hop_length
    )
    
    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack
    )
    
    # Convert to times
    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sr,
        hop_length=hop_length
    )
    
    return onset_times


def get_beats_in_segment(
    beat_times: np.ndarray,
    start_sec: float,
    end_sec: float
) -> np.ndarray:
    """Get beat times that fall within a segment.
    
    Args:
        beat_times: Array of beat times (seconds)
        start_sec: Segment start time
        end_sec: Segment end time
        
    Returns:
        Array of beat times within [start_sec, end_sec)
    """
    mask = (beat_times >= start_sec) & (beat_times < end_sec)
    return beat_times[mask]


def compute_onset_beat_alignment(
    onset_times: np.ndarray,
    beat_times: np.ndarray,
    tolerance_sec: float = 0.070
) -> float:
    """Compute alignment score between onsets and beats.
    
    For each onset, check if there's a beat within tolerance.
    Score = ratio of aligned onsets to total onsets.
    
    Args:
        onset_times: Array of onset times (seconds)
        beat_times: Array of beat times (seconds)
        tolerance_sec: Alignment tolerance (default: ±70ms)
        
    Returns:
        Alignment score in [0, 1]
    """
    if len(onset_times) == 0:
        return 0.0
    
    if len(beat_times) == 0:
        return 0.0
    
    aligned_count = 0
    
    for onset in onset_times:
        # Find closest beat
        distances = np.abs(beat_times - onset)
        min_distance = np.min(distances)
        
        if min_distance <= tolerance_sec:
            aligned_count += 1
    
    return aligned_count / len(onset_times)


def compute_beat_sync_cue(
    vocal_audio: np.ndarray,
    beat_times: np.ndarray,
    start_sec: float,
    end_sec: float,
    sr: int = 44100,
    tolerance_sec: float = 0.070,
    min_beats: int = 2,
    min_onsets: int = 1
) -> Tuple[Optional[float], bool, Dict]:
    """Compute beat synchrony cue b(t) for a segment.
    
    Args:
        vocal_audio: Segment vocal audio
        beat_times: Beat times for the full track
        start_sec: Segment start time
        end_sec: Segment end time
        sr: Sample rate
        tolerance_sec: Alignment tolerance
        min_beats: Minimum beats required
        min_onsets: Minimum onsets required
        
    Returns:
        Tuple of:
            - b_value: Beat sync score (None if unavailable)
            - available: Whether b(t) was computable
            - info: Dictionary with debug info
    """
    info = {}
    
    # Get beats in segment
    segment_beats = get_beats_in_segment(beat_times, start_sec, end_sec)
    info['n_beats'] = len(segment_beats)
    
    # Check beat availability
    if len(segment_beats) < min_beats:
        info['unavailable_reason'] = f"too_few_beats ({len(segment_beats)} < {min_beats})"
        return None, False, info
    
    # Detect onsets in vocal
    onset_times_abs = detect_vocal_onsets(vocal_audio, sr=sr)
    
    # Convert to absolute times (relative to track start)
    onset_times = onset_times_abs + start_sec
    info['n_onsets'] = len(onset_times)
    
    # Check onset availability
    if len(onset_times) < min_onsets:
        info['unavailable_reason'] = f"too_few_onsets ({len(onset_times)} < {min_onsets})"
        # Still return a value (0) but mark as available
        # since this is a valid measurement (no onsets = no sync)
        return 0.0, True, info
    
    # Compute alignment
    b_value = compute_onset_beat_alignment(
        onset_times=onset_times,
        beat_times=segment_beats,
        tolerance_sec=tolerance_sec
    )
    
    info['alignment_score'] = b_value
    
    return b_value, True, info


class BeatSyncComputer:
    """Computes beat synchrony cue b(t) for track segments.
    
    Usage:
        computer = BeatSyncComputer(tolerance_sec=0.070, min_beats=2)
        
        # Set track data
        computer.set_track_data(vocal_audio, beat_times)
        
        # Compute for segment
        b_value, available, info = computer.compute_segment(
            start_sec, end_sec, start_sample, end_sample
        )
    """
    
    def __init__(
        self,
        sr: int = 44100,
        tolerance_sec: float = 0.070,
        min_beats: int = 2,
        min_onsets: int = 1
    ):
        """Initialize BeatSyncComputer.
        
        Args:
            sr: Sample rate
            tolerance_sec: Alignment tolerance (Blueprint: 70ms)
            min_beats: Minimum beats per segment
            min_onsets: Minimum onsets per segment
        """
        self.sr = sr
        self.tolerance_sec = tolerance_sec
        self.min_beats = min_beats
        self.min_onsets = min_onsets
        
        # Track data
        self._vocal_audio: Optional[np.ndarray] = None
        self._beat_times: Optional[np.ndarray] = None
    
    def set_track_data(
        self,
        vocal_audio: np.ndarray,
        beat_times: np.ndarray
    ):
        """Set track-level data.
        
        Args:
            vocal_audio: Full track vocal audio
            beat_times: Beat times for full track
        """
        self._vocal_audio = vocal_audio
        self._beat_times = np.array(beat_times)
    
    def compute_segment(
        self,
        start_sec: float,
        end_sec: float,
        start_sample: int,
        end_sample: int
    ) -> Tuple[Optional[float], bool, Dict]:
        """Compute b(t) for a segment.
        
        Args:
            start_sec: Segment start time
            end_sec: Segment end time
            start_sample: Segment start sample
            end_sample: Segment end sample
            
        Returns:
            Tuple of (b_value, available, info)
        """
        if self._vocal_audio is None or self._beat_times is None:
            raise RuntimeError("Track data not set. Call set_track_data() first.")
        
        # Extract segment audio
        segment_audio = self._vocal_audio[start_sample:end_sample]
        
        return compute_beat_sync_cue(
            vocal_audio=segment_audio,
            beat_times=self._beat_times,
            start_sec=start_sec,
            end_sec=end_sec,
            sr=self.sr,
            tolerance_sec=self.tolerance_sec,
            min_beats=self.min_beats,
            min_onsets=self.min_onsets
        )
    
    def compute_track(
        self,
        segments: List[dict]
    ) -> Tuple[List[Optional[float]], List[bool], List[Dict]]:
        """Compute b(t) for all segments in a track.
        
        Args:
            segments: List of segment dicts
            
        Returns:
            Tuple of (b_values, availabilities, infos)
        """
        b_values = []
        availabilities = []
        infos = []
        
        for seg in segments:
            b_val, avail, info = self.compute_segment(
                start_sec=seg['start_sec'],
                end_sec=seg['end_sec'],
                start_sample=seg['start_sample'],
                end_sample=seg['end_sample']
            )
            b_values.append(b_val)
            availabilities.append(avail)
            infos.append(info)
        
        return b_values, availabilities, infos


if __name__ == "__main__":
    print("=== Beat Sync Cue Test ===\n")
    
    import librosa
    
    # Create synthetic test
    sr = 44100
    duration = 3.0
    
    # Create beat times at 120 BPM (0.5s intervals)
    beat_times = np.arange(0, duration, 0.5)  # [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    print(f"Beat times: {beat_times}")
    print(f"BPM: 120")
    
    # Test 1: Onsets aligned with beats
    print("\n--- Test 1: Aligned onsets ---")
    onset_times_aligned = np.array([0.02, 0.52, 1.01, 1.48, 2.03])  # Within ±70ms of beats
    b_aligned = compute_onset_beat_alignment(onset_times_aligned, beat_times, tolerance_sec=0.070)
    print(f"Onset times: {onset_times_aligned}")
    print(f"b(t) = {b_aligned:.3f} (expected: ~1.0)")
    
    # Test 2: Onsets off-beat
    print("\n--- Test 2: Off-beat onsets ---")
    onset_times_off = np.array([0.25, 0.75, 1.25, 1.75, 2.25])  # Between beats
    b_off = compute_onset_beat_alignment(onset_times_off, beat_times, tolerance_sec=0.070)
    print(f"Onset times: {onset_times_off}")
    print(f"b(t) = {b_off:.3f} (expected: ~0.0)")
    
    # Test 3: Mixed onsets
    print("\n--- Test 3: Mixed onsets ---")
    onset_times_mixed = np.array([0.02, 0.25, 1.01, 1.75, 2.03])  # 3 aligned, 2 off
    b_mixed = compute_onset_beat_alignment(onset_times_mixed, beat_times, tolerance_sec=0.070)
    print(f"Onset times: {onset_times_mixed}")
    print(f"b(t) = {b_mixed:.3f} (expected: ~0.6)")
    
    # Test 4: Full pipeline with synthetic audio
    print("\n--- Test 4: Full pipeline ---")
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create audio with impulses at beat times
    audio = np.zeros_like(t)
    for beat in beat_times:
        idx = int(beat * sr)
        if idx < len(audio):
            # Add a short click
            click_len = int(0.01 * sr)
            audio[idx:idx+click_len] = np.sin(2 * np.pi * 1000 * np.arange(click_len) / sr) * 0.5
    
    # Test with segment
    b_val, avail, info = compute_beat_sync_cue(
        vocal_audio=audio,
        beat_times=beat_times,
        start_sec=0.0,
        end_sec=3.0,
        sr=sr
    )
    
    print(f"Segment [0, 3]s:")
    print(f"  n_beats: {info['n_beats']}")
    print(f"  n_onsets: {info['n_onsets']}")
    print(f"  available: {avail}")
    print(f"  b(t) = {b_val:.3f}" if b_val is not None else "  b(t) = N/A")
    
    print("\n✓ Beat sync module ready!")
