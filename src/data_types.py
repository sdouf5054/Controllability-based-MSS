"""
FLX4-Net Track A Data Types

Core data structures used throughout the pipeline.
All structures are dataclasses for type safety and IDE support.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class Segment:
    """A single audio segment for analysis.
    
    Represents a time window in a track, used as the basic unit
    for all Track A computations.
    
    Attributes:
        seg_id: Unique identifier (format: "{track_id}_{seg_idx:04d}")
        track_id: Parent track identifier
        seg_idx: Segment index within track (0-based)
        start_sec: Start time in seconds
        end_sec: End time in seconds
        start_sample: Start sample index
        end_sample: End sample index
        fold_id: Cross-validation fold (track-level)
        gt_vocal_rms_db: GT vocal RMS in dB (for silence detection)
        is_silent: Whether segment is flagged as silent
    """
    seg_id: str
    track_id: str
    seg_idx: int
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int
    fold_id: int
    gt_vocal_rms_db: float = 0.0
    is_silent: bool = False
    
    @property
    def duration_sec(self) -> float:
        """Segment duration in seconds."""
        return self.end_sec - self.start_sec
    
    @property
    def duration_samples(self) -> int:
        """Segment duration in samples."""
        return self.end_sample - self.start_sample
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class TrackInfo:
    """Information about a single track in the dataset.
    
    Attributes:
        track_id: Unique track identifier
        mixture_path: Path to mixture audio file
        gt_vocal_path: Path to ground truth vocal stem
        gt_other_paths: Optional paths to other GT stems
        duration_sec: Track duration in seconds
        duration_samples: Track duration in samples
        fold_id: Cross-validation fold assignment
    """
    track_id: str
    mixture_path: str
    gt_vocal_path: str
    duration_sec: float
    duration_samples: int
    fold_id: int
    gt_other_paths: Optional[dict] = None  # e.g., {"drums": path, "bass": path}
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "TrackInfo":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class BeatResult:
    """Beat tracking result for a single track.
    
    Stores beat times and availability information.
    Serializable to JSON for caching.
    
    Attributes:
        track_id: Track identifier
        backend: Beat tracker used ("essentia" | "madmom" | "librosa")
        backend_version: Version string for reproducibility
        beat_times: List of beat times in seconds
        confidence: Rhythm confidence (Essentia only, None for others)
        bpm: Estimated BPM
        track_available: Whether beats are reliable at track level
        unavailable_reason: Reason if track_available=False
    """
    track_id: str
    backend: str
    backend_version: str
    beat_times: List[float]
    confidence: Optional[float]
    bpm: Optional[float]
    track_available: bool
    unavailable_reason: Optional[str] = None
    
    @property
    def n_beats(self) -> int:
        """Number of detected beats."""
        return len(self.beat_times)
    
    def get_beats_in_range(self, start_sec: float, end_sec: float) -> List[float]:
        """Get beats within a time range.
        
        Args:
            start_sec: Range start
            end_sec: Range end
            
        Returns:
            List of beat times within [start_sec, end_sec)
        """
        return [b for b in self.beat_times if start_sec <= b < end_sec]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "BeatResult":
        """Create from dictionary."""
        return cls(**d)
    
    def save_json(self, path: str | Path) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, path: str | Path) -> "BeatResult":
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class DifficultyResult:
    """Segment-level difficulty metrics.
    
    Attributes:
        seg_id: Segment identifier
        sisdr_vocal: Scale-Invariant SDR (Primary metric)
        sdr_vocal: Standard SDR (Reference metric)
        valid: Whether computation was successful
        error_msg: Error message if valid=False
    """
    seg_id: str
    sisdr_vocal: float
    sdr_vocal: float
    valid: bool = True
    error_msg: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "DifficultyResult":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class ProxyResult:
    """Artifact proxy measurements for a segment.
    
    These are computed without GT, used to estimate
    separation difficulty in Track B.
    
    Attributes:
        seg_id: Segment identifier
        leakage_lowfreq: Low-frequency leakage proxy (50-150 Hz energy ratio)
        instability: Adjacent segment dissimilarity
        transient_leakage: Percussive/transient energy ratio
        valid: Whether computation was successful
    """
    seg_id: str
    leakage_lowfreq: float
    instability: float
    transient_leakage: float
    valid: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "ProxyResult":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class CueResult:
    """Cue measurements for a segment.
    
    The four controllability cues computed from vocal audio.
    
    Attributes:
        seg_id: Segment identifier
        r: Repetition cue (short-term similarity, past 12s)
        b: Beat synchrony cue (onset-beat alignment), NaN if unavailable
        s: Structure cue (long-range similarity, >= 30s)
        p: Processing cue (texture signature)
        beat_available: Whether b(t) was computable
        r_candidate_count: Number of candidates considered for r
        s_candidate_count: Number of candidates considered for s
    """
    seg_id: str
    r: float
    b: Optional[float]  # None = NaN (beat unavailable)
    s: float
    p: float
    beat_available: bool
    r_candidate_count: int = 0
    s_candidate_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "CueResult":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class SilenceCalibrationResult:
    """Result of silence threshold calibration.
    
    Attributes:
        initial_threshold_db: Initial threshold used
        final_threshold_db: Calibrated threshold
        calibration_method: Method used ("gap" | "5th_percentile" | "fixed")
        n_segments_total: Total number of segments
        n_segments_silent: Number of segments flagged silent
        silent_ratio: Ratio of silent segments
        rms_stats: RMS distribution statistics
    """
    initial_threshold_db: float
    final_threshold_db: float
    calibration_method: str
    n_segments_total: int
    n_segments_silent: int
    silent_ratio: float
    rms_stats: dict = field(default_factory=dict)  # min, max, mean, median, std, percentiles
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass 
class ProxyAssociation:
    """Association between a proxy and difficulty anchor.
    
    Attributes:
        proxy_name: Name of the proxy
        spearman_rho: Spearman correlation coefficient
        p_value: P-value for the correlation
        n_samples: Number of valid samples used
    """
    proxy_name: str
    spearman_rho: float
    p_value: float
    n_samples: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CueContamination:
    """Contamination analysis for a single cue.
    
    Compares cue computed from estimated vocal vs GT vocal.
    
    Attributes:
        cue_name: Name of the cue (r, b, s, p)
        correlation: Pearson correlation between est and gt
        mean_diff: Mean difference (est - gt)
        std_diff: Std of differences
        n_samples: Number of valid samples
    """
    cue_name: str
    correlation: float
    mean_diff: float
    std_diff: float
    n_samples: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BeatCoverage:
    """Beat tracking coverage statistics.
    
    Attributes:
        n_tracks_total: Total number of tracks
        n_tracks_available: Tracks with reliable beats
        n_tracks_unavailable: Tracks with unreliable/missing beats
        track_coverage_ratio: Ratio of available tracks
        n_segments_total: Total segments
        n_segments_beat_available: Segments where b(t) was computed
        segment_coverage_ratio: Ratio of segments with b(t)
        unavailable_reasons: Count of each unavailability reason
    """
    n_tracks_total: int
    n_tracks_available: int
    n_tracks_unavailable: int
    track_coverage_ratio: float
    n_segments_total: int
    n_segments_beat_available: int
    segment_coverage_ratio: float
    unavailable_reasons: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrackASummary:
    """Complete summary of Track A results.
    
    This is what gets saved to summary.json.
    """
    # Dataset info
    dataset_name: str
    n_tracks: int
    n_segments_total: int
    n_segments_valid: int
    
    # Silence info
    silence_info: SilenceCalibrationResult
    
    # Difficulty distribution
    difficulty_stats: dict  # mean, std, quartiles, etc.
    
    # Proxy associations
    proxy_associations: List[ProxyAssociation]
    
    # Cue contamination
    cue_contaminations: List[CueContamination]
    
    # Beat coverage
    beat_coverage: BeatCoverage
    
    # Config snapshot (for reproducibility)
    config_snapshot: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "n_tracks": self.n_tracks,
            "n_segments_total": self.n_segments_total,
            "n_segments_valid": self.n_segments_valid,
            "silence_info": self.silence_info.to_dict(),
            "difficulty_stats": self.difficulty_stats,
            "proxy_associations": [p.to_dict() for p in self.proxy_associations],
            "cue_contaminations": [c.to_dict() for c in self.cue_contaminations],
            "beat_coverage": self.beat_coverage.to_dict(),
            "config_snapshot": self.config_snapshot
        }
    
    def save_json(self, path: str | Path) -> None:
        """Save summary to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Test data types
    print("=== Testing Data Types ===\n")
    
    # Test Segment
    seg = Segment(
        seg_id="track001_0005",
        track_id="track001",
        seg_idx=5,
        start_sec=7.5,
        end_sec=10.5,
        start_sample=330750,
        end_sample=463050,
        fold_id=0,
        gt_vocal_rms_db=-25.3,
        is_silent=False
    )
    print(f"Segment: {seg.seg_id}")
    print(f"  Duration: {seg.duration_sec}s ({seg.duration_samples} samples)")
    print(f"  Silent: {seg.is_silent}")
    print()
    
    # Test BeatResult
    beat = BeatResult(
        track_id="track001",
        backend="essentia",
        backend_version="2.1_beta5",
        beat_times=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        confidence=0.85,
        bpm=120.0,
        track_available=True
    )
    print(f"BeatResult: {beat.track_id}")
    print(f"  Backend: {beat.backend} v{beat.backend_version}")
    print(f"  Beats: {beat.n_beats}, BPM: {beat.bpm}")
    print(f"  Available: {beat.track_available}")
    print(f"  Beats in [1.0, 3.0): {beat.get_beats_in_range(1.0, 3.0)}")
    print()
    
    # Test CueResult
    cue = CueResult(
        seg_id="track001_0005",
        r=0.75,
        b=0.62,
        s=0.45,
        p=0.30,
        beat_available=True,
        r_candidate_count=8,
        s_candidate_count=15
    )
    print(f"CueResult: {cue.seg_id}")
    print(f"  r={cue.r:.2f}, b={cue.b:.2f}, s={cue.s:.2f}, p={cue.p:.2f}")
    print(f"  Beat available: {cue.beat_available}")
    print()
    
    print("✓ All data types working correctly!")
