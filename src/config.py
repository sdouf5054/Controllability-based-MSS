"""
FLX4-Net Track A Configuration Module

Loads and validates configuration from YAML file.
Provides typed access to all configuration parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import yaml


@dataclass
class PathsConfig:
    """Path configuration."""
    musdb_root: str
    medleyvox_root: str = "./Dataset/MedleyVox"
    mss_training_root: str = "./Music-Source-Separation-Training"
    output_root: str = "./outputs"
    cache_root: str = "./cache"
    metadata_root: str = "./metadata"
    tables_root: str = "./tables"
    results_root: str = "./results"
    
    def __post_init__(self):
        """Convert to Path objects and create directories."""
        self.musdb_root = Path(self.musdb_root)
        self.medleyvox_root = Path(self.medleyvox_root)
        self.mss_training_root = Path(self.mss_training_root)
        self.output_root = Path(self.output_root)
        self.cache_root = Path(self.cache_root)
        self.metadata_root = Path(self.metadata_root)
        self.tables_root = Path(self.tables_root)
        self.results_root = Path(self.results_root)
    
    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        for path in [self.output_root, self.cache_root, self.metadata_root, 
                     self.tables_root, self.results_root]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.cache_root / "separated").mkdir(exist_ok=True)
        (self.cache_root / "beat_tracking").mkdir(exist_ok=True)
        (self.results_root / "trackA").mkdir(exist_ok=True)
        (self.results_root / "trackA" / "figures").mkdir(exist_ok=True)


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    mono: bool = True


@dataclass
class SegmentationConfig:
    """Segmentation configuration.
    
    Fixed policy: Window=3.0s, Hop=1.5s (50% overlap)
    """
    window_sec: float = 3.0
    hop_sec: float = 1.5
    
    @property
    def window_samples(self) -> int:
        """Window size in samples (at 44100 Hz)."""
        return int(self.window_sec * 44100)
    
    @property
    def hop_samples(self) -> int:
        """Hop size in samples (at 44100 Hz)."""
        return int(self.hop_sec * 44100)


@dataclass
class SilenceConfig:
    """Silence detection configuration.
    
    Policy: flag_and_separate
    - Primary analysis: is_silent=False only
    - Supplementary: is_silent=True separately
    """
    initial_threshold_db: float = -40.0
    calibration_method: Literal["gap_or_5th_percentile", "fixed"] = "gap_or_5th_percentile"
    percentile_cutoff: float = 5.0


@dataclass
class SeparatorConfig:
    """Source separation configuration.
    
    Uses Music-Source-Separation-Training for inference.
    """
    name: str = "mss_training"
    model_type: str = "htdemucs"  # htdemucs, mel_band_roformer, bs_roformer 등
    config_path: str = "./Music-Source-Separation-Training/configs/config_musdb18_htdemucs.yaml"
    checkpoint_path: Optional[str] = None  # None이면 기본 pretrained 모델
    output_alignment: Literal["truncate_to_shorter", "pad_to_longer"] = "truncate_to_shorter"
    max_length_diff_samples: int = 1000  # ~23ms at 44.1kHz


@dataclass
class EssentiaConfig:
    """Essentia-specific beat tracking config."""
    min_confidence: float = 0.1


@dataclass
class MadmomConfig:
    """Madmom-specific beat tracking config."""
    min_beats_per_minute: float = 0.5
    max_ibi_cv: float = 0.5


@dataclass
class BeatTrackingConfig:
    """Beat tracking configuration."""
    backend: Literal["essentia", "madmom", "librosa"] = "essentia"
    min_beats_per_track: int = 8
    min_beats_per_segment: int = 2
    essentia: EssentiaConfig = field(default_factory=EssentiaConfig)
    madmom: MadmomConfig = field(default_factory=MadmomConfig)


@dataclass
class SimilarityConfig:
    """Similarity computation configuration for cues."""
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 8000
    pooling: Literal["mean", "median"] = "mean"
    normalize: Literal["l2", "none"] = "l2"
    metric: Literal["cosine"] = "cosine"


@dataclass
class RepetitionConfig:
    """Repetition cue r(t) configuration.
    
    Short-term: only looks at past 12 seconds.
    """
    max_lookback_sec: float = 12.0
    aggregation: Literal["max", "top_k_mean"] = "max"
    top_k: int = 3


@dataclass
class StructureConfig:
    """Structure cue s(t) configuration.
    
    Long-range: only compares segments >= 30 seconds apart.
    """
    min_distance_sec: float = 30.0
    aggregation: Literal["top_k_mean"] = "top_k_mean"
    top_k: int = 3


@dataclass
class BeatSyncConfig:
    """Beat synchrony cue b(t) configuration."""
    tolerance_sec: float = 0.070  # ±70ms


@dataclass
class ProcessingConfig:
    """Processing cue p(t) configuration."""
    enabled: bool = True


@dataclass
class CuesConfig:
    """All cues configuration."""
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    beat_sync: BeatSyncConfig = field(default_factory=BeatSyncConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


@dataclass
class MetricsConfig:
    """Difficulty metrics configuration."""
    primary: Literal["sisdr"] = "sisdr"
    compute_sdr: bool = True


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    correlation_method: Literal["spearman", "pearson"] = "spearman"
    figure_format: str = "png"
    figure_dpi: int = 150


@dataclass
class TrackAConfig:
    """Complete Track A configuration.
    
    Usage:
        config = TrackAConfig.from_yaml("configs/trackA_config.yaml")
        print(config.segmentation.window_sec)  # 3.0
        print(config.cues.repetition.max_lookback_sec)  # 12.0
    """
    paths: PathsConfig
    audio: AudioConfig
    segmentation: SegmentationConfig
    silence: SilenceConfig
    separator: SeparatorConfig
    beat_tracking: BeatTrackingConfig
    cues: CuesConfig
    metrics: MetricsConfig
    analysis: AnalysisConfig
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackAConfig":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            TrackAConfig instance
        """
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        
        # Parse nested configs
        paths = PathsConfig(**raw.get('paths', {}))
        audio = AudioConfig(**raw.get('audio', {}))
        segmentation = SegmentationConfig(**raw.get('segmentation', {}))
        silence = SilenceConfig(**raw.get('silence', {}))
        separator = SeparatorConfig(**raw.get('separator', {}))
        
        # Beat tracking with nested configs
        bt_raw = raw.get('beat_tracking', {})
        beat_tracking = BeatTrackingConfig(
            backend=bt_raw.get('backend', 'essentia'),
            min_beats_per_track=bt_raw.get('min_beats_per_track', 8),
            min_beats_per_segment=bt_raw.get('min_beats_per_segment', 2),
            essentia=EssentiaConfig(**bt_raw.get('essentia', {})),
            madmom=MadmomConfig(**bt_raw.get('madmom', {}))
        )
        
        # Cues with nested configs
        cues_raw = raw.get('cues', {})
        cues = CuesConfig(
            similarity=SimilarityConfig(**cues_raw.get('similarity', {})),
            repetition=RepetitionConfig(**cues_raw.get('repetition', {})),
            structure=StructureConfig(**cues_raw.get('structure', {})),
            beat_sync=BeatSyncConfig(**cues_raw.get('beat_sync', {})),
            processing=ProcessingConfig(**cues_raw.get('processing', {}))
        )
        
        metrics = MetricsConfig(**raw.get('metrics', {}))
        analysis = AnalysisConfig(**raw.get('analysis', {}))
        
        return cls(
            paths=paths,
            audio=audio,
            segmentation=segmentation,
            silence=silence,
            separator=separator,
            beat_tracking=beat_tracking,
            cues=cues,
            metrics=metrics,
            analysis=analysis
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (for logging/saving)."""
        import dataclasses
        
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        return _to_dict(self)


# Convenience function
def load_config(path: str | Path = "configs/trackA_config.yaml") -> TrackAConfig:
    """Load Track A configuration from YAML file.
    
    Args:
        path: Path to config file (default: configs/trackA_config.yaml)
        
    Returns:
        TrackAConfig instance
    """
    return TrackAConfig.from_yaml(path)


if __name__ == "__main__":
    # Test loading config
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/trackA_config.yaml"
    
    try:
        config = load_config(config_path)
        print("✓ Config loaded successfully!")
        print(f"\n=== Key Settings ===")
        print(f"MUSDB root: {config.paths.musdb_root}")
        print(f"Sample rate: {config.audio.sample_rate}")
        print(f"Segment: {config.segmentation.window_sec}s window, {config.segmentation.hop_sec}s hop")
        print(f"Silence threshold: {config.silence.initial_threshold_db} dB")
        print(f"Beat backend: {config.beat_tracking.backend}")
        print(f"r(t) lookback: {config.cues.repetition.max_lookback_sec}s")
        print(f"s(t) min distance: {config.cues.structure.min_distance_sec}s")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        raise
