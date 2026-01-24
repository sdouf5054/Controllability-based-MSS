"""
FLX4-Net Track A Source Package

Core modules for the Track A analysis pipeline.
"""

from .config import TrackAConfig, load_config
from .data_types import (
    Segment,
    TrackInfo,
    BeatResult,
    DifficultyResult,
    ProxyResult,
    CueResult,
    SilenceCalibrationResult,
    ProxyAssociation,
    CueContamination,
    BeatCoverage,
    TrackASummary,
)
from .audio_utils import (
    load_audio,
    compute_rms,
    compute_rms_db,
    align_lengths,
    get_segment_audio,
    seconds_to_samples,
    samples_to_seconds,
    get_audio_duration,
    calibrate_silence_threshold,
)
from .separation import (
    separate_track,
    load_separated_vocal,
    align_separated_with_gt,
    get_cache_path,
    is_cached,
    get_separation_stats,
)
from .beat_tracking import (
    run_beat_tracking,
    load_beat_result,
    save_beat_result,
    get_beat_cache_path,
    is_beat_cached,
)
from .segmentation import (
    create_segments_for_track,
    create_segment_index,
    compute_segment_rms,
    add_silence_flags,
    calibrate_and_flag_silence,
    save_segments,
    load_segments,
    print_segment_summary,
)

__all__ = [
    # Config
    "TrackAConfig",
    "load_config",
    # Data types
    "Segment",
    "TrackInfo",
    "BeatResult",
    "DifficultyResult",
    "ProxyResult",
    "CueResult",
    "SilenceCalibrationResult",
    "ProxyAssociation",
    "CueContamination",
    "BeatCoverage",
    "TrackASummary",
    # Audio utils
    "load_audio",
    "compute_rms",
    "compute_rms_db",
    "align_lengths",
    "get_segment_audio",
    "seconds_to_samples",
    "samples_to_seconds",
    "get_audio_duration",
    "calibrate_silence_threshold",
]
