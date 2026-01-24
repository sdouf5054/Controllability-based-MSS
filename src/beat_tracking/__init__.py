"""
FLX4-Net Beat Tracking Module

Provides beat tracking functionality with multiple backends.
Currently using librosa (essentia/madmom not supported on Windows).
"""

from .librosa_tracker import (
    detect_beats_librosa,
    check_track_availability,
    check_segment_availability,
    get_beats_in_range,
)
from .tracker import (
    run_beat_tracking,
    load_beat_result,
    save_beat_result,
    get_beat_cache_path,
    is_beat_cached,
    get_beat_tracking_stats,
)

__all__ = [
    # Librosa tracker
    "detect_beats_librosa",
    "check_track_availability",
    "check_segment_availability",
    "get_beats_in_range",
    # Main tracker interface
    "run_beat_tracking",
    "load_beat_result",
    "save_beat_result",
    "get_beat_cache_path",
    "is_beat_cached",
    "get_beat_tracking_stats",
]
