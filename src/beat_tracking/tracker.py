"""
FLX4-Net Beat Tracker Interface

Unified interface for beat tracking with caching support.
Delegates to backend-specific implementations (librosa).
"""

import json
from pathlib import Path
from typing import Optional, Literal

import numpy as np

from ..data_types import BeatResult
from ..audio_utils import load_audio
from .librosa_tracker import create_beat_result_librosa


# Type alias for backend
BeatBackend = Literal["librosa", "essentia", "madmom"]


def get_beat_cache_path(
    cache_root: Path,
    backend: str,
    track_id: str
) -> Path:
    """Get cache path for beat tracking result.
    
    Cache structure:
        cache_root/beat_tracking/{backend}/{track_id}/beats.json
    
    Args:
        cache_root: Cache root directory
        backend: Beat tracking backend name
        track_id: Track identifier
        
    Returns:
        Path to cached beats.json file
    """
    return cache_root / "beat_tracking" / backend / track_id / "beats.json"


def is_beat_cached(
    cache_root: Path,
    backend: str,
    track_id: str
) -> bool:
    """Check if beat tracking result is cached.
    
    Args:
        cache_root: Cache root directory
        backend: Beat tracking backend name
        track_id: Track identifier
        
    Returns:
        True if cached file exists
    """
    cache_path = get_beat_cache_path(cache_root, backend, track_id)
    return cache_path.exists()


def save_beat_result(result: BeatResult, path: Path) -> None:
    """Save beat tracking result to JSON file.
    
    Args:
        result: BeatResult object
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = result.to_dict()
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_beat_result(path: Path) -> Optional[BeatResult]:
    """Load beat tracking result from JSON file.
    
    Args:
        path: Path to beats.json file
        
    Returns:
        BeatResult object, or None if file not found
    """
    if not path.exists():
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return BeatResult.from_dict(data)
    except Exception as e:
        print(f"Error loading beat result from {path}: {e}")
        return None


def run_beat_tracking(
    mixture_path: Path,
    track_id: str,
    cache_root: Path,
    backend: str = "librosa",
    sr: int = 44100,
    min_beats_per_track: int = 8,
    force_rerun: bool = False
) -> Optional[BeatResult]:
    """Run beat tracking on mixture audio with caching.
    
    Args:
        mixture_path: Path to mixture audio file
        track_id: Track identifier
        cache_root: Cache root directory
        backend: Beat tracking backend ("librosa")
        sr: Sample rate
        min_beats_per_track: Minimum beats for track availability
        force_rerun: Force re-computation even if cached
        
    Returns:
        BeatResult object, or None if failed
    """
    # Check cache
    cache_path = get_beat_cache_path(cache_root, backend, track_id)
    
    if cache_path.exists() and not force_rerun:
        result = load_beat_result(cache_path)
        if result:
            return result
    
    # Load audio
    try:
        audio, _ = load_audio(mixture_path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading audio {mixture_path}: {e}")
        return None
    
    duration_sec = len(audio) / sr
    
    # Run beat tracking based on backend
    if backend == "librosa":
        result = create_beat_result_librosa(
            track_id=track_id,
            audio=audio,
            sr=sr,
            min_beats_per_track=min_beats_per_track,
            duration_sec=duration_sec
        )
    elif backend == "essentia":
        # Essentia not supported on Windows
        print(f"Warning: Essentia backend not available, falling back to librosa")
        result = create_beat_result_librosa(
            track_id=track_id,
            audio=audio,
            sr=sr,
            min_beats_per_track=min_beats_per_track,
            duration_sec=duration_sec
        )
    elif backend == "madmom":
        # madmom not supported on Windows
        print(f"Warning: madmom backend not available, falling back to librosa")
        result = create_beat_result_librosa(
            track_id=track_id,
            audio=audio,
            sr=sr,
            min_beats_per_track=min_beats_per_track,
            duration_sec=duration_sec
        )
    else:
        print(f"Unknown backend: {backend}, using librosa")
        result = create_beat_result_librosa(
            track_id=track_id,
            audio=audio,
            sr=sr,
            min_beats_per_track=min_beats_per_track,
            duration_sec=duration_sec
        )
    
    # Save to cache
    save_beat_result(result, cache_path)
    
    return result


def get_beat_tracking_stats(cache_root: Path, backend: str) -> dict:
    """Get statistics about cached beat tracking results.
    
    Args:
        cache_root: Cache root directory
        backend: Beat tracking backend name
        
    Returns:
        Dictionary with stats
    """
    beat_dir = cache_root / "beat_tracking" / backend
    
    if not beat_dir.exists():
        return {
            "n_tracks": 0,
            "n_available": 0,
            "n_unavailable": 0,
            "coverage_ratio": 0.0
        }
    
    track_dirs = [d for d in beat_dir.iterdir() if d.is_dir()]
    
    n_available = 0
    n_unavailable = 0
    unavailable_reasons: dict = {}
    
    for track_dir in track_dirs:
        beats_file = track_dir / "beats.json"
        if beats_file.exists():
            result = load_beat_result(beats_file)
            if result:
                if result.track_available:
                    n_available += 1
                else:
                    n_unavailable += 1
                    reason = result.unavailable_reason or "unknown"
                    unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1
    
    n_total = n_available + n_unavailable
    
    return {
        "n_tracks": n_total,
        "n_available": n_available,
        "n_unavailable": n_unavailable,
        "coverage_ratio": n_available / n_total if n_total > 0 else 0.0,
        "unavailable_reasons": unavailable_reasons,
        "cache_path": str(beat_dir)
    }


if __name__ == "__main__":
    print("=== Beat Tracker Interface Test ===\n")
    print("This module provides:")
    print("  - run_beat_tracking(): Run beat detection with caching")
    print("  - load_beat_result(): Load cached result")
    print("  - save_beat_result(): Save result to cache")
    print("  - get_beat_tracking_stats(): Get cache statistics")
    print("\nUse 03_run_beat_tracking.py to run batch beat tracking.")
