"""
FLX4-Net Track A Processing Cue p(t)

Measures spectral texture/processing characteristics of the vocal.
Captures effects like reverb, compression, EQ, and overall "polish".

Blueprint Policy:
    - Weak auxiliary cue (least predictive expected)
    - Based on spectral statistics
    - Normalized to [0, 1] range

Features computed:
    1. Spectral centroid (brightness)
    2. Spectral bandwidth (spread)
    3. Spectral flatness (noise-like vs tonal)
    4. Spectral rolloff (high-freq content)

Interpretation:
    - High p(t): Heavily processed, bright, wide spectrum
                 → May indicate reverb/effects → harder to blend
    - Low p(t):  Dry, natural, narrow spectrum
                 → Cleaner signal → easier to control
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


def compute_spectral_features(
    audio: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict[str, float]:
    """Compute spectral features for processing cue.
    
    Args:
        audio: Audio signal (1D)
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        Dictionary with spectral features
    """
    import librosa
    
    # Check for silent audio
    if np.max(np.abs(audio)) < 1e-6:
        return {
            'centroid': 0.0,
            'bandwidth': 0.0,
            'flatness': 0.0,
            'rolloff': 0.0
        }
    
    # Compute spectral features
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=n_fft, hop_length=hop_length
    )
    
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Average over time
    return {
        'centroid': float(np.mean(centroid)),
        'bandwidth': float(np.mean(bandwidth)),
        'flatness': float(np.mean(flatness)),
        'rolloff': float(np.mean(rolloff))
    }


def normalize_feature(
    value: float,
    min_val: float,
    max_val: float
) -> float:
    """Normalize a feature value to [0, 1] range.
    
    Args:
        value: Feature value
        min_val: Expected minimum
        max_val: Expected maximum
        
    Returns:
        Normalized value clipped to [0, 1]
    """
    if max_val <= min_val:
        return 0.0
    
    normalized = (value - min_val) / (max_val - min_val)
    return float(np.clip(normalized, 0.0, 1.0))


def compute_processing_cue(
    audio: np.ndarray,
    sr: int = 44100,
    feature_weights: Optional[Dict[str, float]] = None,
    normalization_params: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[float, Dict]:
    """Compute processing cue p(t) for a segment.
    
    Combines multiple spectral features into a single score.
    
    Args:
        audio: Audio segment (1D)
        sr: Sample rate
        feature_weights: Weights for each feature (default: equal)
        normalization_params: (min, max) for each feature
        
    Returns:
        Tuple of (p_value, feature_dict)
    """
    # Default weights (equal contribution)
    if feature_weights is None:
        feature_weights = {
            'centroid': 0.25,
            'bandwidth': 0.25,
            'flatness': 0.25,
            'rolloff': 0.25
        }
    
    # Default normalization params (empirically determined)
    # These are rough estimates for vocal audio at 44.1kHz
    if normalization_params is None:
        normalization_params = {
            'centroid': (500.0, 4000.0),      # Hz
            'bandwidth': (500.0, 3000.0),      # Hz
            'flatness': (0.001, 0.1),          # Ratio
            'rolloff': (1000.0, 8000.0)        # Hz
        }
    
    # Compute raw features
    features = compute_spectral_features(audio, sr)
    
    # Normalize each feature
    normalized_features = {}
    for name, value in features.items():
        if name in normalization_params:
            min_val, max_val = normalization_params[name]
            normalized_features[name] = normalize_feature(value, min_val, max_val)
        else:
            normalized_features[name] = value
    
    # Compute weighted sum
    p_value = 0.0
    for name, weight in feature_weights.items():
        if name in normalized_features:
            p_value += weight * normalized_features[name]
    
    # Build info dict
    info = {
        'raw_features': features,
        'normalized_features': normalized_features,
        'weights': feature_weights
    }
    
    return float(p_value), info


class ProcessingComputer:
    """Computes processing cue p(t) for track segments.
    
    Usage:
        computer = ProcessingComputer(sr=44100)
        computer.set_track_audio(vocal_audio)
        p_value, info = computer.compute_segment(start_sample, end_sample)
    """
    
    def __init__(
        self,
        sr: int = 44100,
        feature_weights: Optional[Dict[str, float]] = None,
        normalization_params: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """Initialize ProcessingComputer.
        
        Args:
            sr: Sample rate
            feature_weights: Weights for combining features
            normalization_params: (min, max) for normalizing features
        """
        self.sr = sr
        self.feature_weights = feature_weights
        self.normalization_params = normalization_params
        
        self._vocal_audio: Optional[np.ndarray] = None
    
    def set_track_audio(self, vocal_audio: np.ndarray):
        """Set track vocal audio.
        
        Args:
            vocal_audio: Full track vocal audio
        """
        self._vocal_audio = vocal_audio
    
    def compute_segment(
        self,
        start_sample: int,
        end_sample: int
    ) -> Tuple[float, Dict]:
        """Compute p(t) for a segment.
        
        Args:
            start_sample: Segment start sample
            end_sample: Segment end sample
            
        Returns:
            Tuple of (p_value, info)
        """
        if self._vocal_audio is None:
            raise RuntimeError("Track audio not set. Call set_track_audio() first.")
        
        segment_audio = self._vocal_audio[start_sample:end_sample]
        
        return compute_processing_cue(
            audio=segment_audio,
            sr=self.sr,
            feature_weights=self.feature_weights,
            normalization_params=self.normalization_params
        )
    
    def compute_track(
        self,
        segments: List[dict]
    ) -> Tuple[List[float], List[Dict]]:
        """Compute p(t) for all segments.
        
        Args:
            segments: List of segment dicts
            
        Returns:
            Tuple of (p_values, infos)
        """
        p_values = []
        infos = []
        
        for seg in segments:
            p_val, info = self.compute_segment(
                start_sample=seg['start_sample'],
                end_sample=seg['end_sample']
            )
            p_values.append(p_val)
            infos.append(info)
        
        return p_values, infos


def calibrate_normalization_params(
    audio_samples: List[np.ndarray],
    sr: int = 44100,
    percentile_low: float = 5,
    percentile_high: float = 95
) -> Dict[str, Tuple[float, float]]:
    """Calibrate normalization parameters from sample audio.
    
    Useful for adjusting normalization to a specific dataset.
    
    Args:
        audio_samples: List of audio segments
        sr: Sample rate
        percentile_low: Lower percentile for min
        percentile_high: Upper percentile for max
        
    Returns:
        Dictionary mapping feature names to (min, max) tuples
    """
    all_features = {
        'centroid': [],
        'bandwidth': [],
        'flatness': [],
        'rolloff': []
    }
    
    for audio in audio_samples:
        features = compute_spectral_features(audio, sr)
        for name, value in features.items():
            if name in all_features:
                all_features[name].append(value)
    
    normalization_params = {}
    for name, values in all_features.items():
        if len(values) > 0:
            min_val = float(np.percentile(values, percentile_low))
            max_val = float(np.percentile(values, percentile_high))
            normalization_params[name] = (min_val, max_val)
    
    return normalization_params


if __name__ == "__main__":
    print("=== Processing Cue Test ===\n")
    
    import librosa
    
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Test 1: Pure sine wave (simple, unprocessed)
    print("--- Test 1: Pure sine wave (440 Hz) ---")
    audio_pure = np.sin(2 * np.pi * 440 * t) * 0.5
    p_pure, info_pure = compute_processing_cue(audio_pure, sr)
    print(f"p(t) = {p_pure:.3f}")
    print(f"Raw features:")
    for k, v in info_pure['raw_features'].items():
        print(f"  {k}: {v:.2f}")
    
    # Test 2: Wide spectrum noise (heavily processed feel)
    print("\n--- Test 2: White noise (wide spectrum) ---")
    np.random.seed(42)
    audio_noise = np.random.randn(len(t)) * 0.3
    p_noise, info_noise = compute_processing_cue(audio_noise, sr)
    print(f"p(t) = {p_noise:.3f}")
    print(f"Raw features:")
    for k, v in info_noise['raw_features'].items():
        print(f"  {k}: {v:.2f}")
    
    # Test 3: Harmonic rich signal (realistic vocal-like)
    print("\n--- Test 3: Harmonic signal (vocal-like) ---")
    audio_harmonic = np.zeros_like(t)
    for harmonic in [1, 2, 3, 4, 5]:
        audio_harmonic += np.sin(2 * np.pi * 220 * harmonic * t) / harmonic
    audio_harmonic *= 0.3
    p_harmonic, info_harmonic = compute_processing_cue(audio_harmonic, sr)
    print(f"p(t) = {p_harmonic:.3f}")
    print(f"Raw features:")
    for k, v in info_harmonic['raw_features'].items():
        print(f"  {k}: {v:.2f}")
    
    # Test 4: Silent audio
    print("\n--- Test 4: Silent audio ---")
    audio_silent = np.zeros(int(sr * duration))
    p_silent, info_silent = compute_processing_cue(audio_silent, sr)
    print(f"p(t) = {p_silent:.3f}")
    
    print("\nExpected pattern:")
    print("  Pure sine < Harmonic < Noise")
    print(f"  Actual: {p_pure:.3f} < {p_harmonic:.3f} < {p_noise:.3f}")
    
    print("\n✓ Processing module ready!")
