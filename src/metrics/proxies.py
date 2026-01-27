"""
FLX4-Net Track A Artifact Proxies

Computes proxy measures for separation artifacts WITHOUT ground truth.
These proxies are used in Track B to estimate difficulty without GT.

In Track A, we validate these proxies by correlating them with actual
SI-SDR (which requires GT). Strong correlation → useful proxy for Track B.

Proxy Definitions:
    1. leakage_lowfreq: Low-frequency energy ratio (50-150 Hz)
       - High value → likely drum/bass leakage
       - Computed from separated vocal
       
    2. instability: Temporal instability between adjacent frames
       - High value → separation artifacts causing discontinuities
       - Computed from separated vocal
       
    3. transient_leakage: Energy in percussive/transient components
       - High value → likely drum leakage
       - Uses harmonic-percussive separation
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings

from ..data_types import ProxyResult
from ..audio_utils import load_audio


# Constants
EPS = 1e-8


def compute_lowfreq_energy_ratio(
    audio: np.ndarray,
    sr: int = 44100,
    low_freq_min: float = 50.0,
    low_freq_max: float = 150.0,
    n_fft: int = 2048
) -> float:
    """Compute ratio of low-frequency energy to total energy.
    
    High ratio in separated vocals suggests drum/bass leakage.
    
    Args:
        audio: Audio signal (1D)
        sr: Sample rate
        low_freq_min: Low band minimum frequency (Hz)
        low_freq_max: Low band maximum frequency (Hz)
        n_fft: FFT size
        
    Returns:
        Ratio of low-freq energy to total energy (0-1)
    """
    import librosa
    
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft)
    magnitude = np.abs(stft)
    
    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Find bins in low frequency range
    low_mask = (freqs >= low_freq_min) & (freqs <= low_freq_max)
    
    # Compute energies
    total_energy = np.sum(magnitude ** 2) + EPS
    low_energy = np.sum(magnitude[low_mask, :] ** 2)
    
    ratio = low_energy / total_energy
    
    return float(ratio)


def compute_temporal_instability(
    audio: np.ndarray,
    sr: int = 44100,
    frame_length: int = 2048,
    hop_length: int = 512
) -> float:
    """Compute temporal instability as frame-to-frame spectral flux.
    
    High instability suggests separation artifacts causing
    unnatural discontinuities.
    
    Args:
        audio: Audio signal (1D)
        sr: Sample rate
        frame_length: Frame length for analysis
        hop_length: Hop length between frames
        
    Returns:
        Mean spectral flux (higher = more unstable)
    """
    import librosa
    
    # Compute mel spectrogram (log scale for perceptual relevance)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=128
    )
    log_mel = librosa.power_to_db(mel_spec + EPS)
    
    # Compute frame-to-frame differences
    diff = np.diff(log_mel, axis=1)
    
    # Mean absolute difference (spectral flux)
    flux = np.mean(np.abs(diff))
    
    return float(flux)


def compute_transient_energy_ratio(
    audio: np.ndarray,
    sr: int = 44100,
    margin: float = 3.0
) -> float:
    """Compute ratio of percussive/transient energy to total.
    
    Uses harmonic-percussive source separation (HPSS).
    High ratio in separated vocals suggests drum leakage.
    
    Args:
        audio: Audio signal (1D)
        sr: Sample rate
        margin: HPSS margin parameter (higher = stricter separation)
        
    Returns:
        Ratio of percussive energy to total (0-1)
    """
    import librosa
    
    # Harmonic-percussive separation
    try:
        harmonic, percussive = librosa.effects.hpss(audio, margin=margin)
    except Exception:
        # Fallback if HPSS fails
        return 0.0
    
    # Compute energies
    total_energy = np.sum(audio ** 2) + EPS
    percussive_energy = np.sum(percussive ** 2)
    
    ratio = percussive_energy / total_energy
    
    return float(ratio)


def compute_segment_proxies(
    audio: np.ndarray,
    sr: int,
    seg_id: str,
    low_freq_range: Tuple[float, float] = (50.0, 150.0)
) -> ProxyResult:
    """Compute all proxy measures for a single segment.
    
    Args:
        audio: Segment audio (1D)
        sr: Sample rate
        seg_id: Segment identifier
        low_freq_range: (min, max) Hz for low-frequency analysis
        
    Returns:
        ProxyResult with all proxy values
    """
    try:
        # Check for silent/near-silent segment
        if np.max(np.abs(audio)) < 1e-6:
            return ProxyResult(
                seg_id=seg_id,
                leakage_lowfreq=0.0,
                instability=0.0,
                transient_leakage=0.0,
                valid=False
            )
        
        # Compute proxies
        leakage_lowfreq = compute_lowfreq_energy_ratio(
            audio, sr,
            low_freq_min=low_freq_range[0],
            low_freq_max=low_freq_range[1]
        )
        
        instability = compute_temporal_instability(audio, sr)
        
        transient_leakage = compute_transient_energy_ratio(audio, sr)
        
        return ProxyResult(
            seg_id=seg_id,
            leakage_lowfreq=leakage_lowfreq,
            instability=instability,
            transient_leakage=transient_leakage,
            valid=True
        )
        
    except Exception as e:
        return ProxyResult(
            seg_id=seg_id,
            leakage_lowfreq=float('nan'),
            instability=float('nan'),
            transient_leakage=float('nan'),
            valid=False
        )


def compute_track_proxies(
    vocal_path: Path,
    segments: List[dict],
    sr: int = 44100,
    low_freq_range: Tuple[float, float] = (50.0, 150.0)
) -> Tuple[List[ProxyResult], Dict]:
    """Compute proxies for all segments in a track.
    
    Args:
        vocal_path: Path to separated vocal file
        segments: List of segment dicts with start_sample, end_sample, seg_id
        sr: Sample rate
        low_freq_range: (min, max) Hz for low-frequency analysis
        
    Returns:
        Tuple of (list of ProxyResult, stats dict)
    """
    # Load audio
    audio, _ = load_audio(vocal_path, sr=sr, mono=True)
    
    results = []
    valid_count = 0
    
    for seg in segments:
        start_sample = seg['start_sample']
        end_sample = min(seg['end_sample'], len(audio))
        seg_id = seg['seg_id']
        
        # Extract segment
        segment_audio = audio[start_sample:end_sample]
        
        # Compute proxies
        result = compute_segment_proxies(
            audio=segment_audio,
            sr=sr,
            seg_id=seg_id,
            low_freq_range=low_freq_range
        )
        
        results.append(result)
        if result.valid:
            valid_count += 1
    
    stats = {
        "n_segments": len(results),
        "n_valid": valid_count,
        "n_invalid": len(results) - valid_count,
        "valid_ratio": valid_count / len(results) if results else 0
    }
    
    return results, stats


class ProxyComputer:
    """Manages proxy computation across multiple tracks.
    
    Usage:
        computer = ProxyComputer(
            cache_root=Path("./cache"),
            model_name="htdemucs",
            sr=44100
        )
        results = computer.compute_for_track(track_id, segments)
    """
    
    def __init__(
        self,
        cache_root: Path,
        model_name: str = "htdemucs",
        sr: int = 44100,
        low_freq_range: Tuple[float, float] = (50.0, 150.0)
    ):
        """Initialize ProxyComputer.
        
        Args:
            cache_root: Root cache directory
            model_name: Separator model name
            sr: Sample rate
            low_freq_range: (min, max) Hz for low-frequency analysis
        """
        self.cache_root = Path(cache_root)
        self.model_name = model_name
        self.sr = sr
        self.low_freq_range = low_freq_range
        
        # Statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.tracks_processed = 0
    
    def get_separated_vocal_path(self, track_id: str) -> Path:
        """Get path to cached separated vocal."""
        return self.cache_root / "separated" / self.model_name / track_id / "vocals.wav"
    
    def compute_for_track(
        self,
        track_id: str,
        segments: List[dict]
    ) -> Tuple[List[ProxyResult], Dict]:
        """Compute proxies for all segments in a track.
        
        Args:
            track_id: Track identifier
            segments: List of segment dicts
            
        Returns:
            Tuple of (results list, stats dict)
        """
        vocal_path = self.get_separated_vocal_path(track_id)
        
        if not vocal_path.exists():
            results = [
                ProxyResult(
                    seg_id=seg['seg_id'],
                    leakage_lowfreq=float('nan'),
                    instability=float('nan'),
                    transient_leakage=float('nan'),
                    valid=False
                )
                for seg in segments
            ]
            stats = {
                "n_segments": len(results),
                "n_valid": 0,
                "n_invalid": len(results),
                "valid_ratio": 0,
                "error": "separated_vocal_not_found"
            }
            return results, stats
        
        results, stats = compute_track_proxies(
            vocal_path=vocal_path,
            segments=segments,
            sr=self.sr,
            low_freq_range=self.low_freq_range
        )
        
        # Update statistics
        self.total_segments += stats["n_segments"]
        self.valid_segments += stats["n_valid"]
        self.tracks_processed += 1
        
        return results, stats
    
    def get_overall_stats(self) -> Dict:
        """Get overall statistics."""
        return {
            "tracks_processed": self.tracks_processed,
            "total_segments": self.total_segments,
            "valid_segments": self.valid_segments,
            "invalid_segments": self.total_segments - self.valid_segments,
            "valid_ratio": self.valid_segments / self.total_segments if self.total_segments > 0 else 0
        }
    
    def reset_stats(self):
        """Reset running statistics."""
        self.total_segments = 0
        self.valid_segments = 0
        self.tracks_processed = 0


def compute_proxy_stats(results: List[ProxyResult]) -> Dict:
    """Compute statistics over proxy results.
    
    Args:
        results: List of ProxyResult
        
    Returns:
        Dictionary with statistics for each proxy
    """
    valid_results = [r for r in results if r.valid]
    
    if not valid_results:
        return {
            "n_total": len(results),
            "n_valid": 0,
            "leakage_lowfreq": {"mean": None, "std": None},
            "instability": {"mean": None, "std": None},
            "transient_leakage": {"mean": None, "std": None}
        }
    
    leakage_values = [r.leakage_lowfreq for r in valid_results]
    instability_values = [r.instability for r in valid_results]
    transient_values = [r.transient_leakage for r in valid_results]
    
    def stats_for_values(values):
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "p5": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95))
        }
    
    return {
        "n_total": len(results),
        "n_valid": len(valid_results),
        "leakage_lowfreq": stats_for_values(leakage_values),
        "instability": stats_for_values(instability_values),
        "transient_leakage": stats_for_values(transient_values)
    }


def compute_proxy_difficulty_correlation(
    proxy_df,
    difficulty_df,
    method: str = "spearman"
) -> Dict:
    """Compute correlation between proxies and difficulty (SI-SDR).
    
    This validates whether proxies are useful predictors of separation quality.
    
    Args:
        proxy_df: DataFrame with proxy values
        difficulty_df: DataFrame with SI-SDR values
        method: Correlation method ("spearman" or "pearson")
        
    Returns:
        Dictionary with correlation results for each proxy
    """
    from scipy import stats as scipy_stats
    
    # Merge on seg_id
    merged = proxy_df.merge(difficulty_df, on='seg_id', suffixes=('_proxy', '_diff'))
    
    # Filter valid
    valid_mask = merged['valid_proxy'] & merged['valid_diff']
    merged = merged[valid_mask]
    
    if len(merged) < 10:
        return {"error": "too_few_valid_samples", "n_samples": len(merged)}
    
    proxy_cols = ['leakage_lowfreq', 'instability', 'transient_leakage']
    results = {}
    
    for proxy_name in proxy_cols:
        x = merged[proxy_name].values
        y = merged['sisdr_vocal'].values
        
        # Remove NaN
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]
        
        if len(x) < 10:
            results[proxy_name] = {"error": "too_few_samples"}
            continue
        
        if method == "spearman":
            rho, p_value = scipy_stats.spearmanr(x, y)
        else:
            rho, p_value = scipy_stats.pearsonr(x, y)
        
        results[proxy_name] = {
            "correlation": float(rho),
            "p_value": float(p_value),
            "n_samples": len(x),
            "method": method
        }
    
    return results


if __name__ == "__main__":
    print("=== Proxy Metrics Test ===\n")
    
    import librosa
    
    # Create synthetic test signal
    np.random.seed(42)
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simulate "clean" vocal (mostly harmonic)
    vocal_clean = np.sin(2 * np.pi * 440 * t) * 0.3
    vocal_clean += np.sin(2 * np.pi * 880 * t) * 0.15
    
    # Simulate "leaked" vocal (with low-freq bass and transients)
    bass = np.sin(2 * np.pi * 80 * t) * 0.2  # Bass leakage
    kick = np.zeros_like(t)
    kick[::int(sr * 0.5)] = 0.5  # Kick drum transients
    kick = np.convolve(kick, np.exp(-np.linspace(0, 10, 1000)), mode='same')
    
    vocal_leaked = vocal_clean + bass + kick
    
    print("Clean vocal (no leakage):")
    result_clean = compute_segment_proxies(vocal_clean, sr, "test_clean")
    print(f"  Low-freq ratio: {result_clean.leakage_lowfreq:.4f}")
    print(f"  Instability: {result_clean.instability:.4f}")
    print(f"  Transient ratio: {result_clean.transient_leakage:.4f}")
    print()
    
    print("Leaked vocal (bass + drums):")
    result_leaked = compute_segment_proxies(vocal_leaked, sr, "test_leaked")
    print(f"  Low-freq ratio: {result_leaked.leakage_lowfreq:.4f}")
    print(f"  Instability: {result_leaked.instability:.4f}")
    print(f"  Transient ratio: {result_leaked.transient_leakage:.4f}")
    print()
    
    print("Comparison:")
    print(f"  Low-freq ratio increased: {result_leaked.leakage_lowfreq > result_clean.leakage_lowfreq}")
    print(f"  Transient ratio increased: {result_leaked.transient_leakage > result_clean.transient_leakage}")
    
    print("\n✓ Proxy metrics module ready!")
