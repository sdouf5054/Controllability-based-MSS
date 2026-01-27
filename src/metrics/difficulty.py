"""
FLX4-Net Track A Difficulty Metrics

Computes SI-SDR and SDR between estimated and ground truth vocals.
SI-SDR is the primary metric (Blueprint policy).

Reference:
    SI-SDR: Le Roux et al., "SDR – Half-Baked or Well Done?", ICASSP 2019
    https://arxiv.org/abs/1811.02508
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from ..data_types import DifficultyResult
from ..audio_utils import load_audio, align_lengths


# Constants
EPS = 1e-8  # Small epsilon for numerical stability


def compute_sisdr(
    est: np.ndarray,
    ref: np.ndarray,
    eps: float = EPS
) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    SI-SDR is invariant to scaling, making it more robust than standard SDR.
    Higher values indicate better separation quality.
    
    Formula:
        s_target = (<est, ref> / ||ref||^2) * ref
        e_noise = est - s_target
        SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    
    Args:
        est: Estimated (separated) signal, shape (n_samples,)
        ref: Reference (ground truth) signal, shape (n_samples,)
        eps: Small value for numerical stability
        
    Returns:
        SI-SDR in dB (higher is better)
        
    Raises:
        ValueError: If signals have different lengths
    """
    if est.shape != ref.shape:
        raise ValueError(
            f"Signal shapes must match: est={est.shape}, ref={ref.shape}"
        )
    
    # Ensure 1D
    est = est.flatten()
    ref = ref.flatten()
    
    # Zero-mean (remove DC offset)
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    
    # Check for silent reference
    ref_energy = np.sum(ref ** 2)
    if ref_energy < eps:
        return -np.inf  # Silent reference, undefined SI-SDR
    
    # Compute scaling factor: alpha = <est, ref> / ||ref||^2
    alpha = np.sum(est * ref) / (ref_energy + eps)
    
    # Target projection
    s_target = alpha * ref
    
    # Noise (residual)
    e_noise = est - s_target
    
    # Compute SI-SDR
    s_target_energy = np.sum(s_target ** 2)
    e_noise_energy = np.sum(e_noise ** 2)
    
    if e_noise_energy < eps:
        return np.inf  # Perfect reconstruction (practically impossible)
    
    sisdr = 10 * np.log10((s_target_energy + eps) / (e_noise_energy + eps))
    
    return float(sisdr)


def compute_sdr(
    est: np.ndarray,
    ref: np.ndarray,
    eps: float = EPS
) -> float:
    """Compute Signal-to-Distortion Ratio (SDR).
    
    Standard SDR without scale invariance. Included for reference
    and comparison with existing literature.
    
    Formula:
        SDR = 10 * log10(||ref||^2 / ||est - ref||^2)
    
    Args:
        est: Estimated (separated) signal
        ref: Reference (ground truth) signal
        eps: Small value for numerical stability
        
    Returns:
        SDR in dB (higher is better)
    """
    if est.shape != ref.shape:
        raise ValueError(
            f"Signal shapes must match: est={est.shape}, ref={ref.shape}"
        )
    
    # Ensure 1D
    est = est.flatten()
    ref = ref.flatten()
    
    # Reference energy
    ref_energy = np.sum(ref ** 2)
    if ref_energy < eps:
        return -np.inf  # Silent reference
    
    # Noise energy
    noise = est - ref
    noise_energy = np.sum(noise ** 2)
    
    if noise_energy < eps:
        return np.inf  # Perfect reconstruction
    
    sdr = 10 * np.log10((ref_energy + eps) / (noise_energy + eps))
    
    return float(sdr)


def compute_segment_difficulty(
    est_audio: np.ndarray,
    gt_audio: np.ndarray,
    start_sample: int,
    end_sample: int,
    seg_id: str,
    compute_sdr_flag: bool = True
) -> DifficultyResult:
    """Compute difficulty metrics for a single segment.
    
    Args:
        est_audio: Full estimated vocal track
        gt_audio: Full ground truth vocal track
        start_sample: Segment start sample index
        end_sample: Segment end sample index
        seg_id: Segment identifier
        compute_sdr_flag: Whether to also compute standard SDR
        
    Returns:
        DifficultyResult with SI-SDR and optionally SDR
    """
    try:
        # Extract segment
        est_segment = est_audio[start_sample:end_sample]
        gt_segment = gt_audio[start_sample:end_sample]
        
        # Compute SI-SDR (primary metric)
        sisdr = compute_sisdr(est_segment, gt_segment)
        
        # Compute SDR (reference metric)
        sdr = compute_sdr(est_segment, gt_segment) if compute_sdr_flag else 0.0
        
        # Check for invalid values
        if np.isnan(sisdr) or np.isinf(sisdr):
            return DifficultyResult(
                seg_id=seg_id,
                sisdr_vocal=float('nan'),
                sdr_vocal=float('nan'),
                valid=False,
                error_msg=f"Invalid SI-SDR: {sisdr}"
            )
        
        return DifficultyResult(
            seg_id=seg_id,
            sisdr_vocal=sisdr,
            sdr_vocal=sdr,
            valid=True
        )
        
    except Exception as e:
        return DifficultyResult(
            seg_id=seg_id,
            sisdr_vocal=float('nan'),
            sdr_vocal=float('nan'),
            valid=False,
            error_msg=str(e)
        )


def compute_track_difficulty(
    est_path: Path,
    gt_path: Path,
    segments: List[dict],
    sr: int = 44100,
    compute_sdr_flag: bool = True,
    alignment_policy: str = "truncate_to_shorter"
) -> Tuple[List[DifficultyResult], Dict]:
    """Compute difficulty metrics for all segments in a track.
    
    Args:
        est_path: Path to estimated (separated) vocal file
        gt_path: Path to ground truth vocal file
        segments: List of segment dicts with start_sample, end_sample, seg_id
        sr: Sample rate
        compute_sdr_flag: Whether to also compute SDR
        alignment_policy: How to handle length mismatch
        
    Returns:
        Tuple of (list of DifficultyResult, stats dict)
    """
    # Load audio
    est_audio, _ = load_audio(est_path, sr=sr, mono=True)
    gt_audio, _ = load_audio(gt_path, sr=sr, mono=True)
    
    # Align lengths (Blueprint policy: truncate to shorter)
    est_audio, gt_audio = align_lengths(
        est_audio, gt_audio,
        policy=alignment_policy,
        max_diff_samples=1000,
        warn=True
    )
    
    results = []
    valid_count = 0
    invalid_count = 0
    
    for seg in segments:
        start_sample = seg['start_sample']
        end_sample = seg['end_sample']
        seg_id = seg['seg_id']
        
        # Clamp to actual audio length
        end_sample = min(end_sample, len(est_audio), len(gt_audio))
        start_sample = min(start_sample, end_sample)
        
        result = compute_segment_difficulty(
            est_audio=est_audio,
            gt_audio=gt_audio,
            start_sample=start_sample,
            end_sample=end_sample,
            seg_id=seg_id,
            compute_sdr_flag=compute_sdr_flag
        )
        
        results.append(result)
        
        if result.valid:
            valid_count += 1
        else:
            invalid_count += 1
    
    stats = {
        "n_segments": len(results),
        "n_valid": valid_count,
        "n_invalid": invalid_count,
        "valid_ratio": valid_count / len(results) if results else 0
    }
    
    return results, stats


class DifficultyComputer:
    """Manages difficulty computation across multiple tracks.
    
    Handles loading, caching, and batch processing.
    
    Usage:
        computer = DifficultyComputer(
            cache_root=Path("./cache"),
            model_name="htdemucs",
            sr=44100
        )
        results = computer.compute_for_track(track_id, gt_vocal_path, segments)
    """
    
    def __init__(
        self,
        cache_root: Path,
        model_name: str = "htdemucs",
        sr: int = 44100,
        compute_sdr: bool = True,
        alignment_policy: str = "truncate_to_shorter"
    ):
        """Initialize DifficultyComputer.
        
        Args:
            cache_root: Root cache directory
            model_name: Separator model name (for finding cached vocals)
            sr: Sample rate
            compute_sdr: Whether to compute SDR in addition to SI-SDR
            alignment_policy: How to handle length mismatch
        """
        self.cache_root = Path(cache_root)
        self.model_name = model_name
        self.sr = sr
        self.compute_sdr = compute_sdr
        self.alignment_policy = alignment_policy
        
        # Track statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.tracks_processed = 0
    
    def get_separated_vocal_path(self, track_id: str) -> Path:
        """Get path to cached separated vocal.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Path to vocals.wav
        """
        return self.cache_root / "separated" / self.model_name / track_id / "vocals.wav"
    
    def compute_for_track(
        self,
        track_id: str,
        gt_vocal_path: Path,
        segments: List[dict]
    ) -> Tuple[List[DifficultyResult], Dict]:
        """Compute difficulty for all segments in a track.
        
        Args:
            track_id: Track identifier
            gt_vocal_path: Path to ground truth vocal
            segments: List of segment dicts
            
        Returns:
            Tuple of (results list, stats dict)
        """
        # Get separated vocal path
        est_path = self.get_separated_vocal_path(track_id)
        
        if not est_path.exists():
            # Return invalid results for all segments
            results = [
                DifficultyResult(
                    seg_id=seg['seg_id'],
                    sisdr_vocal=float('nan'),
                    sdr_vocal=float('nan'),
                    valid=False,
                    error_msg=f"Separated vocal not found: {est_path}"
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
        
        # Compute difficulty
        results, stats = compute_track_difficulty(
            est_path=est_path,
            gt_path=Path(gt_vocal_path),
            segments=segments,
            sr=self.sr,
            compute_sdr_flag=self.compute_sdr,
            alignment_policy=self.alignment_policy
        )
        
        # Update running statistics
        self.total_segments += stats["n_segments"]
        self.valid_segments += stats["n_valid"]
        self.tracks_processed += 1
        
        return results, stats
    
    def get_overall_stats(self) -> Dict:
        """Get overall statistics across all processed tracks.
        
        Returns:
            Dictionary with overall stats
        """
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


def compute_difficulty_stats(results: List[DifficultyResult]) -> Dict:
    """Compute statistics over difficulty results.
    
    Args:
        results: List of DifficultyResult
        
    Returns:
        Dictionary with statistics
    """
    valid_results = [r for r in results if r.valid]
    
    if not valid_results:
        return {
            "n_total": len(results),
            "n_valid": 0,
            "sisdr": {"mean": None, "std": None, "min": None, "max": None},
            "sdr": {"mean": None, "std": None, "min": None, "max": None}
        }
    
    sisdr_values = [r.sisdr_vocal for r in valid_results]
    sdr_values = [r.sdr_vocal for r in valid_results]
    
    return {
        "n_total": len(results),
        "n_valid": len(valid_results),
        "sisdr": {
            "mean": float(np.mean(sisdr_values)),
            "std": float(np.std(sisdr_values)),
            "min": float(np.min(sisdr_values)),
            "max": float(np.max(sisdr_values)),
            "median": float(np.median(sisdr_values)),
            "p5": float(np.percentile(sisdr_values, 5)),
            "p25": float(np.percentile(sisdr_values, 25)),
            "p75": float(np.percentile(sisdr_values, 75)),
            "p95": float(np.percentile(sisdr_values, 95))
        },
        "sdr": {
            "mean": float(np.mean(sdr_values)),
            "std": float(np.std(sdr_values)),
            "min": float(np.min(sdr_values)),
            "max": float(np.max(sdr_values)),
            "median": float(np.median(sdr_values))
        }
    }


if __name__ == "__main__":
    print("=== Difficulty Metrics Test ===\n")
    
    # Test with synthetic signals
    np.random.seed(42)
    
    # Create reference signal (simulated vocal)
    t = np.linspace(0, 1, 44100)  # 1 second at 44.1kHz
    ref = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz sine wave
    
    # Test 1: Perfect reconstruction
    est_perfect = ref.copy()
    sisdr_perfect = compute_sisdr(est_perfect, ref)
    sdr_perfect = compute_sdr(est_perfect, ref)
    print(f"Perfect reconstruction:")
    print(f"  SI-SDR: {sisdr_perfect:.1f} dB")
    print(f"  SDR: {sdr_perfect:.1f} dB")
    print()
    
    # Test 2: Scaled version (SI-SDR should be high, SDR lower)
    est_scaled = ref * 2.0
    sisdr_scaled = compute_sisdr(est_scaled, ref)
    sdr_scaled = compute_sdr(est_scaled, ref)
    print(f"Scaled (2x amplitude):")
    print(f"  SI-SDR: {sisdr_scaled:.1f} dB (scale-invariant)")
    print(f"  SDR: {sdr_scaled:.1f} dB")
    print()
    
    # Test 3: Noisy estimate
    noise = np.random.randn(len(ref)) * 0.1
    est_noisy = ref + noise
    sisdr_noisy = compute_sisdr(est_noisy, ref)
    sdr_noisy = compute_sdr(est_noisy, ref)
    print(f"Noisy (SNR ~14dB):")
    print(f"  SI-SDR: {sisdr_noisy:.1f} dB")
    print(f"  SDR: {sdr_noisy:.1f} dB")
    print()
    
    # Test 4: Very noisy
    noise_heavy = np.random.randn(len(ref)) * 0.5
    est_very_noisy = ref + noise_heavy
    sisdr_very_noisy = compute_sisdr(est_very_noisy, ref)
    sdr_very_noisy = compute_sdr(est_very_noisy, ref)
    print(f"Very noisy (SNR ~0dB):")
    print(f"  SI-SDR: {sisdr_very_noisy:.1f} dB")
    print(f"  SDR: {sdr_very_noisy:.1f} dB")
    print()
    
    print("✓ Difficulty metrics module ready!")
