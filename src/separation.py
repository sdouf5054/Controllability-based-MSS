"""
FLX4-Net Track A Separation Module

Runs vocal separation using Music-Source-Separation-Training.
Caches results to avoid redundant computation.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Literal
import warnings

from .config import TrackAConfig
from .audio_utils import load_audio, align_lengths

# Type alias for alignment policy
AlignmentPolicy = Literal["truncate_to_shorter", "pad_to_longer"]


def get_cache_path(
    cache_root: Path,
    model_name: str,
    track_id: str,
    stem: str = "vocals"
) -> Path:
    """Get cache path for separated stem.
    
    Cache structure:
        cache_root/separated/{model_name}/{track_id}/{stem}.wav
    
    Args:
        cache_root: Cache root directory
        model_name: Separator model name (e.g., "htdemucs")
        track_id: Track identifier
        stem: Stem name (default: "vocals")
        
    Returns:
        Path to cached stem file
    """
    return cache_root / "separated" / model_name / track_id / f"{stem}.wav"


def is_cached(
    cache_root: Path,
    model_name: str,
    track_id: str,
    stem: str = "vocals"
) -> bool:
    """Check if separation result is cached.
    
    Args:
        cache_root: Cache root directory
        model_name: Separator model name
        track_id: Track identifier
        stem: Stem name
        
    Returns:
        True if cached file exists
    """
    cache_path = get_cache_path(cache_root, model_name, track_id, stem)
    return cache_path.exists()


def run_mss_inference(
    input_path: Path,
    output_dir: Path,
    mss_root: Path,
    model_type: str = "htdemucs",
    config_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    device_ids: List[int] = [0],
    extract_instrumental: bool = False
) -> bool:
    """Run Music-Source-Separation-Training inference.
    
    Args:
        input_path: Path to input audio file
        output_dir: Output directory for separated stems
        mss_root: Path to MSS Training repository
        model_type: Model type (htdemucs, mel_band_roformer, etc.)
        config_path: Path to model config (optional)
        checkpoint_path: Path to model checkpoint (optional)
        device_ids: GPU device IDs
        extract_instrumental: Only extract vocals/instrumental
        
    Returns:
        True if successful
    """
    inference_script = mss_root / "inference.py"
    
    if not inference_script.exists():
        raise FileNotFoundError(
            f"MSS inference.py not found at {inference_script}. "
            f"Please check mss_root path: {mss_root}"
        )
    
    # Build command
    cmd = [
        "python", str(inference_script),
        "--model_type", model_type,
        "--input_folder", str(input_path.parent),
        "--store_dir", str(output_dir),
        "--device_ids", ",".join(map(str, device_ids))
    ]
    
    # Add config if specified
    if config_path and config_path.exists():
        cmd.extend(["--config_path", str(config_path)])
    else:
        # Use default config based on model type
        default_config = mss_root / "configs" / f"config_musdb18_{model_type}.yaml"
        if default_config.exists():
            cmd.extend(["--config_path", str(default_config)])
    
    # Add checkpoint if specified
    if checkpoint_path and checkpoint_path.exists():
        cmd.extend(["--start_check_point", str(checkpoint_path)])
    
    if extract_instrumental:
        cmd.append("--extract_instrumental")
    
    # Run inference
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(mss_root)
        )
        
        if result.returncode != 0:
            print(f"MSS inference failed:")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error running MSS inference: {e}")
        return False


def run_torchaudio_demucs(
    input_path: Path,
    output_dir: Path,
    device: str = "cuda"
) -> bool:
    """Run separation using torchaudio's built-in Demucs.
    
    Fallback option if MSS Training is not available.
    
    Args:
        input_path: Path to input audio file
        output_dir: Output directory
        device: Device to use ("cuda" or "cpu")
        
    Returns:
        True if successful
    """
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
        
        # Load model
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model = bundle.get_model().to(device)
        sample_rate = bundle.sample_rate
        
        # Load audio
        waveform, sr = torchaudio.load(input_path)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform.to(device)
        
        # Separate
        with torch.no_grad():
            # Add batch dimension
            sources = model(waveform.unsqueeze(0))
            sources = sources.squeeze(0)
        
        # Save stems
        # Demucs order: drums, bass, other, vocals
        stem_names = ["drums", "bass", "other", "vocals"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, stem_name in enumerate(stem_names):
            stem_path = output_dir / f"{stem_name}.wav"
            torchaudio.save(
                str(stem_path),
                sources[i].cpu(),
                sample_rate
            )
        
        return True
        
    except Exception as e:
        print(f"Error running torchaudio Demucs: {e}")
        return False


def separate_track(
    mixture_path: Path,
    cache_root: Path,
    track_id: str,
    model_name: str = "htdemucs",
    mss_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    device_ids: List[int] = [0],
    force_rerun: bool = False,
    use_torchaudio_fallback: bool = True
) -> Optional[Path]:
    """Separate a single track and cache the result.
    
    Args:
        mixture_path: Path to mixture audio
        cache_root: Cache root directory
        track_id: Track identifier
        model_name: Separator model name
        mss_root: Path to MSS Training repo (optional)
        config_path: Model config path (optional)
        checkpoint_path: Model checkpoint path (optional)
        device_ids: GPU device IDs
        force_rerun: Force re-separation even if cached
        use_torchaudio_fallback: Use torchaudio if MSS fails
        
    Returns:
        Path to separated vocals, or None if failed
    """
    # Check cache
    vocal_path = get_cache_path(cache_root, model_name, track_id, "vocals")
    
    if vocal_path.exists() and not force_rerun:
        return vocal_path
    
    # Create output directory
    track_output_dir = vocal_path.parent
    track_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try MSS Training first
    success = False
    
    if mss_root and mss_root.exists():
        # Create temp input folder with just this file
        temp_input_dir = cache_root / "temp_input"
        temp_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy or link input file
        temp_input_path = temp_input_dir / mixture_path.name
        if not temp_input_path.exists():
            shutil.copy2(mixture_path, temp_input_path)
        
        # Create temp output folder
        temp_output_dir = cache_root / "temp_output"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        success = run_mss_inference(
            input_path=temp_input_path,
            output_dir=temp_output_dir,
            mss_root=mss_root,
            model_type=model_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device_ids=device_ids
        )
        
        if success:
            # MSS outputs to: temp_output/{model_name}/{filename_without_ext}/vocals.wav
            # Find the output
            possible_paths = [
                temp_output_dir / model_name / temp_input_path.stem / "vocals.wav",
                temp_output_dir / temp_input_path.stem / "vocals.wav",
                temp_output_dir / "vocals.wav"
            ]
            
            mss_vocal_path = None
            for p in possible_paths:
                if p.exists():
                    mss_vocal_path = p
                    break
            
            if mss_vocal_path:
                # Move to cache
                shutil.move(str(mss_vocal_path), str(vocal_path))
                
                # Also move other stems if present
                for stem in ["drums", "bass", "other"]:
                    stem_src = mss_vocal_path.parent / f"{stem}.wav"
                    stem_dst = track_output_dir / f"{stem}.wav"
                    if stem_src.exists():
                        shutil.move(str(stem_src), str(stem_dst))
            else:
                success = False
                print(f"  Warning: MSS output not found for {track_id}")
        
        # Cleanup temp files
        if temp_input_path.exists():
            temp_input_path.unlink()
    
    # Fallback to torchaudio
    if not success and use_torchaudio_fallback:
        print(f"  Falling back to torchaudio Demucs for {track_id}")
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        success = run_torchaudio_demucs(
            input_path=mixture_path,
            output_dir=track_output_dir,
            device=device
        )
    
    if success and vocal_path.exists():
        return vocal_path
    else:
        return None


def load_separated_vocal(
    cache_root: Path,
    model_name: str,
    track_id: str,
    sr: int = 44100,
    mono: bool = True
) -> Optional[Tuple]:
    """Load separated vocal from cache.
    
    Args:
        cache_root: Cache root directory
        model_name: Separator model name
        track_id: Track identifier
        sr: Target sample rate
        mono: Convert to mono
        
    Returns:
        (audio, sr) tuple, or None if not found
    """
    vocal_path = get_cache_path(cache_root, model_name, track_id, "vocals")
    
    if not vocal_path.exists():
        return None
    
    try:
        audio, sr_loaded = load_audio(vocal_path, sr=sr, mono=mono)
        return audio, sr_loaded
    except Exception as e:
        print(f"Error loading {vocal_path}: {e}")
        return None


def align_separated_with_gt(
    est_audio,
    gt_audio,
    policy: AlignmentPolicy = "truncate_to_shorter",
    max_diff_samples: int = 1000,
    warn: bool = True
):
    """Align separated vocal with ground truth.
    
    Args:
        est_audio: Estimated (separated) audio
        gt_audio: Ground truth audio
        policy: Alignment policy ("truncate_to_shorter" or "pad_to_longer")
        max_diff_samples: Max allowed difference before warning
        warn: Whether to print warnings
        
    Returns:
        (est_aligned, gt_aligned)
    """
    return align_lengths(
        est_audio,
        gt_audio,
        policy=policy,
        max_diff_samples=max_diff_samples,
        warn=warn
    )


def get_separation_stats(cache_root: Path, model_name: str) -> dict:
    """Get statistics about cached separations.
    
    Args:
        cache_root: Cache root directory
        model_name: Separator model name
        
    Returns:
        Dictionary with stats
    """
    sep_dir = cache_root / "separated" / model_name
    
    if not sep_dir.exists():
        return {"n_tracks": 0, "total_size_mb": 0}
    
    track_dirs = [d for d in sep_dir.iterdir() if d.is_dir()]
    
    total_size = 0
    for track_dir in track_dirs:
        for f in track_dir.glob("*.wav"):
            total_size += f.stat().st_size
    
    return {
        "n_tracks": len(track_dirs),
        "total_size_mb": total_size / (1024 * 1024),
        "cache_path": str(sep_dir)
    }


if __name__ == "__main__":
    print("=== Separation Module Test ===")
    print("\nThis module provides:")
    print("  - separate_track(): Run separation with caching")
    print("  - load_separated_vocal(): Load cached result")
    print("  - align_separated_with_gt(): Align est/gt lengths")
    print("\nUse 02_run_separator.py to run batch separation.")
