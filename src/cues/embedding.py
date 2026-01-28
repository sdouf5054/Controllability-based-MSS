"""
FLX4-Net Track A Segment Embedding

Computes log-mel spectrogram embeddings for segments.
Used for similarity-based cues: r(t) and s(t).

Blueprint Policy:
    - log-mel spectrogram (128 mels)
    - L2 normalization
    - Cosine similarity
    - Mean pooling over time
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import warnings

# Type alias
EmbeddingArray = np.ndarray  # Shape: (n_mels,) or (n_segments, n_mels)


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 44100,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: float = 8000.0
) -> np.ndarray:
    """Compute log-mel spectrogram.
    
    Args:
        audio: Audio signal (1D)
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
        fmin: Minimum frequency for mel filterbank
        fmax: Maximum frequency for mel filterbank
        
    Returns:
        Log-mel spectrogram, shape (n_mels, n_frames)
    """
    import librosa
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec + 1e-10, ref=np.max)
    
    # Normalize to [0, 1] range per segment
    # This prevents all embeddings from being in the same negative region
    log_mel_min = log_mel.min()
    log_mel_max = log_mel.max()
    if log_mel_max - log_mel_min > 1e-6:
        log_mel = (log_mel - log_mel_min) / (log_mel_max - log_mel_min)
    else:
        log_mel = np.zeros_like(log_mel)
    
    return log_mel


def pool_spectrogram(
    spectrogram: np.ndarray,
    method: str = "mean"
) -> np.ndarray:
    """Pool spectrogram over time axis to get single embedding.
    
    Args:
        spectrogram: Shape (n_mels, n_frames)
        method: Pooling method ("mean" or "median")
        
    Returns:
        Pooled embedding, shape (n_mels,)
    """
    if method == "mean":
        return np.mean(spectrogram, axis=1)
    elif method == "median":
        return np.median(spectrogram, axis=1)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def normalize_embedding(
    embedding: np.ndarray,
    method: str = "l2"
) -> np.ndarray:
    """Normalize embedding vector.
    
    Args:
        embedding: Shape (n_mels,) or (n_segments, n_mels)
        method: Normalization method ("l2" or "none")
        
    Returns:
        Normalized embedding, same shape as input
    """
    if method == "none":
        return embedding
    
    if method == "l2":
        if embedding.ndim == 1:
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                return embedding / norm
            return embedding
        else:
            # Batch normalization
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            return embedding / norms
    
    raise ValueError(f"Unknown normalization method: {method}")


def compute_segment_embedding(
    audio: np.ndarray,
    sr: int = 44100,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: float = 8000.0,
    pooling: str = "mean",
    normalize: str = "l2"
) -> np.ndarray:
    """Compute embedding for a single audio segment.
    
    Pipeline:
        audio → log-mel → pool → zero-mean → L2 normalize → embedding
    
    Args:
        audio: Audio segment (1D)
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length
        fmin: Minimum frequency
        fmax: Maximum frequency
        pooling: Pooling method ("mean" or "median")
        normalize: Normalization ("l2" or "none")
        
    Returns:
        Embedding vector, shape (n_mels,)
    """
    # Check for silent/empty segment
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-10:
        # Return zero embedding (will have zero norm)
        return np.zeros(n_mels)
    
    # Compute log-mel spectrogram
    log_mel = compute_log_mel_spectrogram(
        audio, sr, n_mels, n_fft, hop_length, fmin, fmax
    )
    
    # Pool over time
    embedding = pool_spectrogram(log_mel, method=pooling)
    
    # Zero-mean normalization (critical for distinguishing different content!)
    # This removes the "DC offset" that makes all embeddings similar
    embedding = embedding - np.mean(embedding)
    
    # L2 normalize
    embedding = normalize_embedding(embedding, method=normalize)
    
    return embedding


def compute_cosine_similarity(
    emb1: np.ndarray,
    emb2: np.ndarray
) -> float:
    """Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding, shape (n_mels,)
        emb2: Second embedding, shape (n_mels,)
        
    Returns:
        Cosine similarity in range [-1, 1]
    """
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0  # One or both embeddings are zero
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def compute_similarity_matrix(
    embeddings: np.ndarray
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: Shape (n_segments, n_mels)
        
    Returns:
        Similarity matrix, shape (n_segments, n_segments)
    """
    # Normalize embeddings (in case not already normalized)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms
    
    # Compute similarity matrix via dot product
    similarity = np.dot(normalized, normalized.T)
    
    return similarity


class EmbeddingComputer:
    """Computes and caches embeddings for a track's segments.
    
    Usage:
        computer = EmbeddingComputer(
            sr=44100,
            n_mels=128,
            pooling="mean",
            normalize="l2"
        )
        
        # Load track audio
        computer.set_track_audio(audio)
        
        # Compute embedding for segment
        emb = computer.get_segment_embedding(start_sample, end_sample)
        
        # Or compute all at once
        embeddings = computer.compute_all_embeddings(segments)
    """
    
    def __init__(
        self,
        sr: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 20.0,
        fmax: float = 8000.0,
        pooling: str = "mean",
        normalize: str = "l2"
    ):
        """Initialize EmbeddingComputer.
        
        Args:
            sr: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length
            fmin: Minimum frequency
            fmax: Maximum frequency
            pooling: Pooling method
            normalize: Normalization method
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.pooling = pooling
        self.normalize = normalize
        
        # Track-level cache
        self._track_audio: Optional[np.ndarray] = None
        self._embeddings_cache: Dict[Tuple[int, int], np.ndarray] = {}
    
    def set_track_audio(self, audio: np.ndarray):
        """Set the current track's audio and clear cache.
        
        Args:
            audio: Full track audio (1D)
        """
        self._track_audio = audio
        self._embeddings_cache.clear()
    
    def get_segment_embedding(
        self,
        start_sample: int,
        end_sample: int
    ) -> np.ndarray:
        """Get embedding for a segment (with caching).
        
        Args:
            start_sample: Start sample index
            end_sample: End sample index
            
        Returns:
            Embedding vector, shape (n_mels,)
        """
        if self._track_audio is None:
            raise RuntimeError("No track audio set. Call set_track_audio() first.")
        
        cache_key = (start_sample, end_sample)
        
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        
        # Extract segment
        segment = self._track_audio[start_sample:end_sample]
        
        # Compute embedding
        embedding = compute_segment_embedding(
            audio=segment,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            pooling=self.pooling,
            normalize=self.normalize
        )
        
        # Cache and return
        self._embeddings_cache[cache_key] = embedding
        return embedding
    
    def compute_all_embeddings(
        self,
        segments: List[dict]
    ) -> np.ndarray:
        """Compute embeddings for all segments.
        
        Args:
            segments: List of segment dicts with start_sample, end_sample
            
        Returns:
            Embeddings matrix, shape (n_segments, n_mels)
        """
        embeddings = []
        
        for seg in segments:
            emb = self.get_segment_embedding(
                seg['start_sample'],
                seg['end_sample']
            )
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def compute_similarity_to_segment(
        self,
        target_idx: int,
        all_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute similarity of target segment to all others.
        
        Args:
            target_idx: Index of target segment
            all_embeddings: All segment embeddings, shape (n_segments, n_mels)
            
        Returns:
            Similarity scores, shape (n_segments,)
        """
        target_emb = all_embeddings[target_idx]
        
        # Normalize target
        target_norm = np.linalg.norm(target_emb)
        if target_norm < 1e-10:
            return np.zeros(len(all_embeddings))
        target_normalized = target_emb / target_norm
        
        # Normalize all
        norms = np.linalg.norm(all_embeddings, axis=1)
        norms = np.maximum(norms, 1e-10)
        all_normalized = all_embeddings / norms[:, np.newaxis]
        
        # Compute similarities
        similarities = np.dot(all_normalized, target_normalized)
        
        return similarities
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embeddings_cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Number of cached embeddings."""
        return len(self._embeddings_cache)


def compute_track_embeddings(
    audio: np.ndarray,
    segments: List[dict],
    sr: int = 44100,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: float = 8000.0,
    pooling: str = "mean",
    normalize: str = "l2"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute embeddings and similarity matrix for a track.
    
    Convenience function for computing all embeddings and
    their pairwise similarities.
    
    Args:
        audio: Full track audio
        segments: List of segment dicts
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length
        fmin: Minimum frequency
        fmax: Maximum frequency
        pooling: Pooling method
        normalize: Normalization method
        
    Returns:
        Tuple of:
            - embeddings: Shape (n_segments, n_mels)
            - similarity_matrix: Shape (n_segments, n_segments)
    """
    computer = EmbeddingComputer(
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        pooling=pooling,
        normalize=normalize
    )
    
    computer.set_track_audio(audio)
    embeddings = computer.compute_all_embeddings(segments)
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    return embeddings, similarity_matrix


if __name__ == "__main__":
    print("=== Embedding Module Test ===\n")
    
    import librosa
    
    # Create synthetic test audio
    np.random.seed(42)
    sr = 44100
    duration = 9.0  # 9 seconds = 3 segments at 3s window, 1.5s hop
    
    # Simulate audio with different sections
    t = np.linspace(0, duration, int(sr * duration))
    
    # Section 1: Low frequency (0-3s)
    audio1 = np.sin(2 * np.pi * 200 * t[:sr*3]) * 0.5
    
    # Section 2: High frequency (3-6s)
    audio2 = np.sin(2 * np.pi * 1000 * t[sr*3:sr*6]) * 0.5
    
    # Section 3: Similar to section 1 (6-9s)
    audio3 = np.sin(2 * np.pi * 200 * t[sr*6:]) * 0.5
    
    audio = np.concatenate([audio1, audio2, audio3])
    
    # Create segments
    segments = [
        {"start_sample": 0, "end_sample": sr * 3, "seg_id": "test_0"},
        {"start_sample": int(sr * 1.5), "end_sample": int(sr * 4.5), "seg_id": "test_1"},
        {"start_sample": sr * 3, "end_sample": sr * 6, "seg_id": "test_2"},
        {"start_sample": int(sr * 4.5), "end_sample": int(sr * 7.5), "seg_id": "test_3"},
        {"start_sample": sr * 6, "end_sample": sr * 9, "seg_id": "test_4"},
    ]
    
    # Compute embeddings
    print("Computing embeddings for 5 segments...")
    embeddings, sim_matrix = compute_track_embeddings(
        audio=audio,
        segments=segments,
        sr=sr
    )
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    print("\nSimilarity matrix (segments 0-4):")
    print("    ", end="")
    for i in range(5):
        print(f"  seg{i}", end="")
    print()
    
    for i in range(5):
        print(f"seg{i}", end="")
        for j in range(5):
            print(f"  {sim_matrix[i,j]:.2f}", end="")
        print()
    
    print("\nExpected pattern:")
    print("  - seg0 and seg4 should be similar (both low freq)")
    print("  - seg2 should be different from seg0, seg4 (high freq)")
    
    # Verify
    print(f"\nseg0-seg4 similarity: {sim_matrix[0, 4]:.3f} (should be high)")
    print(f"seg0-seg2 similarity: {sim_matrix[0, 2]:.3f} (should be low)")
    
    print("\n✓ Embedding module ready!")
