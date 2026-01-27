"""
FLX4-Net Cues Module

Computes the four controllability cues for Track A:
    - r(t): Repetition cue (short-term similarity)
    - s(t): Structure cue (long-range similarity)
    - b(t): Beat synchrony cue (onset-beat alignment)
    - p(t): Processing cue (texture signature)
"""

from .embedding import (
    compute_segment_embedding,
    compute_log_mel_spectrogram,
    compute_cosine_similarity,
    compute_similarity_matrix,
    compute_track_embeddings,
    EmbeddingComputer
)

from .repetition import (
    get_repetition_candidates,
    compute_repetition_cue,
    compute_track_repetition,
    RepetitionComputer
)

from .structure import (
    get_structure_candidates,
    compute_structure_cue,
    compute_track_structure,
    analyze_structure_coverage,
    StructureComputer
)

from .beat_sync import (
    detect_vocal_onsets,
    get_beats_in_segment,
    compute_onset_beat_alignment,
    compute_beat_sync_cue,
    BeatSyncComputer
)

from .processing import (
    compute_spectral_features,
    compute_processing_cue,
    ProcessingComputer,
    calibrate_normalization_params
)

__all__ = [
    # Embedding
    "compute_segment_embedding",
    "compute_log_mel_spectrogram",
    "compute_cosine_similarity",
    "compute_similarity_matrix",
    "compute_track_embeddings",
    "EmbeddingComputer",
    # Repetition r(t)
    "get_repetition_candidates",
    "compute_repetition_cue",
    "compute_track_repetition",
    "RepetitionComputer",
    # Structure s(t)
    "get_structure_candidates",
    "compute_structure_cue",
    "compute_track_structure",
    "analyze_structure_coverage",
    "StructureComputer",
    # Beat sync b(t)
    "detect_vocal_onsets",
    "get_beats_in_segment",
    "compute_onset_beat_alignment",
    "compute_beat_sync_cue",
    "BeatSyncComputer",
    # Processing p(t)
    "compute_spectral_features",
    "compute_processing_cue",
    "ProcessingComputer",
    "calibrate_normalization_params",
]
