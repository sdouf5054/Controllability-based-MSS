"""
FLX4-Net Track A Structure Cue s(t)

Measures long-range structural similarity by comparing current segment
to distant segments (>= 30 seconds apart).

Blueprint Policy:
    - min_distance: 30 seconds
    - Considers BOTH past and future segments (unlike r(t))
    - Aggregation: top-k mean similarity
    - Captures verse-chorus-verse patterns

Interpretation:
    - High s(t): Current segment has structural echoes elsewhere
                 → Common pattern (verse, chorus) → More controllable
    - Low s(t):  Current segment is structurally unique (bridge, intro)
                 → Less predictable → Harder to control

Key Difference from r(t):
    - r(t): Short-term (past 12s only) → Local repetition
    - s(t): Long-range (>= 30s away) → Global structure
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def get_structure_candidates(
    current_idx: int,
    segments: List[dict],
    min_distance_sec: float = 30.0
) -> List[int]:
    """Get candidate segment indices for structure comparison.
    
    Candidates are segments at least min_distance_sec away
    from the current segment (both past and future).
    
    Args:
        current_idx: Index of current segment
        segments: List of segment dicts with start_sec, end_sec
        min_distance_sec: Minimum time distance (default: 30s)
        
    Returns:
        List of candidate segment indices
    """
    current_seg = segments[current_idx]
    current_start = current_seg['start_sec']
    current_end = current_seg['end_sec']
    current_center = (current_start + current_end) / 2
    
    candidates = []
    
    for i, seg in enumerate(segments):
        if i == current_idx:
            continue
        
        seg_start = seg['start_sec']
        seg_end = seg['end_sec']
        seg_center = (seg_start + seg_end) / 2
        
        # Compute distance between segment centers
        distance = abs(seg_center - current_center)
        
        if distance >= min_distance_sec:
            candidates.append(i)
    
    return candidates


def compute_structure_cue(
    current_idx: int,
    similarity_matrix: np.ndarray,
    segments: List[dict],
    min_distance_sec: float = 30.0,
    aggregation: str = "top_k_mean",
    top_k: int = 3
) -> Tuple[float, int]:
    """Compute structure cue s(t) for a single segment.
    
    Args:
        current_idx: Index of current segment
        similarity_matrix: Pairwise similarity matrix (n_seg, n_seg)
        segments: List of segment dicts
        min_distance_sec: Minimum distance threshold
        aggregation: How to aggregate ("top_k_mean" recommended)
        top_k: Number of top similarities to average
        
    Returns:
        Tuple of (s_value, n_candidates)
    """
    # Get candidates
    candidates = get_structure_candidates(
        current_idx=current_idx,
        segments=segments,
        min_distance_sec=min_distance_sec
    )
    
    if len(candidates) == 0:
        # No candidates: track is too short
        return 0.0, 0
    
    # Get similarities to candidates
    similarities = similarity_matrix[current_idx, candidates]
    
    # Aggregate
    if aggregation == "top_k_mean":
        k = min(top_k, len(similarities))
        top_k_sims = np.sort(similarities)[-k:]
        s_value = float(np.mean(top_k_sims))
    elif aggregation == "max":
        s_value = float(np.max(similarities))
    elif aggregation == "mean":
        s_value = float(np.mean(similarities))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return s_value, len(candidates)


def compute_track_structure(
    similarity_matrix: np.ndarray,
    segments: List[dict],
    min_distance_sec: float = 30.0,
    aggregation: str = "top_k_mean",
    top_k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute structure cue for all segments in a track.
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        segments: List of segment dicts
        min_distance_sec: Minimum distance threshold
        aggregation: Aggregation method
        top_k: Number of top similarities
        
    Returns:
        Tuple of:
            - s_values: Array of s(t) values, shape (n_segments,)
            - candidate_counts: Array of candidate counts
    """
    n_segments = len(segments)
    s_values = np.zeros(n_segments)
    candidate_counts = np.zeros(n_segments, dtype=int)
    
    for i in range(n_segments):
        s_val, n_cand = compute_structure_cue(
            current_idx=i,
            similarity_matrix=similarity_matrix,
            segments=segments,
            min_distance_sec=min_distance_sec,
            aggregation=aggregation,
            top_k=top_k
        )
        s_values[i] = s_val
        candidate_counts[i] = n_cand
    
    return s_values, candidate_counts


class StructureComputer:
    """Computes structure cue s(t) for track segments.
    
    Usage:
        computer = StructureComputer(
            min_distance_sec=30.0,
            aggregation="top_k_mean",
            top_k=3
        )
        s_values, counts = computer.compute(similarity_matrix, segments)
    """
    
    def __init__(
        self,
        min_distance_sec: float = 30.0,
        aggregation: str = "top_k_mean",
        top_k: int = 3
    ):
        """Initialize StructureComputer.
        
        Args:
            min_distance_sec: Minimum distance threshold (Blueprint: 30s)
            aggregation: "top_k_mean", "max", or "mean"
            top_k: For top_k_mean aggregation
        """
        self.min_distance_sec = min_distance_sec
        self.aggregation = aggregation
        self.top_k = top_k
    
    def compute(
        self,
        similarity_matrix: np.ndarray,
        segments: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute s(t) for all segments.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            segments: List of segment dicts
            
        Returns:
            Tuple of (s_values, candidate_counts)
        """
        return compute_track_structure(
            similarity_matrix=similarity_matrix,
            segments=segments,
            min_distance_sec=self.min_distance_sec,
            aggregation=self.aggregation,
            top_k=self.top_k
        )
    
    def compute_single(
        self,
        current_idx: int,
        similarity_matrix: np.ndarray,
        segments: List[dict]
    ) -> Tuple[float, int]:
        """Compute s(t) for a single segment.
        
        Args:
            current_idx: Segment index
            similarity_matrix: Pairwise similarity matrix
            segments: List of segment dicts
            
        Returns:
            Tuple of (s_value, n_candidates)
        """
        return compute_structure_cue(
            current_idx=current_idx,
            similarity_matrix=similarity_matrix,
            segments=segments,
            min_distance_sec=self.min_distance_sec,
            aggregation=self.aggregation,
            top_k=self.top_k
        )


def analyze_structure_coverage(
    segments: List[dict],
    min_distance_sec: float = 30.0
) -> Dict:
    """Analyze structure candidate coverage for a track.
    
    Useful for understanding how many segments have
    valid structure candidates.
    
    Args:
        segments: List of segment dicts
        min_distance_sec: Minimum distance threshold
        
    Returns:
        Dictionary with coverage statistics
    """
    n_segments = len(segments)
    
    if n_segments == 0:
        return {
            "n_segments": 0,
            "n_with_candidates": 0,
            "coverage_ratio": 0,
            "track_duration_sec": 0,
            "min_distance_sec": min_distance_sec
        }
    
    track_duration = max(seg['end_sec'] for seg in segments)
    
    n_with_candidates = 0
    total_candidates = 0
    
    for i in range(n_segments):
        candidates = get_structure_candidates(i, segments, min_distance_sec)
        if len(candidates) > 0:
            n_with_candidates += 1
            total_candidates += len(candidates)
    
    return {
        "n_segments": n_segments,
        "n_with_candidates": n_with_candidates,
        "n_without_candidates": n_segments - n_with_candidates,
        "coverage_ratio": n_with_candidates / n_segments,
        "avg_candidates": total_candidates / n_segments if n_segments > 0 else 0,
        "track_duration_sec": track_duration,
        "min_distance_sec": min_distance_sec,
        "note": "Segments without candidates are near track edges (< 30s from start/end)"
    }


if __name__ == "__main__":
    print("=== Structure Cue Test ===\n")
    
    # Create test scenario: 2-minute track
    # Segments every 1.5s with 3s window
    # Total: ~80 segments
    duration_sec = 120.0
    hop_sec = 1.5
    window_sec = 3.0
    
    n_segments = int((duration_sec - window_sec) / hop_sec) + 1
    segments = []
    for i in range(n_segments):
        start = i * hop_sec
        end = start + window_sec
        segments.append({
            'seg_id': f'test_{i:04d}',
            'seg_idx': i,
            'start_sec': start,
            'end_sec': end
        })
    
    print(f"Track duration: {duration_sec}s")
    print(f"Number of segments: {n_segments}")
    print(f"First segment: [{segments[0]['start_sec']}, {segments[0]['end_sec']}]s")
    print(f"Last segment: [{segments[-1]['start_sec']}, {segments[-1]['end_sec']}]s")
    
    # Analyze coverage
    coverage = analyze_structure_coverage(segments, min_distance_sec=30.0)
    print(f"\nStructure coverage analysis:")
    print(f"  Segments with candidates: {coverage['n_with_candidates']}/{coverage['n_segments']}")
    print(f"  Coverage ratio: {coverage['coverage_ratio']:.1%}")
    print(f"  Avg candidates per segment: {coverage['avg_candidates']:.1f}")
    
    # Create similarity matrix with structure pattern
    # Simulate: verse at 0-30s, chorus at 30-60s, verse at 60-90s, outro at 90-120s
    np.random.seed(42)
    sim_matrix = np.random.rand(n_segments, n_segments) * 0.3
    sim_matrix = (sim_matrix + sim_matrix.T) / 2
    
    # Verse segments (0-30s and 60-90s) should be similar
    verse1_range = range(0, 20)  # ~0-30s
    verse2_range = range(40, 60)  # ~60-90s
    
    for i in verse1_range:
        for j in verse2_range:
            sim_matrix[i, j] = 0.85
            sim_matrix[j, i] = 0.85
    
    # Chorus segments (30-60s) similar to each other
    chorus_range = range(20, 40)  # ~30-60s
    for i in chorus_range:
        for j in chorus_range:
            if i != j:
                sim_matrix[i, j] = 0.80
    
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Compute structure cue
    print("\nComputing s(t) with min_distance=30s...")
    computer = StructureComputer(min_distance_sec=30.0, aggregation="top_k_mean", top_k=3)
    s_values, counts = computer.compute(sim_matrix, segments)
    
    # Show selected results
    print("\nSelected results:")
    print("  idx  time    s(t)   candidates  section")
    print("  ---  ------  ----   ----------  -------")
    for i in [0, 10, 20, 30, 40, 50, 60, 70, n_segments-1]:
        if i < n_segments:
            section = "verse1" if i < 20 else ("chorus" if i < 40 else ("verse2" if i < 60 else "outro"))
            print(f"  {i:3d}  {segments[i]['start_sec']:5.1f}s  {s_values[i]:.3f}  {counts[i]:3d}         {section}")
    
    print(f"\nMean s(t): {np.mean(s_values):.3f}")
    print(f"Std s(t):  {np.std(s_values):.3f}")
    
    # Expected: verse1 and verse2 segments should have high s(t)
    verse1_mean = np.mean(s_values[list(verse1_range)])
    verse2_mean = np.mean(s_values[list(verse2_range)])
    outro_mean = np.mean(s_values[60:])
    
    print(f"\nBy section:")
    print(f"  Verse 1 (0-30s) mean s(t): {verse1_mean:.3f}")
    print(f"  Verse 2 (60-90s) mean s(t): {verse2_mean:.3f}")
    print(f"  Outro (90-120s) mean s(t): {outro_mean:.3f}")
    
    print("\n✓ Structure module ready!")
