"""
FLX4-Net Track A Repetition Cue r(t)

Measures short-term repetition by comparing current segment
to recent past segments (within 12 seconds).

Blueprint Policy:
    - max_lookback: 12 seconds
    - Only considers PAST segments (no future lookahead)
    - Aggregation: max similarity among candidates
    - Candidates must not overlap with current segment

Interpretation:
    - High r(t): Current segment is similar to recent content
                 → Easier for DJ to control (predictable)
    - Low r(t):  Current segment is novel/different
                 → Harder to blend (unexpected content)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def get_repetition_candidates(
    current_idx: int,
    segments: List[dict],
    max_lookback_sec: float = 12.0,
    min_gap_sec: float = 0.0
) -> List[int]:
    """Get candidate segment indices for repetition comparison.
    
    Candidates are past segments within the lookback window
    that don't overlap with the current segment.
    
    Args:
        current_idx: Index of current segment
        segments: List of segment dicts with start_sec, end_sec
        max_lookback_sec: Maximum time to look back (default: 12s)
        min_gap_sec: Minimum gap between segments (default: 0)
        
    Returns:
        List of candidate segment indices
    """
    if current_idx == 0:
        return []  # No past segments
    
    current_seg = segments[current_idx]
    current_start = current_seg['start_sec']
    
    candidates = []
    
    for i in range(current_idx):
        past_seg = segments[i]
        past_end = past_seg['end_sec']
        past_start = past_seg['start_sec']
        
        # Check: past segment must end before current starts
        if past_end > current_start - min_gap_sec:
            continue  # Overlaps or too close
        
        # Check: within lookback window
        time_diff = current_start - past_end
        if time_diff <= max_lookback_sec:
            candidates.append(i)
    
    return candidates


def compute_repetition_cue(
    current_idx: int,
    similarity_matrix: np.ndarray,
    segments: List[dict],
    max_lookback_sec: float = 12.0,
    aggregation: str = "max",
    top_k: int = 3
) -> Tuple[float, int]:
    """Compute repetition cue r(t) for a single segment.
    
    Args:
        current_idx: Index of current segment
        similarity_matrix: Pairwise similarity matrix (n_seg, n_seg)
        segments: List of segment dicts
        max_lookback_sec: Maximum lookback window
        aggregation: How to aggregate similarities ("max" or "top_k_mean")
        top_k: Number of top similarities for top_k_mean
        
    Returns:
        Tuple of (r_value, n_candidates)
    """
    # Get candidates
    candidates = get_repetition_candidates(
        current_idx=current_idx,
        segments=segments,
        max_lookback_sec=max_lookback_sec
    )
    
    if len(candidates) == 0:
        # No candidates: return 0 (or could use NaN)
        return 0.0, 0
    
    # Get similarities to candidates
    similarities = similarity_matrix[current_idx, candidates]
    
    # Aggregate
    if aggregation == "max":
        r_value = float(np.max(similarities))
    elif aggregation == "top_k_mean":
        k = min(top_k, len(similarities))
        top_k_sims = np.sort(similarities)[-k:]
        r_value = float(np.mean(top_k_sims))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return r_value, len(candidates)


def compute_track_repetition(
    similarity_matrix: np.ndarray,
    segments: List[dict],
    max_lookback_sec: float = 12.0,
    aggregation: str = "max",
    top_k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute repetition cue for all segments in a track.
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        segments: List of segment dicts
        max_lookback_sec: Maximum lookback window
        aggregation: Aggregation method
        top_k: Number of top similarities (if using top_k_mean)
        
    Returns:
        Tuple of:
            - r_values: Array of r(t) values, shape (n_segments,)
            - candidate_counts: Array of candidate counts
    """
    n_segments = len(segments)
    r_values = np.zeros(n_segments)
    candidate_counts = np.zeros(n_segments, dtype=int)
    
    for i in range(n_segments):
        r_val, n_cand = compute_repetition_cue(
            current_idx=i,
            similarity_matrix=similarity_matrix,
            segments=segments,
            max_lookback_sec=max_lookback_sec,
            aggregation=aggregation,
            top_k=top_k
        )
        r_values[i] = r_val
        candidate_counts[i] = n_cand
    
    return r_values, candidate_counts


class RepetitionComputer:
    """Computes repetition cue r(t) for track segments.
    
    Usage:
        computer = RepetitionComputer(
            max_lookback_sec=12.0,
            aggregation="max"
        )
        r_values, counts = computer.compute(similarity_matrix, segments)
    """
    
    def __init__(
        self,
        max_lookback_sec: float = 12.0,
        aggregation: str = "max",
        top_k: int = 3
    ):
        """Initialize RepetitionComputer.
        
        Args:
            max_lookback_sec: Maximum lookback window (Blueprint: 12s)
            aggregation: "max" or "top_k_mean"
            top_k: For top_k_mean aggregation
        """
        self.max_lookback_sec = max_lookback_sec
        self.aggregation = aggregation
        self.top_k = top_k
    
    def compute(
        self,
        similarity_matrix: np.ndarray,
        segments: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute r(t) for all segments.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            segments: List of segment dicts
            
        Returns:
            Tuple of (r_values, candidate_counts)
        """
        return compute_track_repetition(
            similarity_matrix=similarity_matrix,
            segments=segments,
            max_lookback_sec=self.max_lookback_sec,
            aggregation=self.aggregation,
            top_k=self.top_k
        )
    
    def compute_single(
        self,
        current_idx: int,
        similarity_matrix: np.ndarray,
        segments: List[dict]
    ) -> Tuple[float, int]:
        """Compute r(t) for a single segment.
        
        Args:
            current_idx: Segment index
            similarity_matrix: Pairwise similarity matrix
            segments: List of segment dicts
            
        Returns:
            Tuple of (r_value, n_candidates)
        """
        return compute_repetition_cue(
            current_idx=current_idx,
            similarity_matrix=similarity_matrix,
            segments=segments,
            max_lookback_sec=self.max_lookback_sec,
            aggregation=self.aggregation,
            top_k=self.top_k
        )


if __name__ == "__main__":
    print("=== Repetition Cue Test ===\n")
    
    # Create test scenario
    # Segments at 0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5 seconds
    # Window: 3s, Hop: 1.5s
    n_segments = 10
    segments = []
    for i in range(n_segments):
        start = i * 1.5
        end = start + 3.0
        segments.append({
            'seg_id': f'test_{i:04d}',
            'seg_idx': i,
            'start_sec': start,
            'end_sec': end
        })
    
    print("Segments:")
    for seg in segments:
        print(f"  {seg['seg_id']}: [{seg['start_sec']:.1f}, {seg['end_sec']:.1f}]s")
    
    # Create fake similarity matrix
    # Assume seg0, seg2, seg4 are similar (verse)
    # seg1, seg3, seg5 are similar (chorus)
    np.random.seed(42)
    sim_matrix = np.eye(n_segments) * 0.5 + np.random.rand(n_segments, n_segments) * 0.3
    sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Symmetric
    
    # Make some segments more similar
    for i in [0, 2, 4, 6, 8]:
        for j in [0, 2, 4, 6, 8]:
            if i != j:
                sim_matrix[i, j] = 0.85
    
    for i in [1, 3, 5, 7, 9]:
        for j in [1, 3, 5, 7, 9]:
            if i != j:
                sim_matrix[i, j] = 0.80
    
    np.fill_diagonal(sim_matrix, 1.0)
    
    print("\nComputing r(t) with max_lookback=12s...")
    computer = RepetitionComputer(max_lookback_sec=12.0, aggregation="max")
    r_values, counts = computer.compute(sim_matrix, segments)
    
    print("\nResults:")
    print("  idx  start   r(t)   candidates")
    print("  ---  -----   ----   ----------")
    for i, (r, c) in enumerate(zip(r_values, counts)):
        print(f"  {i:3d}  {segments[i]['start_sec']:5.1f}s  {r:.3f}  {c}")
    
    print(f"\nMean r(t): {np.mean(r_values):.3f}")
    print(f"Std r(t):  {np.std(r_values):.3f}")
    
    print("\n✓ Repetition module ready!")
