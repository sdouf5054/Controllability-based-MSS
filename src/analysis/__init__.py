"""
FLX4-Net Analysis Module

Provides correlation analysis, visualization, and summary statistics
for Track A results.
"""

from .trackA_analysis import (
    compute_cue_difficulty_correlation,
    compute_proxy_difficulty_correlation,
    compute_cue_contamination,
    compute_beat_coverage,
    generate_trackA_summary,
    TrackAAnalyzer
)

__all__ = [
    "compute_cue_difficulty_correlation",
    "compute_proxy_difficulty_correlation", 
    "compute_cue_contamination",
    "compute_beat_coverage",
    "generate_trackA_summary",
    "TrackAAnalyzer"
]
