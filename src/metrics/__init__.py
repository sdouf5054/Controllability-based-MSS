"""
FLX4-Net Metrics Module

Provides difficulty metrics (SI-SDR, SDR) and artifact proxies
for vocal separation evaluation.
"""

from .difficulty import (
    compute_sisdr,
    compute_sdr,
    compute_segment_difficulty,
    compute_track_difficulty,
    compute_difficulty_stats,
    DifficultyComputer
)

from .proxies import (
    compute_lowfreq_energy_ratio,
    compute_temporal_instability,
    compute_transient_energy_ratio,
    compute_segment_proxies,
    compute_track_proxies,
    compute_proxy_stats,
    compute_proxy_difficulty_correlation,
    ProxyComputer
)

__all__ = [
    # Difficulty
    "compute_sisdr",
    "compute_sdr",
    "compute_segment_difficulty",
    "compute_track_difficulty",
    "compute_difficulty_stats",
    "DifficultyComputer",
    # Proxies
    "compute_lowfreq_energy_ratio",
    "compute_temporal_instability",
    "compute_transient_energy_ratio",
    "compute_segment_proxies",
    "compute_track_proxies",
    "compute_proxy_stats",
    "compute_proxy_difficulty_correlation",
    "ProxyComputer"
]
