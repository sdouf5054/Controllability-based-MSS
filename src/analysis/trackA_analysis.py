"""
FLX4-Net Track A Analysis

Performs correlation analysis between:
    - Cues (r, s, b, p) and Difficulty (SI-SDR)
    - Proxies and Difficulty (SI-SDR)
    - Cue contamination (est vs GT)

Also computes beat coverage statistics and generates final summary.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats as scipy_stats
import json
import warnings

from ..data_types import (
    ProxyAssociation,
    CueContamination,
    BeatCoverage,
    TrackASummary,
    SilenceCalibrationResult,
)


def compute_correlation(
    x: np.ndarray, y: np.ndarray, method: str = "spearman"
) -> Tuple[float, float, int]:
    """Compute correlation between two arrays.

    Args:
        x: First array
        y: Second array
        method: "spearman" or "pearson"

    Returns:
        Tuple of (correlation, p_value, n_samples)
    """
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]

    n = len(x_valid)

    if n < 3:
        return float("nan"), float("nan"), n

    if method == "spearman":
        rho, p = scipy_stats.spearmanr(x_valid, y_valid)
    elif method == "pearson":
        rho, p = scipy_stats.pearsonr(x_valid, y_valid)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(rho), float(p), n


def compute_cue_difficulty_correlation(
    cues_df: pd.DataFrame, difficulty_df: pd.DataFrame, method: str = "spearman"
) -> Dict[str, Dict]:
    """Compute correlation between cues and SI-SDR.

    Args:
        cues_df: DataFrame with cue values (r, s, b, p)
        difficulty_df: DataFrame with sisdr_vocal
        method: Correlation method

    Returns:
        Dictionary with correlation results for each cue
    """
    # Merge on seg_id
    merged = cues_df.merge(
        difficulty_df[["seg_id", "sisdr_vocal", "valid"]], on="seg_id"
    )

    # Filter valid difficulty
    merged = merged[merged["valid"] == True]

    results = {}

    for cue_name in ["r", "s", "b", "p"]:
        if cue_name == "b":
            # b(t) may have None values - filter to beat_available
            cue_data = merged[merged["beat_available"] == True]
            if len(cue_data) == 0:
                results[cue_name] = {
                    "correlation": float("nan"),
                    "p_value": float("nan"),
                    "n_samples": 0,
                    "method": method,
                    "interpretation": "no_valid_samples",
                }
                continue
            x = cue_data[cue_name].values
        else:
            cue_data = merged
            x = merged[cue_name].values

        y = cue_data["sisdr_vocal"].values

        rho, p, n = compute_correlation(x, y, method)

        # Interpret correlation
        if np.isnan(rho):
            interpretation = "insufficient_data"
        elif abs(rho) < 0.1:
            interpretation = "negligible"
        elif abs(rho) < 0.3:
            interpretation = "weak"
        elif abs(rho) < 0.5:
            interpretation = "moderate"
        elif abs(rho) < 0.7:
            interpretation = "strong"
        else:
            interpretation = "very_strong"

        results[cue_name] = {
            "correlation": rho,
            "p_value": p,
            "n_samples": n,
            "method": method,
            "interpretation": interpretation,
            "significant": p < 0.05 if not np.isnan(p) else False,
        }

    return results


def compute_proxy_difficulty_correlation(
    proxies_df: pd.DataFrame, difficulty_df: pd.DataFrame, method: str = "spearman"
) -> List[ProxyAssociation]:
    """Compute correlation between proxies and SI-SDR.

    Args:
        proxies_df: DataFrame with proxy values
        difficulty_df: DataFrame with sisdr_vocal
        method: Correlation method

    Returns:
        List of ProxyAssociation objects
    """
    # Merge on seg_id
    merged = proxies_df.merge(
        difficulty_df[["seg_id", "sisdr_vocal", "valid"]],
        on="seg_id",
        suffixes=("_proxy", "_diff"),
    )

    # Filter valid
    if "valid_proxy" in merged.columns and "valid_diff" in merged.columns:
        merged = merged[
            (merged["valid_proxy"] == True) & (merged["valid_diff"] == True)
        ]
    elif "valid" in merged.columns:
        merged = merged[merged["valid"] == True]

    proxy_cols = ["leakage_lowfreq", "instability", "transient_leakage"]
    results = []

    for proxy_name in proxy_cols:
        if proxy_name not in merged.columns:
            continue

        x = merged[proxy_name].values
        y = merged["sisdr_vocal"].values

        rho, p, n = compute_correlation(x, y, method)

        results.append(
            ProxyAssociation(
                proxy_name=proxy_name, spearman_rho=rho, p_value=p, n_samples=n
            )
        )

    return results


def compute_cue_contamination(
    cues_est_df: pd.DataFrame, cues_gt_df: pd.DataFrame
) -> List[CueContamination]:
    """Compute contamination between cues from estimated vs GT vocals.

    Contamination measures how different the cue values are when
    computed from separated vocals vs ground truth vocals.

    Args:
        cues_est_df: Cues computed from estimated (separated) vocals
        cues_gt_df: Cues computed from ground truth vocals

    Returns:
        List of CueContamination objects
    """
    # Merge on seg_id
    merged = cues_est_df.merge(cues_gt_df, on="seg_id", suffixes=("_est", "_gt"))

    results = []

    for cue_name in ["r", "s", "b", "p"]:
        est_col = f"{cue_name}_est"
        gt_col = f"{cue_name}_gt"

        if est_col not in merged.columns or gt_col not in merged.columns:
            continue

        # For b(t), filter to where both are available
        if cue_name == "b":
            valid_mask = (merged["beat_available_est"] == True) & (
                merged["beat_available_gt"] == True
            )
            data = merged[valid_mask]
        else:
            data = merged

        est_values = data[est_col].values
        gt_values = data[gt_col].values

        # Remove NaN
        valid = ~(np.isnan(est_values) | np.isnan(gt_values))
        est_valid = est_values[valid]
        gt_valid = gt_values[valid]

        if len(est_valid) < 3:
            results.append(
                CueContamination(
                    cue_name=cue_name,
                    correlation=float("nan"),
                    mean_diff=float("nan"),
                    std_diff=float("nan"),
                    n_samples=len(est_valid),
                )
            )
            continue

        # Compute correlation
        corr, _ = scipy_stats.pearsonr(est_valid, gt_valid)

        # Compute difference statistics
        diff = est_valid - gt_valid
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))

        results.append(
            CueContamination(
                cue_name=cue_name,
                correlation=float(corr),
                mean_diff=mean_diff,
                std_diff=std_diff,
                n_samples=len(est_valid),
            )
        )

    return results


def compute_beat_coverage(
    cues_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    beat_results: Optional[Dict[str, Dict]] = None,
) -> BeatCoverage:
    """Compute beat tracking coverage statistics.

    Args:
        cues_df: DataFrame with beat_available column
        segments_df: DataFrame with track_id column
        beat_results: Optional dict of track beat availability

    Returns:
        BeatCoverage object
    """
    n_segments_total = len(cues_df)
    n_segments_beat_available = int(cues_df["beat_available"].sum())

    # Get unique tracks
    if "track_id" in segments_df.columns:
        # Merge to get track_id for each segment
        merged = cues_df.merge(segments_df[["seg_id", "track_id"]], on="seg_id")
        unique_tracks = merged["track_id"].unique()
        n_tracks_total = len(unique_tracks)

        # Count tracks with at least one beat-available segment
        tracks_with_beats = merged[merged["beat_available"] == True][
            "track_id"
        ].unique()
        n_tracks_available = len(tracks_with_beats)
    else:
        n_tracks_total = 0
        n_tracks_available = 0

    return BeatCoverage(
        n_tracks_total=n_tracks_total,
        n_tracks_available=n_tracks_available,
        n_tracks_unavailable=n_tracks_total - n_tracks_available,
        track_coverage_ratio=(
            n_tracks_available / n_tracks_total if n_tracks_total > 0 else 0
        ),
        n_segments_total=n_segments_total,
        n_segments_beat_available=n_segments_beat_available,
        segment_coverage_ratio=(
            n_segments_beat_available / n_segments_total if n_segments_total > 0 else 0
        ),
    )


def generate_trackA_summary(
    dataset_name: str,
    manifest_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    cues_df: pd.DataFrame,
    proxies_df: pd.DataFrame,
    silence_result: Optional[SilenceCalibrationResult] = None,
    cue_correlations: Optional[Dict] = None,
    proxy_associations: Optional[List[ProxyAssociation]] = None,
    cue_contaminations: Optional[List[CueContamination]] = None,
    beat_coverage: Optional[BeatCoverage] = None,
    config_snapshot: Optional[Dict] = None,
) -> TrackASummary:
    """Generate complete Track A summary.

    Args:
        dataset_name: Name of dataset
        manifest_df: Track manifest
        segments_df: Segment index
        difficulty_df: Difficulty results
        cues_df: Cue results
        proxies_df: Proxy results
        silence_result: Silence calibration result
        cue_correlations: Cue-difficulty correlations
        proxy_associations: Proxy-difficulty associations
        cue_contaminations: Cue contamination results
        beat_coverage: Beat coverage statistics
        config_snapshot: Configuration snapshot

    Returns:
        TrackASummary object
    """
    # Basic counts
    n_tracks = len(manifest_df)
    n_segments_total = len(segments_df)
    n_segments_valid = len(difficulty_df[difficulty_df["valid"] == True])

    # Difficulty statistics
    valid_difficulty = difficulty_df[difficulty_df["valid"] == True]["sisdr_vocal"]
    if len(valid_difficulty) > 0:
        difficulty_stats = {
            "mean": float(np.mean(valid_difficulty)),
            "std": float(np.std(valid_difficulty)),
            "min": float(np.min(valid_difficulty)),
            "max": float(np.max(valid_difficulty)),
            "median": float(np.median(valid_difficulty)),
            "p5": float(np.percentile(valid_difficulty, 5)),
            "p25": float(np.percentile(valid_difficulty, 25)),
            "p75": float(np.percentile(valid_difficulty, 75)),
            "p95": float(np.percentile(valid_difficulty, 95)),
        }
    else:
        difficulty_stats = {}

    # Add cue correlations to difficulty stats
    if cue_correlations:
        difficulty_stats["cue_correlations"] = cue_correlations

    # Default silence result
    if silence_result is None:
        silence_result = SilenceCalibrationResult(
            initial_threshold_db=-40.0,
            final_threshold_db=-40.0,
            calibration_method="unknown",
            n_segments_total=n_segments_total,
            n_segments_silent=0,
            silent_ratio=0.0,
        )

    # Default proxy associations
    if proxy_associations is None:
        proxy_associations = []

    # Default cue contaminations
    if cue_contaminations is None:
        cue_contaminations = []

    # Default beat coverage
    if beat_coverage is None:
        beat_coverage = compute_beat_coverage(cues_df, segments_df)

    return TrackASummary(
        dataset_name=dataset_name,
        n_tracks=n_tracks,
        n_segments_total=n_segments_total,
        n_segments_valid=n_segments_valid,
        silence_info=silence_result,
        difficulty_stats=difficulty_stats,
        proxy_associations=proxy_associations,
        cue_contaminations=cue_contaminations,
        beat_coverage=beat_coverage,
        config_snapshot=config_snapshot or {},
    )


class TrackAAnalyzer:
    """Complete Track A analysis pipeline.

    Usage:
        analyzer = TrackAAnalyzer(
            tables_root=Path("tables"),
            results_root=Path("results/trackA")
        )

        # Load data
        analyzer.load_all_data()

        # Run analysis
        summary = analyzer.run_full_analysis()

        # Save results
        analyzer.save_results()
    """

    def __init__(
        self,
        tables_root: Path,
        results_root: Path,
        metadata_root: Optional[Path] = None,
    ):
        """Initialize analyzer.

        Args:
            tables_root: Path to tables directory
            results_root: Path to results directory
            metadata_root: Path to metadata directory
        """
        self.tables_root = Path(tables_root)
        self.results_root = Path(results_root)
        self.metadata_root = (
            Path(metadata_root)
            if metadata_root
            else self.tables_root.parent / "metadata"
        )

        # DataFrames
        self.manifest_df: Optional[pd.DataFrame] = None
        self.segments_df: Optional[pd.DataFrame] = None
        self.difficulty_df: Optional[pd.DataFrame] = None
        self.cues_df: Optional[pd.DataFrame] = None
        self.proxies_df: Optional[pd.DataFrame] = None

        # Results
        self.cue_correlations: Optional[Dict] = None
        self.proxy_associations: Optional[List[ProxyAssociation]] = None
        self.beat_coverage: Optional[BeatCoverage] = None
        self.summary: Optional[TrackASummary] = None

    def load_all_data(self):
        """Load all required data files."""
        print("Loading data files...")

        # Manifest
        manifest_path = self.metadata_root / "trackA_manifest.csv"
        if manifest_path.exists():
            self.manifest_df = pd.read_csv(manifest_path)
            print(f"  Manifest: {len(self.manifest_df)} tracks")

        # Segments
        segments_path = self.tables_root / "segments_trackA.parquet"
        if segments_path.exists():
            self.segments_df = pd.read_parquet(segments_path)
            print(f"  Segments: {len(self.segments_df)} segments")

        # Difficulty
        difficulty_path = self.tables_root / "difficulty_trackA.parquet"
        if difficulty_path.exists():
            self.difficulty_df = pd.read_parquet(difficulty_path)
            print(f"  Difficulty: {len(self.difficulty_df)} records")

        # Cues
        cues_path = self.tables_root / "cues_trackA.parquet"
        if cues_path.exists():
            self.cues_df = pd.read_parquet(cues_path)
            print(f"  Cues: {len(self.cues_df)} records")

        # Proxies
        proxies_path = self.tables_root / "proxies_trackA.parquet"
        if proxies_path.exists():
            self.proxies_df = pd.read_parquet(proxies_path)
            print(f"  Proxies: {len(self.proxies_df)} records")

    def compute_correlations(self, method: str = "spearman"):
        """Compute all correlations."""
        print("\nComputing correlations...")

        if self.cues_df is not None and self.difficulty_df is not None:
            self.cue_correlations = compute_cue_difficulty_correlation(
                self.cues_df, self.difficulty_df, method
            )
            print("  Cue-Difficulty correlations computed")

        if self.proxies_df is not None and self.difficulty_df is not None:
            self.proxy_associations = compute_proxy_difficulty_correlation(
                self.proxies_df, self.difficulty_df, method
            )
            print("  Proxy-Difficulty correlations computed")

    def compute_coverage(self):
        """Compute beat coverage statistics."""
        if self.cues_df is not None and self.segments_df is not None:
            self.beat_coverage = compute_beat_coverage(self.cues_df, self.segments_df)
            print("  Beat coverage computed")

    def run_full_analysis(
        self, dataset_name: str = "MUSDB18-HQ", config_snapshot: Optional[Dict] = None
    ) -> TrackASummary:
        """Run complete analysis pipeline.

        Args:
            dataset_name: Name of dataset
            config_snapshot: Configuration snapshot

        Returns:
            TrackASummary object
        """
        # Compute correlations
        self.compute_correlations()

        # Compute coverage
        self.compute_coverage()

        # Load silence calibration if available
        silence_result = None
        silence_path = self.results_root / "silence_calibration.json"
        if silence_path.exists():
            with open(silence_path, "r") as f:
                silence_data = json.load(f)
                silence_result = SilenceCalibrationResult(
                    initial_threshold_db=silence_data.get("initial_threshold_db", -40),
                    final_threshold_db=silence_data.get("final_threshold_db", -40),
                    calibration_method=silence_data.get(
                        "calibration_method", "unknown"
                    ),
                    n_segments_total=silence_data.get("n_segments_total", 0),
                    n_segments_silent=silence_data.get("n_segments_silent", 0),
                    silent_ratio=silence_data.get("silent_ratio", 0),
                    rms_stats=silence_data.get("rms_stats", {}),
                )

        # Generate summary
        self.summary = generate_trackA_summary(
            dataset_name=dataset_name,
            manifest_df=self.manifest_df,
            segments_df=self.segments_df,
            difficulty_df=self.difficulty_df,
            cues_df=self.cues_df,
            proxies_df=self.proxies_df,
            silence_result=silence_result,
            cue_correlations=self.cue_correlations,
            proxy_associations=self.proxy_associations,
            cue_contaminations=[],  # Skip contamination for now
            beat_coverage=self.beat_coverage,
            config_snapshot=config_snapshot,
        )

        return self.summary

    def save_results(self, output_path: Optional[Path] = None):
        """Save analysis results to JSON.

        Args:
            output_path: Output path (default: results_root/trackA_summary.json)
        """
        if output_path is None:
            output_path = self.results_root / "trackA_summary.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.summary is not None:
            self.summary.save_json(output_path)
            print(f"\nSaved summary to {output_path}")

        # Also save correlation details
        if self.cue_correlations is not None:
            corr_path = self.results_root / "cue_correlations.json"
            with open(corr_path, "w") as f:
                json.dump(self.cue_correlations, f, indent=2)
            print(f"Saved correlations to {corr_path}")

    def print_summary(self):
        """Print analysis summary to console."""
        if self.summary is None:
            print("No summary available. Run run_full_analysis() first.")
            return

        print("\n" + "=" * 60)
        print("TRACK A ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nDataset: {self.summary.dataset_name}")
        print(f"Tracks: {self.summary.n_tracks}")
        print(
            f"Segments: {self.summary.n_segments_total} total, {self.summary.n_segments_valid} valid"
        )

        # Difficulty stats
        print("\n--- Difficulty (SI-SDR) ---")
        ds = self.summary.difficulty_stats
        if ds.get("mean") is not None:
            print(f"  Mean: {ds['mean']:.2f} dB")
            print(f"  Std:  {ds['std']:.2f} dB")
            print(f"  Range: [{ds['min']:.2f}, {ds['max']:.2f}] dB")

        # Cue correlations
        if self.cue_correlations:
            print("\n--- Cue-Difficulty Correlations (Spearman) ---")
            for cue_name in ["r", "s", "b", "p"]:
                if cue_name in self.cue_correlations:
                    cc = self.cue_correlations[cue_name]
                    sig = "*" if cc.get("significant", False) else ""
                    print(
                        f"  {cue_name}(t): ρ = {cc['correlation']:+.3f}{sig} "
                        f"(p={cc['p_value']:.4f}, n={cc['n_samples']})"
                    )

        # Proxy correlations
        if self.proxy_associations:
            print("\n--- Proxy-Difficulty Correlations (Spearman) ---")
            for pa in self.proxy_associations:
                sig = "*" if pa.p_value < 0.05 else ""
                print(
                    f"  {pa.proxy_name}: ρ = {pa.spearman_rho:+.3f}{sig} "
                    f"(p={pa.p_value:.4f}, n={pa.n_samples})"
                )

        # Beat coverage
        bc = self.summary.beat_coverage
        print("\n--- Beat Coverage ---")
        print(
            f"  Tracks: {bc.n_tracks_available}/{bc.n_tracks_total} ({bc.track_coverage_ratio:.1%})"
        )
        print(
            f"  Segments: {bc.n_segments_beat_available}/{bc.n_segments_total} ({bc.segment_coverage_ratio:.1%})"
        )

        print("\n" + "=" * 60)


if __name__ == "__main__":
    print("=== Track A Analysis Module Test ===\n")

    # Create synthetic test data
    np.random.seed(42)
    n_segments = 100

    # Difficulty
    sisdr = np.random.normal(10, 2, n_segments)
    difficulty_df = pd.DataFrame(
        {
            "seg_id": [f"seg_{i:04d}" for i in range(n_segments)],
            "sisdr_vocal": sisdr,
            "valid": [True] * n_segments,
        }
    )

    # Cues (correlated with difficulty)
    r_values = 0.3 * sisdr / sisdr.max() + 0.5 + np.random.normal(0, 0.1, n_segments)
    s_values = 0.2 * sisdr / sisdr.max() + 0.4 + np.random.normal(0, 0.1, n_segments)
    b_values = 0.4 * sisdr / sisdr.max() + 0.3 + np.random.normal(0, 0.1, n_segments)
    p_values = -0.1 * sisdr / sisdr.max() + 0.5 + np.random.normal(0, 0.1, n_segments)

    cues_df = pd.DataFrame(
        {
            "seg_id": [f"seg_{i:04d}" for i in range(n_segments)],
            "r": np.clip(r_values, 0, 1),
            "s": np.clip(s_values, 0, 1),
            "b": np.clip(b_values, 0, 1),
            "p": np.clip(p_values, 0, 1),
            "beat_available": [True] * n_segments,
        }
    )

    # Compute correlations
    print("Computing cue-difficulty correlations...")
    correlations = compute_cue_difficulty_correlation(cues_df, difficulty_df)

    print("\nResults:")
    for cue_name, result in correlations.items():
        print(
            f"  {cue_name}(t): ρ = {result['correlation']:+.3f}, "
            f"p = {result['p_value']:.4f}, "
            f"interpretation = {result['interpretation']}"
        )

    print("\n✓ Analysis module ready!")
