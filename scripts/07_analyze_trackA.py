#!/usr/bin/env python3
"""
Script 07: Analyze Track A Results

Performs complete analysis of Track A pipeline results:
    - Cue-Difficulty correlations
    - Proxy-Difficulty correlations
    - Beat coverage statistics
    - Generates final summary report

Usage:
    python scripts/07_analyze_trackA.py --config configs/trackA_config.yaml
    python scripts/07_analyze_trackA.py --dry-run
    python scripts/07_analyze_trackA.py --no-plots

Input:
    - tables/difficulty_trackA.parquet
    - tables/cues_trackA.parquet
    - tables/proxies_trackA.parquet
    - tables/segments_trackA.parquet
    - metadata/trackA_manifest.csv

Output:
    - results/trackA/trackA_summary.json
    - results/trackA/cue_correlations.json
    - results/trackA/figures/ (optional plots)
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.analysis import (
    TrackAAnalyzer,
    compute_cue_difficulty_correlation,
    compute_proxy_difficulty_correlation,
    compute_beat_coverage,
)


def check_required_files(tables_root: Path, metadata_root: Path) -> Dict[str, bool]:
    """Check which required files exist."""
    files = {
        "manifest": metadata_root / "trackA_manifest.csv",
        "segments": tables_root / "segments_trackA.parquet",
        "difficulty": tables_root / "difficulty_trackA.parquet",
        "cues": tables_root / "cues_trackA.parquet",
        "proxies": tables_root / "proxies_trackA.parquet",
    }

    return {name: path.exists() for name, path in files.items()}


def print_correlation_table(correlations: Dict, title: str):
    """Print correlation results as a formatted table."""
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Cue':<15} {'ρ':>10} {'p-value':>12} {'n':>8} {'Sig.':>6}")
    print("-" * 60)

    for name, result in correlations.items():
        rho = result.get("correlation", float("nan"))
        p = result.get("p_value", float("nan"))
        n = result.get("n_samples", 0)
        sig = "*" if result.get("significant", False) else ""

        if np.isnan(rho):
            print(f"{name:<15} {'N/A':>10} {'N/A':>12} {n:>8} {sig:>6}")
        else:
            print(f"{name:<15} {rho:>+10.4f} {p:>12.4e} {n:>8} {sig:>6}")

    print("-" * 60)
    print("* = significant at p < 0.05")


def print_proxy_table(associations: list, title: str):
    """Print proxy correlation results."""
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Proxy':<20} {'ρ':>10} {'p-value':>12} {'n':>8} {'Sig.':>6}")
    print("-" * 60)

    for pa in associations:
        sig = "*" if pa.p_value < 0.05 else ""
        if np.isnan(pa.spearman_rho):
            print(
                f"{pa.proxy_name:<20} {'N/A':>10} {'N/A':>12} {pa.n_samples:>8} {sig:>6}"
            )
        else:
            print(
                f"{pa.proxy_name:<20} {pa.spearman_rho:>+10.4f} {pa.p_value:>12.4e} {pa.n_samples:>8} {sig:>6}"
            )

    print("-" * 60)


def generate_plots(
    cues_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    proxies_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate analysis plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("\nWarning: matplotlib/seaborn not available. Skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge data
    merged = cues_df.merge(
        difficulty_df[["seg_id", "sisdr_vocal", "valid"]], on="seg_id"
    )
    merged = merged[merged["valid"] == True]

    # 1. Cue distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, cue_name in zip(axes.flat, ["r", "s", "b", "p"]):
        if cue_name == "b":
            data = merged[merged["beat_available"] == True][cue_name].dropna()
        else:
            data = merged[cue_name].dropna()

        ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel(f"{cue_name}(t)")
        ax.set_ylabel("Count")
        ax.set_title(f"{cue_name}(t) Distribution")

    plt.tight_layout()
    plt.savefig(output_dir / "cue_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved cue_distributions.png")

    # 2. Cue vs Difficulty scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, cue_name in zip(axes.flat, ["r", "s", "b", "p"]):
        if cue_name == "b":
            plot_data = merged[merged["beat_available"] == True]
        else:
            plot_data = merged

        x = plot_data[cue_name].values
        y = plot_data["sisdr_vocal"].values

        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]

        ax.scatter(x, y, alpha=0.5, s=10)

        # Add regression line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=2)

        ax.set_xlabel(f"{cue_name}(t)")
        ax.set_ylabel("SI-SDR (dB)")
        ax.set_title(f"{cue_name}(t) vs Difficulty")

    plt.tight_layout()
    plt.savefig(output_dir / "cue_vs_difficulty.png", dpi=150)
    plt.close()
    print(f"  Saved cue_vs_difficulty.png")

    # 3. SI-SDR distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    sisdr_values = merged["sisdr_vocal"].dropna()
    ax.hist(sisdr_values, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        sisdr_values.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {sisdr_values.mean():.2f} dB",
    )
    ax.axvline(
        sisdr_values.median(),
        color="g",
        linestyle="--",
        label=f"Median: {sisdr_values.median():.2f} dB",
    )
    ax.set_xlabel("SI-SDR (dB)")
    ax.set_ylabel("Count")
    ax.set_title("SI-SDR Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "sisdr_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved sisdr_distribution.png")

    # 4. Correlation heatmap
    if proxies_df is not None:
        # Merge all data
        all_merged = merged.merge(proxies_df, on="seg_id", suffixes=("", "_proxy"))

        cols = [
            "r",
            "s",
            "p",
            "sisdr_vocal",
            "leakage_lowfreq",
            "instability",
            "transient_leakage",
        ]
        available_cols = [c for c in cols if c in all_merged.columns]

        if len(available_cols) > 1:
            corr_data = all_merged[available_cols].dropna()
            corr_matrix = corr_data.corr(method="spearman")

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax
            )
            ax.set_title("Correlation Matrix (Spearman)")

            plt.tight_layout()
            plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
            plt.close()
            print(f"  Saved correlation_heatmap.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Track A results and generate summary report"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trackA_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--method",
        type=str,
        default="spearman",
        choices=["spearman", "pearson"],
        help="Correlation method",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    print("=" * 60)
    print("Script 07: Analyze Track A Results")
    print("=" * 60)

    # Load config
    print(f"\nLoading config from {args.config}")
    config = load_config(args.config)

    tables_root = Path(config.paths.tables_root)
    results_root = Path(config.paths.results_root) / "trackA"
    metadata_root = Path(config.paths.metadata_root)

    # Check required files
    print("\nChecking required files...")
    file_status = check_required_files(tables_root, metadata_root)

    for name, exists in file_status.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")

    missing = [name for name, exists in file_status.items() if not exists]
    if missing:
        print(f"\nMissing required files: {', '.join(missing)}")
        print("Run the corresponding scripts first:")
        if "manifest" in missing:
            print("  python scripts/00_build_manifest.py")
        if "segments" in missing:
            print("  python scripts/01_make_segments.py")
        if "difficulty" in missing:
            print("  python scripts/04_compute_difficulty.py")
        if "cues" in missing:
            print("  python scripts/06_compute_cues.py")
        if "proxies" in missing:
            print("  python scripts/05_compute_proxies.py")
        return

    # Dry run
    if args.dry_run:
        print("\n[DRY RUN] Would perform analysis with:")
        print(f"  Correlation method: {args.method}")
        print(f"  Generate plots: {not args.no_plots}")
        print(f"\nOutput files:")
        print(f"  {results_root / 'trackA_summary.json'}")
        print(f"  {results_root / 'cue_correlations.json'}")
        if not args.no_plots:
            print(f"  {results_root / 'figures/'}")
        return

    # Initialize analyzer
    analyzer = TrackAAnalyzer(
        tables_root=tables_root, results_root=results_root, metadata_root=metadata_root
    )

    # Load data
    analyzer.load_all_data()

    # Run analysis
    print("\nRunning analysis...")
    config_snapshot = {"correlation_method": args.method, "config_path": args.config}
    summary = analyzer.run_full_analysis(
        dataset_name="MUSDB18-HQ", config_snapshot=config_snapshot
    )

    # Print correlation tables
    if analyzer.cue_correlations:
        print_correlation_table(
            analyzer.cue_correlations, "Cue-Difficulty Correlations (Spearman)"
        )

    if analyzer.proxy_associations:
        print_proxy_table(
            analyzer.proxy_associations, "Proxy-Difficulty Correlations (Spearman)"
        )

    # Print beat coverage
    if analyzer.beat_coverage:
        bc = analyzer.beat_coverage
        print("\nBeat Coverage:")
        print(
            f"  Tracks: {bc.n_tracks_available}/{bc.n_tracks_total} ({bc.track_coverage_ratio:.1%})"
        )
        print(
            f"  Segments: {bc.n_segments_beat_available}/{bc.n_segments_total} ({bc.segment_coverage_ratio:.1%})"
        )

    # Print full summary
    analyzer.print_summary()

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(
            cues_df=analyzer.cues_df,
            difficulty_df=analyzer.difficulty_df,
            proxies_df=analyzer.proxies_df,
            output_dir=results_root / "figures",
        )

    # Save results
    analyzer.save_results()

    print("\n" + "=" * 60)
    print("✓ Track A analysis complete!")
    print("=" * 60)

    # Key findings
    print("\nKey Findings:")

    if analyzer.cue_correlations:
        # Find strongest correlation
        valid_corrs = {
            k: v
            for k, v in analyzer.cue_correlations.items()
            if not np.isnan(v.get("correlation", float("nan")))
        }

        if valid_corrs:
            best_cue = max(
                valid_corrs.keys(), key=lambda k: abs(valid_corrs[k]["correlation"])
            )
            best_corr = valid_corrs[best_cue]["correlation"]
            print(
                f"  - Strongest cue correlation: {best_cue}(t) with ρ = {best_corr:+.3f}"
            )

            # Count significant correlations
            n_sig = sum(1 for v in valid_corrs.values() if v.get("significant", False))
            print(f"  - Significant cue correlations: {n_sig}/4")

    if analyzer.proxy_associations:
        valid_proxies = [
            p for p in analyzer.proxy_associations if not np.isnan(p.spearman_rho)
        ]
        if valid_proxies:
            best_proxy = max(valid_proxies, key=lambda p: abs(p.spearman_rho))
            print(
                f"  - Strongest proxy correlation: {best_proxy.proxy_name} with ρ = {best_proxy.spearman_rho:+.3f}"
            )

    print()


if __name__ == "__main__":
    main()
