#!/usr/bin/env python3
"""
Stock Clustering Pipeline Runner

Run with detailed logging and statistics output.

Usage:
    python run.py                    # Default: sample of 50 stocks, 6 months
    python run.py --full             # Full S&P 500 universe, 18 months
    python run.py --tickers AAPL MSFT GOOGL  # Specific tickers
    python run.py --months 12        # Custom period
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    """Print a section header."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print()
    print(f"--- {title} ---")


def print_stats(label: str, value, unit: str = "") -> None:
    """Print a statistic."""
    if isinstance(value, float):
        print(f"  {label:<30} {value:>12.4f} {unit}")
    else:
        print(f"  {label:<30} {value:>12} {unit}")


def print_table(headers: list, rows: list, col_widths: list = None) -> None:
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                      for i in range(len(headers))]

    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {header_line}")
    print(f"  {'-' * sum(col_widths)}")

    for row in rows:
        row_line = "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(f"  {row_line}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def run_pipeline_with_logging(
    tickers: list[str] = None,
    period_months: int = 6,
    clustering_method: str = "hierarchical",
    output_dir: str = "./output",
    use_full_universe: bool = False,
) -> dict:
    """Run the clustering pipeline with detailed logging."""

    from price_correlation.clustering import (
        cluster_hierarchical, cut_dendrogram, find_optimal_eps,
        cluster_dbscan, find_optimal_k
    )
    from price_correlation.correlation import (
        compute_correlation_matrix, correlation_to_distance,
        get_condensed_distance, get_highly_correlated_pairs
    )
    from price_correlation.export import export_all
    from price_correlation.ingestion import fetch_price_history
    from price_correlation.preprocess import preprocess_pipeline
    from price_correlation.universe import get_full_universe, get_sample_tickers
    from price_correlation.validation import (
        compute_silhouette, compute_cluster_stats, get_cluster_members,
        generate_tsne_plot
    )

    pipeline_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("STOCK CLUSTERING PIPELINE")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output dir: {output_dir.absolute()}")
    print(f"  Method:     {clustering_method}")
    print(f"  Period:     {period_months} months")

    # =========================================================================
    # STEP 1: Universe Definition
    # =========================================================================
    print_header("STEP 1: UNIVERSE DEFINITION")
    step_start = time.time()

    if tickers:
        universe = tickers
        logger.info(f"Using {len(universe)} provided tickers")
    elif use_full_universe:
        logger.info("Fetching full S&P 500 + NASDAQ-100 universe...")
        universe = get_full_universe()
        logger.info(f"Retrieved {len(universe)} tickers")
    else:
        universe = get_sample_tickers(50)
        logger.info(f"Using sample of {len(universe)} tickers")

    print_stats("Total tickers", len(universe))
    print_stats("Sample tickers", ", ".join(universe[:10]) + "...")
    print_stats("Step duration", time.time() - step_start, "sec")

    # =========================================================================
    # STEP 2: Data Ingestion
    # =========================================================================
    print_header("STEP 2: DATA INGESTION")
    step_start = time.time()

    logger.info(f"Fetching {period_months} months of price history...")
    logger.info("This may take a few minutes for large universes...")

    prices = fetch_price_history(universe, period_months=period_months)

    print_subheader("Price Data Statistics")
    print_stats("Trading days", len(prices))
    print_stats("Tickers fetched", len(prices.columns))
    print_stats("Date range", f"{prices.index[0].date()} to {prices.index[-1].date()}")
    print_stats("Total data points", prices.size)
    print_stats("Missing values", prices.isna().sum().sum())
    print_stats("Missing %", 100 * prices.isna().sum().sum() / prices.size, "%")

    # Price statistics
    print_subheader("Price Summary (Latest)")
    latest_prices = prices.iloc[-1].dropna()
    print_stats("Min price", latest_prices.min(), "$")
    print_stats("Max price", latest_prices.max(), "$")
    print_stats("Median price", latest_prices.median(), "$")
    print_stats("Mean price", latest_prices.mean(), "$")

    print_stats("Step duration", time.time() - step_start, "sec")

    # =========================================================================
    # STEP 3: Preprocessing
    # =========================================================================
    print_header("STEP 3: PREPROCESSING")
    step_start = time.time()

    logger.info("Cleaning data and computing log returns...")

    returns = preprocess_pipeline(prices, min_history_pct=0.85)
    valid_tickers = list(returns.columns)

    print_subheader("Preprocessing Results")
    print_stats("Input tickers", len(prices.columns))
    print_stats("Output tickers", len(valid_tickers))
    print_stats("Filtered out", len(prices.columns) - len(valid_tickers))
    print_stats("Trading days (returns)", len(returns))

    # Returns statistics
    print_subheader("Returns Statistics")
    print_stats("Mean of means", returns.mean().mean())
    print_stats("Mean of stds", returns.std().mean())
    print_stats("Min return", returns.min().min())
    print_stats("Max return", returns.max().max())

    # Show dropped tickers
    dropped = set(prices.columns) - set(valid_tickers)
    if dropped:
        print_subheader(f"Dropped Tickers ({len(dropped)})")
        print(f"  {', '.join(sorted(dropped)[:20])}")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped) - 20} more")

    print_stats("Step duration", time.time() - step_start, "sec")

    # =========================================================================
    # STEP 4: Correlation Analysis
    # =========================================================================
    print_header("STEP 4: CORRELATION ANALYSIS")
    step_start = time.time()

    logger.info("Computing correlation matrix...")

    corr_matrix = compute_correlation_matrix(returns)
    dist_matrix = correlation_to_distance(corr_matrix)
    condensed_dist = get_condensed_distance(returns)

    # Correlation statistics
    n = len(valid_tickers)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]

    print_subheader("Correlation Matrix")
    print_stats("Matrix size", f"{n} x {n}")
    print_stats("Unique pairs", len(upper_tri))

    print_subheader("Correlation Distribution")
    print_stats("Min correlation", upper_tri.min())
    print_stats("Max correlation", upper_tri.max())
    print_stats("Mean correlation", upper_tri.mean())
    print_stats("Median correlation", np.median(upper_tri))
    print_stats("Std deviation", upper_tri.std())

    # Correlation buckets
    print_subheader("Correlation Buckets")
    buckets = [
        ("Highly positive (>0.7)", (upper_tri > 0.7).sum()),
        ("Positive (0.3 to 0.7)", ((upper_tri > 0.3) & (upper_tri <= 0.7)).sum()),
        ("Weak (-0.3 to 0.3)", ((upper_tri >= -0.3) & (upper_tri <= 0.3)).sum()),
        ("Negative (-0.7 to -0.3)", ((upper_tri >= -0.7) & (upper_tri < -0.3)).sum()),
        ("Highly negative (<-0.7)", (upper_tri < -0.7).sum()),
    ]
    for label, count in buckets:
        pct = 100 * count / len(upper_tri)
        print(f"  {label:<30} {count:>8} ({pct:>5.1f}%)")

    # Top correlated pairs
    print_subheader("Top 10 Most Correlated Pairs")
    top_pairs = get_highly_correlated_pairs(corr_matrix, valid_tickers, threshold=0.0)[:10]
    rows = [(p["ticker_a"], p["ticker_b"], f"{p['correlation']:.4f}") for p in top_pairs]
    print_table(["Ticker A", "Ticker B", "Correlation"], rows, [12, 12, 12])

    print_stats("Step duration", time.time() - step_start, "sec")

    # =========================================================================
    # STEP 5: Clustering
    # =========================================================================
    print_header("STEP 5: CLUSTERING")
    step_start = time.time()

    if clustering_method == "dbscan":
        logger.info("Finding optimal DBSCAN epsilon...")
        eps = find_optimal_eps(dist_matrix)
        logger.info(f"Optimal eps: {eps:.4f}")

        logger.info("Running DBSCAN clustering...")
        labels = cluster_dbscan(dist_matrix, eps=eps)

        print_subheader("DBSCAN Parameters")
        print_stats("Epsilon (eps)", eps)
        print_stats("Min samples", 5)
    else:
        logger.info("Building hierarchical dendrogram...")
        Z = cluster_hierarchical(condensed_dist, method="average")

        logger.info("Finding optimal number of clusters...")
        best_k, best_score = find_optimal_k(Z, dist_matrix, max_k=min(30, n // 3))
        logger.info(f"Optimal k: {best_k} (silhouette: {best_score:.4f})")

        labels = cut_dendrogram(Z, n_clusters=best_k) - 1

        print_subheader("Hierarchical Parameters")
        print_stats("Linkage method", "average")
        print_stats("Optimal clusters (k)", best_k)
        print_stats("Silhouette at k", best_score)

    print_stats("Step duration", time.time() - step_start, "sec")

    # =========================================================================
    # STEP 6: Validation & Analysis
    # =========================================================================
    print_header("STEP 6: VALIDATION & ANALYSIS")
    step_start = time.time()

    silhouette = compute_silhouette(dist_matrix, labels)
    stats = compute_cluster_stats(labels, valid_tickers)
    members = get_cluster_members(labels, valid_tickers)

    print_subheader("Clustering Quality")
    print_stats("Silhouette score", silhouette)
    quality = "Excellent" if silhouette > 0.5 else "Good" if silhouette > 0.25 else "Fair" if silhouette > 0.1 else "Poor"
    print_stats("Quality assessment", quality)

    print_subheader("Cluster Statistics")
    print_stats("Number of clusters", stats["n_clusters"])
    print_stats("Noise points", stats["n_noise"])
    print_stats("Clustered stocks", stats["n_total"] - stats["n_noise"])
    print_stats("Largest cluster", stats["largest_cluster"])
    print_stats("Smallest cluster", stats["smallest_cluster"])
    print_stats("Mean cluster size", stats["mean_cluster_size"])

    # Cluster breakdown
    print_subheader("Cluster Breakdown")
    rows = []
    for cluster_id in sorted(members.keys()):
        if cluster_id == -1:
            label = "NOISE"
        else:
            label = f"Cluster {cluster_id}"

        cluster_tickers = members[cluster_id]
        size = len(cluster_tickers)
        preview = ", ".join(cluster_tickers[:5])
        if size > 5:
            preview += f" (+{size - 5})"

        rows.append((label, size, preview))

    print_table(["Cluster", "Size", "Members"], rows, [12, 8, 45])

    print_stats("Step duration", time.time() - step_start, "sec")

    # =========================================================================
    # STEP 7: Visualization
    # =========================================================================
    print_header("STEP 7: VISUALIZATION")
    step_start = time.time()

    try:
        viz_path = output_dir / "cluster_visualization.png"
        logger.info(f"Generating t-SNE visualization...")
        generate_tsne_plot(dist_matrix, labels, valid_tickers, str(viz_path))
        logger.info(f"Saved: {viz_path}")
        print_stats("Visualization", str(viz_path))
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        print_stats("Visualization", "SKIPPED (matplotlib not available)")

    print_stats("Step duration", time.time() - step_start, "sec")

    # =========================================================================
    # STEP 8: Export
    # =========================================================================
    print_header("STEP 8: EXPORT")
    step_start = time.time()

    logger.info("Exporting results...")

    output_files = export_all(
        labels, valid_tickers, corr_matrix, output_dir,
        correlation_threshold=0.7
    )

    print_subheader("Output Files")
    for name, path in output_files.items():
        size = path.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  {name:<25} {size_str:>12}  {path}")

    # Show sample of JSON output
    print_subheader("JSON Output Preview")
    with open(output_files["clusters_json"]) as f:
        clusters_json = json.load(f)

    for cluster in clusters_json[:3]:
        cid = cluster["cluster"]
        label = "NOISE" if cid == -1 else f"Cluster {cid}"
        print(f"  {label}: {cluster['size']} stocks")
        print(f"    {cluster['members'][:8]}...")

    # Parquet info
    print_subheader("Parquet Files Info")
    clusters_df = pd.read_parquet(output_files["clusters_parquet"])
    print_stats("Clusters table rows", len(clusters_df))
    print_stats("Clusters table cols", list(clusters_df.columns))

    corr_df = pd.read_parquet(output_files["correlations_parquet"])
    print_stats("Correlations pairs", len(corr_df))
    if len(corr_df) > 0:
        print_stats("Correlation range", f"{corr_df['correlation'].min():.3f} to {corr_df['correlation'].max():.3f}")

    print_stats("Step duration", time.time() - step_start, "sec")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - pipeline_start

    print_header("PIPELINE COMPLETE")
    print_subheader("Summary")
    print_stats("Total stocks processed", len(valid_tickers))
    print_stats("Clusters found", stats["n_clusters"])
    print_stats("Noise points", stats["n_noise"])
    print_stats("Silhouette score", silhouette)
    print_stats("Total execution time", format_duration(total_time))

    print_subheader("Output Location")
    print(f"  {output_dir.absolute()}")
    for path in output_files.values():
        print(f"    - {path.name}")

    print()
    print("=" * 70)
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    return {
        "n_stocks": len(valid_tickers),
        "n_clusters": stats["n_clusters"],
        "n_noise": stats["n_noise"],
        "silhouette": silhouette,
        "execution_time": total_time,
        "output_files": {k: str(v) for k, v in output_files.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Stock Clustering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Sample 50 stocks, 6 months
  python run.py --full                   # Full universe, 18 months
  python run.py --months 12              # Sample 50 stocks, 12 months
  python run.py --tickers AAPL MSFT GOOGL JPM XOM
  python run.py --full --method dbscan   # Full universe with DBSCAN
        """
    )

    parser.add_argument(
        "--full", action="store_true",
        help="Use full S&P 500 + NASDAQ-100 universe"
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Specific tickers to analyze"
    )
    parser.add_argument(
        "--months", type=int, default=6,
        help="Lookback period in months (default: 6)"
    )
    parser.add_argument(
        "--method", choices=["hierarchical", "dbscan"], default="hierarchical",
        help="Clustering method (default: hierarchical)"
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)"
    )

    args = parser.parse_args()

    if args.full:
        period = 18
    else:
        period = args.months

    try:
        result = run_pipeline_with_logging(
            tickers=args.tickers,
            period_months=period,
            clustering_method=args.method,
            output_dir=args.output,
            use_full_universe=args.full,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
