#!/usr/bin/env python3
"""
Stock Clustering Pipeline with FMP Data Source

Run with full stock universe from Financial Modeling Prep API.

Usage:
    python run_fmp.py                           # Default config (full universe)
    python run_fmp.py --config config/sample_filtered.yaml
    python run_fmp.py --market-cap-min 1000000000  # $1B+ stocks

Environment:
    FMP_API_KEY - Your FMP API key (required)
                  Get free key at: https://financialmodelingprep.com/developer
"""

import argparse
import json
import logging
import os
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
        print(f"  {label:<35} {value:>12.4f} {unit}")
    else:
        print(f"  {label:<35} {str(value):>12} {unit}")


def print_progress(current: int, total: int, symbol: str) -> None:
    """Print progress for price fetching."""
    pct = 100 * current / total
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r  [{bar}] {pct:5.1f}% ({current}/{total}) {symbol:<10}", end="", flush=True)
    if current == total:
        print()


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


def format_market_cap(value: float) -> str:
    """Format market cap in human-readable form."""
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"


def run_fmp_pipeline(
    config_path: str | None = None,
    market_cap_min: int | None = None,
    market_cap_max: int | None = None,
    days: int = 180,
    output_dir: str = "./output",
    clustering_method: str = "hierarchical",
) -> dict:
    """Run the full pipeline with FMP data source."""

    from price_correlation.clustering import (
        cluster_hierarchical, cut_dendrogram, find_optimal_eps,
        cluster_dbscan, find_optimal_k
    )
    from price_correlation.correlation import (
        compute_correlation_matrix, correlation_to_distance,
        get_condensed_distance, get_highly_correlated_pairs
    )
    from price_correlation.export import export_all
    from price_correlation.fmp_client import (
        FMPClient, load_config, get_fmp_universe
    )
    from price_correlation.preprocess import preprocess_pipeline
    from price_correlation.validation import (
        compute_silhouette, compute_cluster_stats, get_cluster_members,
        generate_tsne_plot
    )

    pipeline_start = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    if config_path:
        config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = load_config()
        logger.info("Using default config")

    # Override config with command line args
    if market_cap_min:
        config.setdefault("filters", {}).setdefault("market_cap", {})["min"] = market_cap_min
    if market_cap_max:
        config.setdefault("filters", {}).setdefault("market_cap", {})["max"] = market_cap_max

    print_header("FMP STOCK CLUSTERING PIPELINE")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config:     {config_path or 'default.yaml'}")
    print(f"  Output:     {output_path.absolute()}")
    print(f"  Days:       {days}")
    print(f"  Method:     {clustering_method}")

    # Check API key
    api_key = config.get("api", {}).get("key") or os.environ.get("FMP_API_KEY")
    if not api_key:
        print("\n  ERROR: FMP_API_KEY environment variable not set!")
        print("  Get your free API key at: https://financialmodelingprep.com/developer")
        print("  Then run: export FMP_API_KEY=your_key_here")
        sys.exit(1)

    # =========================================================================
    # STEP 1: Fetch Universe
    # =========================================================================
    print_header("STEP 1: FETCH STOCK UNIVERSE FROM FMP")
    step_start = time.time()

    filters = config.get("filters", {})
    print_subheader("Active Filters")

    market_cap = filters.get("market_cap", {})
    if market_cap.get("min"):
        print_stats("Market Cap Min", format_market_cap(market_cap["min"]))
    if market_cap.get("max"):
        print_stats("Market Cap Max", format_market_cap(market_cap["max"]))

    volume = filters.get("volume", {})
    if volume.get("min"):
        print_stats("Volume Min", f"{volume['min']:,}")

    print_stats("Exchanges", ", ".join(filters.get("exchanges", ["NYSE", "NASDAQ"])))

    if filters.get("sectors"):
        print_stats("Sectors", ", ".join(filters["sectors"]))
    else:
        print_stats("Sectors", "All")

    print_stats("Active Trading Only", filters.get("is_actively_trading", True))

    print_subheader("Fetching from FMP API")
    logger.info("Connecting to FMP stock screener...")

    universe = get_fmp_universe(config=config)

    print_subheader("Universe Statistics")
    print_stats("Total stocks found", len(universe))

    if universe:
        # Market cap distribution
        market_caps = [s.get("marketCap", 0) for s in universe if s.get("marketCap")]
        if market_caps:
            print_stats("Market cap range", f"{format_market_cap(min(market_caps))} - {format_market_cap(max(market_caps))}")
            print_stats("Median market cap", format_market_cap(np.median(market_caps)))

        # Sector distribution
        sectors = {}
        for s in universe:
            sector = s.get("sector", "Unknown")
            sectors[sector] = sectors.get(sector, 0) + 1

        print_subheader("Sector Distribution")
        for sector, count in sorted(sectors.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / len(universe)
            print(f"  {sector:<30} {count:>6} ({pct:>5.1f}%)")

        # Exchange distribution
        exchanges = {}
        for s in universe:
            ex = s.get("exchangeShortName", s.get("exchange", "Unknown"))
            exchanges[ex] = exchanges.get(ex, 0) + 1

        print_subheader("Exchange Distribution")
        for ex, count in sorted(exchanges.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(universe)
            print(f"  {ex:<30} {count:>6} ({pct:>5.1f}%)")

    tickers = sorted([s["symbol"] for s in universe if s.get("symbol")])
    print_stats("Step duration", format_duration(time.time() - step_start))

    if not tickers:
        print("\n  ERROR: No stocks found matching filters!")
        sys.exit(1)

    # =========================================================================
    # STEP 2: Fetch Price Data
    # =========================================================================
    print_header("STEP 2: FETCH HISTORICAL PRICES")
    step_start = time.time()

    print_subheader("Price Fetch Configuration")
    print_stats("Symbols to fetch", len(tickers))
    print_stats("Days of history", days)
    print_stats("Estimated API calls", len(tickers))

    estimated_time = len(tickers) * 0.3  # ~0.3s per request with rate limiting
    print_stats("Estimated time", format_duration(estimated_time))

    print_subheader("Fetching Prices (this may take a while)")

    client = FMPClient(config=config)
    prices = client.get_batch_historical_prices(
        tickers,
        days=days,
        progress_callback=print_progress,
    )

    print_subheader("Price Data Statistics")
    print_stats("Trading days fetched", len(prices))
    print_stats("Tickers with data", len(prices.columns))
    print_stats("Tickers failed", len(tickers) - len(prices.columns))

    if len(prices) > 0:
        print_stats("Date range", f"{prices.index[0].date()} to {prices.index[-1].date()}")
        print_stats("Total data points", prices.size)
        print_stats("Missing values", prices.isna().sum().sum())
        missing_pct = 100 * prices.isna().sum().sum() / prices.size
        print_stats("Missing %", f"{missing_pct:.2f}%")

    print_stats("Step duration", format_duration(time.time() - step_start))

    if prices.empty or len(prices.columns) < 10:
        print("\n  ERROR: Not enough price data fetched!")
        sys.exit(1)

    # =========================================================================
    # STEP 3: Preprocessing
    # =========================================================================
    print_header("STEP 3: PREPROCESSING")
    step_start = time.time()

    logger.info("Computing log returns and normalizing...")

    returns = preprocess_pipeline(prices, min_history_pct=0.85)
    valid_tickers = list(returns.columns)

    print_subheader("Preprocessing Results")
    print_stats("Input tickers", len(prices.columns))
    print_stats("Output tickers", len(valid_tickers))
    print_stats("Filtered out", len(prices.columns) - len(valid_tickers))

    print_subheader("Returns Statistics")
    print_stats("Mean of means", returns.mean().mean())
    print_stats("Mean of stds", returns.std().mean())

    print_stats("Step duration", format_duration(time.time() - step_start))

    # =========================================================================
    # STEP 4: Correlation Analysis
    # =========================================================================
    print_header("STEP 4: CORRELATION ANALYSIS")
    step_start = time.time()

    logger.info("Computing correlation matrix...")

    corr_matrix = compute_correlation_matrix(returns)
    dist_matrix = correlation_to_distance(corr_matrix)
    condensed_dist = get_condensed_distance(returns)

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

    # Top pairs
    print_subheader("Top 10 Most Correlated Pairs")
    top_pairs = get_highly_correlated_pairs(corr_matrix, valid_tickers, threshold=0.0)[:10]
    for p in top_pairs:
        print(f"  {p['ticker_a']:<8} - {p['ticker_b']:<8}  {p['correlation']:.4f}")

    print_stats("Step duration", format_duration(time.time() - step_start))

    # =========================================================================
    # STEP 5: Clustering
    # =========================================================================
    print_header("STEP 5: CLUSTERING")
    step_start = time.time()

    if clustering_method == "dbscan":
        eps = find_optimal_eps(dist_matrix)
        labels = cluster_dbscan(dist_matrix, eps=eps)
        print_subheader("DBSCAN Parameters")
        print_stats("Epsilon", eps)
    else:
        Z = cluster_hierarchical(condensed_dist, method="average")
        best_k, best_score = find_optimal_k(Z, dist_matrix, max_k=min(50, n // 5))
        labels = cut_dendrogram(Z, n_clusters=best_k) - 1
        print_subheader("Hierarchical Parameters")
        print_stats("Optimal k", best_k)
        print_stats("Silhouette at k", best_score)

    print_stats("Step duration", format_duration(time.time() - step_start))

    # =========================================================================
    # STEP 6: Validation
    # =========================================================================
    print_header("STEP 6: VALIDATION & ANALYSIS")
    step_start = time.time()

    silhouette = compute_silhouette(dist_matrix, labels)
    stats = compute_cluster_stats(labels, valid_tickers)
    members = get_cluster_members(labels, valid_tickers)

    print_subheader("Clustering Quality")
    print_stats("Silhouette score", silhouette)

    print_subheader("Cluster Statistics")
    print_stats("Number of clusters", stats["n_clusters"])
    print_stats("Noise points", stats["n_noise"])
    print_stats("Largest cluster", stats["largest_cluster"])
    print_stats("Mean cluster size", stats["mean_cluster_size"])

    # Cluster details
    print_subheader("Cluster Details")
    for cluster_id in sorted(members.keys()):
        if cluster_id == -1:
            label = "NOISE"
        else:
            label = f"Cluster {cluster_id}"

        cluster_tickers = members[cluster_id]
        preview = ", ".join(cluster_tickers[:8])
        if len(cluster_tickers) > 8:
            preview += f" (+{len(cluster_tickers) - 8})"
        print(f"  {label:<15} [{len(cluster_tickers):>4}] {preview}")

    print_stats("Step duration", format_duration(time.time() - step_start))

    # =========================================================================
    # STEP 7: Visualization
    # =========================================================================
    print_header("STEP 7: VISUALIZATION")
    step_start = time.time()

    try:
        viz_path = output_path / "cluster_visualization.png"
        logger.info("Generating t-SNE visualization...")
        generate_tsne_plot(dist_matrix, labels, valid_tickers, str(viz_path))
        print_stats("Visualization saved", str(viz_path))
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        print_stats("Visualization", "SKIPPED")

    print_stats("Step duration", format_duration(time.time() - step_start))

    # =========================================================================
    # STEP 8: Export
    # =========================================================================
    print_header("STEP 8: EXPORT")
    step_start = time.time()

    output_files = export_all(
        labels, valid_tickers, corr_matrix, output_path,
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
            size_str = f"{size} B"
        print(f"  {name:<25} {size_str:>10}  {path}")

    # Save universe metadata
    universe_path = output_path / "universe_metadata.json"
    with open(universe_path, "w") as f:
        json.dump(universe, f, indent=2)
    print(f"  {'universe_metadata':<25} {'-':>10}  {universe_path}")

    print_stats("Step duration", format_duration(time.time() - step_start))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - pipeline_start

    print_header("PIPELINE COMPLETE")
    print_subheader("Summary")
    print_stats("Universe size", len(tickers))
    print_stats("Stocks with data", len(valid_tickers))
    print_stats("Clusters found", stats["n_clusters"])
    print_stats("Noise points", stats["n_noise"])
    print_stats("Silhouette score", silhouette)
    print_stats("Total execution time", format_duration(total_time))

    print_subheader("Output Files")
    print(f"  {output_path.absolute()}/")
    for path in output_files.values():
        print(f"    {path.name}")
    print(f"    universe_metadata.json")

    print()
    print("=" * 70)
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    return {
        "universe_size": len(tickers),
        "stocks_processed": len(valid_tickers),
        "n_clusters": stats["n_clusters"],
        "n_noise": stats["n_noise"],
        "silhouette": silhouette,
        "execution_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Stock Clustering with FMP Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  FMP_API_KEY    Your Financial Modeling Prep API key (required)
                 Get free key at: https://financialmodelingprep.com/developer

Examples:
  # Full universe (all stocks)
  export FMP_API_KEY=your_key
  python run_fmp.py

  # Use filtered config
  python run_fmp.py --config config/sample_filtered.yaml

  # Large cap only ($10B+)
  python run_fmp.py --market-cap-min 10000000000

  # Mid cap ($2B - $10B)
  python run_fmp.py --market-cap-min 2000000000 --market-cap-max 10000000000
        """
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to config YAML file (default: config/default.yaml)"
    )
    parser.add_argument(
        "--market-cap-min", type=int,
        help="Minimum market cap in USD (e.g., 1000000000 for $1B)"
    )
    parser.add_argument(
        "--market-cap-max", type=int,
        help="Maximum market cap in USD"
    )
    parser.add_argument(
        "--days", type=int, default=180,
        help="Days of price history (default: 180)"
    )
    parser.add_argument(
        "--method", choices=["hierarchical", "dbscan"], default="hierarchical",
        help="Clustering method (default: hierarchical)"
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory (default: ./output)"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("FMP_API_KEY"):
        print("\nERROR: FMP_API_KEY environment variable not set!\n")
        print("Get your free API key at: https://financialmodelingprep.com/developer")
        print("Then run: export FMP_API_KEY=your_key_here\n")
        sys.exit(1)

    try:
        run_fmp_pipeline(
            config_path=args.config,
            market_cap_min=args.market_cap_min,
            market_cap_max=args.market_cap_max,
            days=args.days,
            output_dir=args.output,
            clustering_method=args.method,
        )
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
