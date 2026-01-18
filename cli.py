#!/usr/bin/env python3
"""
Stock Clustering CLI

Unified command-line interface for stock correlation clustering.
Run the full pipeline or execute individual steps.

Usage:
    python cli.py run                    # Full pipeline with defaults
    python cli.py run --source fmp       # Use FMP data source
    python cli.py universe               # Fetch universe only
    python cli.py prices                 # Fetch prices only
    python cli.py cluster                # Run clustering only
    python cli.py --help                 # Show help

Environment:
    FMP_API_KEY - Required for FMP data source
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Logging & Display Helpers
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def supports_color():
    """Check if terminal supports colors."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


USE_COLORS = supports_color()


def c(text: str, color: str) -> str:
    """Apply color to text if supported."""
    if USE_COLORS:
        return f"{color}{text}{Colors.END}"
    return text


def print_header(title: str) -> None:
    """Print a section header."""
    width = 74
    print()
    print(c("=" * width, Colors.CYAN))
    print(c(f"  {title}", Colors.BOLD + Colors.CYAN))
    print(c("=" * width, Colors.CYAN))


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print()
    print(c(f"─── {title} ───", Colors.BLUE))


def print_stat(label: str, value, unit: str = "") -> None:
    """Print a statistic."""
    if isinstance(value, float):
        val_str = f"{value:>14.4f}"
    else:
        val_str = f"{str(value):>14}"
    print(f"  {label:<40} {c(val_str, Colors.GREEN)} {unit}")


def print_success(msg: str) -> None:
    """Print success message."""
    print(c(f"  ✓ {msg}", Colors.GREEN))


def print_error(msg: str) -> None:
    """Print error message."""
    print(c(f"  ✗ {msg}", Colors.RED))


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(c(f"  ⚠ {msg}", Colors.YELLOW))


def print_progress(current: int, total: int, item: str = "") -> None:
    """Print progress bar."""
    pct = 100 * current / total if total > 0 else 0
    bar_width = 35
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    item_display = item[:15].ljust(15) if item else ""
    print(f"\r  [{c(bar, Colors.CYAN)}] {pct:5.1f}% ({current}/{total}) {item_display}", end="", flush=True)
    if current == total:
        print()


def format_duration(seconds: float) -> str:
    """Format duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def format_number(value: float) -> str:
    """Format large numbers."""
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:,.0f}"


# ============================================================================
# State Management
# ============================================================================

class PipelineState:
    """Manages pipeline state between steps."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.work_dir / f"{name}.pkl"

    def save(self, name: str, data) -> None:
        with open(self._path(name), "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Saved state: {name}")

    def load(self, name: str):
        path = self._path(name)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def clear(self) -> None:
        for f in self.work_dir.glob("*.pkl"):
            f.unlink()


# ============================================================================
# Pipeline Steps
# ============================================================================

def step_universe(args, state: PipelineState) -> list[dict]:
    """Step 1: Fetch stock universe."""
    print_header("STEP 1: FETCH STOCK UNIVERSE")
    step_start = time.time()

    if args.source == "fmp":
        from price_correlation.fmp_client import load_config, get_fmp_universe

        # Check API key
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print_error("FMP_API_KEY environment variable not set!")
            print("  Get your free key at: https://financialmodelingprep.com/developer")
            print("  Then run: export FMP_API_KEY=your_key_here")
            sys.exit(1)

        # Load config
        if args.config:
            config = load_config(args.config)
            print_stat("Config file", args.config)
        else:
            config = load_config()
            print_stat("Config file", "default.yaml")

        # Override with CLI args
        if args.market_cap_min:
            config.setdefault("filters", {}).setdefault("market_cap", {})["min"] = args.market_cap_min
        if args.market_cap_max:
            config.setdefault("filters", {}).setdefault("market_cap", {})["max"] = args.market_cap_max

        # Show filters
        print_subheader("Active Filters")
        filters = config.get("filters", {})

        market_cap = filters.get("market_cap", {})
        if market_cap.get("min"):
            print_stat("Market Cap Min", format_number(market_cap["min"]))
        else:
            print_stat("Market Cap Min", "None")
        if market_cap.get("max"):
            print_stat("Market Cap Max", format_number(market_cap["max"]))

        print_stat("Exchanges", ", ".join(filters.get("exchanges", ["NYSE", "NASDAQ"])))

        if filters.get("sectors"):
            print_stat("Sectors", ", ".join(filters["sectors"]))
        else:
            print_stat("Sectors", "All")

        print_subheader("Fetching from FMP API")
        universe = get_fmp_universe(config=config)

    else:
        # yfinance source
        from price_correlation.universe import get_full_universe, get_sample_tickers

        print_stat("Data source", "yfinance")

        if args.tickers:
            tickers = args.tickers
            universe = [{"symbol": t} for t in tickers]
            print_stat("Mode", "Custom tickers")
        elif args.full:
            print_stat("Mode", "Full (S&P 500 + NASDAQ-100)")
            tickers = get_full_universe()
            universe = [{"symbol": t} for t in tickers]
        else:
            print_stat("Mode", f"Sample ({args.sample_size} stocks)")
            tickers = get_sample_tickers(args.sample_size)
            universe = [{"symbol": t} for t in tickers]

    # Stats
    print_subheader("Universe Statistics")
    print_stat("Total stocks", len(universe))

    if universe and "marketCap" in universe[0]:
        market_caps = [s.get("marketCap", 0) for s in universe if s.get("marketCap")]
        if market_caps:
            print_stat("Market cap range", f"{format_number(min(market_caps))} - {format_number(max(market_caps))}")
            print_stat("Median market cap", format_number(np.median(market_caps)))

        # Sector distribution
        sectors = {}
        for s in universe:
            sector = s.get("sector", "Unknown")
            sectors[sector] = sectors.get(sector, 0) + 1

        print_subheader("Sector Distribution (Top 10)")
        for sector, count in sorted(sectors.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / len(universe)
            bar = "█" * int(pct / 5)
            print(f"  {sector:<25} {count:>5} ({pct:>5.1f}%) {c(bar, Colors.CYAN)}")

    # Sample tickers
    tickers = [s["symbol"] for s in universe if s.get("symbol")]
    print_subheader("Sample Tickers")
    print(f"  {', '.join(tickers[:15])}...")

    # Save state
    state.save("universe", universe)
    state.save("tickers", tickers)

    print_subheader("Step Complete")
    print_stat("Duration", format_duration(time.time() - step_start))
    print_success(f"Universe saved: {len(tickers)} tickers")

    return universe


def step_prices(args, state: PipelineState) -> pd.DataFrame:
    """Step 2: Fetch price data."""
    print_header("STEP 2: FETCH PRICE DATA")
    step_start = time.time()

    # Load universe
    tickers = state.load("tickers")
    if not tickers:
        print_error("No universe found. Run 'universe' step first.")
        sys.exit(1)

    print_stat("Tickers to fetch", len(tickers))
    print_stat("Days of history", args.days)

    if args.source == "fmp":
        from price_correlation.fmp_client import FMPClient, load_config

        config = load_config(args.config) if args.config else load_config()
        client = FMPClient(config=config)

        print_subheader("Fetching from FMP API")
        estimated_time = len(tickers) * 0.3
        print_stat("Estimated time", format_duration(estimated_time))
        print()

        prices = client.get_batch_historical_prices(
            tickers,
            days=args.days,
            progress_callback=print_progress,
        )
    else:
        from price_correlation.ingestion import fetch_price_history

        print_subheader("Fetching from yfinance")
        prices = fetch_price_history(tickers, period_months=args.days // 30)

    # Stats
    print_subheader("Price Data Statistics")
    print_stat("Trading days", len(prices))
    print_stat("Tickers with data", len(prices.columns))
    print_stat("Tickers failed", len(tickers) - len(prices.columns))

    if len(prices) > 0:
        print_stat("Date range", f"{prices.index[0].date()} to {prices.index[-1].date()}")
        print_stat("Total data points", f"{prices.size:,}")

        missing = prices.isna().sum().sum()
        missing_pct = 100 * missing / prices.size
        print_stat("Missing values", f"{missing:,} ({missing_pct:.2f}%)")

    # Save state
    state.save("prices", prices)

    print_subheader("Step Complete")
    print_stat("Duration", format_duration(time.time() - step_start))
    print_success(f"Prices saved: {len(prices.columns)} tickers, {len(prices)} days")

    return prices


def step_preprocess(args, state: PipelineState) -> pd.DataFrame:
    """Step 3: Preprocess data."""
    print_header("STEP 3: PREPROCESS DATA")
    step_start = time.time()

    from price_correlation.preprocess import preprocess_pipeline

    # Load prices
    prices = state.load("prices")
    if prices is None:
        print_error("No price data found. Run 'prices' step first.")
        sys.exit(1)

    print_stat("Input tickers", len(prices.columns))
    print_stat("Input days", len(prices))
    print_stat("Min history required", f"{args.min_history * 100:.0f}%")

    print_subheader("Processing")
    returns = preprocess_pipeline(prices, min_history_pct=args.min_history)
    valid_tickers = list(returns.columns)

    print_subheader("Preprocessing Results")
    print_stat("Output tickers", len(valid_tickers))
    print_stat("Filtered out", len(prices.columns) - len(valid_tickers))
    print_stat("Output days", len(returns))

    print_subheader("Returns Statistics")
    print_stat("Mean of means", returns.mean().mean())
    print_stat("Mean of stds", returns.std().mean())
    print_stat("Min return", returns.min().min())
    print_stat("Max return", returns.max().max())

    # Dropped tickers
    dropped = set(prices.columns) - set(valid_tickers)
    if dropped and len(dropped) <= 20:
        print_subheader(f"Dropped Tickers ({len(dropped)})")
        print(f"  {', '.join(sorted(dropped))}")
    elif dropped:
        print_subheader(f"Dropped Tickers ({len(dropped)})")
        print(f"  {', '.join(sorted(dropped)[:20])}... +{len(dropped)-20} more")

    # Save state
    state.save("returns", returns)
    state.save("valid_tickers", valid_tickers)

    print_subheader("Step Complete")
    print_stat("Duration", format_duration(time.time() - step_start))
    print_success(f"Returns saved: {len(valid_tickers)} tickers")

    return returns


def step_correlate(args, state: PipelineState) -> tuple:
    """Step 4: Compute correlations."""
    print_header("STEP 4: COMPUTE CORRELATIONS")
    step_start = time.time()

    from price_correlation.correlation import (
        compute_correlation_matrix, correlation_to_distance,
        get_condensed_distance, get_highly_correlated_pairs
    )

    # Load returns
    returns = state.load("returns")
    valid_tickers = state.load("valid_tickers")
    if returns is None:
        print_error("No preprocessed data found. Run 'preprocess' step first.")
        sys.exit(1)

    print_stat("Tickers", len(valid_tickers))

    print_subheader("Computing Correlation Matrix")
    corr_matrix = compute_correlation_matrix(returns)
    dist_matrix = correlation_to_distance(corr_matrix)
    condensed_dist = get_condensed_distance(returns)

    n = len(valid_tickers)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]

    print_subheader("Correlation Matrix")
    print_stat("Matrix size", f"{n} x {n}")
    print_stat("Unique pairs", f"{len(upper_tri):,}")

    print_subheader("Correlation Distribution")
    print_stat("Min", upper_tri.min())
    print_stat("Max", upper_tri.max())
    print_stat("Mean", upper_tri.mean())
    print_stat("Median", np.median(upper_tri))
    print_stat("Std Dev", upper_tri.std())

    # Distribution buckets
    print_subheader("Correlation Buckets")
    buckets = [
        ("Strong positive (>0.7)", (upper_tri > 0.7).sum()),
        ("Moderate positive (0.3-0.7)", ((upper_tri > 0.3) & (upper_tri <= 0.7)).sum()),
        ("Weak (-0.3 to 0.3)", ((upper_tri >= -0.3) & (upper_tri <= 0.3)).sum()),
        ("Moderate negative (-0.7 to -0.3)", ((upper_tri >= -0.7) & (upper_tri < -0.3)).sum()),
        ("Strong negative (<-0.7)", (upper_tri < -0.7).sum()),
    ]
    for label, count in buckets:
        pct = 100 * count / len(upper_tri)
        bar = "█" * int(pct / 3)
        print(f"  {label:<30} {count:>8} ({pct:>5.1f}%) {c(bar, Colors.CYAN)}")

    # Top pairs
    print_subheader("Top 15 Most Correlated Pairs")
    top_pairs = get_highly_correlated_pairs(corr_matrix, valid_tickers, threshold=0.0)[:15]
    for p in top_pairs:
        corr_bar = "█" * int(p['correlation'] * 10)
        print(f"  {p['ticker_a']:<8} ↔ {p['ticker_b']:<8}  {p['correlation']:.4f}  {c(corr_bar, Colors.GREEN)}")

    # Save state
    state.save("corr_matrix", corr_matrix)
    state.save("dist_matrix", dist_matrix)
    state.save("condensed_dist", condensed_dist)

    print_subheader("Step Complete")
    print_stat("Duration", format_duration(time.time() - step_start))
    print_success("Correlation matrices saved")

    return corr_matrix, dist_matrix, condensed_dist


def step_cluster(args, state: PipelineState) -> np.ndarray:
    """Step 5: Cluster stocks."""
    print_header("STEP 5: CLUSTER STOCKS")
    step_start = time.time()

    from price_correlation.clustering import (
        cluster_hierarchical, cut_dendrogram, find_optimal_eps,
        cluster_dbscan, find_optimal_k
    )
    from price_correlation.validation import compute_silhouette

    # Load data
    dist_matrix = state.load("dist_matrix")
    condensed_dist = state.load("condensed_dist")
    valid_tickers = state.load("valid_tickers")

    if dist_matrix is None:
        print_error("No correlation data found. Run 'correlate' step first.")
        sys.exit(1)

    n = len(valid_tickers)
    print_stat("Stocks to cluster", n)
    print_stat("Method", args.method)

    if args.method == "dbscan":
        print_subheader("DBSCAN Clustering")
        print("  Finding optimal epsilon...")
        eps = find_optimal_eps(dist_matrix)
        print_stat("Optimal epsilon", eps)
        print_stat("Min samples", 5)

        print("  Running DBSCAN...")
        labels = cluster_dbscan(dist_matrix, eps=eps)

    else:
        print_subheader("Hierarchical Clustering")
        print("  Building dendrogram...")
        Z = cluster_hierarchical(condensed_dist, method="average")

        if args.n_clusters:
            best_k = args.n_clusters
            print_stat("Clusters (specified)", best_k)
        else:
            print("  Finding optimal k...")
            best_k, best_score = find_optimal_k(Z, dist_matrix, max_k=min(50, n // 5))
            print_stat("Optimal k", best_k)
            print_stat("Silhouette at k", best_score)

        print("  Cutting dendrogram...")
        labels = cut_dendrogram(Z, n_clusters=best_k) - 1

    # Compute quality metrics
    print_subheader("Clustering Quality")
    silhouette = compute_silhouette(dist_matrix, labels)
    print_stat("Silhouette score", silhouette)

    quality = "Excellent" if silhouette > 0.5 else "Good" if silhouette > 0.25 else "Fair" if silhouette > 0.1 else "Weak"
    print_stat("Quality", quality)

    # Cluster stats
    from collections import Counter
    label_counts = Counter(labels)
    noise_count = label_counts.pop(-1, 0)
    cluster_sizes = sorted(label_counts.values(), reverse=True)

    print_subheader("Cluster Statistics")
    print_stat("Number of clusters", len(label_counts))
    print_stat("Noise points", noise_count)
    print_stat("Largest cluster", max(cluster_sizes) if cluster_sizes else 0)
    print_stat("Smallest cluster", min(cluster_sizes) if cluster_sizes else 0)
    print_stat("Mean cluster size", np.mean(cluster_sizes) if cluster_sizes else 0)

    # Cluster details
    print_subheader("Cluster Details")
    members = {}
    for ticker, label in zip(valid_tickers, labels):
        members.setdefault(int(label), []).append(ticker)

    for cluster_id in sorted(members.keys()):
        if cluster_id == -1:
            label = c("NOISE", Colors.YELLOW)
        else:
            label = f"Cluster {cluster_id}"

        cluster_tickers = sorted(members[cluster_id])
        preview = ", ".join(cluster_tickers[:10])
        if len(cluster_tickers) > 10:
            preview += f" (+{len(cluster_tickers) - 10})"
        print(f"  {label:<18} [{len(cluster_tickers):>4}] {preview}")

    # Save state
    state.save("labels", labels)
    state.save("silhouette", silhouette)

    print_subheader("Step Complete")
    print_stat("Duration", format_duration(time.time() - step_start))
    print_success(f"Clustering complete: {len(label_counts)} clusters")

    return labels


def step_export(args, state: PipelineState) -> dict:
    """Step 6: Export results."""
    print_header("STEP 6: EXPORT RESULTS")
    step_start = time.time()

    from price_correlation.export import export_all
    from price_correlation.validation import generate_tsne_plot

    # Load data
    labels = state.load("labels")
    valid_tickers = state.load("valid_tickers")
    corr_matrix = state.load("corr_matrix")
    dist_matrix = state.load("dist_matrix")
    universe = state.load("universe")

    if labels is None:
        print_error("No clustering results found. Run 'cluster' step first.")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_stat("Output directory", str(output_dir.absolute()))

    # Export clusters
    print_subheader("Exporting Files")
    output_files = export_all(
        labels, valid_tickers, corr_matrix, output_dir,
        correlation_threshold=args.correlation_threshold
    )

    for name, path in output_files.items():
        size = path.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} B"
        print_success(f"{name}: {size_str}")

    # Save universe metadata
    if universe and len(universe) > 0 and "marketCap" in universe[0]:
        universe_path = output_dir / "universe_metadata.json"
        with open(universe_path, "w") as f:
            json.dump(universe, f, indent=2)
        print_success(f"universe_metadata: {universe_path}")

    # Visualization
    if args.visualize:
        print_subheader("Generating Visualization")
        try:
            viz_path = output_dir / "cluster_visualization.png"
            generate_tsne_plot(dist_matrix, labels, valid_tickers, str(viz_path))
            print_success(f"Visualization: {viz_path}")
        except Exception as e:
            print_warning(f"Visualization skipped: {e}")

    print_subheader("Step Complete")
    print_stat("Duration", format_duration(time.time() - step_start))
    print_success(f"Results exported to {output_dir}")

    return output_files


def run_full_pipeline(args, state: PipelineState) -> dict:
    """Run all pipeline steps."""
    print_header("STOCK CLUSTERING PIPELINE")
    print(f"  {c('Started:', Colors.BOLD)} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {c('Source:', Colors.BOLD)}  {args.source}")
    print(f"  {c('Output:', Colors.BOLD)}  {args.output}")

    pipeline_start = time.time()

    # Clear previous state
    state.clear()

    # Run all steps
    step_universe(args, state)
    step_prices(args, state)
    step_preprocess(args, state)
    step_correlate(args, state)
    step_cluster(args, state)
    step_export(args, state)

    # Summary
    total_time = time.time() - pipeline_start
    silhouette = state.load("silhouette")
    labels = state.load("labels")
    valid_tickers = state.load("valid_tickers")

    from collections import Counter
    label_counts = Counter(labels)
    noise_count = label_counts.pop(-1, 0)

    print_header("PIPELINE COMPLETE")
    print_stat("Total stocks processed", len(valid_tickers))
    print_stat("Clusters found", len(label_counts))
    print_stat("Noise points", noise_count)
    print_stat("Silhouette score", silhouette)
    print_stat("Total time", format_duration(total_time))

    print()
    print(c("=" * 74, Colors.GREEN))
    print(c(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.GREEN))
    print(c("=" * 74, Colors.GREEN))
    print()

    return {
        "n_stocks": len(valid_tickers),
        "n_clusters": len(label_counts),
        "n_noise": noise_count,
        "silhouette": silhouette,
        "time": total_time,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stock Correlation Clustering CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  run          Run full pipeline (all steps)
  universe     Fetch stock universe
  prices       Fetch price data
  preprocess   Compute returns and normalize
  correlate    Compute correlation matrix
  cluster      Run clustering algorithm
  export       Export results to files

Examples:
  # Full pipeline with yfinance (quick test)
  python cli.py run

  # Full pipeline with FMP (all stocks)
  export FMP_API_KEY=your_key
  python cli.py run --source fmp

  # Large cap stocks only
  python cli.py run --source fmp --market-cap-min 10000000000

  # Run individual steps
  python cli.py universe --source fmp
  python cli.py prices
  python cli.py preprocess
  python cli.py correlate
  python cli.py cluster --method dbscan
  python cli.py export

  # Use config file
  python cli.py run --source fmp --config config/sample_filtered.yaml
        """
    )

    # Global arguments
    parser.add_argument(
        "--source", choices=["yfinance", "fmp"], default="yfinance",
        help="Data source: yfinance (default) or fmp"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML config file (for FMP source)"
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--work-dir", default="./.pipeline_state",
        help="Directory for pipeline state (default: ./.pipeline_state)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run command
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--full", action="store_true", help="Use full universe (yfinance)")
    run_parser.add_argument("--sample-size", type=int, default=50, help="Sample size (yfinance)")
    run_parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    run_parser.add_argument("--market-cap-min", type=int, help="Min market cap (FMP)")
    run_parser.add_argument("--market-cap-max", type=int, help="Max market cap (FMP)")
    run_parser.add_argument("--days", type=int, default=180, help="Days of history")
    run_parser.add_argument("--min-history", type=float, default=0.85, help="Min history %")
    run_parser.add_argument("--method", choices=["hierarchical", "dbscan"], default="hierarchical")
    run_parser.add_argument("--n-clusters", type=int, help="Number of clusters")
    run_parser.add_argument("--correlation-threshold", type=float, default=0.7)
    run_parser.add_argument("--visualize", action="store_true", default=True)
    run_parser.add_argument("--no-visualize", action="store_false", dest="visualize")

    # universe command
    uni_parser = subparsers.add_parser("universe", help="Fetch stock universe")
    uni_parser.add_argument("--full", action="store_true")
    uni_parser.add_argument("--sample-size", type=int, default=50)
    uni_parser.add_argument("--tickers", nargs="+")
    uni_parser.add_argument("--market-cap-min", type=int)
    uni_parser.add_argument("--market-cap-max", type=int)

    # prices command
    prices_parser = subparsers.add_parser("prices", help="Fetch price data")
    prices_parser.add_argument("--days", type=int, default=180)

    # preprocess command
    prep_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    prep_parser.add_argument("--min-history", type=float, default=0.85)

    # correlate command
    corr_parser = subparsers.add_parser("correlate", help="Compute correlations")

    # cluster command
    clust_parser = subparsers.add_parser("cluster", help="Run clustering")
    clust_parser.add_argument("--method", choices=["hierarchical", "dbscan"], default="hierarchical")
    clust_parser.add_argument("--n-clusters", type=int)

    # export command
    exp_parser = subparsers.add_parser("export", help="Export results")
    exp_parser.add_argument("--correlation-threshold", type=float, default=0.7)
    exp_parser.add_argument("--visualize", action="store_true", default=True)
    exp_parser.add_argument("--no-visualize", action="store_false", dest="visualize")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Initialize state
    state = PipelineState(Path(args.work_dir))

    try:
        if args.command == "run":
            run_full_pipeline(args, state)
        elif args.command == "universe":
            step_universe(args, state)
        elif args.command == "prices":
            step_prices(args, state)
        elif args.command == "preprocess":
            step_preprocess(args, state)
        elif args.command == "correlate":
            step_correlate(args, state)
        elif args.command == "cluster":
            step_cluster(args, state)
        elif args.command == "export":
            step_export(args, state)

    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
