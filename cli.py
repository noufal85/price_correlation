#!/usr/bin/env python3
"""
Stock Clustering CLI

Interactive menu-based command-line interface for stock correlation clustering.
Run the full pipeline or execute individual steps.

Usage:
    python cli.py                  # Launch interactive menu
    python cli.py --help           # Show help

Environment:
    FMP_API_KEY - Required for FMP data source
"""

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
    DIM = '\033[2m'


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


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


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

    def get_status(self) -> dict:
        """Get status of each pipeline step."""
        return {
            "universe": self.exists("universe"),
            "prices": self.exists("prices"),
            "returns": self.exists("returns"),
            "corr_matrix": self.exists("corr_matrix"),
            "labels": self.exists("labels"),
        }


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Pipeline configuration."""

    def __init__(self):
        self.source = "yfinance"
        self.config_file = None
        self.output_dir = "./output"
        self.work_dir = "./.pipeline_state"
        self.days = 180
        self.min_history = 0.85
        self.method = "hierarchical"
        self.n_clusters = None
        self.market_cap_min = None
        self.market_cap_max = None
        self.sample_size = 50
        self.full = False
        self.tickers = None
        self.correlation_threshold = 0.7
        self.visualize = True


# ============================================================================
# Pipeline Steps
# ============================================================================

def step_universe(config: Config, state: PipelineState) -> list[dict]:
    """Step 1: Fetch stock universe."""
    print_header("STEP 1: FETCH STOCK UNIVERSE")
    step_start = time.time()

    if config.source == "fmp":
        from price_correlation.fmp_client import load_config, get_fmp_universe

        # Check API key
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print_error("FMP_API_KEY environment variable not set!")
            print("  Get your free key at: https://financialmodelingprep.com/developer")
            print("  Then run: export FMP_API_KEY=your_key_here")
            return None

        # Load config
        if config.config_file:
            fmp_config = load_config(config.config_file)
            print_stat("Config file", config.config_file)
        else:
            fmp_config = load_config()
            print_stat("Config file", "default.yaml")

        # Override with CLI args
        if config.market_cap_min:
            fmp_config.setdefault("filters", {}).setdefault("market_cap", {})["min"] = config.market_cap_min
        if config.market_cap_max:
            fmp_config.setdefault("filters", {}).setdefault("market_cap", {})["max"] = config.market_cap_max

        # Show filters
        print_subheader("Active Filters")
        filters = fmp_config.get("filters", {})

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
        universe = get_fmp_universe(config=fmp_config)

    else:
        # yfinance source
        from price_correlation.universe import get_full_universe, get_sample_tickers

        print_stat("Data source", "yfinance")

        if config.tickers:
            tickers = config.tickers
            universe = [{"symbol": t} for t in tickers]
            print_stat("Mode", "Custom tickers")
        elif config.full:
            print_stat("Mode", "Full (S&P 500 + NASDAQ-100)")
            tickers = get_full_universe()
            universe = [{"symbol": t} for t in tickers]
        else:
            print_stat("Mode", f"Sample ({config.sample_size} stocks)")
            tickers = get_sample_tickers(config.sample_size)
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


def step_prices(config: Config, state: PipelineState) -> pd.DataFrame:
    """Step 2: Fetch price data."""
    print_header("STEP 2: FETCH PRICE DATA")
    step_start = time.time()

    # Load universe
    tickers = state.load("tickers")
    if not tickers:
        print_error("No universe found. Run step 1 first.")
        return None

    print_stat("Tickers to fetch", len(tickers))
    print_stat("Days of history", config.days)

    if config.source == "fmp":
        from price_correlation.fmp_client import FMPClient, load_config

        fmp_config = load_config(config.config_file) if config.config_file else load_config()
        client = FMPClient(config=fmp_config)

        print_subheader("Fetching from FMP API")
        estimated_time = len(tickers) * 0.3
        print_stat("Estimated time", format_duration(estimated_time))
        print()

        prices = client.get_batch_historical_prices(
            tickers,
            days=config.days,
            progress_callback=print_progress,
        )
    else:
        from price_correlation.ingestion import fetch_price_history

        print_subheader("Fetching from yfinance")
        prices = fetch_price_history(tickers, period_months=config.days // 30)

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


def step_preprocess(config: Config, state: PipelineState) -> pd.DataFrame:
    """Step 3: Preprocess data."""
    print_header("STEP 3: PREPROCESS DATA")
    step_start = time.time()

    from price_correlation.preprocess import preprocess_pipeline

    # Load prices
    prices = state.load("prices")
    if prices is None:
        print_error("No price data found. Run step 2 first.")
        return None

    print_stat("Input tickers", len(prices.columns))
    print_stat("Input days", len(prices))
    print_stat("Min history required", f"{config.min_history * 100:.0f}%")

    print_subheader("Processing")
    returns = preprocess_pipeline(prices, min_history_pct=config.min_history)
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


def step_correlate(config: Config, state: PipelineState) -> tuple:
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
        print_error("No preprocessed data found. Run step 3 first.")
        return None

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


def step_cluster(config: Config, state: PipelineState) -> np.ndarray:
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
        print_error("No correlation data found. Run step 4 first.")
        return None

    n = len(valid_tickers)
    print_stat("Stocks to cluster", n)
    print_stat("Method", config.method)

    if config.method == "dbscan":
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

        if config.n_clusters:
            best_k = config.n_clusters
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


def step_export(config: Config, state: PipelineState) -> dict:
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
        print_error("No clustering results found. Run step 5 first.")
        return None

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_stat("Output directory", str(output_dir.absolute()))

    # Export clusters
    print_subheader("Exporting Files")
    output_files = export_all(
        labels, valid_tickers, corr_matrix, output_dir,
        correlation_threshold=config.correlation_threshold
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
    if config.visualize:
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


def run_full_pipeline(config: Config, state: PipelineState) -> dict:
    """Run all pipeline steps."""
    print_header("STOCK CLUSTERING PIPELINE")
    print(f"  {c('Started:', Colors.BOLD)} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {c('Source:', Colors.BOLD)}  {config.source}")
    print(f"  {c('Output:', Colors.BOLD)}  {config.output_dir}")

    pipeline_start = time.time()

    # Clear previous state
    state.clear()

    # Run all steps
    result = step_universe(config, state)
    if result is None:
        return None

    result = step_prices(config, state)
    if result is None:
        return None

    result = step_preprocess(config, state)
    if result is None:
        return None

    result = step_correlate(config, state)
    if result is None:
        return None

    result = step_cluster(config, state)
    if result is None:
        return None

    step_export(config, state)

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
# Interactive Menu
# ============================================================================

def print_menu(config: Config, state: PipelineState) -> None:
    """Print the interactive menu."""
    status = state.get_status()

    print()
    print(c("=" * 60, Colors.CYAN))
    print(c("       STOCK CLUSTERING PIPELINE - INTERACTIVE MENU", Colors.BOLD + Colors.CYAN))
    print(c("=" * 60, Colors.CYAN))
    print()

    # Current settings
    print(c("  Current Settings:", Colors.BOLD))
    print(f"    Source:      {c(config.source.upper(), Colors.GREEN)}")
    if config.source == "fmp":
        if config.market_cap_min:
            print(f"    Market Cap:  {c(format_number(config.market_cap_min) + '+', Colors.GREEN)}")
        if config.config_file:
            print(f"    Config:      {c(config.config_file, Colors.GREEN)}")
    else:
        mode = "Full" if config.full else f"Sample ({config.sample_size})"
        print(f"    Mode:        {c(mode, Colors.GREEN)}")
    print(f"    Days:        {c(str(config.days), Colors.GREEN)}")
    print(f"    Method:      {c(config.method, Colors.GREEN)}")
    print(f"    Output:      {c(config.output_dir, Colors.GREEN)}")
    print()

    # Status indicators
    def status_icon(done: bool) -> str:
        return c("✓", Colors.GREEN) if done else c("○", Colors.DIM)

    print(c("─" * 60, Colors.BLUE))
    print()
    print(c("  Pipeline Steps:", Colors.BOLD))
    print()
    print(f"    {c('1', Colors.YELLOW)}  {status_icon(status['universe'])}  Fetch Universe       - Get list of stocks to analyze")
    print(f"    {c('2', Colors.YELLOW)}  {status_icon(status['prices'])}  Fetch Prices         - Download historical price data")
    print(f"    {c('3', Colors.YELLOW)}  {status_icon(status['returns'])}  Preprocess           - Compute returns & normalize")
    print(f"    {c('4', Colors.YELLOW)}  {status_icon(status['corr_matrix'])}  Correlations         - Build correlation matrix")
    print(f"    {c('5', Colors.YELLOW)}  {status_icon(status['labels'])}  Cluster              - Run clustering algorithm")
    print(f"    {c('6', Colors.YELLOW)}     Export               - Save results to files")
    print()
    print(c("─" * 60, Colors.BLUE))
    print()
    print(c("  Actions:", Colors.BOLD))
    print()
    print(f"    {c('7', Colors.YELLOW)}     Run Full Pipeline    - Execute all steps (1-6)")
    print(f"    {c('8', Colors.YELLOW)}     Settings             - Change configuration")
    print(f"    {c('9', Colors.YELLOW)}     Clear State          - Reset pipeline state")
    print(f"    {c('0', Colors.YELLOW)}     Exit")
    print()
    print(c("=" * 60, Colors.CYAN))
    print()


def print_settings_menu(config: Config) -> None:
    """Print the settings menu."""
    print()
    print(c("─" * 50, Colors.BLUE))
    print(c("  Settings", Colors.BOLD + Colors.BLUE))
    print(c("─" * 50, Colors.BLUE))
    print()
    print(f"    {c('1', Colors.YELLOW)}  Data Source        [{c(config.source, Colors.GREEN)}]")
    print(f"    {c('2', Colors.YELLOW)}  Days of History    [{c(str(config.days), Colors.GREEN)}]")
    print(f"    {c('3', Colors.YELLOW)}  Clustering Method  [{c(config.method, Colors.GREEN)}]")
    print(f"    {c('4', Colors.YELLOW)}  Min History %      [{c(f'{config.min_history*100:.0f}%', Colors.GREEN)}]")
    print(f"    {c('5', Colors.YELLOW)}  Output Directory   [{c(config.output_dir, Colors.GREEN)}]")
    if config.source == "yfinance":
        mode = "Full" if config.full else f"Sample ({config.sample_size})"
        print(f"    {c('6', Colors.YELLOW)}  Universe Mode      [{c(mode, Colors.GREEN)}]")
    else:
        mcap = format_number(config.market_cap_min) if config.market_cap_min else "None"
        print(f"    {c('6', Colors.YELLOW)}  Market Cap Min     [{c(mcap, Colors.GREEN)}]")
        print(f"    {c('7', Colors.YELLOW)}  Config File        [{c(config.config_file or 'default', Colors.GREEN)}]")
    print()
    print(f"    {c('0', Colors.YELLOW)}  Back to main menu")
    print()


def handle_settings(config: Config) -> None:
    """Handle settings menu."""
    while True:
        print_settings_menu(config)
        choice = input(c("  Enter choice: ", Colors.BOLD)).strip()

        if choice == "0":
            break
        elif choice == "1":
            print()
            print(f"    Current: {config.source}")
            print("    Options: yfinance, fmp")
            new_val = input("    New value: ").strip().lower()
            if new_val in ["yfinance", "fmp"]:
                config.source = new_val
                print_success(f"Source set to {new_val}")
            else:
                print_error("Invalid choice")
        elif choice == "2":
            print()
            print(f"    Current: {config.days}")
            new_val = input("    New value (days): ").strip()
            try:
                config.days = int(new_val)
                print_success(f"Days set to {config.days}")
            except ValueError:
                print_error("Invalid number")
        elif choice == "3":
            print()
            print(f"    Current: {config.method}")
            print("    Options: hierarchical, dbscan")
            new_val = input("    New value: ").strip().lower()
            if new_val in ["hierarchical", "dbscan"]:
                config.method = new_val
                print_success(f"Method set to {new_val}")
            else:
                print_error("Invalid choice")
        elif choice == "4":
            print()
            print(f"    Current: {config.min_history*100:.0f}%")
            new_val = input("    New value (0-100): ").strip()
            try:
                val = float(new_val) / 100
                if 0 <= val <= 1:
                    config.min_history = val
                    print_success(f"Min history set to {val*100:.0f}%")
                else:
                    print_error("Value must be between 0 and 100")
            except ValueError:
                print_error("Invalid number")
        elif choice == "5":
            print()
            print(f"    Current: {config.output_dir}")
            new_val = input("    New value: ").strip()
            if new_val:
                config.output_dir = new_val
                print_success(f"Output directory set to {new_val}")
        elif choice == "6":
            if config.source == "yfinance":
                print()
                print(f"    Current: {'Full' if config.full else f'Sample ({config.sample_size})'}")
                print("    Options: full, sample")
                new_val = input("    New value: ").strip().lower()
                if new_val == "full":
                    config.full = True
                    print_success("Mode set to Full")
                elif new_val == "sample":
                    config.full = False
                    size = input("    Sample size: ").strip()
                    try:
                        config.sample_size = int(size)
                        print_success(f"Mode set to Sample ({config.sample_size})")
                    except ValueError:
                        print_error("Invalid number")
                else:
                    print_error("Invalid choice")
            else:
                print()
                print(f"    Current: {format_number(config.market_cap_min) if config.market_cap_min else 'None'}")
                new_val = input("    New value (USD, or 'none'): ").strip().lower()
                if new_val == "none":
                    config.market_cap_min = None
                    print_success("Market cap min cleared")
                else:
                    try:
                        # Support shorthand like 1B, 500M
                        val = new_val.upper().replace("B", "000000000").replace("M", "000000").replace("K", "000")
                        config.market_cap_min = int(val)
                        print_success(f"Market cap min set to {format_number(config.market_cap_min)}")
                    except ValueError:
                        print_error("Invalid number (use format like 1000000000 or 1B)")
        elif choice == "7" and config.source == "fmp":
            print()
            print(f"    Current: {config.config_file or 'default'}")
            print("    Available: default.yaml, sample_filtered.yaml, sample_smallcap.yaml")
            new_val = input("    Config file path (or 'default'): ").strip()
            if new_val.lower() == "default":
                config.config_file = None
                print_success("Using default config")
            elif new_val:
                if os.path.exists(new_val):
                    config.config_file = new_val
                    print_success(f"Config file set to {new_val}")
                else:
                    # Try config directory
                    cfg_path = f"config/{new_val}"
                    if os.path.exists(cfg_path):
                        config.config_file = cfg_path
                        print_success(f"Config file set to {cfg_path}")
                    else:
                        print_error(f"File not found: {new_val}")


def interactive_menu():
    """Run interactive menu loop."""
    config = Config()
    state = PipelineState(Path(config.work_dir))

    print()
    print(c("  Stock Clustering Pipeline v1.0", Colors.BOLD + Colors.GREEN))
    print(c("  Type a number and press Enter to select an option", Colors.DIM))

    while True:
        try:
            print_menu(config, state)
            choice = input(c("  Enter choice: ", Colors.BOLD)).strip()

            if choice == "0":
                print()
                print(c("  Goodbye!", Colors.GREEN))
                print()
                break
            elif choice == "1":
                step_universe(config, state)
                input(c("\n  Press Enter to continue...", Colors.DIM))
            elif choice == "2":
                step_prices(config, state)
                input(c("\n  Press Enter to continue...", Colors.DIM))
            elif choice == "3":
                step_preprocess(config, state)
                input(c("\n  Press Enter to continue...", Colors.DIM))
            elif choice == "4":
                step_correlate(config, state)
                input(c("\n  Press Enter to continue...", Colors.DIM))
            elif choice == "5":
                step_cluster(config, state)
                input(c("\n  Press Enter to continue...", Colors.DIM))
            elif choice == "6":
                step_export(config, state)
                input(c("\n  Press Enter to continue...", Colors.DIM))
            elif choice == "7":
                run_full_pipeline(config, state)
                input(c("\n  Press Enter to continue...", Colors.DIM))
            elif choice == "8":
                handle_settings(config)
            elif choice == "9":
                print()
                confirm = input(c("  Clear all pipeline state? (y/n): ", Colors.YELLOW)).strip().lower()
                if confirm == "y":
                    state.clear()
                    print_success("Pipeline state cleared")
                else:
                    print("  Cancelled")
                input(c("\n  Press Enter to continue...", Colors.DIM))
            else:
                print_error("Invalid choice. Enter a number 0-9.")
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n")
            print(c("  Interrupted. Returning to menu...", Colors.YELLOW))
            print()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stock Correlation Clustering CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run without arguments to launch interactive menu.

Examples:
  python cli.py                    # Interactive menu
  python cli.py --help             # Show help
        """
    )

    parser.add_argument(
        "--non-interactive", "-n", action="store_true",
        help="Exit if run without arguments (for scripting)"
    )

    args = parser.parse_args()

    if args.non_interactive:
        parser.print_help()
        sys.exit(0)

    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\n  Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
