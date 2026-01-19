"""Step-by-step pipeline execution with caching and state management."""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .cache import get_cache, TTL_PRICES, TTL_UNIVERSE
from .clustering import cluster_hierarchical, cut_dendrogram, find_optimal_k, cluster_dbscan, find_optimal_eps
from .correlation import compute_correlation_matrix, correlation_to_distance, get_condensed_distance, get_highly_correlated_pairs
from .export import export_all
from .pipeline_state import PipelineStateManager, get_state_manager, PIPELINE_STEPS
from .preprocess import preprocess_pipeline
from .universe import get_sample_tickers
from .validation import compute_cluster_stats, compute_silhouette

logger = logging.getLogger(__name__)


@dataclass
class StepConfig:
    """Configuration for pipeline steps."""

    # Data source
    data_source: str = "sample"  # "sample", "fmp_all", "fmp_filtered"
    filters: dict = field(default_factory=dict)
    max_stocks: int = 0

    # Date range
    start_date: str | None = None
    end_date: str | None = None
    period_months: int = 18

    # Preprocessing
    min_history_pct: float = 0.90
    remove_market_factor: bool = False

    # Clustering
    clustering_method: str = "hierarchical"
    n_clusters: int | None = None

    # Export
    output_dir: str = "./output"
    correlation_threshold: float = 0.7

    def to_dict(self) -> dict:
        return {
            "data_source": self.data_source,
            "filters": self.filters,
            "max_stocks": self.max_stocks,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "period_months": self.period_months,
            "min_history_pct": self.min_history_pct,
            "remove_market_factor": self.remove_market_factor,
            "clustering_method": self.clustering_method,
            "n_clusters": self.n_clusters,
            "output_dir": self.output_dir,
            "correlation_threshold": self.correlation_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _cache_key_universe(config: StepConfig) -> str:
    """Generate cache key for universe data."""
    import hashlib
    filter_str = str(sorted(config.filters.items())) if config.filters else "none"
    hash_val = hashlib.md5(f"{config.data_source}:{filter_str}:{config.max_stocks}".encode()).hexdigest()[:12]
    return f"price_correlation:universe:{hash_val}"


def _cache_key_prices(tickers: list[str], start: str, end: str) -> str:
    """Generate cache key for price data."""
    import hashlib
    tickers_hash = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:12]
    return f"price_correlation:prices:{tickers_hash}:{start}:{end}"


# =============================================================================
# Step 1: Universe
# =============================================================================

def run_step_universe(
    config: StepConfig,
    state_manager: PipelineStateManager,
    progress_callback: Callable[[str], None] | None = None,
) -> list[str]:
    """
    Step 1: Fetch stock universe.

    Returns list of ticker symbols.
    """
    state_manager.state.mark_step_started("universe")
    state_manager.save_state()

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        log("Step 1: Fetching stock universe...")
        cache = get_cache()

        # Check cache first
        cache_key = _cache_key_universe(config)
        if cache and cache.is_connected:
            cached = cache.get(cache_key)
            if cached:
                import pickle
                tickers = pickle.loads(cached)
                log(f"  [Cache Hit] Loaded {len(tickers)} tickers from cache")
                state_manager.store_step_data("universe", tickers, {"count": len(tickers)})
                return tickers

        # Fetch based on data source
        if config.data_source == "sample":
            sample_size = config.max_stocks if config.max_stocks > 0 else 50
            tickers = get_sample_tickers(sample_size)
            log(f"  Using sample tickers: {len(tickers)}")

        elif config.data_source in ("fmp_all", "fmp_filtered"):
            from .fmp_client import FMPClient

            api_key = os.environ.get("FMP_API_KEY")
            if not api_key:
                raise ValueError("FMP_API_KEY required for FMP data source")

            client = FMPClient(api_key=api_key)

            if config.data_source == "fmp_all":
                log("  Fetching full universe from FMP...")
                stocks = client.get_full_universe_iterative(
                    progress_callback=lambda msg: log(f"    {msg}"),
                    split_threshold=475,
                )
            else:
                filters = config.filters or {}
                log(f"  Fetching from FMP with filters: {filters}")
                stocks = client.get_stock_screener(
                    market_cap_min=filters.get("market_cap_min"),
                    market_cap_max=filters.get("market_cap_max"),
                    volume_min=filters.get("volume_min"),
                    volume_max=filters.get("volume_max"),
                    progress_callback=lambda msg: log(f"    {msg}"),
                )

            tickers = [s["symbol"] for s in stocks]
            log(f"  Found {len(tickers)} stocks from FMP")

            # Apply max_stocks limit
            if config.max_stocks > 0 and len(tickers) > config.max_stocks:
                tickers = tickers[:config.max_stocks]
                log(f"  Limited to {len(tickers)} stocks")

        else:
            from .universe import get_full_universe
            tickers = get_full_universe()
            log(f"  Using full universe: {len(tickers)}")

        # Cache the result
        if cache and cache.is_connected:
            import pickle
            cache.set(cache_key, pickle.dumps(tickers), TTL_UNIVERSE)
            log(f"  [Cached] Universe data")

        state_manager.store_step_data("universe", tickers, {"count": len(tickers)})
        log(f"  Universe: {len(tickers)} tickers")
        return tickers

    except Exception as e:
        state_manager.state.mark_error("universe", str(e))
        state_manager.save_state()
        raise


# =============================================================================
# Step 2: Prices
# =============================================================================

def run_step_prices(
    config: StepConfig,
    state_manager: PipelineStateManager,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Step 2: Fetch price data for all tickers.

    Requires: universe step complete.
    Returns DataFrame of prices.
    """
    state_manager.state.mark_step_started("prices")
    state_manager.save_state()

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        # Load universe from previous step
        tickers = state_manager.load_step_data("universe")
        if tickers is None:
            raise ValueError("Universe data not found. Run universe step first.")

        log(f"Step 2: Fetching price data for {len(tickers)} tickers...")

        # Calculate date range
        end_date = config.end_date or datetime.now().strftime("%Y-%m-%d")
        if config.start_date:
            start_date = config.start_date
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=config.period_months * 30)
            start_date = start_dt.strftime("%Y-%m-%d")

        cache = get_cache()
        cache_key = _cache_key_prices(tickers, start_date, end_date)

        # Check cache
        if cache and cache.is_connected:
            cached = cache.get(cache_key)
            if cached:
                import pickle
                prices = pickle.loads(cached)
                log(f"  [Cache Hit] Loaded prices from cache ({prices.shape[1]} tickers, {prices.shape[0]} days)")
                state_manager.store_step_data("prices", prices, {
                    "tickers": prices.shape[1],
                    "days": prices.shape[0],
                    "start_date": start_date,
                    "end_date": end_date,
                })
                return prices

        # Fetch prices with progress
        from .ingestion import fetch_price_history

        def price_progress(current: int, total: int, message: str):
            log(f"  [Price Fetch] {message}")

        prices = fetch_price_history(
            tickers,
            start_date=start_date,
            end_date=end_date,
            period_months=config.period_months,
            progress_callback=price_progress,
        )

        log(f"  Fetched: {prices.shape[1]} tickers, {prices.shape[0]} days")

        # Cache the result
        if cache and cache.is_connected:
            import pickle
            try:
                serialized = pickle.dumps(prices)
                if len(serialized) < 100 * 1024 * 1024:  # Under 100MB
                    cache.set(cache_key, serialized, TTL_PRICES)
                    log(f"  [Cached] Price data ({len(serialized) / 1024 / 1024:.1f} MB)")
            except Exception as e:
                log(f"  [Cache Skip] Data too large or error: {e}")

        state_manager.store_step_data("prices", prices, {
            "tickers": prices.shape[1],
            "days": prices.shape[0],
            "start_date": start_date,
            "end_date": end_date,
        })
        return prices

    except Exception as e:
        state_manager.state.mark_error("prices", str(e))
        state_manager.save_state()
        raise


# =============================================================================
# Step 3: Preprocess
# =============================================================================

def run_step_preprocess(
    config: StepConfig,
    state_manager: PipelineStateManager,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Step 3: Preprocess prices - compute returns, normalize.

    Requires: prices step complete.
    Returns DataFrame of normalized returns.
    """
    state_manager.state.mark_step_started("preprocess")
    state_manager.save_state()

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        prices = state_manager.load_step_data("prices")
        if prices is None:
            raise ValueError("Price data not found. Run prices step first.")

        log(f"Step 3: Preprocessing {prices.shape[1]} tickers...")

        returns = preprocess_pipeline(
            prices,
            min_history_pct=config.min_history_pct,
            remove_market=config.remove_market_factor,
        )

        valid_tickers = list(returns.columns)
        dropped = prices.shape[1] - len(valid_tickers)

        log(f"  After preprocessing: {len(valid_tickers)} tickers ({dropped} dropped)")

        state_manager.store_step_data("preprocess", returns, {
            "tickers": len(valid_tickers),
            "days": returns.shape[0],
            "dropped": dropped,
        })
        return returns

    except Exception as e:
        state_manager.state.mark_error("preprocess", str(e))
        state_manager.save_state()
        raise


# =============================================================================
# Step 4: Correlation
# =============================================================================

def run_step_correlation(
    config: StepConfig,
    state_manager: PipelineStateManager,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Step 4: Compute correlation and distance matrices.

    Requires: preprocess step complete.
    Returns dict with corr_matrix, dist_matrix, condensed_dist, tickers.
    """
    state_manager.state.mark_step_started("correlation")
    state_manager.save_state()

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        returns = state_manager.load_step_data("preprocess")
        if returns is None:
            raise ValueError("Preprocessed data not found. Run preprocess step first.")

        log(f"Step 4: Computing correlations for {returns.shape[1]} tickers...")

        corr_matrix = compute_correlation_matrix(returns)
        dist_matrix = correlation_to_distance(corr_matrix)
        condensed_dist = get_condensed_distance(returns)

        log(f"  Correlation matrix: {corr_matrix.shape}")

        result = {
            "corr_matrix": corr_matrix,
            "dist_matrix": dist_matrix,
            "condensed_dist": condensed_dist,
            "tickers": list(returns.columns),
        }

        state_manager.store_step_data("correlation", result, {
            "matrix_shape": list(corr_matrix.shape),
            "tickers": len(returns.columns),
        })
        return result

    except Exception as e:
        state_manager.state.mark_error("correlation", str(e))
        state_manager.save_state()
        raise


# =============================================================================
# Step 5: Clustering
# =============================================================================

def run_step_clustering(
    config: StepConfig,
    state_manager: PipelineStateManager,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Step 5: Run clustering algorithm.

    Requires: correlation step complete.
    Returns dict with labels, stats, method info.
    """
    state_manager.state.mark_step_started("clustering")
    state_manager.save_state()

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        corr_data = state_manager.load_step_data("correlation")
        if corr_data is None:
            raise ValueError("Correlation data not found. Run correlation step first.")

        dist_matrix = corr_data["dist_matrix"]
        condensed_dist = corr_data["condensed_dist"]
        tickers = corr_data["tickers"]

        log(f"Step 5: Clustering {len(tickers)} stocks...")

        if config.clustering_method == "dbscan":
            eps = find_optimal_eps(dist_matrix)
            labels = cluster_dbscan(dist_matrix, eps=eps)
            method_info = f"DBSCAN (eps={eps:.3f})"
        else:
            Z = cluster_hierarchical(condensed_dist, method="average")
            if config.n_clusters:
                best_k = config.n_clusters
                best_score = compute_silhouette(dist_matrix, cut_dendrogram(Z, n_clusters=best_k) - 1)
            else:
                best_k, best_score = find_optimal_k(Z, dist_matrix)
            labels = cut_dendrogram(Z, n_clusters=best_k) - 1
            method_info = f"Hierarchical (k={best_k})"

        log(f"  Method: {method_info}")

        silhouette = compute_silhouette(dist_matrix, labels)
        stats = compute_cluster_stats(labels, tickers)

        log(f"  Silhouette score: {silhouette:.3f}")
        log(f"  Clusters: {stats['n_clusters']}, Noise: {stats['n_noise']}")

        result = {
            "labels": labels,
            "tickers": tickers,
            "stats": stats,
            "silhouette": silhouette,
            "method": config.clustering_method,
            "method_info": method_info,
        }

        state_manager.store_step_data("clustering", result, {
            "n_clusters": stats["n_clusters"],
            "n_noise": stats["n_noise"],
            "silhouette": round(silhouette, 4),
            "method": method_info,
        })
        return result

    except Exception as e:
        state_manager.state.mark_error("clustering", str(e))
        state_manager.save_state()
        raise


# =============================================================================
# Step 6: Export
# =============================================================================

def run_step_export(
    config: StepConfig,
    state_manager: PipelineStateManager,
    progress_callback: Callable[[str], None] | None = None,
    start_time: float | None = None,
) -> dict:
    """
    Step 6: Export results to files and database.

    Requires: clustering and correlation steps complete.
    Returns dict with export info.
    """
    state_manager.state.mark_step_started("export")
    state_manager.save_state()

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        clustering_data = state_manager.load_step_data("clustering")
        corr_data = state_manager.load_step_data("correlation")

        if clustering_data is None:
            raise ValueError("Clustering data not found. Run clustering step first.")
        if corr_data is None:
            raise ValueError("Correlation data not found. Run correlation step first.")

        log("Step 6: Exporting results...")

        labels = clustering_data["labels"]
        tickers = clustering_data["tickers"]
        stats = clustering_data["stats"]
        silhouette = clustering_data["silhouette"]
        corr_matrix = corr_data["corr_matrix"]

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        execution_time = time.time() - start_time if start_time else 0

        output_files = export_all(
            labels,
            tickers,
            corr_matrix,
            output_dir,
            correlation_threshold=config.correlation_threshold,
            n_clusters=stats["n_clusters"],
            n_noise=stats["n_noise"],
            silhouette_score=silhouette,
            clustering_method=config.clustering_method,
            execution_time_seconds=execution_time,
        )

        for name, item in output_files.items():
            if name == "db_export":
                if item.get("success"):
                    log(f"  {name}: {item['clusters_exported']} clusters, "
                        f"{item['correlations_exported']} correlations exported")
                else:
                    log(f"  {name}: {item.get('message', 'failed')}")
            else:
                log(f"  {name}: {item}")

        # Summary
        log("")
        log("=" * 50)
        log("PIPELINE COMPLETED SUCCESSFULLY")
        log("=" * 50)
        log(f"  Stocks processed: {len(tickers)}")
        log(f"  Clusters found: {stats['n_clusters']}")
        log(f"  Silhouette score: {silhouette:.3f}")
        log(f"  Total time: {execution_time:.1f}s")
        log("=" * 50)

        result = {
            "output_files": output_files,
            "execution_time": execution_time,
            "n_stocks": len(tickers),
            "n_clusters": stats["n_clusters"],
            "silhouette": silhouette,
        }

        state_manager.store_step_data("export", result, {
            "files": list(output_files.keys()),
            "execution_time": round(execution_time, 1),
        })
        return result

    except Exception as e:
        state_manager.state.mark_error("export", str(e))
        state_manager.save_state()
        raise


# =============================================================================
# Run single step or full pipeline
# =============================================================================

STEP_FUNCTIONS = {
    "universe": run_step_universe,
    "prices": run_step_prices,
    "preprocess": run_step_preprocess,
    "correlation": run_step_correlation,
    "clustering": run_step_clustering,
    "export": run_step_export,
}


def run_single_step(
    step: str,
    config: StepConfig | dict,
    session_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Run a single pipeline step.

    Args:
        step: Step name (universe, prices, preprocess, correlation, clustering, export)
        config: Step configuration
        session_id: Session ID for state management
        progress_callback: Optional callback for progress messages

    Returns:
        Step result data
    """
    if isinstance(config, dict):
        config = StepConfig.from_dict(config)

    state_manager = get_state_manager(session_id)
    state_manager.set_config(config.to_dict())

    if step not in STEP_FUNCTIONS:
        raise ValueError(f"Unknown step: {step}. Valid steps: {list(STEP_FUNCTIONS.keys())}")

    step_func = STEP_FUNCTIONS[step]

    # Export step needs start_time
    if step == "export":
        return step_func(config, state_manager, progress_callback, start_time=time.time())

    return step_func(config, state_manager, progress_callback)


def run_from_step(
    start_step: str,
    config: StepConfig | dict,
    session_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Run pipeline from a specific step onwards.

    Args:
        start_step: Step to start from
        config: Step configuration
        session_id: Session ID for state management
        progress_callback: Optional callback for progress messages

    Returns:
        Final result from export step
    """
    if isinstance(config, dict):
        config = StepConfig.from_dict(config)

    state_manager = get_state_manager(session_id)
    state_manager.set_config(config.to_dict())

    # Clear from the starting step
    state_manager.clear_step(start_step)

    start_idx = PIPELINE_STEPS.index(start_step)
    steps_to_run = PIPELINE_STEPS[start_idx:]

    pipeline_start = time.time()
    result = None

    for step in steps_to_run:
        step_func = STEP_FUNCTIONS[step]
        if step == "export":
            result = step_func(config, state_manager, progress_callback, start_time=pipeline_start)
        else:
            result = step_func(config, state_manager, progress_callback)

    return result


def run_full_pipeline(
    config: StepConfig | dict,
    session_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Run the complete pipeline from scratch.

    Args:
        config: Pipeline configuration
        session_id: Optional session ID (creates new if not provided)
        progress_callback: Optional callback for progress messages

    Returns:
        Final result from export step
    """
    return run_from_step("universe", config, session_id, progress_callback)
