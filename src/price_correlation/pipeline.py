"""Pipeline orchestrator - run the complete clustering workflow."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from .clustering import auto_cluster, cluster_dbscan, cluster_hierarchical, cut_dendrogram, find_optimal_eps, find_optimal_k
from .correlation import compute_correlation_matrix, correlation_to_distance, get_condensed_distance
from .export import export_all
from .ingestion import fetch_price_history_cached
from .preprocess import preprocess_pipeline
from .universe import get_full_universe, get_sample_tickers
from .validation import compute_cluster_stats, compute_silhouette, generate_tsne_plot, print_cluster_summary


@dataclass
class PipelineConfig:
    """Configuration for the clustering pipeline."""

    start_date: str | None = None
    end_date: str | None = None
    period_months: int = 18
    min_history_pct: float = 0.90
    remove_market_factor: bool = False
    clustering_method: str = "hierarchical"
    n_clusters: int | None = None
    output_dir: str = "./output"
    correlation_threshold: float = 0.7
    visualize: bool = True
    # Legacy options (backward compatible)
    use_sample: bool = False
    sample_size: int = 50
    tickers: list[str] = field(default_factory=list)
    # New data source options
    data_source: str = "sample"  # "sample", "fmp_all", "fmp_filtered"
    filters: dict = field(default_factory=dict)
    max_stocks: int = 0  # 0 = no limit


def run_pipeline(config: PipelineConfig | dict | None = None) -> dict:
    """
    Run the complete stock clustering pipeline.

    Args:
        config: PipelineConfig or dict with configuration

    Returns:
        Summary dictionary with results
    """
    if config is None:
        config = PipelineConfig()
    elif isinstance(config, dict):
        config = PipelineConfig(**config)

    start_time = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Getting stock universe...", flush=True)

    # Determine data source (support both old and new config formats)
    data_source = config.data_source
    if data_source == "sample" and config.use_sample:
        # Legacy format
        data_source = "sample"
    elif data_source == "sample" and config.tickers:
        # Explicit tickers provided
        data_source = "explicit"

    if data_source == "explicit" or config.tickers:
        tickers = config.tickers
        print(f"  Using provided tickers: {len(tickers)}", flush=True)

    elif data_source == "sample":
        # Use hardcoded sample tickers
        sample_size = config.max_stocks if config.max_stocks > 0 else config.sample_size
        if sample_size == 0:
            sample_size = 50
        tickers = get_sample_tickers(sample_size)
        print(f"  Using sample tickers: {len(tickers)}", flush=True)

    elif data_source in ("fmp_all", "fmp_filtered"):
        # Use FMP API
        import os
        from .fmp_client import FMPClient

        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable required for FMP data source")

        client = FMPClient(api_key=api_key)
        filters = config.filters or {}

        if data_source == "fmp_all":
            print("  Fetching full universe from FMP (no filters)...", flush=True)
            stocks = client.get_full_universe_iterative(
                progress_callback=lambda msg: print(f"    {msg}", flush=True),
                split_threshold=475,
            )
        else:
            # fmp_filtered
            mcap_min = filters.get("market_cap_min")
            mcap_max = filters.get("market_cap_max")
            vol_min = filters.get("volume_min")
            vol_max = filters.get("volume_max")

            filter_desc = []
            if mcap_min or mcap_max:
                if mcap_min and mcap_max:
                    filter_desc.append(f"Market Cap: ${mcap_min/1e9:.0f}B-${mcap_max/1e9:.0f}B")
                elif mcap_min:
                    filter_desc.append(f"Market Cap: >=${mcap_min/1e9:.0f}B")
                else:
                    filter_desc.append(f"Market Cap: <=${mcap_max/1e9:.0f}B")
            if vol_min:
                filter_desc.append(f"Volume: >={vol_min/1000:.0f}K")

            print(f"  Fetching from FMP with filters: {', '.join(filter_desc) or 'none'}", flush=True)

            stocks = client.get_stock_screener(
                market_cap_min=mcap_min,
                market_cap_max=mcap_max,
                volume_min=vol_min,
                volume_max=vol_max,
                progress_callback=lambda msg: print(f"    {msg}", flush=True),
            )

        tickers = [s["symbol"] for s in stocks]
        print(f"  Found {len(tickers)} stocks from FMP", flush=True)

        # Apply max_stocks limit
        if config.max_stocks > 0 and len(tickers) > config.max_stocks:
            tickers = tickers[:config.max_stocks]
            print(f"  Limited to {len(tickers)} stocks (max_stocks={config.max_stocks})", flush=True)

    else:
        # Fallback to full universe from Wikipedia
        tickers = get_full_universe()
        print(f"  Using full universe: {len(tickers)}", flush=True)

    print(f"  Universe: {len(tickers)} tickers", flush=True)

    print("Step 2: Fetching price data...", flush=True)

    def price_progress(current: int, total: int, message: str):
        """Progress callback for price fetching."""
        print(f"  [Price Fetch] {message}", flush=True)

    prices = fetch_price_history_cached(
        tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        period_months=config.period_months,
        progress_callback=price_progress,
    )
    print(f"  Fetched: {prices.shape[1]} tickers, {prices.shape[0]} days", flush=True)

    print("Step 3: Preprocessing...")
    returns = preprocess_pipeline(
        prices,
        min_history_pct=config.min_history_pct,
        remove_market=config.remove_market_factor,
    )
    valid_tickers = list(returns.columns)
    print(f"  After preprocessing: {len(valid_tickers)} tickers")

    print("Step 4: Computing correlations...")
    corr_matrix = compute_correlation_matrix(returns)
    dist_matrix = correlation_to_distance(corr_matrix)
    condensed_dist = get_condensed_distance(returns)
    print(f"  Correlation matrix: {corr_matrix.shape}")

    print("Step 5: Clustering...")
    if config.clustering_method == "dbscan":
        eps = find_optimal_eps(dist_matrix)
        labels = cluster_dbscan(dist_matrix, eps=eps)
        method_info = f"DBSCAN (eps={eps:.3f})"
    else:
        Z = cluster_hierarchical(condensed_dist, method="average")
        if config.n_clusters:
            best_k = config.n_clusters
        else:
            best_k, _ = find_optimal_k(Z, dist_matrix)
        labels = cut_dendrogram(Z, n_clusters=best_k) - 1
        method_info = f"Hierarchical (k={best_k})"
    print(f"  Method: {method_info}")

    print("Step 6: Validating...")
    silhouette = compute_silhouette(dist_matrix, labels)
    stats = compute_cluster_stats(labels, valid_tickers)
    print(f"  Silhouette score: {silhouette:.3f}")
    print(f"  Clusters: {stats['n_clusters']}, Noise: {stats['n_noise']}")

    if config.visualize:
        print("Step 7: Generating visualization...")
        viz_path = output_dir / "cluster_visualization.png"
        generate_tsne_plot(dist_matrix, labels, valid_tickers, str(viz_path))
        print(f"  Saved: {viz_path}")

    print("Step 8: Exporting results...")
    output_files = export_all(
        labels,
        valid_tickers,
        corr_matrix,
        output_dir,
        correlation_threshold=config.correlation_threshold,
        n_clusters=stats["n_clusters"],
        n_noise=stats["n_noise"],
        silhouette_score=silhouette,
        clustering_method=config.clustering_method,
        execution_time_seconds=time.time() - start_time,
    )
    for name, item in output_files.items():
        if name == "db_export":
            if item.get("success"):
                print(f"  {name}: {item['clusters_exported']} clusters, "
                      f"{item['correlations_exported']} correlations exported")
            else:
                print(f"  {name}: {item.get('message', 'failed')}")
        else:
            print(f"  {name}: {item}")

    elapsed = time.time() - start_time

    print_cluster_summary(labels, valid_tickers)

    # Print clear completion summary
    print("", flush=True)
    print("=" * 50, flush=True)
    print("PIPELINE COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 50, flush=True)
    print(f"  Stocks processed: {len(valid_tickers)}", flush=True)
    print(f"  Clusters found: {stats['n_clusters']}", flush=True)
    print(f"  Silhouette score: {silhouette:.3f}", flush=True)
    print(f"  Total time: {elapsed:.1f}s", flush=True)
    print("=" * 50, flush=True)

    return {
        "n_stocks_input": len(tickers),
        "n_stocks_processed": len(valid_tickers),
        "n_clusters": stats["n_clusters"],
        "n_noise": stats["n_noise"],
        "silhouette_score": silhouette,
        "clustering_method": config.clustering_method,
        "output_files": [str(p) for p in output_files.values()],
        "execution_time_seconds": elapsed,
    }


def run_sample_pipeline() -> dict:
    """Run pipeline on sample data for testing."""
    config = PipelineConfig(
        use_sample=True,
        sample_size=30,
        period_months=6,
        visualize=False,
    )
    return run_pipeline(config)


if __name__ == "__main__":
    result = run_pipeline()
    print(f"\nCompleted in {result['execution_time_seconds']:.1f}s")
