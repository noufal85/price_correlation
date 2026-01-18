"""Pipeline orchestrator - run the complete clustering workflow."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from .clustering import auto_cluster, cluster_dbscan, cluster_hierarchical, cut_dendrogram, find_optimal_eps, find_optimal_k
from .correlation import compute_correlation_matrix, correlation_to_distance, get_condensed_distance
from .export import export_all
from .ingestion import fetch_price_history
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
    use_sample: bool = False
    sample_size: int = 50
    tickers: list[str] = field(default_factory=list)


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

    print("Step 1: Getting stock universe...")
    if config.tickers:
        tickers = config.tickers
    elif config.use_sample:
        tickers = get_sample_tickers(config.sample_size)
    else:
        tickers = get_full_universe()
    print(f"  Universe: {len(tickers)} tickers")

    print("Step 2: Fetching price data...")
    prices = fetch_price_history(
        tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        period_months=config.period_months,
    )
    print(f"  Fetched: {prices.shape[1]} tickers, {prices.shape[0]} days")

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
    )
    for name, path in output_files.items():
        print(f"  {name}: {path}")

    elapsed = time.time() - start_time

    print_cluster_summary(labels, valid_tickers)

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
