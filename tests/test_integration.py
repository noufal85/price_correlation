"""Integration tests using real data (no mocks)."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from price_correlation.clustering import (
    auto_cluster,
    cluster_dbscan,
    cluster_hierarchical,
    cut_dendrogram,
    find_optimal_eps,
)
from price_correlation.correlation import (
    compute_correlation_matrix,
    correlation_to_distance,
    get_condensed_distance,
)
from price_correlation.export import export_all
from price_correlation.ingestion import fetch_price_history
from price_correlation.pipeline import PipelineConfig, run_pipeline
from price_correlation.preprocess import preprocess_pipeline
from price_correlation.validation import compute_cluster_stats, compute_silhouette


class TestDataPipeline:
    """Test data ingestion and preprocessing with real data."""

    def test_fetch_and_preprocess(self):
        """
        Fetch real prices → preprocess → validate output.

        Tests: fetch_price_history, preprocess_pipeline
        """
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "BAC", "XOM"]

        prices = fetch_price_history(tickers, period_months=6)

        assert isinstance(prices, pd.DataFrame)
        assert len(prices) > 50
        assert len(prices.columns) >= 5

        returns = preprocess_pipeline(prices, min_history_pct=0.8)

        assert isinstance(returns, pd.DataFrame)
        assert len(returns.columns) >= 5

        mean_of_means = returns.mean().mean()
        mean_of_stds = returns.std().mean()

        assert abs(mean_of_means) < 0.1
        assert abs(mean_of_stds - 1.0) < 0.1

        assert not returns.isna().any().any()


class TestCorrelationClustering:
    """Test correlation and clustering with real data."""

    def test_correlation_and_clustering(self):
        """
        Compute correlations → cluster → validate.

        Tests: compute_correlation_matrix, correlation_to_distance,
               get_condensed_distance, cluster_dbscan, cluster_hierarchical,
               compute_silhouette
        """
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "JPM", "BAC", "GS", "WFC", "C",
            "XOM", "CVX", "COP", "SLB", "EOG",
        ]

        prices = fetch_price_history(tickers, period_months=6)
        returns = preprocess_pipeline(prices, min_history_pct=0.8)
        valid_tickers = list(returns.columns)

        corr_matrix = compute_correlation_matrix(returns)

        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert corr_matrix.shape[0] == len(valid_tickers)
        assert np.allclose(corr_matrix, corr_matrix.T)
        assert np.allclose(np.diag(corr_matrix), 1.0)

        dist_matrix = correlation_to_distance(corr_matrix)

        assert np.allclose(np.diag(dist_matrix), 0.0)
        assert dist_matrix.min() >= 0
        assert dist_matrix.max() <= 2.0

        condensed = get_condensed_distance(returns)
        expected_len = len(valid_tickers) * (len(valid_tickers) - 1) // 2
        assert len(condensed) == expected_len

        eps = find_optimal_eps(dist_matrix)
        dbscan_labels = cluster_dbscan(dist_matrix, eps=eps)
        assert len(dbscan_labels) == len(valid_tickers)

        Z = cluster_hierarchical(condensed, method="average")
        hier_labels = cut_dendrogram(Z, n_clusters=3)
        assert len(hier_labels) == len(valid_tickers)
        assert set(hier_labels) == {1, 2, 3}

        silhouette = compute_silhouette(dist_matrix, hier_labels - 1)
        assert -1 <= silhouette <= 1


class TestFullPipeline:
    """Test complete pipeline with real data."""

    def test_full_pipeline_small(self):
        """
        Run complete pipeline on small dataset.

        Tests: run_pipeline (end-to-end orchestrator)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                tickers=[
                    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
                    "JPM", "BAC", "GS",
                    "XOM", "CVX",
                ],
                period_months=3,
                output_dir=tmpdir,
                visualize=False,
                clustering_method="hierarchical",
            )

            result = run_pipeline(config)

            assert result["n_stocks_processed"] >= 8
            assert result["n_clusters"] >= 1
            assert -1 <= result["silhouette_score"] <= 1
            assert result["execution_time_seconds"] > 0

            output_dir = Path(tmpdir)
            assert (output_dir / "stock_clusters.json").exists()
            assert (output_dir / "equity_clusters.parquet").exists()
            assert (output_dir / "pair_correlations.parquet").exists()

            clusters_df = pd.read_parquet(output_dir / "equity_clusters.parquet")
            assert "ticker" in clusters_df.columns
            assert "cluster_id" in clusters_df.columns
            assert len(clusters_df) >= 8

            import json
            with open(output_dir / "stock_clusters.json") as f:
                clusters_json = json.load(f)
            assert isinstance(clusters_json, list)
            assert all("cluster" in c and "members" in c for c in clusters_json)


class TestExport:
    """Test export functionality with real clustering results."""

    def test_export_formats(self):
        """
        Cluster real data → export → verify files.

        Tests: export_all, file format correctness
        """
        tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "XOM", "CVX", "JNJ"]

        prices = fetch_price_history(tickers, period_months=3)
        returns = preprocess_pipeline(prices, min_history_pct=0.8)
        valid_tickers = list(returns.columns)

        corr_matrix = compute_correlation_matrix(returns)
        dist_matrix = correlation_to_distance(corr_matrix)
        condensed = get_condensed_distance(returns)

        labels, _ = auto_cluster(dist_matrix, condensed, method="hierarchical")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_files = export_all(
                labels,
                valid_tickers,
                corr_matrix,
                tmpdir,
                correlation_threshold=0.5,
            )

            for path in output_files.values():
                assert path.exists()

            clusters_df = pd.read_parquet(output_files["clusters_parquet"])
            assert len(clusters_df) == len(valid_tickers)

            corr_df = pd.read_parquet(output_files["correlations_parquet"])
            assert all(abs(corr_df["correlation"]) >= 0.5)
