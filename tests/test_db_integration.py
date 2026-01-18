"""Integration tests for TimescaleDB functionality.

These tests require a running TimescaleDB instance.
Skip if database is not available.
"""

import os
from datetime import date

import numpy as np
import pytest

# Set test environment
os.environ.setdefault("ENABLE_DB_EXPORT", "true")


def db_available():
    """Check if TimescaleDB is available."""
    try:
        from price_correlation.db import TimescaleClient, get_db_config

        config = get_db_config()
        client = TimescaleClient(**config)
        client.connect()
        return client.is_connected
    except Exception:
        return False


@pytest.fixture
def db_client():
    """Create a TimescaleDB client for testing."""
    from price_correlation.db import TimescaleClient, get_db_config

    config = get_db_config()
    client = TimescaleClient(**config)
    client.connect()
    yield client
    client.close()


@pytest.fixture
def sample_data():
    """Generate sample clustering data for testing."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    labels = np.array([0, 0, 1, 1, 2])

    # Create a sample correlation matrix
    n = len(tickers)
    corr_matrix = np.eye(n)
    # Add some correlations
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.85
    corr_matrix[2, 3] = corr_matrix[3, 2] = 0.78
    corr_matrix[0, 2] = corr_matrix[2, 0] = 0.45

    return {
        "tickers": tickers,
        "labels": labels,
        "corr_matrix": corr_matrix,
        "n_clusters": 3,
        "n_noise": 0,
        "silhouette_score": 0.45,
        "clustering_method": "hierarchical",
    }


@pytest.mark.skipif(not db_available(), reason="TimescaleDB not available")
class TestTimescaleDBIntegration:
    """Integration tests for TimescaleDB operations."""

    def test_connection(self, db_client):
        """Test that we can connect to TimescaleDB."""
        assert db_client.is_connected

    def test_export_clusters(self, db_client, sample_data):
        """Test exporting cluster assignments."""
        analysis_date = date.today()

        count = db_client.export_clusters(
            labels=sample_data["labels"],
            tickers=sample_data["tickers"],
            analysis_date=analysis_date,
        )

        assert count == len(sample_data["tickers"])

    def test_export_correlations(self, db_client, sample_data):
        """Test exporting correlation pairs."""
        analysis_date = date.today()

        count = db_client.export_correlations(
            corr_matrix=sample_data["corr_matrix"],
            tickers=sample_data["tickers"],
            threshold=0.7,
            analysis_date=analysis_date,
        )

        # Should export pairs above threshold (0.85 and 0.78)
        assert count == 2

    def test_export_run_metadata(self, db_client, sample_data):
        """Test exporting run metadata."""
        analysis_date = date.today()

        # Should not raise
        db_client.export_run_metadata(
            n_stocks_processed=len(sample_data["tickers"]),
            n_clusters=sample_data["n_clusters"],
            n_noise=sample_data["n_noise"],
            silhouette_score=sample_data["silhouette_score"],
            clustering_method=sample_data["clustering_method"],
            execution_time_seconds=10.5,
            analysis_date=analysis_date,
        )

    def test_full_export_pipeline(self, sample_data):
        """Test the high-level export function."""
        from price_correlation.db import export_to_timescaledb

        result = export_to_timescaledb(
            labels=sample_data["labels"],
            tickers=sample_data["tickers"],
            corr_matrix=sample_data["corr_matrix"],
            n_clusters=sample_data["n_clusters"],
            n_noise=sample_data["n_noise"],
            silhouette_score=sample_data["silhouette_score"],
            clustering_method=sample_data["clustering_method"],
            execution_time_seconds=10.5,
            correlation_threshold=0.7,
        )

        assert result["success"]
        assert result["clusters_exported"] == len(sample_data["tickers"])
        assert result["correlations_exported"] == 2

    def test_query_exported_data(self, db_client, sample_data):
        """Test querying exported data."""
        analysis_date = date.today()

        # First export the data
        db_client.export_clusters(
            labels=sample_data["labels"],
            tickers=sample_data["tickers"],
            analysis_date=analysis_date,
        )

        # Query the data
        conn = db_client.connect()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT ticker, cluster_id FROM equity_clusters WHERE analysis_date = %s",
            (analysis_date,),
        )
        results = cursor.fetchall()
        cursor.close()

        assert len(results) == len(sample_data["tickers"])

        # Check specific assignments
        result_dict = {r[0]: r[1] for r in results}
        assert result_dict["AAPL"] == 0
        assert result_dict["MSFT"] == 0
        assert result_dict["META"] == 2


class TestDBConfig:
    """Test database configuration."""

    def test_get_db_config(self):
        """Test getting DB configuration from environment."""
        from price_correlation.db import get_db_config

        config = get_db_config()

        assert "host" in config
        assert "port" in config
        assert "database" in config
        assert "user" in config
        assert "password" in config
        assert isinstance(config["port"], int)

    def test_is_db_export_enabled(self):
        """Test DB export enabled check."""
        from price_correlation.db import is_db_export_enabled

        # Should return True by default in test environment
        original = os.environ.get("ENABLE_DB_EXPORT")

        os.environ["ENABLE_DB_EXPORT"] = "true"
        assert is_db_export_enabled()

        os.environ["ENABLE_DB_EXPORT"] = "false"
        assert not is_db_export_enabled()

        # Restore
        if original:
            os.environ["ENABLE_DB_EXPORT"] = original

    def test_db_stats_when_disabled(self):
        """Test DB stats when export is disabled."""
        from price_correlation.db import get_db_stats

        original = os.environ.get("ENABLE_DB_EXPORT")
        os.environ["ENABLE_DB_EXPORT"] = "false"

        stats = get_db_stats()
        assert not stats["enabled"]

        if original:
            os.environ["ENABLE_DB_EXPORT"] = original
