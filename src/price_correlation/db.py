"""TimescaleDB client for storing clustering results."""

import logging
import os
import time
from datetime import date
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0


def get_db_config() -> dict:
    """Get TimescaleDB configuration from environment variables."""
    return {
        "host": os.environ.get("TIMESCALE_HOST", "192.168.68.88"),
        "port": int(os.environ.get("TIMESCALE_PORT", 5432)),
        "database": os.environ.get("TIMESCALE_DB", "timescaledb"),
        "user": os.environ.get("TIMESCALE_USER", "postgres"),
        "password": os.environ.get("TIMESCALE_PASSWORD", "password"),
    }


def is_db_export_enabled() -> bool:
    """Check if DB export is enabled via environment."""
    return os.environ.get("ENABLE_DB_EXPORT", "true").lower() in ("true", "1", "yes")


def retry_with_backoff(
    max_attempts: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
):
    """
    Decorator for DB operations with exponential backoff.

    Retries on connection and temporary errors.
    Does not retry on syntax or constraint violations.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import psycopg2

            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (
                    psycopg2.OperationalError,
                    psycopg2.InterfaceError,
                    ConnectionError,
                ) as e:
                    last_error = e
                    if attempt == max_attempts - 1:
                        raise

                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"DB operation failed, retry {attempt + 1}/{max_attempts} "
                        f"in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                except (
                    psycopg2.ProgrammingError,
                    psycopg2.IntegrityError,
                ) as e:
                    # Don't retry syntax errors or constraint violations
                    raise

            raise last_error

        return wrapper

    return decorator


class TimescaleClient:
    """Client for TimescaleDB operations with connection pooling and retry logic."""

    def __init__(
        self,
        host: str = "192.168.68.88",
        port: int = 5432,
        database: str = "timescaledb",
        user: str = "postgres",
        password: str = "password",
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self):
        """Establish database connection."""
        if self._conn is not None and not self._conn.closed:
            return self._conn

        import psycopg2

        try:
            self._conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=10,
            )
            self._conn.autocommit = False
            logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}")
            return self._conn
        except Exception as e:
            logger.error(f"TimescaleDB connection failed: {e}")
            raise

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._conn is not None and not self._conn.closed

    def init_schema(self) -> None:
        """Initialize database schema from SQL file."""
        schema_path = Path(__file__).parent.parent.parent / "scripts" / "init_timescaledb.sql"

        if not schema_path.exists():
            logger.warning(f"Schema file not found: {schema_path}")
            return

        conn = self.connect()
        cursor = conn.cursor()

        try:
            with open(schema_path) as f:
                sql = f.read()
            cursor.execute(sql)
            conn.commit()
            logger.info("Database schema initialized")
        except Exception as e:
            conn.rollback()
            logger.error(f"Schema initialization failed: {e}")
            raise
        finally:
            cursor.close()

    @retry_with_backoff()
    def export_clusters(
        self,
        labels: np.ndarray,
        tickers: list[str],
        analysis_date: date | None = None,
    ) -> int:
        """
        Export cluster assignments to database.

        Args:
            labels: Cluster labels array
            tickers: List of ticker symbols
            analysis_date: Analysis date (defaults to today)

        Returns:
            Number of rows inserted
        """
        if analysis_date is None:
            analysis_date = date.today()

        conn = self.connect()
        cursor = conn.cursor()

        try:
            # Delete existing entries for this date
            cursor.execute(
                "DELETE FROM equity_clusters WHERE analysis_date = %s",
                (analysis_date,),
            )

            # Insert new entries
            values = [
                (analysis_date, ticker, int(label))
                for ticker, label in zip(tickers, labels)
            ]

            from psycopg2.extras import execute_values

            execute_values(
                cursor,
                """
                INSERT INTO equity_clusters (analysis_date, ticker, cluster_id)
                VALUES %s
                ON CONFLICT (analysis_date, ticker)
                DO UPDATE SET cluster_id = EXCLUDED.cluster_id
                """,
                values,
            )

            conn.commit()
            logger.info(f"Exported {len(values)} cluster assignments")
            return len(values)

        except Exception as e:
            conn.rollback()
            logger.error(f"Cluster export failed: {e}")
            raise
        finally:
            cursor.close()

    @retry_with_backoff()
    def export_correlations(
        self,
        corr_matrix: np.ndarray,
        tickers: list[str],
        threshold: float = 0.7,
        analysis_date: date | None = None,
    ) -> int:
        """
        Export significant correlations to database.

        Args:
            corr_matrix: Correlation matrix
            tickers: List of ticker symbols
            threshold: Minimum correlation threshold
            analysis_date: Analysis date (defaults to today)

        Returns:
            Number of rows inserted
        """
        if analysis_date is None:
            analysis_date = date.today()

        conn = self.connect()
        cursor = conn.cursor()

        try:
            # Delete existing entries for this date
            cursor.execute(
                "DELETE FROM pair_correlations WHERE analysis_date = %s",
                (analysis_date,),
            )

            # Build values for significant correlations
            n = len(tickers)
            values = []
            for i in range(n):
                for j in range(i + 1, n):
                    corr = corr_matrix[i, j]
                    if abs(corr) >= threshold:
                        values.append(
                            (analysis_date, tickers[i], tickers[j], float(corr))
                        )

            if not values:
                conn.commit()
                logger.info("No correlations above threshold to export")
                return 0

            from psycopg2.extras import execute_values

            execute_values(
                cursor,
                """
                INSERT INTO pair_correlations
                    (analysis_date, ticker_a, ticker_b, correlation)
                VALUES %s
                ON CONFLICT (analysis_date, ticker_a, ticker_b)
                DO UPDATE SET correlation = EXCLUDED.correlation
                """,
                values,
            )

            conn.commit()
            logger.info(f"Exported {len(values)} correlation pairs")
            return len(values)

        except Exception as e:
            conn.rollback()
            logger.error(f"Correlation export failed: {e}")
            raise
        finally:
            cursor.close()

    @retry_with_backoff()
    def export_run_metadata(
        self,
        n_stocks_processed: int,
        n_clusters: int,
        n_noise: int = 0,
        silhouette_score: float | None = None,
        clustering_method: str | None = None,
        execution_time_seconds: float | None = None,
        analysis_date: date | None = None,
    ) -> None:
        """Export analysis run metadata to database."""
        if analysis_date is None:
            analysis_date = date.today()

        conn = self.connect()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO analysis_runs
                    (analysis_date, n_stocks_processed, n_clusters, n_noise,
                     silhouette_score, clustering_method, execution_time_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (analysis_date)
                DO UPDATE SET
                    n_stocks_processed = EXCLUDED.n_stocks_processed,
                    n_clusters = EXCLUDED.n_clusters,
                    n_noise = EXCLUDED.n_noise,
                    silhouette_score = EXCLUDED.silhouette_score,
                    clustering_method = EXCLUDED.clustering_method,
                    execution_time_seconds = EXCLUDED.execution_time_seconds,
                    created_at = NOW()
                """,
                (
                    analysis_date,
                    n_stocks_processed,
                    n_clusters,
                    n_noise,
                    silhouette_score,
                    clustering_method,
                    execution_time_seconds,
                ),
            )

            conn.commit()
            logger.info(f"Exported run metadata for {analysis_date}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Metadata export failed: {e}")
            raise
        finally:
            cursor.close()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# Global client instance
_client: TimescaleClient | None = None


def get_db_client() -> TimescaleClient | None:
    """Get or create global database client instance."""
    global _client

    if not is_db_export_enabled():
        return None

    if _client is None:
        config = get_db_config()
        _client = TimescaleClient(**config)
        try:
            _client.connect()
        except Exception as e:
            logger.warning(f"Could not connect to TimescaleDB: {e}")
            _client = None

    return _client


def export_to_timescaledb(
    labels: np.ndarray,
    tickers: list[str],
    corr_matrix: np.ndarray,
    n_clusters: int,
    n_noise: int = 0,
    silhouette_score: float | None = None,
    clustering_method: str | None = None,
    execution_time_seconds: float | None = None,
    correlation_threshold: float = 0.7,
    analysis_date: date | None = None,
) -> dict:
    """
    High-level function to export all results to TimescaleDB.

    Args:
        labels: Cluster labels
        tickers: List of ticker symbols
        corr_matrix: Correlation matrix
        n_clusters: Number of clusters found
        n_noise: Number of noise points
        silhouette_score: Clustering quality score
        clustering_method: Method used for clustering
        execution_time_seconds: Pipeline execution time
        correlation_threshold: Min correlation for pair export
        analysis_date: Analysis date (defaults to today)

    Returns:
        Dictionary with export statistics
    """
    if not is_db_export_enabled():
        return {
            "success": False,
            "message": "DB export disabled",
            "clusters_exported": 0,
            "correlations_exported": 0,
        }

    client = get_db_client()
    if client is None:
        return {
            "success": False,
            "message": "Could not connect to TimescaleDB",
            "clusters_exported": 0,
            "correlations_exported": 0,
        }

    if analysis_date is None:
        analysis_date = date.today()

    try:
        # Export clusters
        clusters_count = client.export_clusters(labels, tickers, analysis_date)

        # Export correlations
        corr_count = client.export_correlations(
            corr_matrix, tickers, correlation_threshold, analysis_date
        )

        # Export run metadata
        client.export_run_metadata(
            n_stocks_processed=len(tickers),
            n_clusters=n_clusters,
            n_noise=n_noise,
            silhouette_score=silhouette_score,
            clustering_method=clustering_method,
            execution_time_seconds=execution_time_seconds,
            analysis_date=analysis_date,
        )

        return {
            "success": True,
            "message": "Export complete",
            "clusters_exported": clusters_count,
            "correlations_exported": corr_count,
            "analysis_date": str(analysis_date),
        }

    except Exception as e:
        logger.error(f"TimescaleDB export failed: {e}")
        return {
            "success": False,
            "message": str(e),
            "clusters_exported": 0,
            "correlations_exported": 0,
        }


def get_db_stats() -> dict:
    """Get database connection status and stats."""
    config = get_db_config()

    if not is_db_export_enabled():
        return {
            "enabled": False,
            "connected": False,
            "host": config["host"],
            "port": config["port"],
        }

    client = get_db_client()

    return {
        "enabled": True,
        "connected": client is not None and client.is_connected,
        "host": config["host"],
        "port": config["port"],
        "database": config["database"],
    }
