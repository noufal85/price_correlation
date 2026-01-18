"""Export module - save results to JSON and Parquet."""

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .db import export_to_timescaledb, is_db_export_enabled

logger = logging.getLogger(__name__)


def export_clusters_json(
    labels: np.ndarray,
    tickers: list[str],
    output_path: str | Path,
) -> None:
    """
    Export cluster assignments to JSON.

    Format:
        [{"cluster": 0, "members": [...], "size": N}, ...]
    """
    clusters: dict[int, list[str]] = {}

    for ticker, label in zip(tickers, labels):
        label_int = int(label)
        if label_int not in clusters:
            clusters[label_int] = []
        clusters[label_int].append(ticker)

    output = []
    for cluster_id in sorted(clusters.keys()):
        members = sorted(clusters[cluster_id])
        output.append({
            "cluster": cluster_id,
            "members": members,
            "size": len(members),
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def export_clusters_parquet(
    labels: np.ndarray,
    tickers: list[str],
    output_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """
    Export cluster assignments to Parquet.

    Schema:
        analysis_date, ticker, cluster_id, [optional metadata columns]
    """
    records = []
    analysis_date = date.today().isoformat()

    for ticker, label in zip(tickers, labels):
        record = {
            "analysis_date": analysis_date,
            "ticker": ticker,
            "cluster_id": int(label),
        }

        if metadata and ticker in metadata:
            record.update(metadata[ticker])

        records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_path, compression="snappy", index=False)


def export_correlations_parquet(
    corr_matrix: np.ndarray,
    tickers: list[str],
    output_path: str | Path,
    threshold: float = 0.7,
) -> None:
    """
    Export significant correlations to Parquet (sparse format).

    Only stores pairs with |correlation| >= threshold.
    """
    n = len(tickers)
    records = []

    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                records.append({
                    "ticker_a": tickers[i],
                    "ticker_b": tickers[j],
                    "correlation": float(corr),
                })

    df = pd.DataFrame(records)
    df = df.sort_values("correlation", ascending=False)
    df.to_parquet(output_path, compression="snappy", index=False)


def export_correlations_json(
    corr_matrix: np.ndarray,
    tickers: list[str],
    output_path: str | Path,
    threshold: float = 0.7,
) -> None:
    """
    Export significant correlations to JSON.

    Only stores pairs with |correlation| >= threshold.
    """
    n = len(tickers)
    records = []

    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                records.append({
                    "ticker_a": tickers[i],
                    "ticker_b": tickers[j],
                    "correlation": round(float(corr), 4),
                })

    # Sort by correlation descending
    records.sort(key=lambda x: x["correlation"], reverse=True)

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)


def export_equity_clusters_json(
    labels: np.ndarray,
    tickers: list[str],
    output_path: str | Path,
) -> None:
    """
    Export detailed equity cluster assignments to JSON.

    Format:
        [{"ticker": "AAPL", "cluster_id": 0, "analysis_date": "2024-01-01"}, ...]
    """
    analysis_date = date.today().isoformat()
    records = []

    for ticker, label in zip(tickers, labels):
        records.append({
            "ticker": ticker,
            "cluster_id": int(label),
            "analysis_date": analysis_date,
        })

    # Sort by cluster_id, then ticker
    records.sort(key=lambda x: (x["cluster_id"], x["ticker"]))

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)


def export_all(
    labels: np.ndarray,
    tickers: list[str],
    corr_matrix: np.ndarray,
    output_dir: str | Path,
    correlation_threshold: float = 0.7,
    export_json: bool = True,
    export_parquet: bool = True,
    export_db: bool | None = None,
    n_clusters: int | None = None,
    n_noise: int = 0,
    silhouette_score: float | None = None,
    clustering_method: str | None = None,
    execution_time_seconds: float | None = None,
) -> dict[str, Path | dict]:
    """
    Export all results to output directory and optionally to TimescaleDB.

    Args:
        labels: Cluster labels for each ticker
        tickers: List of ticker symbols
        corr_matrix: Correlation matrix
        output_dir: Output directory path
        correlation_threshold: Min correlation for pair export
        export_json: Export JSON format files
        export_parquet: Export Parquet format files
        export_db: Export to TimescaleDB (None=auto based on env)
        n_clusters: Number of clusters (for DB metadata)
        n_noise: Number of noise points (for DB metadata)
        silhouette_score: Clustering quality score (for DB metadata)
        clustering_method: Method used (for DB metadata)
        execution_time_seconds: Pipeline time (for DB metadata)

    Returns:
        Dictionary of output file paths and DB export result
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # JSON exports
    if export_json:
        # Cluster summary (grouped by cluster)
        json_path = output_dir / "stock_clusters.json"
        export_clusters_json(labels, tickers, json_path)
        output_files["clusters_json"] = json_path

        # Equity clusters (per-ticker format)
        equity_json = output_dir / "equity_clusters.json"
        export_equity_clusters_json(labels, tickers, equity_json)
        output_files["equity_json"] = equity_json

        # Pair correlations
        pairs_json = output_dir / "pair_correlations.json"
        export_correlations_json(
            corr_matrix, tickers, pairs_json, threshold=correlation_threshold
        )
        output_files["pairs_json"] = pairs_json

    # Parquet exports
    if export_parquet:
        clusters_parquet = output_dir / "equity_clusters.parquet"
        export_clusters_parquet(labels, tickers, clusters_parquet)
        output_files["clusters_parquet"] = clusters_parquet

        corr_parquet = output_dir / "pair_correlations.parquet"
        export_correlations_parquet(
            corr_matrix, tickers, corr_parquet, threshold=correlation_threshold
        )
        output_files["correlations_parquet"] = corr_parquet

    # TimescaleDB export
    should_export_db = export_db if export_db is not None else is_db_export_enabled()
    if should_export_db:
        # Calculate n_clusters from labels if not provided
        if n_clusters is None:
            from collections import Counter

            label_counts = Counter(labels)
            n_clusters = len([k for k in label_counts if k != -1])
            n_noise = label_counts.get(-1, 0)

        logger.info("Exporting to TimescaleDB...")
        db_result = export_to_timescaledb(
            labels=labels,
            tickers=tickers,
            corr_matrix=corr_matrix,
            n_clusters=n_clusters,
            n_noise=n_noise,
            silhouette_score=silhouette_score,
            clustering_method=clustering_method,
            execution_time_seconds=execution_time_seconds,
            correlation_threshold=correlation_threshold,
        )
        output_files["db_export"] = db_result

        if db_result["success"]:
            logger.info(
                f"DB export complete: {db_result['clusters_exported']} clusters, "
                f"{db_result['correlations_exported']} correlations"
            )
        else:
            logger.warning(f"DB export failed: {db_result['message']}")

    return output_files
