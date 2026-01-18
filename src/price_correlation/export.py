"""Export module - save results to JSON and Parquet."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


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


def export_all(
    labels: np.ndarray,
    tickers: list[str],
    corr_matrix: np.ndarray,
    output_dir: str | Path,
    correlation_threshold: float = 0.7,
) -> dict[str, Path]:
    """
    Export all results to output directory.

    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "stock_clusters.json"
    clusters_parquet = output_dir / "equity_clusters.parquet"
    corr_parquet = output_dir / "pair_correlations.parquet"

    export_clusters_json(labels, tickers, json_path)
    export_clusters_parquet(labels, tickers, clusters_parquet)
    export_correlations_parquet(
        corr_matrix, tickers, corr_parquet, threshold=correlation_threshold
    )

    return {
        "clusters_json": json_path,
        "clusters_parquet": clusters_parquet,
        "correlations_parquet": corr_parquet,
    }
