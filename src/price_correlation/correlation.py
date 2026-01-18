"""Correlation engine - compute correlation and distance matrices."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def compute_correlation_matrix(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise Pearson correlation matrix.

    Args:
        returns_df: DataFrame (rows=dates, columns=tickers)

    Returns:
        Symmetric NxN correlation matrix
    """
    returns_matrix = returns_df.values.T
    return np.corrcoef(returns_matrix)


def correlation_to_distance(
    corr_matrix: np.ndarray,
    method: str = "sqrt",
) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.

    Args:
        corr_matrix: NxN correlation matrix
        method: "sqrt" for d=sqrt(2*(1-rho)) or "simple" for d=1-rho

    Returns:
        NxN distance matrix with diagonal = 0
    """
    if method == "sqrt":
        dist = np.sqrt(2 * (1 - corr_matrix))
    else:
        dist = 1 - corr_matrix

    np.fill_diagonal(dist, 0)
    return dist


def get_condensed_distance(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Compute condensed distance array for scipy hierarchical clustering.

    Uses correlation distance directly via pdist.

    Returns:
        Condensed array of length N*(N-1)/2
    """
    returns_matrix = returns_df.values.T
    return pdist(returns_matrix, metric="correlation")


def get_highly_correlated_pairs(
    corr_matrix: np.ndarray,
    tickers: list[str],
    threshold: float = 0.7,
) -> list[dict]:
    """
    Extract pairs with correlation above threshold.

    Returns:
        List of {"ticker_a", "ticker_b", "correlation"} dicts
    """
    n = len(tickers)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                pairs.append({
                    "ticker_a": tickers[i],
                    "ticker_b": tickers[j],
                    "correlation": float(corr),
                })

    return sorted(pairs, key=lambda x: -abs(x["correlation"]))


def distance_matrix_from_condensed(
    condensed: np.ndarray,
) -> np.ndarray:
    """Convert condensed distance array to full square matrix."""
    return squareform(condensed)
