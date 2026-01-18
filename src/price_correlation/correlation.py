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
    corr = np.corrcoef(returns_matrix)

    # Replace NaN with 0 (uncorrelated) for stocks with insufficient data
    corr = np.nan_to_num(corr, nan=0.0)

    return corr


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
    max_pairs: int = 10000,
) -> list[dict]:
    """
    Extract pairs with correlation above threshold.

    Uses vectorized numpy operations for speed with large matrices.

    Args:
        corr_matrix: NxN correlation matrix
        tickers: List of ticker symbols
        threshold: Minimum absolute correlation
        max_pairs: Maximum pairs to return (for memory efficiency)

    Returns:
        List of {"ticker_a", "ticker_b", "correlation"} dicts
    """
    n = len(tickers)

    # Get upper triangle indices (excluding diagonal)
    i_upper, j_upper = np.triu_indices(n, k=1)

    # Get all upper triangle correlations
    upper_corrs = corr_matrix[i_upper, j_upper]

    # Find indices where correlation exceeds threshold
    mask = np.abs(upper_corrs) >= threshold
    filtered_i = i_upper[mask]
    filtered_j = j_upper[mask]
    filtered_corrs = upper_corrs[mask]

    # Sort by absolute correlation descending
    sort_idx = np.argsort(-np.abs(filtered_corrs))

    # Limit to max_pairs for memory efficiency
    sort_idx = sort_idx[:max_pairs]

    # Build result list
    pairs = []
    for idx in sort_idx:
        pairs.append({
            "ticker_a": tickers[filtered_i[idx]],
            "ticker_b": tickers[filtered_j[idx]],
            "correlation": float(filtered_corrs[idx]),
        })

    return pairs


def distance_matrix_from_condensed(
    condensed: np.ndarray,
) -> np.ndarray:
    """Convert condensed distance array to full square matrix."""
    return squareform(condensed)
