"""Preprocessing - clean data and compute normalized returns."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def clean_price_data(
    price_df: pd.DataFrame,
    min_history_pct: float = 0.90,
) -> pd.DataFrame:
    """
    Clean price data by forward-filling and filtering sparse tickers.

    Args:
        price_df: DataFrame of prices (index=dates, columns=tickers)
        min_history_pct: Minimum required non-null percentage

    Returns:
        Cleaned DataFrame with sparse tickers removed
    """
    df = price_df.copy()
    df = df.ffill()

    total_days = len(df)
    required_days = int(total_days * min_history_pct)

    valid_tickers = []
    for col in df.columns:
        non_null = df[col].notna().sum()
        if non_null >= required_days:
            valid_tickers.append(col)

    return df[valid_tickers]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price series.

    Formula: r_t = ln(P_t) - ln(P_{t-1})
    """
    log_prices = np.log(prices)
    returns = log_prices.diff().iloc[1:]
    return returns


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple percentage returns."""
    return prices.pct_change().iloc[1:]


def zscore_normalize(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalize each column.

    Result: each column has mean ≈ 0, std ≈ 1
    """
    return (returns - returns.mean()) / returns.std()


def remove_market_factor(
    returns: pd.DataFrame,
    n_components: int = 1,
) -> pd.DataFrame:
    """
    Remove market factor using PCA.

    Removes the first n principal components (typically market beta).
    """
    returns_clean = returns.dropna(axis=1, how="any")
    returns_matrix = returns_clean.values

    pca = PCA(n_components=n_components)
    market_components = pca.fit_transform(returns_matrix)
    reconstructed = pca.inverse_transform(market_components)

    residuals = returns_matrix - reconstructed
    return pd.DataFrame(
        residuals,
        index=returns_clean.index,
        columns=returns_clean.columns,
    )


def preprocess_pipeline(
    prices: pd.DataFrame,
    min_history_pct: float = 0.90,
    remove_market: bool = False,
) -> pd.DataFrame:
    """
    Run full preprocessing pipeline.

    Steps:
        1. Clean and filter data
        2. Compute log returns
        3. Z-score normalize
        4. Optionally remove market factor
    """
    cleaned = clean_price_data(prices, min_history_pct)
    returns = compute_log_returns(cleaned)
    returns = returns.dropna(axis=1, how="any")
    normalized = zscore_normalize(returns)

    if remove_market:
        normalized = remove_market_factor(normalized)

    return normalized
