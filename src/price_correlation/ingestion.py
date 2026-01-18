"""Data ingestion - fetch historical price data."""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from .cache import (
    TTL_PRICES,
    cache_key_for_prices,
    deserialize_dataframe,
    get_cache,
    serialize_dataframe,
)

logger = logging.getLogger(__name__)


def fetch_price_history(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    period_months: int = 18,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for multiple tickers.

    Args:
        tickers: List of stock symbols
        start_date: Start date as "YYYY-MM-DD" (optional)
        end_date: End date as "YYYY-MM-DD" (optional)
        period_months: Lookback period if dates not specified

    Returns:
        DataFrame with dates as index and tickers as columns
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=period_months * 30)
        start_date = start_dt.strftime("%Y-%m-%d")

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    return prices


def fetch_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Fetch price history for a single ticker."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.Series(dtype=float, name=ticker)

    return data["Close"].rename(ticker)


def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Get US trading days between two dates."""
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    return spy.index


def fetch_price_history_cached(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    period_months: int = 18,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices with Redis caching.

    Uses cache if available, falls back to API on cache miss.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=period_months * 30)
        start_date = start_dt.strftime("%Y-%m-%d")

    cache = get_cache()
    cache_key = cache_key_for_prices(tickers, start_date, end_date)

    # Try cache first
    if cache and cache.is_connected:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Cache hit for {len(tickers)} tickers")
            try:
                return deserialize_dataframe(cached_data)
            except Exception as e:
                logger.warning(f"Cache deserialize failed: {e}")

    # Cache miss - fetch from API
    logger.info(f"Cache miss - fetching {len(tickers)} tickers from API")
    prices = fetch_price_history(tickers, start_date, end_date, period_months)

    # Store in cache
    if cache and cache.is_connected:
        try:
            cache.set(cache_key, serialize_dataframe(prices), TTL_PRICES)
            logger.info(f"Cached {len(tickers)} tickers price data")
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

    return prices
