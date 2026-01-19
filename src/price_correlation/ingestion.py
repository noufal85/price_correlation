"""Data ingestion - fetch historical price data."""

import logging
from datetime import datetime, timedelta
from typing import Callable

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

# Default batch size for fetching prices
DEFAULT_BATCH_SIZE = 50


def fetch_price_history(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    period_months: int = 18,
    progress_callback: Callable[[int, int, str], None] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for multiple tickers.

    Args:
        tickers: List of stock symbols
        start_date: Start date as "YYYY-MM-DD" (optional)
        end_date: End date as "YYYY-MM-DD" (optional)
        period_months: Lookback period if dates not specified
        progress_callback: Optional callback(current, total, message) for progress updates
        batch_size: Number of tickers to fetch per batch

    Returns:
        DataFrame with dates as index and tickers as columns
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=period_months * 30)
        start_date = start_dt.strftime("%Y-%m-%d")

    total_tickers = len(tickers)

    # For small lists, fetch all at once
    if total_tickers <= batch_size:
        if progress_callback:
            progress_callback(0, total_tickers, f"Fetching {total_tickers} tickers...")

        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if progress_callback:
            progress_callback(total_tickers, total_tickers, f"Fetched {total_tickers} tickers")

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers

        return prices

    # For larger lists, fetch in batches with progress reporting
    all_prices = []
    fetched = 0
    failed_tickers = []

    for i in range(0, total_tickers, batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_tickers + batch_size - 1) // batch_size

        if progress_callback:
            pct = int((fetched / total_tickers) * 100)
            progress_callback(
                fetched,
                total_tickers,
                f"Batch {batch_num}/{total_batches}: Fetching {len(batch)} tickers ({pct}%)"
            )

        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    batch_prices = data["Close"]
                else:
                    batch_prices = data[["Close"]]
                    batch_prices.columns = batch
                all_prices.append(batch_prices)

            fetched += len(batch)

        except Exception as e:
            logger.warning(f"Batch {batch_num} failed: {e}")
            failed_tickers.extend(batch)
            fetched += len(batch)

    if progress_callback:
        msg = f"Downloaded {fetched} tickers"
        if failed_tickers:
            msg += f" ({len(failed_tickers)} failed)"
        progress_callback(total_tickers, total_tickers, msg)

    if not all_prices:
        return pd.DataFrame()

    # Combine all batches
    prices = pd.concat(all_prices, axis=1)

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
    progress_callback: Callable[[int, int, str], None] | None = None,
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
    prices = fetch_price_history(
        tickers, start_date, end_date, period_months,
        progress_callback=progress_callback
    )

    # Store in cache
    if cache and cache.is_connected:
        try:
            cache.set(cache_key, serialize_dataframe(prices), TTL_PRICES)
            logger.info(f"Cached {len(tickers)} tickers price data")
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

    return prices
