"""Universe management - fetch NYSE/NASDAQ stock tickers."""

import logging

import pandas as pd
import yfinance as yf

from .cache import (
    TTL_UNIVERSE,
    cache_key_for_universe,
    deserialize_json,
    get_cache,
    serialize_json,
)

logger = logging.getLogger(__name__)


def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia as a reliable baseline."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return sorted(set(tickers))


def get_nasdaq100_tickers() -> list[str]:
    """Fetch NASDAQ-100 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for table in tables:
        if "Ticker" in table.columns:
            return sorted(table["Ticker"].tolist())
        if "Symbol" in table.columns:
            return sorted(table["Symbol"].tolist())
    return []


def get_dow_tickers() -> list[str]:
    """Fetch Dow Jones Industrial Average tickers."""
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = pd.read_html(url)
    for table in tables:
        if "Symbol" in table.columns:
            return sorted(table["Symbol"].tolist())
    return []


def validate_tickers(tickers: list[str]) -> list[str]:
    """Validate tickers by checking if they have price data."""
    valid = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if info.get("regularMarketPrice") is not None:
                valid.append(ticker)
        except Exception:
            continue
    return valid


def get_full_universe(include_validation: bool = False) -> list[str]:
    """
    Get combined universe of major US stocks.

    Combines S&P 500, NASDAQ-100, and Dow Jones for comprehensive coverage.
    """
    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()
    dow = get_dow_tickers()

    combined = sorted(set(sp500 + nasdaq100 + dow))

    if include_validation:
        combined = validate_tickers(combined)

    return combined


def get_sample_tickers(n: int = 50) -> list[str]:
    """Get a sample of tickers for testing.

    If n > 50, fetches additional tickers from the full universe.
    """
    # Core sample tickers (well-known, liquid stocks)
    samples = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC",
        "XOM", "CVX", "PFE", "ABBV", "KO", "PEP", "MRK", "TMO", "COST",
        "CSCO", "ABT", "ACN", "NKE", "MCD", "LLY", "DHR", "TXN", "NEE",
        "PM", "UPS", "ORCL", "RTX", "LOW", "IBM", "GS", "CAT", "AMGN",
        "DE", "SBUX", "AXP", "BLK", "MDLZ"
    ]

    if n <= len(samples):
        return samples[:n]

    # Need more tickers - fetch from full universe
    import random
    try:
        full_universe = get_full_universe()
        # Remove duplicates with existing samples
        additional = [t for t in full_universe if t not in samples]
        random.shuffle(additional)
        # Add additional tickers to reach n
        samples.extend(additional[:n - len(samples)])
        return samples[:n]
    except Exception as e:
        print(f"  Warning: Could not fetch additional tickers: {e}")
        return samples


def get_full_universe_cached(
    source: str = "yfinance",
    include_validation: bool = False,
    filters: dict | None = None,
) -> list[str]:
    """
    Get stock universe with Redis caching.

    Args:
        source: Data source ('yfinance' or 'fmp')
        include_validation: Whether to validate tickers
        filters: Optional filters for cache key generation

    Returns:
        List of ticker symbols
    """
    cache = get_cache()
    cache_key = cache_key_for_universe(source, filters)

    # Try cache first
    if cache and cache.is_connected:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            logger.info("Cache hit for universe")
            try:
                return deserialize_json(cached_data)
            except Exception as e:
                logger.warning(f"Cache deserialize failed: {e}")

    # Cache miss - fetch from source
    logger.info(f"Cache miss - fetching universe from {source}")
    tickers = get_full_universe(include_validation=include_validation)

    # Store in cache
    if cache and cache.is_connected:
        try:
            cache.set(cache_key, serialize_json(tickers), TTL_UNIVERSE)
            logger.info(f"Cached {len(tickers)} tickers")
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

    return tickers
