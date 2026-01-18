"""Universe management - fetch NYSE/NASDAQ stock tickers."""

import pandas as pd
import yfinance as yf


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
    """Get a sample of tickers for testing."""
    samples = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC",
        "XOM", "CVX", "PFE", "ABBV", "KO", "PEP", "MRK", "TMO", "COST",
        "CSCO", "ABT", "ACN", "NKE", "MCD", "LLY", "DHR", "TXN", "NEE",
        "PM", "UPS", "ORCL", "RTX", "LOW", "IBM", "GS", "CAT", "AMGN",
        "DE", "SBUX", "AXP", "BLK", "MDLZ"
    ]
    return samples[:n]
