"""FMP (Financial Modeling Prep) API client for stock data."""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://financialmodelingprep.com/api/v3"


def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file, or None for default

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override API key from environment if not set
    if config.get("api", {}).get("key") is None:
        config.setdefault("api", {})["key"] = os.environ.get("FMP_API_KEY")

    return config


def get_api_key(config: dict | None = None) -> str:
    """Get API key from config or environment."""
    if config and config.get("api", {}).get("key"):
        return config["api"]["key"]

    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise ValueError(
            "FMP API key not found. Set FMP_API_KEY environment variable or "
            "specify in config file. Get a free key at: "
            "https://financialmodelingprep.com/developer"
        )
    return api_key


class FMPClient:
    """Client for Financial Modeling Prep API."""

    def __init__(
        self,
        api_key: str | None = None,
        config: dict | None = None,
        config_path: str | Path | None = None,
    ):
        """
        Initialize FMP client.

        Args:
            api_key: FMP API key (overrides config/env)
            config: Configuration dictionary
            config_path: Path to config file
        """
        if config is None and config_path:
            config = load_config(config_path)
        elif config is None:
            config = {}

        self.config = config
        self.api_key = api_key or get_api_key(config)
        self.base_url = config.get("api", {}).get("base_url", DEFAULT_BASE_URL)
        self.request_delay = config.get("api", {}).get("request_delay", 0.25)
        self.price_batch_size = config.get("api", {}).get("price_batch_size", 5)

        self._last_request_time = 0

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict | None = None) -> dict | list:
        """Make API request with rate limiting."""
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params["apikey"] = self.api_key

        logger.debug(f"FMP request: {endpoint}")

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Check for API error responses
        if isinstance(data, dict) and "Error Message" in data:
            raise ValueError(f"FMP API error: {data['Error Message']}")

        return data

    def get_stock_screener(
        self,
        market_cap_min: int | None = None,
        market_cap_max: int | None = None,
        volume_min: int | None = None,
        volume_max: int | None = None,
        exchanges: list[str] | None = None,
        sectors: list[str] | None = None,
        industries: list[str] | None = None,
        country: str | None = None,
        is_actively_trading: bool = True,
        limit: int = 10000,
    ) -> list[dict]:
        """
        Fetch stocks matching filters using stock screener.

        Returns:
            List of stock dictionaries with symbol, name, marketCap, etc.
        """
        params = {"limit": limit}

        if market_cap_min:
            params["marketCapMoreThan"] = market_cap_min
        if market_cap_max:
            params["marketCapLowerThan"] = market_cap_max
        if volume_min:
            params["volumeMoreThan"] = volume_min
        if volume_max:
            params["volumeLowerThan"] = volume_max
        if is_actively_trading:
            params["isActivelyTrading"] = "true"
        if country:
            params["country"] = country

        all_stocks = []

        # Fetch for each exchange
        target_exchanges = exchanges or ["NYSE", "NASDAQ"]

        for exchange in target_exchanges:
            params["exchange"] = exchange
            logger.info(f"Fetching stocks from {exchange}...")

            stocks = self._request("stock-screener", params)

            if isinstance(stocks, list):
                # Filter by sector if specified
                if sectors:
                    stocks = [s for s in stocks if s.get("sector") in sectors]

                # Filter by industry if specified
                if industries:
                    stocks = [s for s in stocks if s.get("industry") in industries]

                all_stocks.extend(stocks)
                logger.info(f"  Found {len(stocks)} stocks on {exchange}")

        # Remove duplicates by symbol
        seen = set()
        unique_stocks = []
        for stock in all_stocks:
            symbol = stock.get("symbol")
            if symbol and symbol not in seen:
                seen.add(symbol)
                unique_stocks.append(stock)

        logger.info(f"Total unique stocks: {len(unique_stocks)}")
        return unique_stocks

    def get_universe_from_config(self) -> list[dict]:
        """Fetch universe using filters from config."""
        filters = self.config.get("filters", {})

        market_cap = filters.get("market_cap", {})
        volume = filters.get("volume", {})

        return self.get_stock_screener(
            market_cap_min=market_cap.get("min"),
            market_cap_max=market_cap.get("max"),
            volume_min=volume.get("min"),
            volume_max=volume.get("max"),
            exchanges=filters.get("exchanges"),
            sectors=filters.get("sectors"),
            industries=filters.get("industries"),
            country=filters.get("country"),
            is_actively_trading=filters.get("is_actively_trading", True),
        )

    def get_historical_prices(
        self,
        symbol: str,
        days: int = 180,
    ) -> pd.DataFrame:
        """
        Fetch historical prices for a single symbol.

        Returns:
            DataFrame with date index and OHLCV columns
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
        }

        data = self._request(f"historical-price-full/{symbol}", params)

        if not data or "historical" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["historical"])
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        return df

    def get_batch_historical_prices(
        self,
        symbols: list[str],
        days: int = 180,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Fetch historical prices for multiple symbols.

        Args:
            symbols: List of ticker symbols
            days: Number of days of history
            progress_callback: Optional callback(current, total, symbol)

        Returns:
            DataFrame with date index and symbol columns (adjusted close)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        all_prices = {}
        total = len(symbols)
        failed = []

        for i, symbol in enumerate(symbols):
            try:
                if progress_callback:
                    progress_callback(i + 1, total, symbol)

                df = self.get_historical_prices(symbol, days)

                if not df.empty and "adjClose" in df.columns:
                    all_prices[symbol] = df["adjClose"]
                elif not df.empty and "close" in df.columns:
                    all_prices[symbol] = df["close"]
                else:
                    failed.append(symbol)
                    logger.warning(f"No price data for {symbol}")

            except Exception as e:
                failed.append(symbol)
                logger.warning(f"Failed to fetch {symbol}: {e}")

        if failed:
            logger.warning(f"Failed to fetch {len(failed)} symbols: {failed[:10]}...")

        if not all_prices:
            return pd.DataFrame()

        # Combine into single DataFrame
        prices_df = pd.DataFrame(all_prices)
        prices_df = prices_df.sort_index()

        return prices_df


def get_fmp_universe(
    config_path: str | Path | None = None,
    config: dict | None = None,
) -> list[dict]:
    """
    Fetch complete stock universe from FMP using config filters.

    Args:
        config_path: Path to YAML config file
        config: Configuration dictionary (overrides config_path)

    Returns:
        List of stock dictionaries
    """
    if config is None:
        config = load_config(config_path)

    client = FMPClient(config=config)
    return client.get_universe_from_config()


def get_fmp_universe_tickers(
    config_path: str | Path | None = None,
    config: dict | None = None,
) -> list[str]:
    """Get just the ticker symbols from FMP universe."""
    stocks = get_fmp_universe(config_path, config)
    return sorted([s["symbol"] for s in stocks if s.get("symbol")])


def fetch_fmp_prices(
    tickers: list[str],
    days: int = 180,
    api_key: str | None = None,
    config: dict | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetch historical prices for tickers from FMP.

    Args:
        tickers: List of ticker symbols
        days: Number of days of history
        api_key: FMP API key
        config: Configuration dictionary
        progress_callback: Optional callback(current, total, symbol)

    Returns:
        DataFrame with date index and ticker columns
    """
    client = FMPClient(api_key=api_key, config=config)
    return client.get_batch_historical_prices(
        tickers, days=days, progress_callback=progress_callback
    )
