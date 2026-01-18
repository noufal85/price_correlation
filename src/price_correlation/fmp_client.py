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
        progress_callback=None,
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
            if progress_callback:
                progress_callback(f"Fetching {exchange}...")
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
                if progress_callback:
                    progress_callback(f"Found {len(stocks)} on {exchange}")
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

    def get_full_universe_iterative(
        self,
        exchanges: list[str] | None = None,
        sectors: list[str] | None = None,
        is_actively_trading: bool = True,
        progress_callback=None,
        split_threshold: int = 500,
    ) -> list[dict]:
        """
        Fetch COMPLETE stock universe using iterative market cap ranges.

        This method fetches all stocks by iterating through market cap ranges.
        If any range returns >= split_threshold stocks, it automatically splits
        the range in half and refetches to ensure no data is missed.

        Args:
            exchanges: Target exchanges (default: NYSE, NASDAQ)
            sectors: Optional sector filter
            is_actively_trading: Only include actively trading stocks
            progress_callback: Optional callback for progress updates
            split_threshold: Max stocks per range before splitting (default: 500)

        Returns:
            Complete list of all stocks matching criteria
        """
        all_stocks = []
        seen_symbols = set()
        target_exchanges = exchanges or ["NYSE", "NASDAQ"]

        # Initial market cap ranges (will be split if needed)
        ranges_to_process = [
            (1_000_000_000_000, None),             # $1T+
            (100_000_000_000, 1_000_000_000_000),  # $100B - $1T
            (50_000_000_000, 100_000_000_000),     # $50B - $100B
            (10_000_000_000, 50_000_000_000),      # $10B - $50B
            (5_000_000_000, 10_000_000_000),       # $5B - $10B
            (2_000_000_000, 5_000_000_000),        # $2B - $5B
            (1_000_000_000, 2_000_000_000),        # $1B - $2B
            (500_000_000, 1_000_000_000),          # $500M - $1B
            (300_000_000, 500_000_000),            # $300M - $500M
            (100_000_000, 300_000_000),            # $100M - $300M
            (50_000_000, 100_000_000),             # $50M - $100M
            (10_000_000, 50_000_000),              # $10M - $50M
            (None, 10_000_000),                    # < $10M
        ]

        processed_count = 0
        total_initial = len(ranges_to_process)

        while ranges_to_process:
            mcap_min, mcap_max = ranges_to_process.pop(0)
            range_label = self._format_mcap_range(mcap_min, mcap_max)
            processed_count += 1

            if progress_callback:
                progress_callback(f"[{processed_count}] Fetching {range_label}...")

            logger.info(f"Fetching market cap range: {range_label}")

            range_needs_split = False

            for exchange in target_exchanges:
                params = {
                    "limit": 10000,
                    "exchange": exchange,
                    "isActivelyTrading": "true" if is_actively_trading else "false",
                }

                if mcap_min:
                    params["marketCapMoreThan"] = mcap_min
                if mcap_max:
                    params["marketCapLowerThan"] = mcap_max

                try:
                    stocks = self._request("stock-screener", params)

                    if isinstance(stocks, list):
                        # Filter by sector if specified
                        if sectors:
                            stocks = [s for s in stocks if s.get("sector") in sectors]

                        # Check if we hit the threshold - need to split
                        if len(stocks) >= split_threshold:
                            range_needs_split = True
                            logger.warning(
                                f"  {exchange} {range_label}: {len(stocks)} stocks "
                                f"(>= {split_threshold}), will split range"
                            )
                            if progress_callback:
                                progress_callback(
                                    f"  {exchange}: {len(stocks)} stocks - splitting range..."
                                )
                        else:
                            # Add only unseen stocks
                            new_count = 0
                            for stock in stocks:
                                symbol = stock.get("symbol")
                                if symbol and symbol not in seen_symbols:
                                    seen_symbols.add(symbol)
                                    all_stocks.append(stock)
                                    new_count += 1

                            if new_count > 0:
                                logger.info(f"  {exchange} {range_label}: +{new_count} new stocks")
                            if progress_callback:
                                progress_callback(f"  {exchange}: +{new_count} stocks")

                except Exception as e:
                    logger.warning(f"Error fetching {exchange} {range_label}: {e}")

            # If any exchange hit threshold, split the range and add to queue
            if range_needs_split:
                split_ranges = self._split_range(mcap_min, mcap_max)
                if split_ranges:
                    logger.info(f"  Splitting {range_label} into {len(split_ranges)} sub-ranges")
                    # Add split ranges to the front of the queue
                    ranges_to_process = split_ranges + ranges_to_process
                else:
                    # Can't split further (range too small), just fetch what we can
                    logger.warning(f"  Cannot split {range_label} further, fetching as-is")
                    for exchange in target_exchanges:
                        params = {
                            "limit": 10000,
                            "exchange": exchange,
                            "isActivelyTrading": "true" if is_actively_trading else "false",
                        }
                        if mcap_min:
                            params["marketCapMoreThan"] = mcap_min
                        if mcap_max:
                            params["marketCapLowerThan"] = mcap_max

                        try:
                            stocks = self._request("stock-screener", params)
                            if isinstance(stocks, list):
                                if sectors:
                                    stocks = [s for s in stocks if s.get("sector") in sectors]
                                new_count = 0
                                for stock in stocks:
                                    symbol = stock.get("symbol")
                                    if symbol and symbol not in seen_symbols:
                                        seen_symbols.add(symbol)
                                        all_stocks.append(stock)
                                        new_count += 1
                                logger.info(f"  {exchange} {range_label}: +{new_count} new stocks (forced)")
                        except Exception as e:
                            logger.warning(f"Error fetching {exchange} {range_label}: {e}")

        if progress_callback:
            progress_callback(f"Complete: {len(all_stocks)} total stocks")

        logger.info(f"Total unique stocks fetched: {len(all_stocks)}")
        return all_stocks

    def _split_range(
        self,
        mcap_min: int | None,
        mcap_max: int | None,
    ) -> list[tuple[int | None, int | None]]:
        """
        Split a market cap range in half.

        Returns list of 2 sub-ranges, or empty list if can't split.
        """
        # Handle edge cases
        if mcap_min is None and mcap_max is None:
            return []

        if mcap_min is None:
            # Range is "< X", split into "< X/2" and "X/2 to X"
            mid = mcap_max // 2
            if mid < 1_000_000:  # Don't split below $1M
                return []
            return [(None, mid), (mid, mcap_max)]

        if mcap_max is None:
            # Range is "> X", can't easily split (would need to know max)
            # Split into "X to X*10" and "> X*10"
            upper = mcap_min * 10
            return [(mcap_min, upper), (upper, None)]

        # Normal range with both bounds
        if mcap_max - mcap_min < 10_000_000:  # Don't split ranges smaller than $10M
            return []

        mid = (mcap_min + mcap_max) // 2
        return [(mcap_min, mid), (mid, mcap_max)]

    def _format_mcap_range(self, mcap_min: int | None, mcap_max: int | None) -> str:
        """Format market cap range for display."""
        def fmt(val):
            if val is None:
                return "∞"
            if val >= 1_000_000_000_000:
                return f"${val / 1_000_000_000_000:.0f}T"
            if val >= 1_000_000_000:
                return f"${val / 1_000_000_000:.0f}B"
            if val >= 1_000_000:
                return f"${val / 1_000_000:.0f}M"
            return f"${val:,.0f}"

        if mcap_min is None:
            return f"< {fmt(mcap_max)}"
        if mcap_max is None:
            return f"> {fmt(mcap_min)}"
        return f"{fmt(mcap_min)} - {fmt(mcap_max)}"

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
