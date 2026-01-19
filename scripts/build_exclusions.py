#!/usr/bin/env python3
"""
Build and update the stock exclusion cache from FMP API.

This script fetches ETF and index lists from Financial Modeling Prep API
and combines them with manual exclusions to create a complete exclusion list.

Usage:
    python scripts/build_exclusions.py [--force] [--analyze]

Options:
    --force     Force refresh even if cache is not expired
    --analyze   Analyze current correlations to find potential duplicates
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
EXCLUSIONS_CONFIG = CONFIG_DIR / "exclusions.yaml"
DEFAULT_CACHE_FILE = CONFIG_DIR / "exclusion_cache.json"

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def load_config() -> dict:
    """Load exclusions configuration from YAML."""
    if not EXCLUSIONS_CONFIG.exists():
        logger.warning(f"Config file not found: {EXCLUSIONS_CONFIG}")
        return {}

    with open(EXCLUSIONS_CONFIG) as f:
        return yaml.safe_load(f) or {}


def get_api_key() -> str | None:
    """Get FMP API key from environment."""
    return os.environ.get("FMP_API_KEY")


def fetch_etf_list(api_key: str) -> list[str]:
    """Fetch list of all ETF symbols from FMP."""
    logger.info("Fetching ETF list from FMP...")
    url = f"{FMP_BASE_URL}/etf/list"
    params = {"apikey": api_key}

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            # Filter to US exchanges only (NYSE, NASDAQ, AMEX)
            us_etfs = []
            for etf in data:
                symbol = etf.get("symbol", "")
                exchange = etf.get("exchange", "")
                # Include if it looks like a US symbol (no dots, reasonable length)
                if symbol and "." not in symbol and len(symbol) <= 5:
                    us_etfs.append(symbol)
                elif exchange in ("NYSE", "NASDAQ", "AMEX", "NYSE ARCA"):
                    us_etfs.append(symbol)

            logger.info(f"Found {len(us_etfs)} US ETFs (from {len(data)} total)")
            return us_etfs
        else:
            logger.error(f"Unexpected ETF list response: {data}")
            return []

    except Exception as e:
        logger.error(f"Failed to fetch ETF list: {e}")
        return []


def fetch_index_list(api_key: str) -> list[str]:
    """Fetch list of tradable index symbols from FMP."""
    logger.info("Fetching index list from FMP...")
    url = f"{FMP_BASE_URL}/symbol/available-indexes"
    params = {"apikey": api_key}

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            symbols = [item.get("symbol") for item in data if item.get("symbol")]
            logger.info(f"Found {len(symbols)} indexes")
            return symbols
        else:
            logger.error(f"Unexpected index list response: {data}")
            return []

    except Exception as e:
        logger.error(f"Failed to fetch index list: {e}")
        return []


def get_share_class_exclusions(config: dict) -> tuple[set[str], dict[str, list[str]]]:
    """
    Get share class exclusions based on config settings.

    Returns:
        Tuple of (symbols_to_exclude, share_class_groups)
    """
    share_classes = config.get("share_classes", [])
    handling = config.get("settings", {}).get("share_class_handling", "keep_most_liquid")

    groups = {}
    to_exclude = set()

    for group in share_classes:
        if not isinstance(group, list) or len(group) < 2:
            continue

        primary = group[0]
        groups[primary] = group

        if handling == "exclude_all":
            # Exclude all share classes
            to_exclude.update(group)
        elif handling == "keep_first":
            # Keep first, exclude rest
            to_exclude.update(group[1:])
        else:
            # keep_most_liquid - we'll determine at runtime based on volume
            # For now, exclude all but first as default
            to_exclude.update(group[1:])

    return to_exclude, groups


def compile_patterns(config: dict) -> list[re.Pattern]:
    """Compile regex patterns from config."""
    patterns = []

    for pattern_list in ["patterns", "etf_patterns"]:
        for pattern in config.get(pattern_list, []):
            try:
                patterns.append(re.compile(pattern))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

    return patterns


def build_exclusion_cache(
    config: dict,
    api_key: str | None = None,
    force: bool = False
) -> dict:
    """
    Build the complete exclusion cache.

    Args:
        config: Exclusions configuration
        api_key: FMP API key (optional, uses env if not provided)
        force: Force refresh even if cache exists

    Returns:
        Exclusion cache dictionary
    """
    auto_fetch = config.get("auto_fetch", {})
    cache_file = PROJECT_ROOT / auto_fetch.get("cache_file", "config/exclusion_cache.json")
    cache_expiry = auto_fetch.get("cache_expiry_hours", 168)

    # Check existing cache
    if cache_file.exists() and not force:
        try:
            with open(cache_file) as f:
                cache = json.load(f)

            # Check expiry
            created = datetime.fromisoformat(cache.get("created_at", "2000-01-01"))
            if datetime.now() - created < timedelta(hours=cache_expiry):
                logger.info(f"Using cached exclusions (created {created})")
                return cache
            else:
                logger.info("Cache expired, refreshing...")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    # Get API key
    if not api_key:
        api_key = get_api_key()

    # Build fresh exclusion list
    etf_symbols = set()
    index_symbols = set()

    if api_key and auto_fetch.get("enabled", True):
        if auto_fetch.get("fetch_etfs", True):
            etf_symbols = set(fetch_etf_list(api_key))

        if auto_fetch.get("fetch_indexes", True):
            index_symbols = set(fetch_index_list(api_key))
    else:
        logger.warning("No API key or auto-fetch disabled, using patterns only")

    # Get manual exclusions
    manual_excludes = set(config.get("manual_excludes", []))

    # Get share class exclusions
    share_class_excludes, share_class_groups = get_share_class_exclusions(config)

    # Compile patterns
    patterns = compile_patterns(config)
    pattern_strings = [p.pattern for p in patterns]

    # Build cache
    cache = {
        "created_at": datetime.now().isoformat(),
        "etf_count": len(etf_symbols),
        "index_count": len(index_symbols),
        "manual_count": len(manual_excludes),
        "share_class_count": len(share_class_excludes),
        "pattern_count": len(patterns),
        "etfs": sorted(etf_symbols),
        "indexes": sorted(index_symbols),
        "manual": sorted(manual_excludes),
        "share_class_excludes": sorted(share_class_excludes),
        "share_class_groups": share_class_groups,
        "patterns": pattern_strings,
    }

    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    logger.info(f"Saved exclusion cache to {cache_file}")
    logger.info(f"  ETFs: {len(etf_symbols)}")
    logger.info(f"  Indexes: {len(index_symbols)}")
    logger.info(f"  Manual: {len(manual_excludes)}")
    logger.info(f"  Share classes: {len(share_class_excludes)}")
    logger.info(f"  Patterns: {len(patterns)}")

    return cache


def analyze_correlations(threshold: float = 0.98):
    """
    Analyze current correlation data to find potential share class duplicates.

    This queries the database for highly correlated pairs that might be
    share classes or related instruments.
    """
    logger.info(f"Analyzing correlations >= {threshold}...")

    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.environ.get("TIMESCALE_HOST", "192.168.68.88"),
            port=int(os.environ.get("TIMESCALE_PORT", 5432)),
            database=os.environ.get("TIMESCALE_DB", "timescaledb"),
            user=os.environ.get("TIMESCALE_USER", "postgres"),
            password=os.environ.get("TIMESCALE_PASSWORD", "password"),
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ticker_a, ticker_b, correlation, analysis_date
            FROM pair_correlations
            WHERE correlation >= %s
            ORDER BY correlation DESC
            LIMIT 100
        """, (threshold,))

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            logger.info("No pairs found above threshold")
            return

        logger.info(f"\nPotential duplicates (correlation >= {threshold}):")
        logger.info("-" * 60)

        for ticker_a, ticker_b, corr, date in rows:
            logger.info(f"  {corr:.4f}  {ticker_a:8} - {ticker_b:8}  ({date})")

        logger.info("-" * 60)
        logger.info(f"Total: {len(rows)} pairs")
        logger.info("\nReview these pairs and add share classes to config/exclusions.yaml")

    except ImportError:
        logger.error("psycopg2 not installed, cannot analyze database")
    except Exception as e:
        logger.error(f"Database analysis failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Build stock exclusion cache from FMP API"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force refresh even if cache is not expired"
    )
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Analyze correlations to find potential duplicates"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.98,
        help="Correlation threshold for duplicate analysis (default: 0.98)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    if args.analyze:
        threshold = config.get("settings", {}).get(
            "duplicate_correlation_threshold",
            args.threshold
        )
        analyze_correlations(threshold)
        return

    # Build cache
    cache = build_exclusion_cache(config, force=args.force)

    # Summary
    total = (
        cache["etf_count"] +
        cache["index_count"] +
        cache["manual_count"] +
        cache["share_class_count"]
    )
    logger.info(f"\nTotal unique exclusions: ~{total} symbols")
    logger.info("Run with --analyze to find potential duplicates in correlation data")


if __name__ == "__main__":
    main()
