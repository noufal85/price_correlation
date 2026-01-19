"""
Stock exclusion management for correlation analysis.

Handles loading and applying exclusions to filter out ETFs, indexes,
and duplicate share classes that would create artificial correlations.
"""

import json
import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXCLUSIONS_CONFIG = CONFIG_DIR / "exclusions.yaml"
DEFAULT_CACHE_FILE = OUTPUT_DIR / "exclusion_cache.json"


class ExclusionManager:
    """Manages stock exclusions for correlation analysis."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        cache_path: str | Path | None = None,
    ):
        """
        Initialize exclusion manager.

        Args:
            config_path: Path to exclusions.yaml config file
            cache_path: Path to exclusion_cache.json
        """
        self.config_path = Path(config_path) if config_path else EXCLUSIONS_CONFIG
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_FILE

        self._config = None
        self._cache = None
        self._patterns = None
        self._all_excludes = None

    def load_config(self) -> dict:
        """Load exclusions configuration."""
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            logger.warning(f"Exclusions config not found: {self.config_path}")
            self._config = {}
            return self._config

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f) or {}

        logger.info(f"Loaded exclusions config from {self.config_path}")
        return self._config

    def load_cache(self) -> dict:
        """Load exclusion cache (ETFs, indexes from FMP)."""
        if self._cache is not None:
            return self._cache

        if not self.cache_path.exists():
            logger.warning(f"Exclusion cache not found: {self.cache_path}")
            logger.info("Run 'python scripts/build_exclusions.py' to build cache")
            self._cache = {}
            return self._cache

        with open(self.cache_path) as f:
            self._cache = json.load(f)

        logger.info(
            f"Loaded exclusion cache: {self._cache.get('etf_count', 0)} ETFs, "
            f"{self._cache.get('index_count', 0)} indexes"
        )
        return self._cache

    def get_compiled_patterns(self) -> list[re.Pattern]:
        """Get compiled regex patterns for exclusion."""
        if self._patterns is not None:
            return self._patterns

        self._patterns = []
        config = self.load_config()
        cache = self.load_cache()

        # Patterns from config
        for pattern_list in ["patterns", "etf_patterns"]:
            for pattern in config.get(pattern_list, []):
                try:
                    self._patterns.append(re.compile(pattern))
                except re.error as e:
                    logger.warning(f"Invalid pattern '{pattern}': {e}")

        # Patterns from cache (if auto-fetched patterns exist)
        for pattern in cache.get("patterns", []):
            try:
                self._patterns.append(re.compile(pattern))
            except re.error:
                pass

        return self._patterns

    def get_all_excludes(self) -> set[str]:
        """Get set of all symbols to exclude."""
        if self._all_excludes is not None:
            return self._all_excludes

        config = self.load_config()
        cache = self.load_cache()

        excludes = set()

        # From cache (ETFs, indexes)
        excludes.update(cache.get("etfs", []))
        excludes.update(cache.get("indexes", []))

        # Manual excludes from config
        excludes.update(config.get("manual_excludes", []))

        # Share class excludes from cache
        excludes.update(cache.get("share_class_excludes", []))

        self._all_excludes = excludes
        return excludes

    def get_share_class_groups(self) -> dict[str, list[str]]:
        """Get share class groupings (e.g., GOOGL -> [GOOGL, GOOG])."""
        cache = self.load_cache()
        return cache.get("share_class_groups", {})

    def is_excluded(self, symbol: str) -> tuple[bool, str]:
        """
        Check if a symbol should be excluded.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (is_excluded, reason)
        """
        if not symbol:
            return True, "empty symbol"

        # Check direct exclusion set
        if symbol in self.get_all_excludes():
            # Determine the reason
            cache = self.load_cache()
            if symbol in cache.get("etfs", []):
                return True, "ETF"
            if symbol in cache.get("indexes", []):
                return True, "index"
            if symbol in cache.get("share_class_excludes", []):
                return True, "share class duplicate"
            if symbol in self.load_config().get("manual_excludes", []):
                return True, "manual exclude"
            return True, "excluded"

        # Check patterns
        for pattern in self.get_compiled_patterns():
            if pattern.match(symbol):
                return True, f"pattern: {pattern.pattern}"

        return False, ""

    def filter_tickers(
        self,
        tickers: list[str],
        log_exclusions: bool | None = None,
    ) -> list[str]:
        """
        Filter a list of tickers, removing excluded symbols.

        Args:
            tickers: List of ticker symbols
            log_exclusions: Whether to log excluded tickers (uses config default if None)

        Returns:
            Filtered list of tickers
        """
        config = self.load_config()
        if log_exclusions is None:
            log_exclusions = config.get("settings", {}).get("log_exclusions", True)

        filtered = []
        excluded_counts = {
            "ETF": 0,
            "index": 0,
            "share class duplicate": 0,
            "pattern": 0,
            "other": 0,
        }

        for ticker in tickers:
            is_excluded, reason = self.is_excluded(ticker)
            if is_excluded:
                # Categorize the exclusion
                if reason == "ETF":
                    excluded_counts["ETF"] += 1
                elif reason == "index":
                    excluded_counts["index"] += 1
                elif reason == "share class duplicate":
                    excluded_counts["share class duplicate"] += 1
                elif reason.startswith("pattern"):
                    excluded_counts["pattern"] += 1
                else:
                    excluded_counts["other"] += 1
            else:
                filtered.append(ticker)

        # Log summary
        total_excluded = len(tickers) - len(filtered)
        if total_excluded > 0 and log_exclusions:
            logger.info(f"Excluded {total_excluded} symbols:")
            for reason, count in excluded_counts.items():
                if count > 0:
                    logger.info(f"  {reason}: {count}")

        return filtered

    def select_from_share_class(
        self,
        symbols: list[str],
        volumes: dict[str, float] | None = None,
    ) -> str | None:
        """
        Select the best symbol from a share class group.

        Args:
            symbols: List of share class symbols (e.g., ["GOOGL", "GOOG"])
            volumes: Optional dict of symbol -> average volume

        Returns:
            The selected symbol, or None if all should be excluded
        """
        config = self.load_config()
        handling = config.get("settings", {}).get("share_class_handling", "keep_most_liquid")

        if not symbols:
            return None

        if handling == "exclude_all":
            return None

        if handling == "keep_first":
            return symbols[0]

        if handling == "keep_most_liquid" and volumes:
            # Find symbol with highest volume
            best = max(symbols, key=lambda s: volumes.get(s, 0))
            return best

        # Default to first
        return symbols[0]


# Global instance
_manager: ExclusionManager | None = None


def get_exclusion_manager() -> ExclusionManager:
    """Get or create the global exclusion manager instance."""
    global _manager
    if _manager is None:
        _manager = ExclusionManager()
    return _manager


def filter_tickers(tickers: list[str], log_exclusions: bool = True) -> list[str]:
    """
    Convenience function to filter tickers using global manager.

    Args:
        tickers: List of ticker symbols
        log_exclusions: Whether to log exclusions

    Returns:
        Filtered list
    """
    return get_exclusion_manager().filter_tickers(tickers, log_exclusions)


def is_excluded(symbol: str) -> tuple[bool, str]:
    """
    Convenience function to check if symbol is excluded.

    Returns:
        Tuple of (is_excluded, reason)
    """
    return get_exclusion_manager().is_excluded(symbol)
