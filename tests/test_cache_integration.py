"""Integration tests for Redis caching functionality.

These tests require a running Redis instance.
Skip if Redis is not available.
"""

import os

import pandas as pd
import pytest

# Set test environment
os.environ.setdefault("ENABLE_CACHE", "true")


def redis_available():
    """Check if Redis is available."""
    try:
        from price_correlation.cache import RedisCache, get_cache_config

        config = get_cache_config()
        cache = RedisCache(**config)
        cache.connect()
        return cache.is_connected
    except Exception:
        return False


@pytest.fixture
def cache_client():
    """Create a Redis cache client for testing."""
    from price_correlation.cache import RedisCache, get_cache_config

    config = get_cache_config()
    cache = RedisCache(**config)
    cache.connect()
    yield cache
    # Clean up test keys
    cache.clear_pattern("price_correlation:test:*")
    cache.close()


@pytest.fixture
def sample_dataframe():
    """Generate a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "AAPL": [150.0, 151.0, 152.0, 153.0, 154.0],
            "MSFT": [300.0, 301.0, 302.0, 303.0, 304.0],
            "GOOGL": [2800.0, 2810.0, 2820.0, 2830.0, 2840.0],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
class TestRedisCacheIntegration:
    """Integration tests for Redis cache operations."""

    def test_connection(self, cache_client):
        """Test that we can connect to Redis."""
        assert cache_client.is_connected

    def test_set_get_string(self, cache_client):
        """Test basic set/get operations."""
        key = "price_correlation:test:string"
        value = b"test_value"

        cache_client.set(key, value, ttl_seconds=60)
        result = cache_client.get(key)

        assert result == value

    def test_set_get_dataframe(self, cache_client, sample_dataframe):
        """Test caching a DataFrame."""
        from price_correlation.cache import deserialize_dataframe, serialize_dataframe

        key = "price_correlation:test:dataframe"

        # Serialize and cache
        serialized = serialize_dataframe(sample_dataframe)
        cache_client.set(key, serialized, ttl_seconds=60)

        # Retrieve and deserialize
        cached = cache_client.get(key)
        result = deserialize_dataframe(cached)

        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_cache_miss(self, cache_client):
        """Test cache miss returns None."""
        result = cache_client.get("price_correlation:test:nonexistent")
        assert result is None

    def test_delete(self, cache_client):
        """Test deleting a key."""
        key = "price_correlation:test:delete"
        cache_client.set(key, b"value", ttl_seconds=60)

        # Verify it exists
        assert cache_client.get(key) == b"value"

        # Delete and verify
        cache_client.delete(key)
        assert cache_client.get(key) is None

    def test_clear_pattern(self, cache_client):
        """Test clearing keys by pattern."""
        # Set multiple keys
        for i in range(5):
            cache_client.set(f"price_correlation:test:pattern:{i}", b"value", ttl_seconds=60)

        # Clear by pattern
        deleted = cache_client.clear_pattern("price_correlation:test:pattern:*")
        assert deleted == 5

        # Verify all are gone
        for i in range(5):
            assert cache_client.get(f"price_correlation:test:pattern:{i}") is None

    def test_clear_all(self, cache_client):
        """Test clearing all price_correlation cache keys."""
        # Set some test keys
        cache_client.set("price_correlation:test:all:1", b"value", ttl_seconds=60)
        cache_client.set("price_correlation:test:all:2", b"value", ttl_seconds=60)

        # Clear all
        deleted = cache_client.clear_all()
        assert deleted >= 2


class TestCacheConfig:
    """Test cache configuration."""

    def test_get_cache_config(self):
        """Test getting cache configuration from environment."""
        from price_correlation.cache import get_cache_config

        config = get_cache_config()

        assert "host" in config
        assert "port" in config
        assert "db" in config
        assert isinstance(config["port"], int)
        assert isinstance(config["db"], int)

    def test_is_cache_enabled(self):
        """Test cache enabled check."""
        from price_correlation.cache import is_cache_enabled

        original = os.environ.get("ENABLE_CACHE")

        os.environ["ENABLE_CACHE"] = "true"
        assert is_cache_enabled()

        os.environ["ENABLE_CACHE"] = "false"
        assert not is_cache_enabled()

        # Restore
        if original:
            os.environ["ENABLE_CACHE"] = original

    def test_cache_stats_when_disabled(self):
        """Test cache stats when caching is disabled."""
        from price_correlation.cache import get_cache_stats

        original = os.environ.get("ENABLE_CACHE")
        os.environ["ENABLE_CACHE"] = "false"

        stats = get_cache_stats()
        assert not stats["enabled"]

        if original:
            os.environ["ENABLE_CACHE"] = original


class TestCacheKeyGeneration:
    """Test cache key generation functions."""

    def test_cache_key_for_prices(self):
        """Test price cache key generation."""
        from price_correlation.cache import cache_key_for_prices

        tickers = ["AAPL", "MSFT", "GOOGL"]
        key1 = cache_key_for_prices(tickers, "2024-01-01", "2024-06-01")

        # Same tickers in different order should produce same key
        key2 = cache_key_for_prices(["GOOGL", "AAPL", "MSFT"], "2024-01-01", "2024-06-01")
        assert key1 == key2

        # Different tickers should produce different key
        key3 = cache_key_for_prices(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")
        assert key1 != key3

        # Different dates should produce different key
        key4 = cache_key_for_prices(tickers, "2024-01-01", "2024-07-01")
        assert key1 != key4

    def test_cache_key_for_universe(self):
        """Test universe cache key generation."""
        from price_correlation.cache import cache_key_for_universe

        key1 = cache_key_for_universe("fmp", {"min_cap": 1000000000})
        key2 = cache_key_for_universe("fmp", {"min_cap": 1000000000})
        assert key1 == key2

        key3 = cache_key_for_universe("fmp", {"min_cap": 500000000})
        assert key1 != key3

        key4 = cache_key_for_universe("yfinance", {"min_cap": 1000000000})
        assert key1 != key4


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
class TestCachedFunctions:
    """Test the cached wrapper functions."""

    def test_clear_cache_function(self):
        """Test the clear_cache utility function."""
        from price_correlation.cache import clear_cache, get_cache, get_cache_stats

        # Ensure cache is enabled
        original = os.environ.get("ENABLE_CACHE")
        os.environ["ENABLE_CACHE"] = "true"

        stats = get_cache_stats()
        if stats["connected"]:
            result = clear_cache()
            assert result["success"]
            assert "keys_deleted" in result

        if original:
            os.environ["ENABLE_CACHE"] = original
