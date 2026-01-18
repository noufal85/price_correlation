"""Redis caching module for price data and universe."""

import hashlib
import logging
import os
import pickle
from functools import wraps
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

# Default TTLs
TTL_UNIVERSE = 86400  # 24 hours
TTL_PRICES = 21600  # 6 hours


def get_cache_config() -> dict:
    """Get Redis configuration from environment variables."""
    return {
        "host": os.environ.get("REDIS_HOST", "192.168.68.88"),
        "port": int(os.environ.get("REDIS_PORT", 6379)),
        "db": int(os.environ.get("REDIS_DB", 0)),
        "password": os.environ.get("REDIS_PASSWORD") or None,
    }


def is_cache_enabled() -> bool:
    """Check if caching is enabled via environment."""
    return os.environ.get("ENABLE_CACHE", "true").lower() in ("true", "1", "yes")


class RedisCache:
    """Redis cache client with graceful degradation."""

    def __init__(
        self,
        host: str = "192.168.68.88",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._client = None
        self._connected = False

    def connect(self):
        """Connect to Redis server."""
        if self._client is not None:
            return self._client

        try:
            import redis

            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return self._client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self._client = None
            self._connected = False
            return None

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    def get(self, key: str) -> bytes | None:
        """Get value from cache."""
        if not self._connected and not self.connect():
            return None

        try:
            return self._client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET failed for {key}: {e}")
            return None

    def set(self, key: str, value: bytes, ttl_seconds: int = TTL_PRICES) -> bool:
        """Set value in cache with TTL."""
        if not self._connected and not self.connect():
            return False

        try:
            self._client.setex(key, ttl_seconds, value)
            return True
        except Exception as e:
            logger.warning(f"Redis SET failed for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._connected and not self.connect():
            return False

        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE failed for {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern. Returns count of deleted keys."""
        if not self._connected and not self.connect():
            return 0

        try:
            keys = list(self._client.scan_iter(match=pattern, count=1000))
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Redis CLEAR failed for pattern {pattern}: {e}")
            return 0

    def clear_all(self) -> int:
        """Clear all price_correlation cache keys."""
        return self.clear_pattern("price_correlation:*")

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False


# Global cache instance
_cache: RedisCache | None = None


def get_cache() -> RedisCache | None:
    """Get or create global cache instance."""
    global _cache

    if not is_cache_enabled():
        return None

    if _cache is None:
        config = get_cache_config()
        _cache = RedisCache(**config)
        _cache.connect()

    return _cache


def cache_key_for_prices(tickers: list[str], start: str, end: str) -> str:
    """Generate cache key for price data."""
    tickers_str = ",".join(sorted(tickers))
    hash_val = hashlib.md5(tickers_str.encode()).hexdigest()[:12]
    return f"price_correlation:prices:{hash_val}:{start}:{end}"


def cache_key_for_universe(source: str, filters: dict | None = None) -> str:
    """Generate cache key for universe data."""
    filter_str = str(sorted(filters.items())) if filters else "none"
    hash_val = hashlib.md5(filter_str.encode()).hexdigest()[:12]
    return f"price_correlation:universe:{source}:{hash_val}"


def serialize_dataframe(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame for caching."""
    return pickle.dumps(df)


def deserialize_dataframe(data: bytes) -> pd.DataFrame:
    """Deserialize DataFrame from cache."""
    return pickle.loads(data)


def serialize_json(data: Any) -> bytes:
    """Serialize JSON-compatible data for caching."""
    return pickle.dumps(data)


def deserialize_json(data: bytes) -> Any:
    """Deserialize JSON-compatible data from cache."""
    return pickle.loads(data)


def cached(ttl: int = TTL_PRICES, key_func: Callable | None = None):
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds
        key_func: Optional function to generate cache key from args/kwargs
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()

            # Skip caching if disabled or unavailable
            if cache is None or not cache.is_connected:
                return func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key based on function name and args
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key_str = ":".join(key_parts)
                hash_val = hashlib.md5(key_str.encode()).hexdigest()[:16]
                cache_key = f"price_correlation:func:{func.__name__}:{hash_val}"

            # Try to get from cache
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Cache hit for {func.__name__}")
                try:
                    return pickle.loads(cached_data)
                except Exception as e:
                    logger.warning(f"Cache deserialize failed: {e}")

            # Cache miss - call function
            logger.info(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)

            # Store in cache
            try:
                cache.set(cache_key, pickle.dumps(result), ttl)
            except Exception as e:
                logger.warning(f"Cache store failed: {e}")

            return result

        return wrapper

    return decorator


def get_cache_stats() -> dict:
    """Get cache connection status and stats."""
    cache = get_cache()

    if cache is None:
        return {
            "enabled": False,
            "connected": False,
            "host": None,
            "port": None,
        }

    config = get_cache_config()
    return {
        "enabled": is_cache_enabled(),
        "connected": cache.is_connected,
        "host": config["host"],
        "port": config["port"],
    }


def clear_cache() -> dict:
    """Clear all cache entries. Returns stats about cleared keys."""
    cache = get_cache()

    if cache is None:
        return {"success": False, "message": "Cache not enabled", "keys_deleted": 0}

    if not cache.is_connected:
        return {"success": False, "message": "Cache not connected", "keys_deleted": 0}

    count = cache.clear_all()
    return {
        "success": True,
        "message": f"Cleared {count} cache entries",
        "keys_deleted": count,
    }
