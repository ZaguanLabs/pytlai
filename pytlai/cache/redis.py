"""Redis cache implementation for distributed caching."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pytlai.cache.base import TranslationCache

if TYPE_CHECKING:
    from redis import Redis


class RedisCache(TranslationCache):
    """Redis-based translation cache for distributed applications.

    Stores translations in Redis with optional TTL. Suitable for
    multi-process or multi-server deployments.

    Requires the 'redis' optional dependency:
        pip install pytlai[redis]

    Attributes:
        ttl: Time-to-live for cache entries in seconds. 0 means no expiration.
        key_prefix: Prefix for all cache keys in Redis.
    """

    def __init__(
        self,
        url: str | None = None,
        ttl: int = 3600,
        key_prefix: str = "pytlai:",
    ) -> None:
        """Initialize the Redis cache.

        Args:
            url: Redis connection URL. Defaults to REDIS_URL env var
                 or 'redis://localhost:6379'.
            ttl: Time-to-live for cache entries in seconds.
                 Default is 3600 (1 hour). Set to 0 for no expiration.
            key_prefix: Prefix for all cache keys. Default is 'pytlai:'.

        Raises:
            ImportError: If the redis package is not installed.
        """
        try:
            from redis import Redis
        except ImportError as e:
            raise ImportError(
                "Redis support requires the 'redis' package. "
                "Install it with: pip install pytlai[redis]"
            ) from e

        self._ttl = ttl
        self._prefix = key_prefix

        connection_url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis: Redis[str] = Redis.from_url(
            connection_url,
            decode_responses=True,
        )

    def _make_key(self, key: str) -> str:
        """Create a prefixed Redis key."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> str | None:
        """Retrieve a cached translation from Redis.

        Args:
            key: Cache key in format {hash}:{target_lang}.

        Returns:
            The cached translation string, or None if not found.
        """
        return self._redis.get(self._make_key(key))

    def set(self, key: str, value: str) -> None:
        """Store a translation in Redis.

        Args:
            key: Cache key in format {hash}:{target_lang}.
            value: The translated text to cache.
        """
        redis_key = self._make_key(key)
        if self._ttl > 0:
            self._redis.setex(redis_key, self._ttl, value)
        else:
            self._redis.set(redis_key, value)

    def get_many(self, keys: list[str]) -> dict[str, str | None]:
        """Retrieve multiple cached translations from Redis.

        Uses Redis MGET for efficient batch retrieval.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary mapping keys to cached values (or None if not found).
        """
        if not keys:
            return {}

        redis_keys = [self._make_key(k) for k in keys]
        values = self._redis.mget(redis_keys)
        return dict(zip(keys, values, strict=True))

    def set_many(self, items: dict[str, str]) -> None:
        """Store multiple translations in Redis.

        Uses Redis pipeline for efficient batch storage.

        Args:
            items: Dictionary mapping cache keys to translated values.
        """
        if not items:
            return

        pipe = self._redis.pipeline()
        for key, value in items.items():
            redis_key = self._make_key(key)
            if self._ttl > 0:
                pipe.setex(redis_key, self._ttl, value)
            else:
                pipe.set(redis_key, value)
        pipe.execute()

    def clear(self) -> None:
        """Clear all pytlai entries from Redis.

        Only clears keys with the configured prefix.
        """
        cursor = 0
        pattern = f"{self._prefix}*"
        while True:
            cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
            if keys:
                self._redis.delete(*keys)
            if cursor == 0:
                break

    def close(self) -> None:
        """Close the Redis connection."""
        self._redis.close()
