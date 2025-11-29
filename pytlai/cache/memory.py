"""In-memory cache implementation with TTL support."""

import time

from pytlai.cache.base import TranslationCache


class InMemoryCache(TranslationCache):
    """In-memory translation cache with TTL support.

    Simple dictionary-based cache that stores translations in memory.
    Suitable for single-process applications or development.

    Attributes:
        ttl: Time-to-live for cache entries in seconds. 0 means no expiration.
    """

    def __init__(self, ttl: int = 3600) -> None:
        """Initialize the in-memory cache.

        Args:
            ttl: Time-to-live for cache entries in seconds.
                 Default is 3600 (1 hour). Set to 0 for no expiration.
        """
        self._cache: dict[str, str] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl

    def get(self, key: str) -> str | None:
        """Retrieve a cached translation.

        Args:
            key: Cache key in format {hash}:{target_lang}.

        Returns:
            The cached translation string, or None if not found or expired.
        """
        timestamp = self._timestamps.get(key)
        if timestamp is None:
            return None

        # Check TTL expiration
        if self._ttl > 0 and time.time() - timestamp > self._ttl:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            return None

        return self._cache.get(key)

    def set(self, key: str, value: str) -> None:
        """Store a translation in the cache.

        Args:
            key: Cache key in format {hash}:{target_lang}.
            value: The translated text to cache.
        """
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def get_many(self, keys: list[str]) -> dict[str, str | None]:
        """Retrieve multiple cached translations.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary mapping keys to cached values (or None if not found).
        """
        return {key: self.get(key) for key in keys}

    def set_many(self, items: dict[str, str]) -> None:
        """Store multiple translations in the cache.

        Args:
            items: Dictionary mapping cache keys to translated values.
        """
        now = time.time()
        for key, value in items.items():
            self._cache[key] = value
            self._timestamps[key] = now

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._timestamps.clear()

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache (respecting TTL)."""
        return self.get(key) is not None
