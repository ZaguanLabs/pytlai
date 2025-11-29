"""Abstract base class for translation caches."""

from abc import ABC, abstractmethod


class TranslationCache(ABC):
    """Abstract base class for translation caches.

    All cache implementations must inherit from this class and implement
    the get and set methods. Caches store translations keyed by a hash
    of the source text combined with the target language.

    Cache key format: {sha256_hash}:{target_lang}
    Example: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e:es_ES
    """

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Retrieve a cached translation.

        Args:
            key: Cache key in format {hash}:{target_lang}.

        Returns:
            The cached translation string, or None if not found or expired.
        """
        ...

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store a translation in the cache.

        Args:
            key: Cache key in format {hash}:{target_lang}.
            value: The translated text to cache.
        """
        ...

    def get_many(self, keys: list[str]) -> dict[str, str | None]:
        """Retrieve multiple cached translations.

        Default implementation calls get() for each key.
        Subclasses may override for batch optimization.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary mapping keys to cached values (or None if not found).
        """
        return {key: self.get(key) for key in keys}

    def set_many(self, items: dict[str, str]) -> None:
        """Store multiple translations in the cache.

        Default implementation calls set() for each item.
        Subclasses may override for batch optimization.

        Args:
            items: Dictionary mapping cache keys to translated values.
        """
        for key, value in items.items():
            self.set(key, value)

    def clear(self) -> None:
        """Clear all entries from the cache.

        Optional method. Default implementation does nothing.
        Subclasses may override to support cache clearing.
        """
        pass

    def close(self) -> None:
        """Close any connections or resources.

        Optional method. Default implementation does nothing.
        Subclasses may override to clean up resources.
        """
        pass
