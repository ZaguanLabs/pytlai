"""Tests for cache implementations."""

import tempfile
import time
from pathlib import Path

import pytest

from pytlai.cache.memory import InMemoryCache
from pytlai.cache.file import FileCache


class TestInMemoryCache:
    """Tests for InMemoryCache."""

    def test_set_and_get(self) -> None:
        """Test basic set and get operations."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self) -> None:
        """Test getting a non-existent key returns None."""
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self) -> None:
        """Test that entries expire after TTL."""
        cache = InMemoryCache(ttl=1)  # 1 second TTL
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_no_ttl_expiration(self) -> None:
        """Test that TTL=0 means no expiration."""
        cache = InMemoryCache(ttl=0)
        cache.set("key1", "value1")
        # Entry should persist (we can't really test forever, but it shouldn't expire immediately)
        assert cache.get("key1") == "value1"

    def test_set_many(self) -> None:
        """Test setting multiple values at once."""
        cache = InMemoryCache()
        cache.set_many({"key1": "value1", "key2": "value2", "key3": "value3"})

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_get_many(self) -> None:
        """Test getting multiple values at once."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        result = cache.get_many(["key1", "key2", "key3"])
        assert result == {"key1": "value1", "key2": "value2", "key3": None}

    def test_clear(self) -> None:
        """Test clearing the cache."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache) == 0

    def test_len(self) -> None:
        """Test cache length."""
        cache = InMemoryCache()
        assert len(cache) == 0

        cache.set("key1", "value1")
        assert len(cache) == 1

        cache.set("key2", "value2")
        assert len(cache) == 2

    def test_contains(self) -> None:
        """Test membership check."""
        cache = InMemoryCache()
        cache.set("key1", "value1")

        assert "key1" in cache
        assert "key2" not in cache

    def test_overwrite(self) -> None:
        """Test overwriting an existing key."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key1", "value2")

        assert cache.get("key1") == "value2"


class TestFileCache:
    """Tests for FileCache."""

    def test_create_new_cache(self) -> None:
        """Test creating a new file cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            cache = FileCache(path, read_only=False)

            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"

    def test_read_only_missing_file(self) -> None:
        """Test that read-only mode raises error for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"

            with pytest.raises(FileNotFoundError):
                FileCache(path, read_only=True)

    def test_save_and_load_json(self) -> None:
        """Test saving and loading JSON cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"

            # Create and save
            cache1 = FileCache(path, read_only=False)
            cache1.set("key1", "value1", source="Hello")
            cache1.set("key2", "value2", source="World")
            cache1.meta = {"target_lang": "es_ES"}
            cache1.save()

            # Load in new instance
            cache2 = FileCache(path, read_only=True)
            assert cache2.get("key1") == "value1"
            assert cache2.get("key2") == "value2"
            assert cache2.meta.get("target_lang") == "es_ES"

    def test_save_and_load_yaml(self) -> None:
        """Test saving and loading YAML cache."""
        pytest.importorskip("yaml")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.yaml"

            # Create and save
            cache1 = FileCache(path, read_only=False)
            cache1.set("key1", "value1", source="Hello")
            cache1.save()

            # Load in new instance
            cache2 = FileCache(path, read_only=True)
            assert cache2.get("key1") == "value1"

    def test_auto_detect_format(self) -> None:
        """Test auto-detection of file format from extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "cache.json"
            cache = FileCache(json_path, read_only=False)
            assert cache._format == "json"

    def test_len_and_contains(self) -> None:
        """Test length and membership operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            cache = FileCache(path, read_only=False)

            assert len(cache) == 0
            assert "key1" not in cache

            cache.set("key1", "value1")
            assert len(cache) == 1
            assert "key1" in cache

    def test_clear(self) -> None:
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            cache = FileCache(path, read_only=False)

            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.clear()

            assert len(cache) == 0
            assert cache.get("key1") is None
