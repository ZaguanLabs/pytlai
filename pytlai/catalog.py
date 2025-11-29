"""Translation catalog for pytlai."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Literal


class TranslationCatalog:
    """A collection of translations that can be loaded, merged, and queried.

    The catalog provides a unified interface for working with translations
    from multiple sources. It can load from JSON, YAML, or PO files and
    merge translations from different files.

    Example:
        >>> catalog = TranslationCatalog()
        >>> catalog.load("locale/es_ES.json")
        >>> catalog.load("locale/es_ES_custom.json")  # Merge additional
        >>> translation = catalog.get_by_source("Hello")
        >>> print(translation)  # "Hola"
    """

    def __init__(self) -> None:
        """Initialize an empty catalog."""
        # hash -> {"source": str, "target": str, "metadata": dict}
        self._entries: dict[str, dict[str, Any]] = {}
        # source text -> hash (for reverse lookup)
        self._source_index: dict[str, str] = {}
        # Catalog metadata
        self._meta: dict[str, Any] = {}

    def load(
        self,
        file_path: str | Path,
        file_format: Literal["json", "yaml", "po"] | None = None,
        merge: bool = True,
    ) -> int:
        """Load translations from a file.

        Args:
            file_path: Path to the translation file.
            file_format: File format. If None, detected from extension.
            merge: If True, merge with existing entries. If False, replace.

        Returns:
            Number of entries loaded.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If format is not supported.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Translation file not found: {path}")

        # Detect format
        if file_format is None:
            ext = path.suffix.lower()
            format_map = {".json": "json", ".yaml": "yaml", ".yml": "yaml", ".po": "po"}
            file_format = format_map.get(ext)
            if not file_format:
                raise ValueError(f"Cannot determine format for extension: {ext}")

        # Clear if not merging
        if not merge:
            self._entries.clear()
            self._source_index.clear()

        # Load based on format
        if file_format == "json":
            count = self._load_json(path)
        elif file_format == "yaml":
            count = self._load_yaml(path)
        elif file_format == "po":
            count = self._load_po(path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        return count

    def _load_json(self, path: Path) -> int:
        """Load from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Update metadata
        if "meta" in data:
            self._meta.update(data["meta"])

        # Load translations
        translations = data.get("translations", {})
        count = 0

        for hash_key, entry in translations.items():
            if isinstance(entry, dict):
                source = entry.get("source", "")
                target = entry.get("target", "")
            else:
                # Simple format: hash -> target
                source = ""
                target = str(entry)

            self._entries[hash_key] = {
                "source": source,
                "target": target,
                "metadata": {},
            }

            if source:
                self._source_index[source] = hash_key

            count += 1

        return count

    def _load_yaml(self, path: Path) -> int:
        """Load from YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML support requires the 'pyyaml' package. "
                "Install it with: pip install pytlai[yaml]"
            ) from e

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Update metadata
        if "meta" in data:
            self._meta.update(data["meta"])

        # Load translations
        translations = data.get("translations", {})
        count = 0

        for hash_key, entry in translations.items():
            if isinstance(entry, dict):
                source = entry.get("source", "")
                target = entry.get("target", "")
            else:
                source = ""
                target = str(entry)

            self._entries[hash_key] = {
                "source": source,
                "target": target,
                "metadata": {},
            }

            if source:
                self._source_index[source] = hash_key

            count += 1

        return count

    def _load_po(self, path: Path) -> int:
        """Load from PO file."""
        with open(path, encoding="utf-8") as f:
            content = f.read()

        count = 0
        current_msgid: list[str] = []
        current_msgstr: list[str] = []
        current_hash: str | None = None
        in_msgid = False
        in_msgstr = False

        for line in content.split("\n"):
            line_stripped = line.strip()

            # Extract hash from comment
            if line_stripped.startswith("#. hash:"):
                current_hash = line_stripped.split(":", 1)[1].strip()
                continue

            if line_stripped.startswith("msgid "):
                in_msgid = True
                in_msgstr = False
                current_msgid = [self._parse_po_string(line_stripped[6:])]
            elif line_stripped.startswith("msgstr "):
                in_msgid = False
                in_msgstr = True
                current_msgstr = [self._parse_po_string(line_stripped[7:])]
            elif line_stripped.startswith('"') and line_stripped.endswith('"'):
                if in_msgid:
                    current_msgid.append(self._parse_po_string(line_stripped))
                elif in_msgstr:
                    current_msgstr.append(self._parse_po_string(line_stripped))
            elif not line_stripped:
                # End of entry
                if current_msgid and current_msgstr:
                    source = "".join(current_msgid)
                    target = "".join(current_msgstr)

                    if source:  # Skip header
                        # Use provided hash or compute from source
                        if current_hash:
                            hash_key = current_hash
                        else:
                            hash_key = hashlib.sha256(source.encode()).hexdigest()

                        self._entries[hash_key] = {
                            "source": source,
                            "target": target,
                            "metadata": {},
                        }
                        self._source_index[source] = hash_key
                        count += 1

                current_msgid = []
                current_msgstr = []
                current_hash = None
                in_msgid = False
                in_msgstr = False

        return count

    def _parse_po_string(self, s: str) -> str:
        """Parse a PO string value."""
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.replace('\\"', '"').replace("\\n", "\n")

    def get(self, hash_key: str) -> str | None:
        """Get a translation by hash key.

        Args:
            hash_key: The SHA-256 hash of the source text.

        Returns:
            The translated text, or None if not found.
        """
        entry = self._entries.get(hash_key)
        return entry["target"] if entry else None

    def get_by_source(self, source: str) -> str | None:
        """Get a translation by source text.

        Args:
            source: The original source text.

        Returns:
            The translated text, or None if not found.
        """
        # Try direct lookup
        hash_key = self._source_index.get(source)
        if hash_key:
            return self.get(hash_key)

        # Try computing hash
        computed_hash = hashlib.sha256(source.strip().encode()).hexdigest()
        return self.get(computed_hash)

    def get_entry(self, hash_key: str) -> dict[str, Any] | None:
        """Get a full translation entry by hash key.

        Args:
            hash_key: The SHA-256 hash of the source text.

        Returns:
            Dictionary with 'source', 'target', and 'metadata', or None.
        """
        return self._entries.get(hash_key)

    def add(
        self,
        source: str,
        target: str,
        hash_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a translation to the catalog.

        Args:
            source: The source text.
            target: The translated text.
            hash_key: Optional hash key. If None, computed from source.
            metadata: Optional metadata for the entry.

        Returns:
            The hash key for the entry.
        """
        if hash_key is None:
            hash_key = hashlib.sha256(source.strip().encode()).hexdigest()

        self._entries[hash_key] = {
            "source": source,
            "target": target,
            "metadata": metadata or {},
        }
        self._source_index[source] = hash_key

        return hash_key

    def remove(self, hash_key: str) -> bool:
        """Remove a translation from the catalog.

        Args:
            hash_key: The hash key of the entry to remove.

        Returns:
            True if removed, False if not found.
        """
        entry = self._entries.pop(hash_key, None)
        if entry:
            source = entry.get("source", "")
            if source and self._source_index.get(source) == hash_key:
                del self._source_index[source]
            return True
        return False

    def merge(self, other: TranslationCatalog, overwrite: bool = True) -> int:
        """Merge another catalog into this one.

        Args:
            other: The catalog to merge from.
            overwrite: If True, overwrite existing entries.

        Returns:
            Number of entries added or updated.
        """
        count = 0
        for hash_key, entry in other._entries.items():
            if overwrite or hash_key not in self._entries:
                self._entries[hash_key] = entry.copy()
                if entry.get("source"):
                    self._source_index[entry["source"]] = hash_key
                count += 1
        return count

    def to_cache_dict(self, target_lang: str) -> dict[str, str]:
        """Convert catalog to cache-compatible dictionary.

        Args:
            target_lang: Target language code for cache keys.

        Returns:
            Dictionary mapping cache keys to translations.
        """
        result = {}
        for hash_key, entry in self._entries.items():
            cache_key = f"{hash_key}:{target_lang}"
            result[cache_key] = entry["target"]
        return result

    def __len__(self) -> int:
        """Return the number of entries in the catalog."""
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        """Check if a hash key or source text is in the catalog."""
        if key in self._entries:
            return True
        return key in self._source_index

    def __iter__(self) -> Iterator[str]:
        """Iterate over hash keys in the catalog."""
        return iter(self._entries)

    @property
    def meta(self) -> dict[str, Any]:
        """Get catalog metadata."""
        return self._meta

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        """Get all entries (read-only view)."""
        return self._entries.copy()
