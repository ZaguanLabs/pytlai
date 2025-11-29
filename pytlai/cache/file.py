"""File-based cache for offline translation support."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pytlai.cache.base import TranslationCache


class FileCache(TranslationCache):
    """File-based translation cache for offline use.

    Loads translations from JSON, YAML, or PO files. Ideal for shipping
    pre-translated content without requiring AI or Redis at runtime.

    Supports two modes:
    - Read-only: Load existing translations (default)
    - Read-write: Load and save new translations

    File format (JSON):
        {
            "meta": {"source_lang": "en", "target_lang": "es_ES"},
            "translations": {
                "hash:lang": {"source": "Hello", "target": "Hola"}
            }
        }

    Attributes:
        file_path: Path to the cache file.
        file_format: Format of the cache file (json, yaml, po).
        read_only: If True, don't write changes back to file.
    """

    def __init__(
        self,
        file_path: str | Path,
        file_format: Literal["json", "yaml", "po"] | None = None,
        read_only: bool = True,
        auto_save: bool = False,
    ) -> None:
        """Initialize the file cache.

        Args:
            file_path: Path to the cache file.
            file_format: Format of the file. If None, detected from extension.
            read_only: If True, don't write changes back to file.
            auto_save: If True, save after each set() call. Only if not read_only.

        Raises:
            FileNotFoundError: If read_only and file doesn't exist.
            ValueError: If file format is not supported.
        """
        self._path = Path(file_path)
        self._read_only = read_only
        self._auto_save = auto_save and not read_only
        self._dirty = False

        # Detect format from extension if not specified
        if file_format is None:
            ext = self._path.suffix.lower()
            format_map = {".json": "json", ".yaml": "yaml", ".yml": "yaml", ".po": "po"}
            file_format = format_map.get(ext, "json")

        self._format: Literal["json", "yaml", "po"] = file_format

        # Storage
        self._cache: dict[str, str] = {}
        self._sources: dict[str, str] = {}  # hash -> source text (for PO export)
        self._meta: dict[str, Any] = {}

        # Load existing file
        if self._path.exists():
            self._load()
        elif read_only:
            raise FileNotFoundError(f"Cache file not found: {self._path}")

    def _load(self) -> None:
        """Load translations from file."""
        if self._format == "json":
            self._load_json()
        elif self._format == "yaml":
            self._load_yaml()
        elif self._format == "po":
            self._load_po()

    def _load_json(self) -> None:
        """Load translations from JSON file."""
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)

        self._meta = data.get("meta", {})
        translations = data.get("translations", {})

        for key, entry in translations.items():
            if isinstance(entry, dict):
                self._cache[key] = entry.get("target", "")
                if "source" in entry:
                    self._sources[key] = entry["source"]
            else:
                # Simple format: key -> translation
                self._cache[key] = entry

    def _load_yaml(self) -> None:
        """Load translations from YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML support requires the 'pyyaml' package. "
                "Install it with: pip install pytlai[yaml]"
            ) from e

        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._meta = data.get("meta", {})
        translations = data.get("translations", {})

        for key, entry in translations.items():
            if isinstance(entry, dict):
                self._cache[key] = entry.get("target", "")
                if "source" in entry:
                    self._sources[key] = entry["source"]
            else:
                self._cache[key] = entry

    def _load_po(self) -> None:
        """Load translations from PO file.

        PO files use msgid (source) as lookup, so we need to hash them
        to match our cache key format.
        """
        import hashlib

        with open(self._path, encoding="utf-8") as f:
            content = f.read()

        # Simple PO parser - handles basic msgid/msgstr pairs
        current_msgid: list[str] = []
        current_msgstr: list[str] = []
        in_msgid = False
        in_msgstr = False

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("msgid "):
                in_msgid = True
                in_msgstr = False
                current_msgid = [self._parse_po_string(line[6:])]
            elif line.startswith("msgstr "):
                in_msgid = False
                in_msgstr = True
                current_msgstr = [self._parse_po_string(line[7:])]
            elif line.startswith('"') and line.endswith('"'):
                # Continuation line
                if in_msgid:
                    current_msgid.append(self._parse_po_string(line))
                elif in_msgstr:
                    current_msgstr.append(self._parse_po_string(line))
            elif not line:
                # Empty line - end of entry
                if current_msgid and current_msgstr:
                    source = "".join(current_msgid)
                    target = "".join(current_msgstr)
                    if source and target:  # Skip header
                        # Create hash-based key
                        text_hash = hashlib.sha256(source.encode()).hexdigest()
                        target_lang = self._meta.get("target_lang", "unknown")
                        key = f"{text_hash}:{target_lang}"
                        self._cache[key] = target
                        self._sources[key] = source
                current_msgid = []
                current_msgstr = []
                in_msgid = False
                in_msgstr = False

    def _parse_po_string(self, s: str) -> str:
        """Parse a PO string value (remove quotes, unescape)."""
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.replace('\\"', '"').replace("\\n", "\n")

    def get(self, key: str) -> str | None:
        """Retrieve a cached translation.

        Args:
            key: Cache key in format {hash}:{target_lang}.

        Returns:
            The cached translation string, or None if not found.
        """
        return self._cache.get(key)

    def set(self, key: str, value: str, source: str | None = None) -> None:
        """Store a translation in the cache.

        Args:
            key: Cache key in format {hash}:{target_lang}.
            value: The translated text to cache.
            source: Optional source text (for PO export).
        """
        if self._read_only:
            return

        self._cache[key] = value
        if source:
            self._sources[key] = source
        self._dirty = True

        if self._auto_save:
            self.save()

    def save(self) -> None:
        """Save translations to file."""
        if self._read_only or not self._dirty:
            return

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._format == "json":
            self._save_json()
        elif self._format == "yaml":
            self._save_yaml()
        elif self._format == "po":
            self._save_po()

        self._dirty = False

    def _save_json(self) -> None:
        """Save translations to JSON file."""
        translations = {}
        for key, target in self._cache.items():
            entry: dict[str, str] = {"target": target}
            if key in self._sources:
                entry["source"] = self._sources[key]
            translations[key] = entry

        data = {"meta": self._meta, "translations": translations}

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_yaml(self) -> None:
        """Save translations to YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML support requires the 'pyyaml' package. "
                "Install it with: pip install pytlai[yaml]"
            ) from e

        translations = {}
        for key, target in self._cache.items():
            entry: dict[str, str] = {"target": target}
            if key in self._sources:
                entry["source"] = self._sources[key]
            translations[key] = entry

        data = {"meta": self._meta, "translations": translations}

        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def _save_po(self) -> None:
        """Save translations to PO file."""
        lines = [
            '# Translation file generated by pytlai',
            'msgid ""',
            'msgstr ""',
            f'"Content-Type: text/plain; charset=UTF-8\\n"',
            "",
        ]

        for key, target in self._cache.items():
            source = self._sources.get(key, "")
            if source:
                lines.append(f'msgid "{self._escape_po(source)}"')
                lines.append(f'msgstr "{self._escape_po(target)}"')
                lines.append("")

        with open(self._path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _escape_po(self, s: str) -> str:
        """Escape a string for PO format."""
        return s.replace('"', '\\"').replace("\n", "\\n")

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._sources.clear()
        self._dirty = True

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache

    @property
    def meta(self) -> dict[str, Any]:
        """Get cache metadata."""
        return self._meta

    @meta.setter
    def meta(self, value: dict[str, Any]) -> None:
        """Set cache metadata."""
        self._meta = value
        self._dirty = True
