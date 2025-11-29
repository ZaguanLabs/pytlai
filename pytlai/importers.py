"""Translation importers for pytlai."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pytlai.cache.base import TranslationCache
from pytlai.cache.memory import InMemoryCache
from pytlai.catalog import TranslationCatalog


class TranslationImporter:
    """Import existing translations from various file formats.

    Supports importing from JSON, YAML, PO/MO (gettext), and other
    common i18n formats. Imported translations can be used to seed
    a cache or populate a catalog.

    Example:
        >>> importer = TranslationImporter()
        >>> cache = importer.to_cache("locale/es_ES.po", target_lang="es_ES")
        >>> # Use cache with Pytlai - existing translations won't hit AI
        >>> translator = Pytlai(target_lang="es_ES", cache=cache, ...)
    """

    def load(
        self,
        file_path: str | Path,
        file_format: Literal["json", "yaml", "po", "mo", "csv"] | None = None,
    ) -> TranslationCatalog:
        """Load translations from a file into a catalog.

        Args:
            file_path: Path to the translation file.
            file_format: File format. If None, detected from extension.

        Returns:
            TranslationCatalog containing the loaded translations.

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
            format_map = {
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".po": "po",
                ".pot": "po",
                ".mo": "mo",
                ".csv": "csv",
            }
            file_format = format_map.get(ext)
            if not file_format:
                raise ValueError(f"Cannot determine format for extension: {ext}")

        catalog = TranslationCatalog()

        if file_format == "json":
            self._load_json(path, catalog)
        elif file_format == "yaml":
            self._load_yaml(path, catalog)
        elif file_format == "po":
            self._load_po(path, catalog)
        elif file_format == "mo":
            self._load_mo(path, catalog)
        elif file_format == "csv":
            self._load_csv(path, catalog)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        return catalog

    def to_cache(
        self,
        file_path: str | Path,
        target_lang: str,
        file_format: Literal["json", "yaml", "po", "mo", "csv"] | None = None,
        ttl: int = 0,
    ) -> TranslationCache:
        """Load translations directly into a cache.

        Args:
            file_path: Path to the translation file.
            target_lang: Target language code for cache keys.
            file_format: File format. If None, detected from extension.
            ttl: Time-to-live for cache entries. 0 means no expiration.

        Returns:
            InMemoryCache pre-populated with translations.
        """
        catalog = self.load(file_path, file_format)
        cache = InMemoryCache(ttl=ttl)

        # Populate cache
        for hash_key, entry in catalog.entries.items():
            cache_key = f"{hash_key}:{target_lang}"
            cache.set(cache_key, entry["target"])

        return cache

    def _load_json(self, path: Path, catalog: TranslationCatalog) -> None:
        """Load from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle pytlai format
        if "translations" in data:
            for hash_key, entry in data["translations"].items():
                if isinstance(entry, dict):
                    catalog.add(
                        source=entry.get("source", ""),
                        target=entry.get("target", ""),
                        hash_key=hash_key,
                    )
                else:
                    catalog.add(source="", target=str(entry), hash_key=hash_key)
            return

        # Handle simple key-value format (common in i18n)
        for key, value in data.items():
            if key in ("meta", "_meta", "__meta__"):
                continue
            if isinstance(value, str):
                hash_key = hashlib.sha256(key.encode()).hexdigest()
                catalog.add(source=key, target=value, hash_key=hash_key)
            elif isinstance(value, dict) and "message" in value:
                # Chrome extension format
                hash_key = hashlib.sha256(key.encode()).hexdigest()
                catalog.add(source=key, target=value["message"], hash_key=hash_key)

    def _load_yaml(self, path: Path, catalog: TranslationCatalog) -> None:
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

        # Handle pytlai format
        if "translations" in data:
            for hash_key, entry in data["translations"].items():
                if isinstance(entry, dict):
                    catalog.add(
                        source=entry.get("source", ""),
                        target=entry.get("target", ""),
                        hash_key=hash_key,
                    )
                else:
                    catalog.add(source="", target=str(entry), hash_key=hash_key)
            return

        # Handle Rails-style nested YAML
        self._load_yaml_nested(data, catalog, prefix="")

    def _load_yaml_nested(
        self,
        data: dict[str, Any],
        catalog: TranslationCatalog,
        prefix: str,
    ) -> None:
        """Recursively load nested YAML structure."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, str):
                hash_key = hashlib.sha256(full_key.encode()).hexdigest()
                catalog.add(source=full_key, target=value, hash_key=hash_key)
            elif isinstance(value, dict):
                self._load_yaml_nested(value, catalog, full_key)

    def _load_po(self, path: Path, catalog: TranslationCatalog) -> None:
        """Load from PO file."""
        with open(path, encoding="utf-8") as f:
            content = f.read()

        current_msgid: list[str] = []
        current_msgstr: list[str] = []
        current_hash: str | None = None
        in_msgid = False
        in_msgstr = False

        for line in content.split("\n"):
            line_stripped = line.strip()

            # Extract hash from comment if present
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

                    if source and target:  # Skip header and empty
                        catalog.add(
                            source=source,
                            target=target,
                            hash_key=current_hash,
                        )

                current_msgid = []
                current_msgstr = []
                current_hash = None
                in_msgid = False
                in_msgstr = False

    def _load_mo(self, path: Path, catalog: TranslationCatalog) -> None:
        """Load from MO (compiled gettext) file."""
        import struct

        with open(path, "rb") as f:
            # Read magic number
            magic = struct.unpack("I", f.read(4))[0]

            if magic == 0x950412DE:
                # Little endian
                fmt = "<"
            elif magic == 0xDE120495:
                # Big endian
                fmt = ">"
            else:
                raise ValueError(f"Invalid MO file magic number: {magic:08x}")

            # Read header
            version = struct.unpack(f"{fmt}I", f.read(4))[0]
            nstrings = struct.unpack(f"{fmt}I", f.read(4))[0]
            orig_offset = struct.unpack(f"{fmt}I", f.read(4))[0]
            trans_offset = struct.unpack(f"{fmt}I", f.read(4))[0]

            # Read string tables
            f.seek(orig_offset)
            orig_table = [
                struct.unpack(f"{fmt}II", f.read(8)) for _ in range(nstrings)
            ]

            f.seek(trans_offset)
            trans_table = [
                struct.unpack(f"{fmt}II", f.read(8)) for _ in range(nstrings)
            ]

            # Read strings
            for i in range(nstrings):
                orig_len, orig_off = orig_table[i]
                trans_len, trans_off = trans_table[i]

                f.seek(orig_off)
                source = f.read(orig_len).decode("utf-8")

                f.seek(trans_off)
                target = f.read(trans_len).decode("utf-8")

                if source and target:  # Skip header
                    catalog.add(source=source, target=target)

    def _load_csv(self, path: Path, catalog: TranslationCatalog) -> None:
        """Load from CSV file."""
        import csv

        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Try common column names
                source = row.get("source") or row.get("msgid") or row.get("key") or ""
                target = row.get("target") or row.get("msgstr") or row.get("value") or ""
                hash_key = row.get("hash")

                if source and target:
                    catalog.add(source=source, target=target, hash_key=hash_key)

    def _parse_po_string(self, s: str) -> str:
        """Parse a PO string value."""
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.replace('\\"', '"').replace("\\n", "\n")


def seed_cache_from_file(
    file_path: str | Path,
    target_lang: str,
    file_format: Literal["json", "yaml", "po", "mo", "csv"] | None = None,
) -> TranslationCache:
    """Convenience function to create a pre-seeded cache from a file.

    Args:
        file_path: Path to the translation file.
        target_lang: Target language code.
        file_format: File format. If None, detected from extension.

    Returns:
        InMemoryCache pre-populated with translations.

    Example:
        >>> from pytlai import Pytlai
        >>> from pytlai.importers import seed_cache_from_file
        >>>
        >>> cache = seed_cache_from_file("locale/es.po", "es_ES")
        >>> translator = Pytlai(target_lang="es_ES", cache=cache, ...)
    """
    importer = TranslationImporter()
    return importer.to_cache(file_path, target_lang, file_format)
