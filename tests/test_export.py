"""Tests for export, import, and catalog functionality."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from pytlai.catalog import TranslationCatalog
from pytlai.export import TranslationExporter
from pytlai.importers import TranslationImporter, seed_cache_from_file


class TestTranslationExporter:
    """Tests for TranslationExporter."""

    def test_export_json(self) -> None:
        """Test exporting to JSON format."""
        exporter = TranslationExporter()
        translations = {
            "hash1": {"source": "Hello", "target": "Hola"},
            "hash2": {"source": "World", "target": "Mundo"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.json"
            exporter.export(
                translations=translations,
                output_path=path,
                source_lang="en",
                target_lang="es_ES",
            )

            assert path.exists()

            with open(path) as f:
                data = json.load(f)

            assert data["meta"]["source_lang"] == "en"
            assert data["meta"]["target_lang"] == "es_ES"
            assert data["translations"]["hash1"]["target"] == "Hola"
            assert data["translations"]["hash2"]["target"] == "Mundo"

    def test_export_yaml(self) -> None:
        """Test exporting to YAML format."""
        pytest.importorskip("yaml")
        import yaml

        exporter = TranslationExporter()
        translations = {
            "hash1": {"source": "Hello", "target": "Hola"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.yaml"
            exporter.export(
                translations=translations,
                output_path=path,
                target_lang="es_ES",
            )

            assert path.exists()

            with open(path) as f:
                data = yaml.safe_load(f)

            assert data["translations"]["hash1"]["target"] == "Hola"

    def test_export_po(self) -> None:
        """Test exporting to PO format."""
        exporter = TranslationExporter()
        translations = {
            "hash1": {"source": "Hello", "target": "Hola"},
            "hash2": {"source": "World", "target": "Mundo"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.po"
            exporter.export(
                translations=translations,
                output_path=path,
                target_lang="es_ES",
            )

            assert path.exists()

            content = path.read_text()
            assert 'msgid "Hello"' in content
            assert 'msgstr "Hola"' in content
            assert 'msgid "World"' in content
            assert 'msgstr "Mundo"' in content

    def test_export_csv(self) -> None:
        """Test exporting to CSV format."""
        exporter = TranslationExporter()
        translations = {
            "hash1": {"source": "Hello", "target": "Hola"},
            "hash2": {"source": "World", "target": "Mundo"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.csv"
            exporter.export(
                translations=translations,
                output_path=path,
                target_lang="es_ES",
            )

            assert path.exists()

            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            hashes = {r["hash"] for r in rows}
            assert hashes == {"hash1", "hash2"}

    def test_auto_detect_format(self) -> None:
        """Test auto-detection of format from extension."""
        exporter = TranslationExporter()
        translations = {"hash1": {"source": "Hello", "target": "Hola"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            # JSON
            json_path = Path(tmpdir) / "test.json"
            exporter.export(translations, json_path, target_lang="es")
            assert json_path.exists()

            # CSV
            csv_path = Path(tmpdir) / "test.csv"
            exporter.export(translations, csv_path, target_lang="es")
            assert csv_path.exists()

    def test_creates_parent_directories(self) -> None:
        """Test that parent directories are created."""
        exporter = TranslationExporter()
        translations = {"hash1": {"source": "Hello", "target": "Hola"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "translations.json"
            exporter.export(translations, path, target_lang="es")
            assert path.exists()


class TestTranslationCatalog:
    """Tests for TranslationCatalog."""

    def test_add_and_get(self) -> None:
        """Test adding and retrieving translations."""
        catalog = TranslationCatalog()
        hash_key = catalog.add(source="Hello", target="Hola")

        assert catalog.get(hash_key) == "Hola"

    def test_get_by_source(self) -> None:
        """Test retrieving by source text."""
        catalog = TranslationCatalog()
        catalog.add(source="Hello", target="Hola")

        assert catalog.get_by_source("Hello") == "Hola"

    def test_get_entry(self) -> None:
        """Test retrieving full entry."""
        catalog = TranslationCatalog()
        hash_key = catalog.add(source="Hello", target="Hola", metadata={"line": 1})

        entry = catalog.get_entry(hash_key)
        assert entry is not None
        assert entry["source"] == "Hello"
        assert entry["target"] == "Hola"
        assert entry["metadata"]["line"] == 1

    def test_remove(self) -> None:
        """Test removing an entry."""
        catalog = TranslationCatalog()
        hash_key = catalog.add(source="Hello", target="Hola")

        assert catalog.remove(hash_key) is True
        assert catalog.get(hash_key) is None
        assert catalog.remove(hash_key) is False  # Already removed

    def test_merge(self) -> None:
        """Test merging catalogs."""
        catalog1 = TranslationCatalog()
        catalog1.add(source="Hello", target="Hola")

        catalog2 = TranslationCatalog()
        catalog2.add(source="World", target="Mundo")

        count = catalog1.merge(catalog2)
        assert count == 1
        assert catalog1.get_by_source("World") == "Mundo"

    def test_len_and_contains(self) -> None:
        """Test length and membership."""
        catalog = TranslationCatalog()
        assert len(catalog) == 0

        hash_key = catalog.add(source="Hello", target="Hola")
        assert len(catalog) == 1
        assert hash_key in catalog
        assert "Hello" in catalog  # Source text lookup

    def test_load_json(self) -> None:
        """Test loading from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.json"
            data = {
                "meta": {"target_lang": "es_ES"},
                "translations": {
                    "hash1": {"source": "Hello", "target": "Hola"},
                },
            }
            with open(path, "w") as f:
                json.dump(data, f)

            catalog = TranslationCatalog()
            count = catalog.load(path)

            assert count == 1
            assert catalog.get("hash1") == "Hola"
            assert catalog.meta.get("target_lang") == "es_ES"

    def test_to_cache_dict(self) -> None:
        """Test converting to cache dictionary."""
        catalog = TranslationCatalog()
        catalog.add(source="Hello", target="Hola", hash_key="hash1")

        cache_dict = catalog.to_cache_dict("es_ES")
        assert cache_dict == {"hash1:es_ES": "Hola"}


class TestTranslationImporter:
    """Tests for TranslationImporter."""

    def test_load_json(self) -> None:
        """Test loading from JSON file."""
        importer = TranslationImporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.json"
            data = {
                "translations": {
                    "hash1": {"source": "Hello", "target": "Hola"},
                },
            }
            with open(path, "w") as f:
                json.dump(data, f)

            catalog = importer.load(path)
            assert catalog.get("hash1") == "Hola"

    def test_load_simple_json(self) -> None:
        """Test loading simple key-value JSON (common i18n format)."""
        importer = TranslationImporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.json"
            data = {
                "Hello": "Hola",
                "World": "Mundo",
            }
            with open(path, "w") as f:
                json.dump(data, f)

            catalog = importer.load(path)
            assert catalog.get_by_source("Hello") == "Hola"
            assert catalog.get_by_source("World") == "Mundo"

    def test_load_po(self) -> None:
        """Test loading from PO file."""
        importer = TranslationImporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.po"
            content = '''
msgid ""
msgstr ""

msgid "Hello"
msgstr "Hola"

msgid "World"
msgstr "Mundo"
'''
            path.write_text(content)

            catalog = importer.load(path)
            assert catalog.get_by_source("Hello") == "Hola"
            assert catalog.get_by_source("World") == "Mundo"

    def test_load_csv(self) -> None:
        """Test loading from CSV file."""
        importer = TranslationImporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.csv"
            content = '''source,target
Hello,Hola
World,Mundo
'''
            path.write_text(content)

            catalog = importer.load(path)
            assert catalog.get_by_source("Hello") == "Hola"
            assert catalog.get_by_source("World") == "Mundo"

    def test_to_cache(self) -> None:
        """Test creating cache from file."""
        importer = TranslationImporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.json"
            data = {
                "translations": {
                    "hash1": {"source": "Hello", "target": "Hola"},
                },
            }
            with open(path, "w") as f:
                json.dump(data, f)

            cache = importer.to_cache(path, target_lang="es_ES")
            assert cache.get("hash1:es_ES") == "Hola"

    def test_seed_cache_from_file(self) -> None:
        """Test convenience function for seeding cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.json"
            data = {
                "translations": {
                    "hash1": {"source": "Hello", "target": "Hola"},
                },
            }
            with open(path, "w") as f:
                json.dump(data, f)

            cache = seed_cache_from_file(path, "es_ES")
            assert cache.get("hash1:es_ES") == "Hola"


class TestRoundTrip:
    """Test export -> import round trips."""

    def test_json_round_trip(self) -> None:
        """Test JSON export and import."""
        exporter = TranslationExporter()
        importer = TranslationImporter()

        translations = {
            "hash1": {"source": "Hello", "target": "Hola"},
            "hash2": {"source": "World", "target": "Mundo"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.json"

            # Export
            exporter.export(translations, path, target_lang="es_ES")

            # Import
            catalog = importer.load(path)

            assert catalog.get("hash1") == "Hola"
            assert catalog.get("hash2") == "Mundo"

    def test_po_round_trip(self) -> None:
        """Test PO export and import."""
        exporter = TranslationExporter()
        importer = TranslationImporter()

        translations = {
            "hash1": {"source": "Hello", "target": "Hola"},
            "hash2": {"source": "World", "target": "Mundo"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translations.po"

            # Export
            exporter.export(translations, path, target_lang="es_ES")

            # Import
            catalog = importer.load(path)

            assert catalog.get_by_source("Hello") == "Hola"
            assert catalog.get_by_source("World") == "Mundo"
