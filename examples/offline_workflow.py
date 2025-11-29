#!/usr/bin/env python3
"""Example: Offline translation workflow.

This example demonstrates the complete offline workflow:
1. Extract strings from source files
2. Translate using AI and cache results
3. Export translations to language files
4. Use translations at runtime without AI

This is ideal for projects that want to ship pre-translated content
without requiring users to have API keys or internet access.

Requirements:
    - Set OPENAI_API_KEY environment variable (for initial translation)
    - pip install pytlai

Usage:
    python offline_workflow.py
"""

import json
import tempfile
from pathlib import Path

from pytlai import Pytlai
from pytlai.cache import FileCache, InMemoryCache
from pytlai.catalog import TranslationCatalog
from pytlai.export import TranslationExporter
from pytlai.importers import TranslationImporter, seed_cache_from_file
from pytlai.providers import OpenAIProvider


def main() -> None:
    """Demonstrate the offline translation workflow."""
    # Sample content to translate
    html_content = """
    <html>
    <body>
        <h1>Welcome to Our App</h1>
        <p>This is a sample application.</p>
        <p>Click the button to get started.</p>
        <button>Get Started</button>
        <footer>© 2024 Example Corp</footer>
    </body>
    </html>
    """

    # Create a temporary directory for our language files
    with tempfile.TemporaryDirectory() as tmpdir:
        locale_dir = Path(tmpdir) / "locale"
        locale_dir.mkdir()

        print("=" * 60)
        print("STEP 1: TRANSLATE WITH AI AND CACHE")
        print("=" * 60)

        # Create translator with caching
        cache = InMemoryCache(ttl=0)  # No expiration
        translator = Pytlai(
            target_lang="es_ES",
            provider=OpenAIProvider(),
            cache=cache,
        )

        # Translate content
        result = translator.process(html_content)
        print(f"Translated {result.translated_count} items")
        print(f"Translated content preview:")
        print(result.content[:200] + "...")

        print("\n" + "=" * 60)
        print("STEP 2: EXPORT TRANSLATIONS TO FILES")
        print("=" * 60)

        # Build translations dict from cache
        translations = {}
        for key, value in cache._cache.items():
            hash_part = key.split(":")[0]
            # We need to find the source text - in real usage you'd track this
            translations[hash_part] = {
                "source": f"[source for {hash_part[:8]}...]",
                "target": value,
            }

        # Export to different formats
        exporter = TranslationExporter()

        # JSON format
        json_path = locale_dir / "es_ES.json"
        exporter.export(
            translations=translations,
            output_path=json_path,
            source_lang="en",
            target_lang="es_ES",
        )
        print(f"Exported to JSON: {json_path}")

        # PO format (gettext compatible)
        po_path = locale_dir / "es_ES.po"
        exporter.export(
            translations=translations,
            output_path=po_path,
            source_lang="en",
            target_lang="es_ES",
        )
        print(f"Exported to PO: {po_path}")

        # Show JSON content
        print("\nJSON file content:")
        with open(json_path) as f:
            data = json.load(f)
            print(json.dumps(data, indent=2, ensure_ascii=False)[:500] + "...")

        print("\n" + "=" * 60)
        print("STEP 3: USE TRANSLATIONS OFFLINE (NO AI)")
        print("=" * 60)

        # Create a new translator using only the cached file
        # This simulates runtime in a deployed application
        offline_cache = FileCache(json_path, read_only=True)
        offline_translator = Pytlai(
            target_lang="es_ES",
            cache=offline_cache,
            provider=None,  # No AI provider needed!
        )

        # The translations are available from the file
        print(f"Loaded {len(offline_cache)} cached translations")
        print("Translations can now be used without any AI API calls!")

        print("\n" + "=" * 60)
        print("STEP 4: IMPORT EXISTING TRANSLATIONS")
        print("=" * 60)

        # You can also import translations from existing PO files
        importer = TranslationImporter()
        catalog = importer.load(po_path)
        print(f"Imported {len(catalog)} translations from PO file")

        # Convert to cache for use with Pytlai
        imported_cache = importer.to_cache(json_path, target_lang="es_ES")
        print(f"Created cache with {len(imported_cache._cache)} entries")

        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETE")
        print("=" * 60)
        print("""
Summary:
1. Translated content using AI (requires API key)
2. Exported translations to JSON and PO files
3. Loaded translations from files (no API key needed)
4. Imported existing translations from PO files

Your application can now ship with the locale/ directory
and use FileCache for translations without any AI dependencies!
        """)


if __name__ == "__main__":
    main()
