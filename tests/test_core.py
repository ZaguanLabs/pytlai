"""Tests for core Pytlai class."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pytlai import Pytlai, ProcessedContent, TranslationConfig, PythonOptions
from pytlai.cache.memory import InMemoryCache
from pytlai.providers.base import AIProvider


class MockProvider(AIProvider):
    """Mock AI provider for testing."""

    def __init__(self, translations: dict[str, str] | None = None) -> None:
        self.translations = translations or {}
        self.call_count = 0
        self.last_texts: list[str] = []

    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
        text_contexts: list[str] | None = None,
        glossary: dict[str, str] | None = None,
        style: str | None = None,
    ) -> list[str]:
        self.call_count += 1
        self.last_texts = texts
        self.last_text_contexts = text_contexts

        # Return translations if provided, otherwise prefix with "Translated: "
        return [self.translations.get(t, f"Translated: {t}") for t in texts]


class TestPytlaiInit:
    """Tests for Pytlai initialization."""

    def test_init_with_target_lang(self) -> None:
        """Test initialization with target_lang."""
        translator = Pytlai(target_lang="es_ES")
        assert translator.target_lang == "es_ES"
        assert translator.source_lang == "en"

    def test_init_with_config(self) -> None:
        """Test initialization with TranslationConfig."""
        config = TranslationConfig(
            target_lang="fr_FR",
            source_lang="de",
            excluded_terms=["API"],
            context="Technical docs",
        )
        translator = Pytlai(config=config)

        assert translator.target_lang == "fr_FR"
        assert translator.source_lang == "de"
        assert translator.config.excluded_terms == ["API"]

    def test_init_requires_target_lang(self) -> None:
        """Test that initialization requires target_lang or config."""
        with pytest.raises(ValueError, match="target_lang or config"):
            Pytlai()

    def test_init_normalizes_lang_code(self) -> None:
        """Test that language codes are normalized."""
        translator = Pytlai(target_lang="es-ES")
        assert translator.target_lang == "es_ES"


class TestPytlaiProcessHTML:
    """Tests for HTML processing."""

    def test_process_html_simple(self) -> None:
        """Test processing simple HTML."""
        provider = MockProvider({"Hello World": "Hola Mundo"})
        translator = Pytlai(target_lang="es_ES", provider=provider)

        result = translator.process("<p>Hello World</p>")

        assert isinstance(result, ProcessedContent)
        assert "Hola Mundo" in result.content
        assert result.translated_count == 1
        assert result.total_nodes == 1

    def test_process_html_multiple_nodes(self) -> None:
        """Test processing HTML with multiple text nodes."""
        provider = MockProvider({
            "Title": "Título",
            "Paragraph": "Párrafo",
        })
        translator = Pytlai(target_lang="es_ES", provider=provider)

        result = translator.process("<h1>Title</h1><p>Paragraph</p>")

        assert "Título" in result.content
        assert "Párrafo" in result.content
        assert result.translated_count == 2

    def test_process_html_uses_cache(self) -> None:
        """Test that cached translations are used."""
        provider = MockProvider()
        cache = InMemoryCache()

        # Pre-populate cache
        text_hash = hashlib.sha256("Hello".encode()).hexdigest()
        cache.set(f"{text_hash}:es_ES", "Hola")

        translator = Pytlai(target_lang="es_ES", provider=provider, cache=cache)
        result = translator.process("<p>Hello</p>")

        assert "Hola" in result.content
        assert result.cached_count == 1
        assert result.translated_count == 0
        assert provider.call_count == 0  # Provider not called

    def test_process_html_caches_new_translations(self) -> None:
        """Test that new translations are cached."""
        provider = MockProvider({"Hello": "Hola"})
        cache = InMemoryCache()
        translator = Pytlai(target_lang="es_ES", provider=provider, cache=cache)

        # First call
        translator.process("<p>Hello</p>")

        # Check cache
        text_hash = hashlib.sha256("Hello".encode()).hexdigest()
        assert cache.get(f"{text_hash}:es_ES") == "Hola"

    def test_process_html_deduplicates(self) -> None:
        """Test that duplicate texts are only translated once."""
        provider = MockProvider({"Hello": "Hola"})
        translator = Pytlai(target_lang="es_ES", provider=provider)

        result = translator.process("<p>Hello</p><p>Hello</p><p>Hello</p>")

        # Provider should only receive one unique text
        assert len(provider.last_texts) == 1
        assert result.translated_count == 1

    def test_process_html_empty_content(self) -> None:
        """Test processing HTML with no translatable content."""
        provider = MockProvider()
        translator = Pytlai(target_lang="es_ES", provider=provider)

        result = translator.process("<script>code();</script>")

        assert result.total_nodes == 0
        assert result.translated_count == 0
        assert provider.call_count == 0


class TestPytlaiProcessPython:
    """Tests for Python processing."""

    def test_process_python_docstring(self) -> None:
        """Test processing Python docstrings."""
        provider = MockProvider({"A greeting function.": "Una función de saludo."})
        translator = Pytlai(target_lang="es_ES", provider=provider)

        code = '''def greet():
    """A greeting function."""
    pass
'''
        result = translator.process(code, content_type="python")

        assert "Una función de saludo." in result.content
        assert result.translated_count == 1

    def test_process_python_comments(self) -> None:
        """Test processing Python comments."""
        provider = MockProvider({"Initialize the value": "Inicializar el valor"})
        translator = Pytlai(target_lang="es_ES", provider=provider)

        code = '''# Initialize the value
x = 1
'''
        result = translator.process(code, content_type="python")

        assert "Inicializar el valor" in result.content

    def test_process_python_preserves_code(self) -> None:
        """Test that code structure is preserved."""
        provider = MockProvider()
        translator = Pytlai(target_lang="es_ES", provider=provider)

        code = '''def calculate(x, y):
    """Calculate sum."""
    return x + y
'''
        result = translator.process(code, content_type="python")

        assert "def calculate(x, y):" in result.content
        assert "return x + y" in result.content


class TestPytlaiProcessFile:
    """Tests for file processing."""

    def test_process_file_html(self) -> None:
        """Test processing an HTML file."""
        provider = MockProvider({"Hello": "Hola"})
        translator = Pytlai(target_lang="es_ES", provider=provider)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.html"
            path.write_text("<p>Hello</p>")

            result = translator.process_file(path)

            assert "Hola" in result.content

    def test_process_file_python(self) -> None:
        """Test processing a Python file."""
        provider = MockProvider({"A test module.": "Un módulo de prueba."})
        translator = Pytlai(target_lang="es_ES", provider=provider)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.py"
            path.write_text('"""A test module."""\n')

            result = translator.process_file(path)

            assert "Un módulo de prueba." in result.content

    def test_process_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        translator = Pytlai(target_lang="es_ES")

        with pytest.raises(FileNotFoundError):
            translator.process_file("/nonexistent/file.html")

    def test_process_file_unknown_extension(self) -> None:
        """Test error for unknown file extension."""
        translator = Pytlai(target_lang="es_ES")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xyz"
            path.write_text("content")

            with pytest.raises(ValueError, match="Cannot determine content type"):
                translator.process_file(path)


class TestPytlaiTranslateText:
    """Tests for single text translation."""

    def test_translate_text_simple(self) -> None:
        """Test translating a single text string."""
        provider = MockProvider({"Hello": "Hola"})
        translator = Pytlai(target_lang="es_ES", provider=provider)

        result = translator.translate_text("Hello")

        assert result == "Hola"

    def test_translate_text_uses_cache(self) -> None:
        """Test that cached translations are used."""
        provider = MockProvider()
        cache = InMemoryCache()

        text_hash = hashlib.sha256("Hello".encode()).hexdigest()
        cache.set(f"{text_hash}:es_ES", "Hola")

        translator = Pytlai(target_lang="es_ES", provider=provider, cache=cache)
        result = translator.translate_text("Hello")

        assert result == "Hola"
        assert provider.call_count == 0

    def test_translate_text_caches_result(self) -> None:
        """Test that translation result is cached."""
        provider = MockProvider({"Hello": "Hola"})
        cache = InMemoryCache()
        translator = Pytlai(target_lang="es_ES", provider=provider, cache=cache)

        translator.translate_text("Hello")

        text_hash = hashlib.sha256("Hello".encode()).hexdigest()
        assert cache.get(f"{text_hash}:es_ES") == "Hola"

    def test_translate_text_empty_string(self) -> None:
        """Test translating empty string returns empty."""
        translator = Pytlai(target_lang="es_ES")
        result = translator.translate_text("")
        assert result == ""

    def test_translate_text_no_provider(self) -> None:
        """Test error when no provider and not cached."""
        translator = Pytlai(target_lang="es_ES", provider=None)

        with pytest.raises(RuntimeError, match="No AI provider"):
            translator.translate_text("Hello")


class TestPytlaiContentTypeDetection:
    """Tests for content type auto-detection."""

    def test_detect_html_doctype(self) -> None:
        """Test detecting HTML from doctype."""
        translator = Pytlai(target_lang="es_ES")
        result = translator.process("<!DOCTYPE html><html><body>Hi</body></html>")
        # Should not raise - detected as HTML

    def test_detect_html_tag(self) -> None:
        """Test detecting HTML from html tag."""
        translator = Pytlai(target_lang="es_ES")
        result = translator.process("<html><body>Hi</body></html>")
        # Should not raise

    def test_detect_python_shebang(self) -> None:
        """Test detecting Python from shebang."""
        provider = MockProvider()
        translator = Pytlai(target_lang="es_ES", provider=provider)

        code = "#!/usr/bin/env python3\n# Comment\nx = 1"
        result = translator.process(code)
        # Should detect as Python and process comments


class TestPytlaiCacheOnlyMode:
    """Tests for cache-only mode (no provider)."""

    def test_cache_only_with_hits(self) -> None:
        """Test cache-only mode with cache hits."""
        cache = InMemoryCache()
        text_hash = hashlib.sha256("Hello".encode()).hexdigest()
        cache.set(f"{text_hash}:es_ES", "Hola")

        translator = Pytlai(target_lang="es_ES", provider=None, cache=cache)
        result = translator.process("<p>Hello</p>")

        assert "Hola" in result.content
        assert result.cached_count == 1

    def test_cache_only_with_misses(self) -> None:
        """Test cache-only mode with cache misses (no translation)."""
        cache = InMemoryCache()
        translator = Pytlai(target_lang="es_ES", provider=None, cache=cache)

        result = translator.process("<p>Hello</p>")

        # Original text should remain (no provider to translate)
        assert "Hello" in result.content
        assert result.translated_count == 0


class TestPytlaiConfiguration:
    """Tests for configuration options."""

    def test_excluded_terms_passed_to_provider(self) -> None:
        """Test that excluded terms are passed to provider."""
        provider = MagicMock(spec=AIProvider)
        provider.translate.return_value = ["Translated"]

        translator = Pytlai(
            target_lang="es_ES",
            provider=provider,
            excluded_terms=["API", "SDK"],
        )
        translator.process("<p>Hello</p>")

        provider.translate.assert_called_once()
        call_kwargs = provider.translate.call_args
        assert call_kwargs.kwargs.get("excluded_terms") == ["API", "SDK"]

    def test_context_passed_to_provider(self) -> None:
        """Test that context is passed to provider."""
        provider = MagicMock(spec=AIProvider)
        provider.translate.return_value = ["Translated"]

        translator = Pytlai(
            target_lang="es_ES",
            provider=provider,
            context="Technical documentation",
        )
        translator.process("<p>Hello</p>")

        provider.translate.assert_called_once()
        call_kwargs = provider.translate.call_args
        assert call_kwargs.kwargs.get("context") == "Technical documentation"

    def test_python_options(self) -> None:
        """Test Python-specific options."""
        options = PythonOptions(
            translate_docstrings=True,
            translate_comments=False,
        )
        translator = Pytlai(
            target_lang="es_ES",
            python_options=options,
        )

        assert translator.config.python_options.translate_docstrings is True
        assert translator.config.python_options.translate_comments is False
