"""Core translator class for pytlai."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pytlai.cache.base import TranslationCache
from pytlai.cache.memory import InMemoryCache
from pytlai.config import PythonOptions, TranslationConfig, TranslationStyle
from pytlai.languages import get_text_direction, is_rtl, normalize_lang_code
from pytlai.processors.base import ContentProcessor, ProcessedContent, TextNode
from pytlai.processors.html import HTMLProcessor
from pytlai.processors.python import PythonProcessor
from pytlai.providers.base import AIProvider


class Pytlai:
    """Main translator class for pytlai.

    Orchestrates the translation pipeline: content processing, caching,
    and AI translation. Supports HTML and Python source code.

    Example:
        >>> from pytlai import Pytlai
        >>> from pytlai.providers import OpenAIProvider
        >>>
        >>> translator = Pytlai(
        ...     target_lang="es_ES",
        ...     provider=OpenAIProvider(),
        ... )
        >>> result = translator.process("<h1>Hello World</h1>")
        >>> print(result.content)
        <h1>Hola Mundo</h1>

    Attributes:
        target_lang: Target language code.
        source_lang: Source language code.
        provider: AI provider for translations.
        cache: Translation cache.
    """

    # File extension to content type mapping
    EXTENSION_MAP: dict[str, str] = {
        ".html": "html",
        ".htm": "html",
        ".xhtml": "html",
        ".py": "python",
        ".pyw": "python",
    }

    def __init__(
        self,
        target_lang: str | None = None,
        provider: AIProvider | None = None,
        cache: TranslationCache | None = None,
        config: TranslationConfig | None = None,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
        python_options: PythonOptions | None = None,
    ) -> None:
        """Initialize the translator.

        Args:
            target_lang: Target language code (e.g., 'es_ES', 'ja_JP').
                        Required if config is not provided.
            provider: AI provider for translations. Can be None for cache-only mode.
            cache: Translation cache. Defaults to InMemoryCache.
            config: Full configuration object. Overrides individual params.
            source_lang: Source language code. Defaults to 'en'.
            excluded_terms: Terms that should not be translated.
            context: Additional context for AI translations.
            python_options: Options for Python source translation.

        Raises:
            ValueError: If neither target_lang nor config is provided.
        """
        # Build or use config
        if config:
            self._config = config
        elif target_lang:
            self._config = TranslationConfig(
                target_lang=target_lang,
                source_lang=source_lang,
                excluded_terms=excluded_terms or [],
                context=context,
                python_options=python_options or PythonOptions(),
            )
        else:
            raise ValueError("Either target_lang or config must be provided")

        self._provider = provider
        self._cache = cache if cache is not None else InMemoryCache()

        # Initialize processors
        self._processors: dict[str, ContentProcessor] = {
            "html": HTMLProcessor(),
            "python": PythonProcessor(self._config.python_options),
        }

    @property
    def target_lang(self) -> str:
        """Get the target language code."""
        return self._config.target_lang

    @property
    def source_lang(self) -> str:
        """Get the source language code."""
        return self._config.source_lang

    @property
    def config(self) -> TranslationConfig:
        """Get the translation configuration."""
        return self._config

    def process(
        self,
        content: str,
        content_type: Literal["html", "python"] | None = None,
    ) -> ProcessedContent:
        """Translate content.

        Args:
            content: The content to translate (HTML or Python source).
            content_type: Content type ('html' or 'python'). If None,
                         attempts to auto-detect.

        Returns:
            ProcessedContent with translated content and statistics.

        Raises:
            ValueError: If content type cannot be determined.
        """
        # Auto-detect content type if not specified
        if content_type is None:
            content_type = self._detect_content_type(content)

        processor = self._processors.get(content_type)
        if not processor:
            raise ValueError(f"Unsupported content type: {content_type}")

        # Extract translatable nodes
        parsed, nodes = processor.extract(content)

        if not nodes:
            return ProcessedContent(
                content=content,
                translated_count=0,
                cached_count=0,
                total_nodes=0,
            )

        # Translate nodes
        translations, cached_count, translated_count = self._translate_batch(nodes)

        # Apply translations
        if content_type == "html":
            result = processor.apply(parsed, nodes, translations, self.target_lang)
        else:
            result = processor.apply(parsed, nodes, translations)

        return ProcessedContent(
            content=result,
            translated_count=translated_count,
            cached_count=cached_count,
            total_nodes=len(nodes),
        )

    def process_file(
        self,
        file_path: str | Path,
        content_type: Literal["html", "python"] | None = None,
        encoding: str = "utf-8",
    ) -> ProcessedContent:
        """Translate a file.

        Args:
            file_path: Path to the file to translate.
            content_type: Content type. If None, detected from extension.
            encoding: File encoding. Defaults to 'utf-8'.

        Returns:
            ProcessedContent with translated content and statistics.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If content type cannot be determined.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Detect content type from extension if not specified
        if content_type is None:
            content_type = self.EXTENSION_MAP.get(path.suffix.lower())
            if not content_type:
                raise ValueError(
                    f"Cannot determine content type for extension: {path.suffix}"
                )

        content = path.read_text(encoding=encoding)
        return self.process(content, content_type)

    def translate_text(self, text: str) -> str:
        """Translate a single text string.

        Convenience method for translating individual strings without
        content processing. Uses cache and AI provider directly.

        Args:
            text: The text to translate.

        Returns:
            The translated text.

        Raises:
            RuntimeError: If no AI provider is configured and text is not cached.
        """
        text = text.strip()
        if not text:
            return text

        # Check cache
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"{text_hash}:{self.target_lang}"

        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Translate via AI
        if not self._provider:
            raise RuntimeError(
                "No AI provider configured and text not found in cache"
            )

        translations = self._provider.translate(
            texts=[text],
            target_lang=self.target_lang,
            source_lang=self.source_lang,
            excluded_terms=self._config.excluded_terms,
            context=self._config.context,
        )

        translated = translations[0]

        # Cache the result
        self._cache.set(cache_key, translated)

        return translated

    def _translate_batch(
        self, nodes: list[TextNode]
    ) -> tuple[dict[str, str], int, int]:
        """Translate a batch of text nodes.

        Args:
            nodes: List of TextNode objects to translate.

        Returns:
            Tuple of (translations dict, cached_count, translated_count).
        """
        translations: dict[str, str] = {}
        cache_misses: list[TextNode] = []
        seen_hashes: set[str] = set()
        cached_count = 0

        # Check cache for each node
        for node in nodes:
            cache_key = f"{node.hash}:{self.target_lang}"
            cached = self._cache.get(cache_key)

            if cached:
                translations[node.hash] = cached
                cached_count += 1
            elif node.hash not in seen_hashes:
                cache_misses.append(node)
                seen_hashes.add(node.hash)

        # Translate cache misses via AI
        translated_count = 0
        if cache_misses and self._provider:
            texts = [node.text for node in cache_misses]
            # Collect per-text contexts for disambiguation
            text_contexts = [node.context for node in cache_misses]

            results = self._provider.translate(
                texts=texts,
                target_lang=self.target_lang,
                source_lang=self.source_lang,
                excluded_terms=self._config.excluded_terms,
                context=self._config.context,
                text_contexts=text_contexts if any(text_contexts) else None,
                glossary=self._config.glossary,
                style=self._config.style,
            )

            # Cache and store results
            for node, translated in zip(cache_misses, results, strict=True):
                translations[node.hash] = translated
                cache_key = f"{node.hash}:{self.target_lang}"
                self._cache.set(cache_key, translated)
                translated_count += 1

        return translations, cached_count, translated_count

    def _detect_content_type(self, content: str) -> Literal["html", "python"]:
        """Auto-detect content type from content.

        Args:
            content: The content to analyze.

        Returns:
            Detected content type.

        Raises:
            ValueError: If content type cannot be determined.
        """
        content_stripped = content.strip()

        # Check for HTML markers
        if content_stripped.startswith("<!") or content_stripped.startswith("<html"):
            return "html"
        if "</" in content and ">" in content:
            # Likely HTML with closing tags
            return "html"

        # Check for Python markers
        if content_stripped.startswith("#!") and "python" in content_stripped.split("\n")[0]:
            return "python"
        if content_stripped.startswith('"""') or content_stripped.startswith("'''"):
            return "python"
        if "def " in content or "class " in content or "import " in content:
            return "python"

        # Default to HTML for simple text
        return "html"

    def is_source_lang(self, lang_code: str | None = None) -> bool:
        """Check if a language code matches the source language.

        When true, translation can be bypassed.

        Args:
            lang_code: Language code to check. If None, uses target_lang.

        Returns:
            True if the language matches the source language.
        """
        check_lang = lang_code or self.target_lang
        # Normalize: compare base language codes (e.g., 'en-US' -> 'en')
        normalized_check = normalize_lang_code(check_lang).split("_")[0].lower()
        normalized_source = normalize_lang_code(self.source_lang).split("_")[0].lower()
        return normalized_check == normalized_source

    def is_rtl(self, lang_code: str | None = None) -> bool:
        """Check if the target language uses right-to-left text direction.

        Useful for setting dir="rtl" on HTML elements.

        Args:
            lang_code: Language code to check. If None, uses target_lang.

        Returns:
            True if the language is RTL (Arabic, Hebrew, Persian, Urdu, etc.).
        """
        return is_rtl(lang_code or self.target_lang)

    def get_dir(self, lang_code: str | None = None) -> str:
        """Get the text direction for the target language.

        Args:
            lang_code: Language code to check. If None, uses target_lang.

        Returns:
            'rtl' or 'ltr'.
        """
        return get_text_direction(lang_code or self.target_lang)

    @property
    def glossary(self) -> dict[str, str] | None:
        """Get the glossary of preferred translations."""
        return self._config.glossary

    @property
    def style(self) -> TranslationStyle | None:
        """Get the translation style/register."""
        return self._config.style

    def close(self) -> None:
        """Close any resources (cache connections, etc.)."""
        self._cache.close()
