"""Configuration dataclasses for pytlai."""

from dataclasses import dataclass, field
from typing import Literal

# Translation style/register for tone control
TranslationStyle = Literal["formal", "neutral", "casual", "marketing", "technical"]


@dataclass
class AIProviderConfig:
    """Configuration for an AI translation provider.

    Attributes:
        type: The provider type (openai, anthropic, google, or custom).
        api_key: API key for the provider. If None, reads from environment.
        model: Model name to use for translations.
        base_url: Custom API base URL for the provider.
        timeout: Request timeout in seconds.
    """

    type: Literal["openai", "anthropic", "google", "custom"] = "openai"
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: int = 30


@dataclass
class CacheConfig:
    """Configuration for the translation cache.

    Attributes:
        type: Cache backend type (memory, redis, or file).
        ttl: Time-to-live for cache entries in seconds. 0 means no expiration.
        connection_string: Connection string for Redis cache.
        key_prefix: Prefix for all cache keys.
        file_path: Path to cache file for file-based cache.
        file_format: Format for file cache (json, yaml, po).
    """

    type: Literal["memory", "redis", "file"] = "memory"
    ttl: int = 3600
    connection_string: str | None = None
    key_prefix: str = "pytlai:"
    file_path: str | None = None
    file_format: Literal["json", "yaml", "po"] = "json"


@dataclass
class PythonOptions:
    """Options for Python source code translation.

    Attributes:
        translate_docstrings: Whether to translate docstrings.
        translate_comments: Whether to translate comments.
        translate_strings: Whether to translate string literals.
        string_markers: Markers that indicate a string should be translated.
            Common markers include '_(' for gettext and '# translate'.
    """

    translate_docstrings: bool = True
    translate_comments: bool = True
    translate_strings: bool = False
    string_markers: list[str] = field(default_factory=lambda: ["_(", "# translate"])


@dataclass
class TranslationConfig:
    """Main configuration for the Pytlai translator.

    Attributes:
        target_lang: Target language code (e.g., 'es_ES', 'fr_FR', 'ja_JP').
        source_lang: Source language code. Defaults to 'en'.
        excluded_terms: List of terms that should never be translated.
        context: Additional context for the AI to improve translation quality.
            Example: 'Technical documentation for a Python web framework'.
        glossary: Optional dictionary of preferred translations for specific phrases.
            Helps avoid literal translations of idioms and tech jargon.
            Example: {"on the fly": "fortløpende", "cutting-edge": "banebrytende"}
        style: Optional style/register for the translation tone.
            Controls formality and tone of output. Default: 'neutral'.
        python_options: Options specific to Python source translation.
    """

    target_lang: str
    source_lang: str = "en"
    excluded_terms: list[str] = field(default_factory=list)
    context: str | None = None
    glossary: dict[str, str] | None = None
    style: TranslationStyle | None = None
    python_options: PythonOptions = field(default_factory=PythonOptions)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.target_lang:
            raise ValueError("target_lang is required")

        # Normalize language codes to underscore format
        self.target_lang = self.target_lang.replace("-", "_")
        self.source_lang = self.source_lang.replace("-", "_")
