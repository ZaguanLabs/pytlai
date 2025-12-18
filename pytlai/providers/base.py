"""Abstract base class for AI translation providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytlai.config import TranslationStyle


@dataclass
class ModelInfo:
    """Information about an AI model.

    Attributes:
        name: Model name/identifier.
        capabilities: List of capabilities (e.g., 'json_mode', 'streaming').
    """

    name: str
    capabilities: list[str]


class AIProvider(ABC):
    """Abstract base class for AI translation providers.

    All AI provider implementations must inherit from this class and
    implement the translate method. Providers handle the actual
    communication with AI services (OpenAI, Anthropic, etc.).
    """

    @abstractmethod
    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
        text_contexts: list[str] | None = None,
        glossary: dict[str, str] | None = None,
        style: "TranslationStyle | None" = None,
    ) -> list[str]:
        """Translate a batch of texts to the target language.

        Args:
            texts: List of text strings to translate.
            target_lang: Target language code (e.g., 'es_ES', 'ja_JP').
            source_lang: Source language code. Defaults to 'en'.
            excluded_terms: List of terms that should not be translated.
            context: Additional global context to improve translation quality.
            text_contexts: Per-text context strings for disambiguation.
                Each entry corresponds to the text at the same index.
                Example: ["in <button>", "docstring for function 'save'"]
            glossary: Optional dictionary of preferred translations for specific phrases.
            style: Optional style/register for the translation tone.

        Returns:
            List of translated strings in the same order as input.

        Raises:
            TranslationError: If translation fails.
        """
        ...

    def get_model_info(self) -> ModelInfo:
        """Get information about the AI model being used.

        Returns:
            ModelInfo with name and capabilities.
        """
        return ModelInfo(name="unknown", capabilities=[])


class TranslationError(Exception):
    """Exception raised when translation fails.

    Attributes:
        message: Error description.
        provider: Name of the provider that failed.
        original_error: The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
