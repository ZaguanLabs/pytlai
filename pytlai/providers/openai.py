"""OpenAI provider implementation for pytlai."""

from __future__ import annotations

import json
import os
from typing import Any

from pytlai.languages import get_language_name
from pytlai.providers.base import AIProvider, ModelInfo, TranslationError


class OpenAIProvider(AIProvider):
    """OpenAI-based translation provider.

    Uses OpenAI's chat completion API with JSON mode for reliable
    batch translations. Supports GPT-4o, GPT-4o-mini, and other models.

    Attributes:
        model: The OpenAI model to use.
        temperature: Sampling temperature (lower = more deterministic).
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 30,
        temperature: float = 0.3,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use. Defaults to 'gpt-4o-mini'.
            base_url: Custom API base URL. If None, reads from OPENAI_BASE_URL
                      env var or uses OpenAI's default.
            timeout: Request timeout in seconds.
            temperature: Sampling temperature (0.0-2.0). Lower values are
                        more deterministic. Default is 0.3.

        Raises:
            ImportError: If the openai package is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI support requires the 'openai' package. "
                "Install it with: pip install pytlai"
            ) from e

        self._model = model or os.environ.get("OPENAI_MODEL", self.DEFAULT_MODEL)
        self._temperature = temperature

        # Build client kwargs
        client_kwargs: dict[str, Any] = {"timeout": timeout}

        if api_key:
            client_kwargs["api_key"] = api_key

        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self._client = OpenAI(**client_kwargs)

    def _build_system_prompt(
        self,
        target_lang: str,
        source_lang: str,
        excluded_terms: list[str] | None = None,
        context: str | None = None,
    ) -> str:
        """Build the system prompt for translation.

        Args:
            target_lang: Target language code.
            source_lang: Source language code.
            excluded_terms: Terms to exclude from translation.
            context: Additional context for the AI.

        Returns:
            The system prompt string.
        """
        target_name = get_language_name(target_lang)
        source_name = get_language_name(source_lang)

        prompt = f"""# Role
You are an expert native translator. You translate content from {source_name} to {target_name} with the fluency and nuance of a highly educated native speaker.

# Context
{context or "The content is general web or application content."}

# Task
Translate the provided texts into idiomatic {target_name}.

# Style Guide
- **Natural Flow**: Avoid literal translations. Rephrase sentences to sound completely natural to a native speaker.
- **Vocabulary**: Use precise, culturally relevant terminology. Avoid awkward "translationese" or robotic phrasing.
- **Tone**: Maintain the original intent but adapt the wording to fit the target culture's expectations.
- **HTML Safety**: Do NOT translate HTML tags, class names, IDs, or attributes.
- **Interpolation**: Do NOT translate variables (e.g., {{{{name}}}}, {{count}}, %s, {{}}).
- **Code**: Do NOT translate code snippets, function names, or technical identifiers.

# Format
Return ONLY a JSON object with a "translations" key containing an array of strings in the exact same order as the input.
Example: {{"translations": ["translated text 1", "translated text 2"]}}"""

        if excluded_terms:
            terms_list = "\n".join(f"- {term}" for term in excluded_terms)
            prompt += f"""

# Exclusions
Do NOT translate the following terms. Keep them exactly as they appear in the source:
{terms_list}"""

        return prompt

    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
    ) -> list[str]:
        """Translate a batch of texts using OpenAI.

        Args:
            texts: List of text strings to translate.
            target_lang: Target language code (e.g., 'es_ES', 'ja_JP').
            source_lang: Source language code. Defaults to 'en'.
            excluded_terms: List of terms that should not be translated.
            context: Additional context to improve translation quality.

        Returns:
            List of translated strings in the same order as input.

        Raises:
            TranslationError: If the API call fails or response is invalid.
        """
        if not texts:
            return []

        system_prompt = self._build_system_prompt(
            target_lang, source_lang, excluded_terms, context
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
                ],
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise TranslationError(
                    "Empty response from OpenAI",
                    provider="openai",
                )

            return self._parse_response(content, len(texts))

        except json.JSONDecodeError as e:
            raise TranslationError(
                f"Invalid JSON in response: {e}",
                provider="openai",
                original_error=e,
            ) from e
        except Exception as e:
            if isinstance(e, TranslationError):
                raise
            raise TranslationError(
                f"API call failed: {e}",
                provider="openai",
                original_error=e,
            ) from e

    def _parse_response(self, content: str, expected_count: int) -> list[str]:
        """Parse the JSON response from OpenAI.

        Handles both formats:
        - {"translations": ["text1", "text2"]}
        - ["text1", "text2"]

        Args:
            content: JSON string from the API.
            expected_count: Expected number of translations.

        Returns:
            List of translated strings.

        Raises:
            TranslationError: If parsing fails or count mismatch.
        """
        result = json.loads(content)

        # Handle direct array response
        if isinstance(result, list):
            translations = result
        # Handle object with translations key
        elif isinstance(result, dict):
            if "translations" in result:
                translations = result["translations"]
            else:
                # Try to find any array value
                for value in result.values():
                    if isinstance(value, list):
                        translations = value
                        break
                else:
                    raise TranslationError(
                        f"No translations array found in response: {result}",
                        provider="openai",
                    )
        else:
            raise TranslationError(
                f"Unexpected response format: {type(result)}",
                provider="openai",
            )

        if len(translations) != expected_count:
            raise TranslationError(
                f"Expected {expected_count} translations, got {len(translations)}",
                provider="openai",
            )

        return [str(t) for t in translations]

    def get_model_info(self) -> ModelInfo:
        """Get information about the OpenAI model being used.

        Returns:
            ModelInfo with model name and capabilities.
        """
        capabilities = ["json_mode", "batch"]

        # Add streaming capability for supported models
        if "gpt-4" in self._model or "gpt-3.5" in self._model:
            capabilities.append("streaming")

        return ModelInfo(name=self._model, capabilities=capabilities)
