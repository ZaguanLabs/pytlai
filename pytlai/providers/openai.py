"""OpenAI provider implementation for pytlai."""

from __future__ import annotations

import json
import os
from typing import Any

from pytlai.config import TranslationStyle
from pytlai.languages import get_language_name, normalize_lang_code
from pytlai.providers.base import AIProvider, ModelInfo, TranslationError

# Language-specific locale clarifications
LOCALE_CLARIFICATIONS: dict[str, str] = {
    # Norwegian variants
    "nb_NO": "Use Norwegian Bokmål (nb-NO), not Nynorsk.",
    "nb": "Use Norwegian Bokmål (nb-NO), not Nynorsk.",
    "no": "Use Norwegian Bokmål (nb-NO).",
    "nn_NO": "Use Norwegian Nynorsk (nn-NO), not Bokmål.",
    "nn": "Use Norwegian Nynorsk (nn-NO), not Bokmål.",
    # Chinese variants
    "zh_CN": "Use Simplified Chinese characters.",
    "zh_TW": "Use Traditional Chinese characters.",
    "zh": "Use Simplified Chinese characters.",
    # Portuguese variants
    "pt_BR": "Use Brazilian Portuguese conventions.",
    "pt_PT": "Use European Portuguese conventions.",
    "pt": "Use Brazilian Portuguese conventions.",
    # English variants
    "en_GB": "Use British English spelling and conventions.",
    "en_US": "Use American English spelling and conventions.",
    # Spanish variants
    "es_ES": "Use Castilian Spanish (Spain) conventions.",
    "es_MX": "Use Mexican Spanish conventions.",
}

# Style descriptions for translation register
STYLE_DESCRIPTIONS: dict[str, str] = {
    "formal": "Use formal, professional language suitable for official documents or business communication.",
    "neutral": "Use a neutral, professional tone suitable for general web content and documentation.",
    "casual": "Use casual, conversational language suitable for blogs, social media, or friendly communication.",
    "marketing": "Use persuasive, engaging language suitable for marketing copy, landing pages, and promotional content.",
    "technical": "Use precise, technical language suitable for developer documentation, API references, and technical guides.",
}


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
        temperature: float = 0.1,
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
        glossary: dict[str, str] | None = None,
        style: TranslationStyle | None = None,
    ) -> str:
        """Build the system prompt for translation.

        Args:
            target_lang: Target language code.
            source_lang: Source language code.
            excluded_terms: Terms to exclude from translation.
            context: Additional context for the AI.
            glossary: Optional preferred translations for specific phrases.
            style: Optional style/register for the translation.

        Returns:
            The system prompt string.
        """
        target_name = get_language_name(target_lang)
        source_name = get_language_name(source_lang)
        normalized_lang = normalize_lang_code(target_lang)
        locale_hint = LOCALE_CLARIFICATIONS.get(target_lang) or LOCALE_CLARIFICATIONS.get(normalized_lang) or ""
        style_desc = STYLE_DESCRIPTIONS.get(style or "neutral", STYLE_DESCRIPTIONS["neutral"])

        prompt = f"""# Role
You are an expert native translator. You translate content from {source_name} to {target_name} with the fluency and nuance of a highly educated native speaker.

# Context
{f"The content is for: {context}. Adapt the tone to be appropriate for this context." if context else "The content is general web content."}

# Register
{style_desc}

# Task
Translate the provided texts into idiomatic {target_name}.

# Style Guide
- **Natural Flow**: Avoid literal translations. Rephrase sentences to sound completely natural to a native speaker.
- **Vocabulary**: Use precise, culturally relevant terminology. Avoid awkward "translationese" or robotic phrasing.
- **Tone**: Maintain the original intent but adapt the wording to fit the target culture's expectations.
- **Idioms**: Never translate idioms literally. Replace English idioms with natural {target_name} equivalents.
- **HTML/Code Safety**: Do NOT translate HTML tags, class names, IDs, attributes, URLs, email addresses, or content inside backticks or <code> blocks.
- **Interpolation**: Do NOT translate variables or placeholders (e.g., {{{{name}}}}, {{count}}, %s, $1).
- **Formatting**: Preserve meaningful whitespace (leading/trailing spaces, multiple spaces, newlines). Use idiomatic punctuation for the target language.
- **Context Hints**: If you see {{{{__ctx__:...}}}}, use that hint to disambiguate the translation, then REMOVE the hint from your output."""

        # Add locale clarification if available
        if locale_hint:
            prompt += f"\n- **Locale**: {locale_hint}"

        # Add user-provided glossary if available
        if glossary:
            prompt += "\n\n# Glossary\nWhen you encounter these phrases, prefer these translations (unless context demands otherwise):"
            for source, target in glossary.items():
                prompt += f'\n- "{source}" → {target}'

        # Add quality check instruction
        prompt += f"\n\n# Quality Check\nAfter translating each string, verify it sounds like native {target_name} and not a calque. If any phrase sounds like a literal translation, rewrite it naturally."

        # Add format requirements
        prompt += """\n\n# Input Format
Input may be either:
1. A simple array of strings: ["text1", "text2"]
2. An object with items containing text and context: {"items": [{"text": "Run", "context": "in <button>"}, ...]}

# Output Format
Return a valid JSON object with a single key "translations" containing an array of strings in the exact same order as the input.
Example: {"translations": ["translated text 1", "translated text 2"]}
- Do NOT wrap in Markdown code blocks.
- Do NOT include any {{__ctx__:...}} markers in your output."""

        if excluded_terms:
            terms_list = "\n".join(f"- {term}" for term in excluded_terms)
            prompt += f"""

# Exclusions
Do NOT translate the following terms. Keep them exactly as they appear in the source:
{terms_list}"""

        return prompt

    def _build_user_message(
        self,
        texts: list[str],
        text_contexts: list[str] | None = None,
    ) -> str:
        """Build the user message for translation.

        If text_contexts are provided, formats each text with its context
        to help the AI disambiguate translations.

        Args:
            texts: List of texts to translate.
            text_contexts: Optional per-text context strings.

        Returns:
            JSON string for the API request.
        """
        if not text_contexts or not any(text_contexts):
            # Simple format: just the texts
            return json.dumps(texts, ensure_ascii=False)

        # Format with context for disambiguation
        items = []
        for i, text in enumerate(texts):
            ctx = text_contexts[i] if i < len(text_contexts) else ""
            if ctx:
                items.append({"text": text, "context": ctx})
            else:
                items.append({"text": text})

        return json.dumps({"items": items}, ensure_ascii=False)

    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
        text_contexts: list[str] | None = None,
        glossary: dict[str, str] | None = None,
        style: TranslationStyle | None = None,
    ) -> list[str]:
        """Translate a batch of texts using OpenAI.

        Args:
            texts: List of text strings to translate.
            target_lang: Target language code (e.g., 'es_ES', 'ja_JP').
            source_lang: Source language code. Defaults to 'en'.
            excluded_terms: List of terms that should not be translated.
            context: Additional context to improve translation quality.
            text_contexts: Per-text context for disambiguation.
            glossary: Optional preferred translations for specific phrases.
            style: Optional style/register for the translation.

        Returns:
            List of translated strings in the same order as input.

        Raises:
            TranslationError: If the API call fails or response is invalid.
        """
        if not texts:
            return []

        system_prompt = self._build_system_prompt(
            target_lang, source_lang, excluded_terms, context, glossary, style
        )

        # Build user message with optional per-text context
        user_content = self._build_user_message(texts, text_contexts)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
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
        model_name = self._model or self.DEFAULT_MODEL
        return ModelInfo(name=model_name, capabilities=["json_mode", "batch", "streaming"])
