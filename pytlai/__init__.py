"""
pytlai - Python Translation AI

Translate Python scripts or web pages on the fly to any human language.
"""

from pytlai.config import (
    AIProviderConfig,
    CacheConfig,
    PythonOptions,
    TranslationConfig,
    TranslationStyle,
)
from pytlai.core import Pytlai
from pytlai.languages import (
    get_language_name,
    get_text_direction,
    is_rtl,
    normalize_lang_code,
)
from pytlai.processors.base import ProcessedContent

__version__ = "1.0.0"

__all__ = [
    # Main class
    "Pytlai",
    # Config
    "AIProviderConfig",
    "CacheConfig",
    "PythonOptions",
    "TranslationConfig",
    "TranslationStyle",
    # Results
    "ProcessedContent",
    # Language utilities
    "get_language_name",
    "get_text_direction",
    "is_rtl",
    "normalize_lang_code",
    # Version
    "__version__",
]
