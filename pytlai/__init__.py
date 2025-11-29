"""
pytlai - Python Translation AI

Translate Python scripts or web pages on the fly to any human language.
"""

from pytlai.config import (
    AIProviderConfig,
    CacheConfig,
    PythonOptions,
    TranslationConfig,
)
from pytlai.languages import (
    get_language_name,
    get_text_direction,
    is_rtl,
    normalize_lang_code,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AIProviderConfig",
    "CacheConfig",
    "PythonOptions",
    "TranslationConfig",
    "get_language_name",
    "get_text_direction",
    "is_rtl",
    "normalize_lang_code",
]
