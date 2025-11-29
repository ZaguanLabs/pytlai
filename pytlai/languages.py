"""Language codes, names, and RTL detection for pytlai."""

# Right-to-left languages
RTL_LANGUAGES: set[str] = {"ar", "he", "fa", "ur", "ps", "sd", "ug"}

# Short code to full locale mapping
SHORT_CODE_MAP: dict[str, str] = {
    "en": "en_US",
    "de": "de_DE",
    "es": "es_ES",
    "fr": "fr_FR",
    "it": "it_IT",
    "ja": "ja_JP",
    "pt": "pt_BR",
    "zh": "zh_CN",
    "ko": "ko_KR",
    "ru": "ru_RU",
    "ar": "ar_SA",
    "he": "he_IL",
    "fa": "fa_IR",
    "hi": "hi_IN",
    "bn": "bn_BD",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "tr": "tr_TR",
    "vi": "vi_VN",
    "th": "th_TH",
    "sv": "sv_SE",
    "da": "da_DK",
    "fi": "fi_FI",
    "nb": "nb_NO",
    "no": "nb_NO",
    "cs": "cs_CZ",
    "el": "el_GR",
    "hu": "hu_HU",
    "ro": "ro_RO",
    "uk": "uk_UA",
    "id": "id_ID",
    "ms": "ms_MY",
    "tl": "tl_PH",
}

# Full language names for AI prompts
LANGUAGE_NAMES: dict[str, str] = {
    # Tier 1 - High Quality
    "en_US": "English (United States)",
    "en_GB": "English (United Kingdom)",
    "de_DE": "German (Germany)",
    "es_ES": "Spanish (Spain)",
    "es_MX": "Spanish (Mexico)",
    "fr_FR": "French (France)",
    "it_IT": "Italian (Italy)",
    "ja_JP": "Japanese (Japan)",
    "pt_BR": "Portuguese (Brazil)",
    "pt_PT": "Portuguese (Portugal)",
    "zh_CN": "Chinese (Simplified)",
    "zh_TW": "Chinese (Traditional)",
    # Tier 2 - Good Quality
    "ar_SA": "Arabic (Saudi Arabia)",
    "bn_BD": "Bengali (Bangladesh)",
    "cs_CZ": "Czech (Czech Republic)",
    "da_DK": "Danish (Denmark)",
    "el_GR": "Greek (Greece)",
    "fi_FI": "Finnish (Finland)",
    "he_IL": "Hebrew (Israel)",
    "hi_IN": "Hindi (India)",
    "hu_HU": "Hungarian (Hungary)",
    "id_ID": "Indonesian (Indonesia)",
    "ko_KR": "Korean (South Korea)",
    "nl_NL": "Dutch (Netherlands)",
    "nb_NO": "Norwegian Bokmål (Norway)",
    "pl_PL": "Polish (Poland)",
    "ro_RO": "Romanian (Romania)",
    "ru_RU": "Russian (Russia)",
    "sv_SE": "Swedish (Sweden)",
    "th_TH": "Thai (Thailand)",
    "tr_TR": "Turkish (Turkey)",
    "uk_UA": "Ukrainian (Ukraine)",
    "vi_VN": "Vietnamese (Vietnam)",
    # Tier 3 - Functional
    "bg_BG": "Bulgarian (Bulgaria)",
    "ca_ES": "Catalan (Spain)",
    "fa_IR": "Persian (Iran)",
    "hr_HR": "Croatian (Croatia)",
    "lt_LT": "Lithuanian (Lithuania)",
    "lv_LV": "Latvian (Latvia)",
    "ms_MY": "Malay (Malaysia)",
    "sk_SK": "Slovak (Slovakia)",
    "sl_SI": "Slovenian (Slovenia)",
    "sr_RS": "Serbian (Serbia)",
    "sw_KE": "Swahili (Kenya)",
    "tl_PH": "Filipino (Philippines)",
    "ur_PK": "Urdu (Pakistan)",
}


def get_text_direction(lang_code: str) -> str:
    """Get the text direction for a language code.

    Args:
        lang_code: Language code (e.g., 'ar_SA', 'en_US', or short 'ar', 'en').

    Returns:
        'rtl' for right-to-left languages, 'ltr' otherwise.
    """
    # Extract the primary language code
    primary = lang_code.split("_")[0].lower()
    return "rtl" if primary in RTL_LANGUAGES else "ltr"


def is_rtl(lang_code: str) -> bool:
    """Check if a language is right-to-left.

    Args:
        lang_code: Language code (e.g., 'ar_SA', 'he_IL', or short 'ar', 'he').

    Returns:
        True if the language is RTL, False otherwise.
    """
    return get_text_direction(lang_code) == "rtl"


def normalize_lang_code(lang_code: str) -> str:
    """Normalize a language code to full locale format.

    Args:
        lang_code: Language code in any format (e.g., 'en', 'en-US', 'en_US').

    Returns:
        Normalized language code in underscore format (e.g., 'en_US').
    """
    # Replace hyphens with underscores
    normalized = lang_code.replace("-", "_")

    # If it's a short code, expand it
    if "_" not in normalized:
        normalized = SHORT_CODE_MAP.get(normalized.lower(), f"{normalized}_{normalized.upper()}")

    return normalized


def get_language_name(lang_code: str) -> str:
    """Get the full language name for a language code.

    Args:
        lang_code: Language code (e.g., 'es_ES', 'ja_JP').

    Returns:
        Full language name (e.g., 'Spanish (Spain)', 'Japanese (Japan)').
        Falls back to the code itself if not found.
    """
    normalized = normalize_lang_code(lang_code)
    return LANGUAGE_NAMES.get(normalized, normalized)
