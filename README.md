# pytlai

**Python Translation AI** — Translate Python scripts or web pages on the fly to any human language.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **HTML Translation** — Extract text nodes, skip code/scripts, reconstruct with `lang`/`dir` attributes
- **Python Translation** — Translate docstrings and comments while preserving code structure
- **Smart Caching** — Memory, Redis, or file-based caching to avoid re-translating identical content
- **Offline Mode** — Export to JSON/YAML/PO files, ship with your project, no AI at runtime
- **40+ Languages** — Full support for major languages including RTL (Arabic, Hebrew, etc.)
- **Batch Processing** — Deduplicate and batch API calls for efficiency

## Installation

```bash
pip install pytlai
```

With optional dependencies:

```bash
pip install pytlai[redis]  # Redis cache support
pip install pytlai[yaml]   # YAML file support
pip install pytlai[all]    # All optional dependencies
```

## Quick Start

### Translate HTML

```python
from pytlai import Pytlai
from pytlai.providers import OpenAIProvider

translator = Pytlai(
    target_lang="es_ES",
    provider=OpenAIProvider(),
)

result = translator.process("<h1>Hello World</h1><p>Welcome to our site.</p>")
print(result.content)
# <html lang="es-ES" dir="ltr"><h1>Hola Mundo</h1><p>Bienvenido a nuestro sitio.</p></html>
```

### Translate Python Script

```python
from pytlai import Pytlai
from pytlai.providers import OpenAIProvider

translator = Pytlai(
    target_lang="ja_JP",
    provider=OpenAIProvider(),
)

code = '''
def greet(name):
    """Return a greeting message."""
    # Build the greeting
    return f"Hello, {name}!"
'''

result = translator.process(code, content_type="python")
print(result.content)
# def greet(name):
#     """挨拶メッセージを返します。"""
#     # 挨拶を作成する
#     return f"Hello, {name}!"
```

### Translate Files

```python
# Auto-detects content type from extension
result = translator.process_file("app.py")
result = translator.process_file("index.html")
```

## Command Line Interface

```bash
# Translate a file
pytlai translate index.html -l es_ES -o index_es.html

# Translate Python script
pytlai translate app.py -l fr_FR -o app_fr.py -t python

# Extract strings for translation
pytlai extract app.py -o strings.json

# Export to gettext PO format
pytlai export translations.json -o locale/es.po -f po
```

## Offline Workflow

For projects that want to ship translations without runtime AI dependencies:

```python
# 1. During development: translate and export
from pytlai import Pytlai
from pytlai.providers import OpenAIProvider
from pytlai.export import TranslationExporter

translator = Pytlai(target_lang="es_ES", provider=OpenAIProvider())
result = translator.process(content)

# Export translations
exporter = TranslationExporter()
exporter.export_from_cache(translator._cache, "locale/es_ES.json", target_lang="es_ES")
```

```python
# 2. At runtime: use cached translations (no AI needed)
from pytlai import Pytlai
from pytlai.cache import FileCache

translator = Pytlai(
    target_lang="es_ES",
    cache=FileCache("locale/es_ES.json"),
    provider=None,  # No AI provider needed
)
result = translator.process(content)
```

## Configuration

```python
from pytlai import Pytlai, TranslationConfig, PythonOptions
from pytlai.providers import OpenAIProvider
from pytlai.cache import RedisCache

config = TranslationConfig(
    target_lang="fr_FR",
    source_lang="en",
    excluded_terms=["API", "SDK", "pytlai"],  # Never translate these
    context="Technical documentation for a Python library",
    python_options=PythonOptions(
        translate_docstrings=True,
        translate_comments=True,
        translate_strings=False,
    ),
)

translator = Pytlai(
    config=config,
    provider=OpenAIProvider(model="gpt-4o"),
    cache=RedisCache(url="redis://localhost:6379"),
)
```

## Supported Languages

**Tier 1 (High Quality):** English, German, Spanish, French, Italian, Japanese, Portuguese, Chinese (Simplified/Traditional)

**Tier 2 (Good Quality):** Arabic, Bengali, Czech, Danish, Greek, Finnish, Hebrew, Hindi, Hungarian, Indonesian, Korean, Dutch, Norwegian, Polish, Romanian, Russian, Swedish, Thai, Turkish, Ukrainian, Vietnamese

**Tier 3 (Functional):** Bulgarian, Catalan, Persian, Croatian, Lithuanian, Latvian, Malay, Slovak, Slovenian, Serbian, Swahili, Filipino, Urdu

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | Custom API base URL | - |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |

## API Reference

### Pytlai

The main translator class.

```python
Pytlai(
    target_lang: str,              # Target language code (e.g., "es_ES")
    provider: AIProvider = None,   # AI provider (OpenAIProvider, etc.)
    cache: TranslationCache = None, # Cache backend (InMemoryCache, RedisCache, FileCache)
    config: TranslationConfig = None, # Full configuration object
    source_lang: str = "en",       # Source language code
    excluded_terms: list[str] = None, # Terms to never translate
    context: str = None,           # Context for AI translations
)
```

**Methods:**

- `process(content, content_type=None)` → `ProcessedContent` — Translate content
- `process_file(path, content_type=None)` → `ProcessedContent` — Translate a file
- `translate_text(text)` → `str` — Translate a single string

### ProcessedContent

Result of translation.

```python
ProcessedContent(
    content: str,          # Translated content
    translated_count: int, # Number of newly translated items
    cached_count: int,     # Number of cache hits
    total_nodes: int,      # Total translatable nodes found
)
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.
