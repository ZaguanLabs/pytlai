# pytlai Implementation Plan

**Python Translation AI** — Translate Python scripts or web pages on the fly to any human language.

Extends the Tstlai HTML translation engine with Python source code support.

---

## Scope

### HTML Translation (from Tstlai)
- Parse HTML, extract text nodes
- Skip `<script>`, `<style>`, `<code>`, `<pre>`, `<textarea>`, `data-no-translate`
- Batch translate via AI, cache results
- Reconstruct HTML with `lang`/`dir` attributes

### Python Script Translation (new)
- Parse Python source using `ast` module
- Extract translatable content:
  - **Docstrings** — module, class, function/method docstrings
  - **Comments** — `#` line comments
  - **String literals** — optionally, user-facing strings (with heuristics or markers)
- Preserve code structure, indentation, and syntax
- Reconstruct valid Python source

### Existing Language File Support
- **Import PO/MO files** — read existing gettext translations as cache seed
- **Import JSON/YAML** — read existing i18n files from frameworks (Flask-Babel, etc.)
- **Translate missing keys** — identify untranslated strings, translate via AI, merge back
- **Export updated files** — write back to PO/JSON/YAML with new translations

### Web Framework Integration
- **Template strings** — extract from Jinja2, Django templates (future phase)
- **Mixed content** — handle pages with both `_()` wrapped and hard-coded strings
- **Fallback chain** — check language file → cache → AI provider

---

## Project Structure

```
pytlai/
├── pytlai/
│   ├── __init__.py           # Package exports
│   ├── core.py               # Main Pytlai class (orchestrator)
│   ├── config.py             # Configuration dataclasses
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── base.py           # TranslationCache ABC
│   │   ├── memory.py         # InMemoryCache
│   │   ├── redis.py          # RedisCache
│   │   └── file.py           # FileCache (JSON/YAML/PO)
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py           # AIProvider ABC
│   │   └── openai.py         # OpenAIProvider
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── base.py           # ContentProcessor ABC
│   │   ├── html.py           # HTMLProcessor class
│   │   └── python.py         # PythonProcessor class
│   ├── export.py             # TranslationExporter (JSON/YAML/PO/CSV)
│   ├── importers.py          # TranslationImporter (PO/MO/JSON/YAML)
│   ├── catalog.py            # TranslationCatalog (load/merge/lookup)
│   └── languages.py          # Language codes, names, RTL detection
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_cache.py
│   ├── test_providers.py
│   ├── test_html.py
│   ├── test_python.py
│   └── test_export.py
├── examples/
│   ├── translate_html.py
│   ├── translate_script.py
│   └── offline_workflow.py
├── pyproject.toml
├── README.md
└── docs/
    ├── porting-guide.md
    └── pytlai-plan.md
```

---

## Implementation Phases

### Phase 1: Foundation

| Step | File | Description |
|------|------|-------------|
| 1.1 | `pyproject.toml` | Project metadata, dependencies |
| 1.2 | `pytlai/config.py` | `TranslationConfig`, `AIProviderConfig`, `CacheConfig` dataclasses |
| 1.3 | `pytlai/languages.py` | Language codes, names, RTL detection, short-code mapping |

### Phase 2: Cache Layer

| Step | File | Description |
|------|------|-------------|
| 2.1 | `pytlai/cache/base.py` | `TranslationCache` abstract base class |
| 2.2 | `pytlai/cache/memory.py` | `InMemoryCache` with TTL support |
| 2.3 | `pytlai/cache/redis.py` | `RedisCache` with TTL and key prefix |
| 2.4 | `pytlai/cache/file.py` | `FileCache` for JSON/YAML/PO language files |

### Phase 3: AI Providers

| Step | File | Description |
|------|------|-------------|
| 3.1 | `pytlai/providers/base.py` | `AIProvider` abstract base class |
| 3.2 | `pytlai/providers/openai.py` | `OpenAIProvider` with JSON mode, system prompt, response parsing |

### Phase 4: Content Processors

| Step | File | Description |
|------|------|-------------|
| 4.1 | `pytlai/processors/base.py` | `ContentProcessor` ABC with `extract()` and `apply()` methods |
| 4.2 | `pytlai/processors/html.py` | `HTMLProcessor`: parse HTML, extract text nodes, reconstruct |
| 4.3 | `pytlai/processors/python.py` | `PythonProcessor`: parse Python, extract docstrings/comments, reconstruct |

### Phase 5: Core Translator

| Step | File | Description |
|------|------|-------------|
| 5.1 | `pytlai/core.py` | `Pytlai` main class: `process()`, `process_file()`, `translate_text()` |
| 5.2 | `pytlai/__init__.py` | Public API exports |

### Phase 5b: Export & Offline Support

| Step | File | Description |
|------|------|-------------|
| 5b.1 | `pytlai/export.py` | `TranslationExporter`: export to JSON, YAML, PO/POT, CSV |
| 5b.2 | `pytlai/catalog.py` | `TranslationCatalog`: load/merge language files, lookup translations |
| 5b.3 | `pytlai/importers.py` | `TranslationImporter`: import existing PO/MO/JSON/YAML as cache seed |

### Phase 6: Testing

| Step | File | Description |
|------|------|-------------|
| 6.1 | `tests/test_cache.py` | Unit tests for cache implementations (memory, redis, file) |
| 6.2 | `tests/test_html.py` | Unit tests for HTML extraction and reconstruction |
| 6.3 | `tests/test_python.py` | Unit tests for Python docstring/comment extraction |
| 6.4 | `tests/test_export.py` | Unit tests for export formats (JSON, YAML, PO, CSV) |
| 6.5 | `tests/test_core.py` | Integration tests for full translation pipeline |

### Phase 7: CLI (Optional)

| Step | File | Description |
|------|------|-------------|
| 7.1 | `pytlai/cli.py` | Command-line interface: `extract`, `translate`, `export` commands |

### Phase 8: Documentation & Examples

| Step | File | Description |
|------|------|-------------|
| 8.1 | `README.md` | Installation, quick start, API reference |
| 8.2 | `examples/translate_html.py` | HTML translation example |
| 8.3 | `examples/translate_script.py` | Python script translation example |
| 8.4 | `examples/offline_workflow.py` | Extract → translate → export → use offline |

---

## Key Algorithms

### Content Hashing
```python
cache_key = sha256(text.strip()) + ":" + target_lang
```

### Translation Pipeline
1. Detect content type (HTML or Python) or use explicit processor
2. Extract translatable content via processor
3. Hash each text item
4. Check cache for existing translations
5. Deduplicate cache misses
6. Batch translate via AI provider
7. Cache new translations
8. Apply translations back to source
9. Return reconstructed content

### HTML Processing
- **Ignored tags**: `script`, `style`, `code`, `pre`, `textarea`
- **Skip attribute**: `data-no-translate`
- **Output**: Set `lang` and `dir` attributes on `<html>`

### Python Processing
- **Docstrings**: Module, class, function/method docstrings (triple-quoted strings as first statement)
- **Comments**: Lines starting with `#` (excluding shebangs, encoding declarations, type comments)
- **String literals** (optional): Strings marked with `# translate` comment or `_()` wrapper
- **Preserve**: All code structure, indentation, line numbers where possible

### RTL Languages
`ar`, `he`, `fa`, `ur`, `ps`, `sd`, `ug`

---

## Language File Formats

### JSON (default)
```json
{
  "meta": {
    "source_lang": "en",
    "target_lang": "es_ES",
    "generated": "2025-11-29T10:00:00Z"
  },
  "translations": {
    "a591a6d40bf420...": {
      "source": "Hello World",
      "target": "Hola Mundo"
    }
  }
}
```

### YAML
```yaml
meta:
  source_lang: en
  target_lang: es_ES
translations:
  a591a6d40bf420...:
    source: Hello World
    target: Hola Mundo
```

### PO (gettext-compatible)
```po
msgid "Hello World"
msgstr "Hola Mundo"
```

### CSV
```csv
hash,source,target
a591a6d40bf420...,Hello World,Hola Mundo
```

---

## Offline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Development Time                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Extract strings  pytlai extract app.py --output strings.json│
│  2. Translate        pytlai translate strings.json --lang es_ES │
│  3. Export           pytlai export --format po --output locale/ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Runtime (Offline)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  from pytlai import Pytlai                                      │
│  from pytlai.cache import FileCache                             │
│                                                                 │
│  translator = Pytlai(                                           │
│      target_lang="es_ES",                                       │
│      cache=FileCache("locale/es_ES.json"),  # No AI needed      │
│  )                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

```toml
[project]
dependencies = [
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "openai>=1.0",
]

[project.optional-dependencies]
redis = ["redis>=5.0"]
yaml = ["pyyaml>=6.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "mypy>=1.0", "ruff>=0.1"]
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | Model to use (default: `gpt-4o-mini`) |
| `OPENAI_BASE_URL` | Custom API base URL |
| `REDIS_URL` | Redis connection string |

---

## API Design

### Translate HTML

```python
from pytlai import Pytlai
from pytlai.providers import OpenAIProvider

translator = Pytlai(
    target_lang="es_ES",
    provider=OpenAIProvider(),
)

result = translator.process("<h1>Hello World</h1>")
print(result.content)  # <h1>Hola Mundo</h1>
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

### Translate File

```python
# Auto-detects content type from extension
result = translator.process_file("app.py")
result = translator.process_file("index.html")
```

### With Full Configuration

```python
from pytlai import Pytlai, TranslationConfig
from pytlai.providers import OpenAIProvider
from pytlai.cache import RedisCache

config = TranslationConfig(
    target_lang="fr_FR",
    source_lang="en",
    excluded_terms=["Pytlai", "API"],
    context="Technical documentation for a Python library",
    python_options={
        "translate_docstrings": True,
        "translate_comments": True,
        "translate_strings": False,  # Only marked strings
    },
)

translator = Pytlai(
    config=config,
    provider=OpenAIProvider(model="gpt-4o"),
    cache=RedisCache(url="redis://localhost:6379"),
)
```

---

## Project Guidelines

### Version Control
- **Commit after each successful phase** — don't push until ready for review
- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Tag releases with semantic versioning: `v0.1.0`, `v0.2.0`, etc.

### Documentation
- **Keep documentation up to date** with every code change
- `README.md` — installation, quick start, basic usage
- `docs/` — detailed guides, API reference, examples
- Docstrings on all public classes, methods, and functions
- Type hints throughout the codebase

### Changelog
- **Maintain `CHANGELOG.md`** following [Keep a Changelog](https://keepachangelog.com/) format
- Update on every release with: Added, Changed, Deprecated, Removed, Fixed, Security
- Link to relevant issues/PRs where applicable

### Code Quality
- Run tests before committing: `pytest`
- Type checking with `mypy` (add to dev dependencies)
- Linting with `ruff` (add to dev dependencies)
- Format with `ruff format` or `black`
- Aim for >80% test coverage on core functionality

### Dependencies
- Pin minimum versions, not exact versions
- Keep optional dependencies truly optional (redis, yaml)
- Document which features require which optional deps

### Error Handling
- Raise descriptive exceptions with actionable messages
- Never silently swallow errors
- Log warnings for recoverable issues

---

## Next Steps

1. Confirm this plan meets your requirements
2. Begin Phase 1 implementation
3. Iterate through phases with tests at each step
