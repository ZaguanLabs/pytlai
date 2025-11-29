# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-11-29

### Added

- **Core translator** (`Pytlai` class) with `process()`, `process_file()`, and `translate_text()` methods
- **HTML processing** — Extract text nodes, skip script/style/code, set lang/dir attributes
- **Python processing** — Translate docstrings and comments while preserving code structure
- **Cache backends**:
  - `InMemoryCache` — Simple in-memory cache with TTL
  - `RedisCache` — Distributed cache for multi-process deployments
  - `FileCache` — File-based cache for offline mode (JSON/YAML/PO)
- **AI providers**:
  - `OpenAIProvider` — GPT-4o/GPT-4o-mini with JSON mode
- **Export/Import**:
  - `TranslationExporter` — Export to JSON, YAML, PO, CSV
  - `TranslationImporter` — Import from JSON, YAML, PO, MO, CSV
  - `TranslationCatalog` — Load, merge, and query translations
- **Language support**:
  - 40+ languages with RTL detection
  - Language code normalization (en-US → en_US)
- **CLI commands**:
  - `pytlai translate` — Translate files
  - `pytlai extract` — Extract translatable strings
  - `pytlai export` — Export translations to language files
- **Configuration**:
  - `TranslationConfig` — Main configuration dataclass
  - `PythonOptions` — Python-specific translation options
  - Environment variable support (OPENAI_API_KEY, OPENAI_MODEL, etc.)
- **Documentation**:
  - Comprehensive README with examples
  - Example scripts for HTML, Python, and offline workflows
- **Context-aware translation** — Captures surrounding context to disambiguate ambiguous words
  - HTML: parent tag, CSS classes, sibling text, ancestor path
  - Python: function/class name, parameters, return type, surrounding code
- **Testing**:
  - 117 unit and integration tests
  - >80% code coverage on core functionality

### Notes

- Initial release
- Requires Python 3.10+
- OpenAI API key required for AI translations (not needed for cached/offline mode)

[Unreleased]: https://github.com/pytlai/pytlai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pytlai/pytlai/releases/tag/v0.1.0
