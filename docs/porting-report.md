# Porting Report: Tstlai (TypeScript) → Pytlai (Python)

This document describes the experience of porting the Tstlai HTML translation engine from TypeScript to Python, following the porting guide. It covers what worked well, what was problematic, and recommendations for improving the porting process.

---

## Project Overview

| Aspect | Tstlai (Source) | Pytlai (Target) |
|--------|-----------------|-----------------|
| Language | TypeScript | Python 3.10+ |
| HTML Parser | Cheerio | BeautifulSoup4 |
| HTTP Client | fetch/axios | openai SDK |
| Cache | Map / ioredis | dict / redis-py |
| Package Manager | npm | pip / pyproject.toml |
| Test Framework | Jest/Vitest | pytest |

**Porting Duration**: ~8 phases over multiple sessions

**Final Result**: 117 tests passing, full feature parity plus extensions (Python source translation, export/import, CLI)

---

## What Worked Well

### 1. Clear Architecture Separation

The porting guide's separation of concerns made translation straightforward:

```
Cache Layer     → Direct 1:1 mapping (interface + implementations)
AI Provider     → Direct 1:1 mapping (interface + OpenAI implementation)
HTML Processor  → Direct 1:1 mapping (extract/apply pattern)
Main Class      → Direct 1:1 mapping (orchestrator pattern)
```

Each component could be ported and tested independently before integration.

### 2. Algorithm Specifications Were Precise

The pseudocode in the porting guide translated almost directly to Python:

**Guide specification:**
```python
cache_key = f"{item.hash}:{target_lang}"
cached = cache.get(cache_key)
```

**Actual implementation:**
```python
cache_key = f"{node.hash}:{self.target_lang}"
cached = self._cache.get(cache_key)
```

The SHA-256 hashing, cache key format, and batch translation algorithm required zero interpretation.

### 3. Test Cases Provided Clear Acceptance Criteria

The guide's test cases (basic translation, ignored tags, cache hit, RTL) served as an initial test suite skeleton. This made TDD practical from day one.

### 4. Interface-First Design

Defining abstract base classes first (`TranslationCache`, `AIProvider`, `ContentProcessor`) before implementations allowed:
- Parallel development of cache backends
- Easy mocking in tests
- Clear contracts between components

### 5. Language/Library Recommendations

The guide's recommendation of BeautifulSoup4 for Python HTML parsing was spot-on. It provided:
- Mutable DOM tree (essential for in-place text replacement)
- NavigableString for text node access
- Simple serialization back to HTML

---

## What Was Problematic

### 1. Response Format Ambiguity

**Problem**: The guide mentioned two possible API response formats but didn't specify which to prefer or how to handle edge cases.

```json
// Format 1: Direct array
["Translation 1", "Translation 2"]

// Format 2: Object wrapper
{"translations": ["Translation 1", "Translation 2"]}
```

**Solution**: We standardized on requesting `{"translations": [...]}` via the system prompt and added fallback parsing:

```python
if isinstance(result, dict):
    if "translations" in result:
        return result["translations"]
    # Fallback: first array value in the dict
    for value in result.values():
        if isinstance(value, list):
            return value
```

**Recommendation**: Specify the exact expected format and provide robust parsing code.

### 2. Whitespace Preservation Not Addressed

**Problem**: The guide didn't mention how to handle leading/trailing whitespace in text nodes.

```html
<p>  Hello World  </p>
```

Should the translation preserve the spaces? The guide was silent on this.

**Solution**: We preserve original whitespace by storing it and reapplying after translation:

```python
leading = len(original) - len(original.lstrip())
trailing = len(original) - len(original.rstrip())
translated_with_ws = original[:leading] + translated + original[-trailing:] if trailing else original[:leading] + translated
```

**Recommendation**: Add a section on whitespace handling with clear examples.

### 3. Error Handling Underspecified

**Problem**: The guide didn't cover:
- What happens when the AI returns fewer translations than requested?
- How to handle API rate limits?
- What to do with malformed JSON responses?

**Solution**: We added defensive checks and custom exceptions:

```python
if len(results) != len(texts):
    raise TranslationError(
        f"Expected {len(texts)} translations, got {len(results)}"
    )
```

**Recommendation**: Add an "Error Handling" section with common failure modes and recovery strategies.

### 4. No Guidance on Extending Beyond HTML

**Problem**: The guide was HTML-only. When extending to Python source code, we had to design:
- New processor interface methods
- AST-based extraction
- Comment tokenization
- Docstring location tracking

**Solution**: We generalized the `ContentProcessor` interface:

```python
class ContentProcessor(ABC):
    @abstractmethod
    def extract(self, content: str) -> tuple[Any, list[TextNode]]: ...
    
    @abstractmethod
    def apply(self, parsed: Any, nodes: list[TextNode], translations: dict[str, str]) -> str: ...
```

**Recommendation**: The guide should mention extensibility patterns for non-HTML content.

### 5. Context Disambiguation Not Mentioned

**Problem**: Single words like "Run", "Post", "Save" can translate differently based on context. The guide didn't address this.

**Solution**: We added a `context` field to `TextNode` and capture surrounding information:

```python
@dataclass
class TextNode:
    id: str
    text: str
    hash: str
    node_type: str = "text"
    context: str = ""  # NEW: "in <button> | with: Cancel | inside: nav"
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Recommendation**: Add a section on context-aware translation with examples.

### 6. Cache Key Collision Risk

**Problem**: The cache key format `{hash}:{target_lang}` doesn't account for:
- Different source languages (same text, different source → different translation)
- Different AI models (GPT-4 vs GPT-3.5 may translate differently)
- Different contexts (same word in different contexts)

**Solution**: For now, we kept the simple format but documented the limitation. A more robust format would be:

```python
cache_key = f"{hash}:{source_lang}:{target_lang}:{model}"
```

**Recommendation**: Discuss cache key design trade-offs and when to include additional dimensions.

---

## Unexpected Challenges

### 1. BeautifulSoup Text Node Mutation

BeautifulSoup's `NavigableString` objects are immutable. You can't do:
```python
text_node.string = "new value"  # Doesn't work
```

Instead, you must use `replace_with()`:
```python
text_node.replace_with("new value")
```

This wasn't obvious from the guide's DOM mutation examples.

### 2. Python AST Doesn't Preserve Comments

Python's `ast` module parses code structure but discards comments entirely. We had to use the `tokenize` module separately to extract comments, then correlate line numbers.

### 3. Docstring Location Tracking

Triple-quoted strings in Python can span multiple lines. Tracking their exact start/end positions for replacement required careful handling of `ast.Constant` nodes and their `lineno`/`end_lineno` attributes.

### 4. Empty Cache Truthiness

Python's `bool([])` returns `False`. This caused a subtle bug:

```python
# Bug: empty cache is replaced with new InMemoryCache
self._cache = cache or InMemoryCache()

# Fix: explicit None check
self._cache = cache if cache is not None else InMemoryCache()
```

---

## Extensions Beyond the Guide

Pytlai extends beyond the original Tstlai scope:

| Feature | Tstlai | Pytlai |
|---------|--------|--------|
| HTML translation | ✓ | ✓ |
| Python docstrings | ✗ | ✓ |
| Python comments | ✗ | ✓ |
| Export to PO/JSON/YAML/CSV | ✗ | ✓ |
| Import from PO/MO/JSON/YAML | ✗ | ✓ |
| Translation catalog | ✗ | ✓ |
| CLI interface | ✗ | ✓ |
| Context disambiguation | ✗ | ✓ |
| File-based cache | ✗ | ✓ |

---

## Metrics

| Metric | Value |
|--------|-------|
| Total Python files | 18 |
| Lines of code | ~2,500 |
| Test files | 6 |
| Test cases | 117 |
| Test coverage | >80% (core) |
| Dependencies | 4 required, 3 optional |

---

## Recommendations for the Porting Guide

### High Priority

1. **Add error handling section** with common failure modes
2. **Specify exact JSON response format** expected from AI
3. **Add whitespace preservation examples**
4. **Include cache key design discussion**

### Medium Priority

5. **Add extensibility section** for non-HTML content
6. **Include context disambiguation** as an optional feature
7. **Provide more language-specific gotchas** (Python, Go, PHP)
8. **Add performance benchmarks** as targets

### Low Priority

9. **Add export/import specifications** for language files
10. **Include CLI design patterns**
11. **Add logging/debugging recommendations**

---

## Conclusion

The porting guide provided an excellent foundation. The architecture was clean, algorithms were precise, and the phased approach worked well. The main gaps were around edge cases (whitespace, errors, context) and extensibility beyond HTML.

The resulting Pytlai implementation is more feature-rich than the original Tstlai, demonstrating that a well-designed core architecture enables natural extension.

**Time saved by the guide**: Estimated 60-70% compared to designing from scratch.

**Confidence in correctness**: High, due to clear specifications and test cases.

---

## Appendix: File Mapping

| Tstlai (TypeScript) | Pytlai (Python) |
|---------------------|-----------------|
| `src/index.ts` | `pytlai/__init__.py` |
| `src/tstlai.ts` | `pytlai/core.py` |
| `src/config.ts` | `pytlai/config.py` |
| `src/cache/memory.ts` | `pytlai/cache/memory.py` |
| `src/cache/redis.ts` | `pytlai/cache/redis.py` |
| `src/providers/openai.ts` | `pytlai/providers/openai.py` |
| `src/html-processor.ts` | `pytlai/processors/html.py` |
| - | `pytlai/processors/python.py` (new) |
| - | `pytlai/export.py` (new) |
| - | `pytlai/importers.py` (new) |
| - | `pytlai/catalog.py` (new) |
| - | `pytlai/cli.py` (new) |
