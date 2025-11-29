# Improved Porting Guide: Building a Translation Engine

This guide incorporates lessons learned from porting Tstlai (TypeScript) to Pytlai (Python). It provides a more complete specification for implementing a translation engine in any language.

---

## Table of Contents

1. [Quick Start Checklist](#quick-start-checklist)
2. [Architecture Overview](#architecture-overview)
3. [Data Structures](#data-structures)
4. [Phase 1: Cache Layer](#phase-1-cache-layer)
5. [Phase 2: AI Provider](#phase-2-ai-provider)
6. [Phase 3: Content Processor](#phase-3-content-processor)
7. [Phase 4: Main Translator](#phase-4-main-translator)
8. [Phase 5: Error Handling](#phase-5-error-handling)
9. [Phase 6: Context Disambiguation](#phase-6-context-disambiguation)
10. [Phase 7: Testing](#phase-7-testing)
11. [Phase 8: Extensions](#phase-8-extensions)
12. [Language-Specific Notes](#language-specific-notes)
13. [Performance Optimization](#performance-optimization)

---

## Quick Start Checklist

Before starting, ensure you have:

- [ ] SHA-256 hashing library
- [ ] HTML parser with **mutable DOM** (not just parsing)
- [ ] HTTP client for API calls
- [ ] JSON parsing library
- [ ] Optional: Redis client library

**Minimum viable implementation**: ~500 lines of code

**Full implementation with extensions**: ~2,500 lines of code

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Translation Engine                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │   Content    │────▶│    Text      │────▶│   Translation        │ │
│  │   Input      │     │  Extraction  │     │   Pipeline           │ │
│  └──────────────┘     └──────────────┘     └──────────────────────┘ │
│         │                    │                       │               │
│         │                    ▼                       ▼               │
│         │             ┌──────────────┐     ┌──────────────────────┐ │
│         │             │   TextNode   │     │   Cache Layer        │ │
│         │             │   + Context  │     │   (check first)      │ │
│         │             └──────────────┘     └──────────────────────┘ │
│         │                    │                       │               │
│         │                    │              ┌────────┴────────┐      │
│         │                    │              │                 │      │
│         │                    │         cache hit         cache miss  │
│         │                    │              │                 │      │
│         │                    │              ▼                 ▼      │
│         │                    │     ┌──────────────┐  ┌─────────────┐│
│         │                    │     │   Return     │  │ AI Provider ││
│         │                    │     │   Cached     │  │ (batch call)││
│         │                    │     └──────────────┘  └─────────────┘│
│         │                    │              │                 │      │
│         │                    │              └────────┬────────┘      │
│         │                    │                       │               │
│         │                    │                       ▼               │
│         │                    │              ┌──────────────────────┐ │
│         │                    │              │   Store in Cache     │ │
│         │                    │              └──────────────────────┘ │
│         │                    │                       │               │
│         ▼                    ▼                       ▼               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │   Content    │◀────│    Text      │◀────│   Translations       │ │
│  │   Output     │     │  Replacement │     │   (hash → text)      │ │
│  └──────────────┘     └──────────────┘     └──────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Hash-based caching**: Same text → same hash → same cache key
2. **Batch translation**: Send multiple texts in one API call
3. **Deduplication**: Don't translate identical text twice
4. **Separation of concerns**: Cache, Provider, Processor are independent

---

## Data Structures

### TextNode

The fundamental unit of translatable content:

```python
@dataclass
class TextNode:
    id: str           # Unique identifier (UUID)
    text: str         # Original text content
    hash: str         # SHA-256 of text.strip()
    node_type: str    # "html_text", "docstring", "comment", etc.
    context: str      # Surrounding context for disambiguation
    metadata: dict    # Additional info (line number, parent tag, etc.)
```

**Why `context`?** Single words translate differently based on surroundings:
- "Run" in `<button>` → "Ejecutar" (execute)
- "Run" in sports article → "Correr" (physical running)

### TranslationConfig

```python
@dataclass
class TranslationConfig:
    target_lang: str              # e.g., "es_ES", "ja_JP"
    source_lang: str = "en"       # Source language
    excluded_terms: list[str]     # Never translate these (e.g., ["API", "SDK"])
    context: str | None           # Global context (e.g., "Technical documentation")
```

### ProcessedContent (Return Value)

```python
@dataclass
class ProcessedContent:
    content: str          # Translated content
    translated_count: int # Newly translated items
    cached_count: int     # Cache hits
    total_nodes: int      # Total translatable nodes found
```

---

## Phase 1: Cache Layer

### Interface

```python
from abc import ABC, abstractmethod

class TranslationCache(ABC):
    @abstractmethod
    def get(self, key: str) -> str | None:
        """Get cached translation. Returns None if not found or expired."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store translation in cache."""
        pass
    
    def __bool__(self) -> bool:
        """Allow empty caches to be truthy."""
        return True  # IMPORTANT: Don't return False for empty cache
```

### Cache Key Format

```python
def cache_key(text_hash: str, target_lang: str) -> str:
    return f"{text_hash}:{target_lang}"

# Example:
# hash = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
# target_lang = "es_ES"
# key = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e:es_ES"
```

**Advanced cache key** (if needed):
```python
def cache_key(text_hash: str, source_lang: str, target_lang: str, model: str) -> str:
    return f"{text_hash}:{source_lang}:{target_lang}:{model}"
```

### In-Memory Implementation

```python
import time

class InMemoryCache(TranslationCache):
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, str] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl_seconds
    
    def get(self, key: str) -> str | None:
        timestamp = self._timestamps.get(key)
        if timestamp is None:
            return None
        
        if time.time() - timestamp > self._ttl:
            # Expired - clean up
            del self._cache[key]
            del self._timestamps[key]
            return None
        
        return self._cache.get(key)
    
    def set(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def __len__(self) -> int:
        return len(self._cache)
```

### Redis Implementation

```python
import redis

class RedisCache(TranslationCache):
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        ttl_seconds: int = 3600,
        key_prefix: str = "translate:"
    ):
        self._client = redis.from_url(url)
        self._ttl = ttl_seconds
        self._prefix = key_prefix
    
    def get(self, key: str) -> str | None:
        value = self._client.get(self._prefix + key)
        return value.decode() if value else None
    
    def set(self, key: str, value: str) -> None:
        full_key = self._prefix + key
        if self._ttl > 0:
            self._client.setex(full_key, self._ttl, value)
        else:
            self._client.set(full_key, value)
```

---

## Phase 2: AI Provider

### Interface

```python
from abc import ABC, abstractmethod

class AIProvider(ABC):
    @abstractmethod
    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
        text_contexts: list[str] | None = None,  # Per-text context
    ) -> list[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of strings to translate
            target_lang: Target language code (e.g., "es_ES")
            source_lang: Source language code
            excluded_terms: Terms to never translate
            context: Global context for all texts
            text_contexts: Per-text context for disambiguation
        
        Returns:
            List of translated strings in same order as input
        
        Raises:
            TranslationError: If translation fails
        """
        pass
```

### OpenAI Implementation

```python
import json
import openai

class OpenAIProvider(AIProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        base_url: str | None = None,
    ):
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
    
    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
        text_contexts: list[str] | None = None,
    ) -> list[str]:
        if not texts:
            return []
        
        system_prompt = self._build_system_prompt(
            target_lang, source_lang, excluded_terms, context
        )
        user_content = self._build_user_message(texts, text_contexts)
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        
        return self._parse_response(response, len(texts))
    
    def _build_system_prompt(
        self,
        target_lang: str,
        source_lang: str,
        excluded_terms: list[str] | None,
        context: str | None,
    ) -> str:
        prompt = f"""# Role
You are an expert native translator. Translate from {source_lang} to {target_lang}.

# Context
{context or "General web or application content."}

# Style Guide
- Natural flow: Avoid literal translations
- Vocabulary: Use culturally relevant terminology
- Disambiguation: Use provided context to choose correct translation
  - "Run" in <button> → action verb (execute)
  - "Run" in sports → physical running
- HTML Safety: Do NOT translate tags, classes, IDs
- Variables: Do NOT translate {{name}}, {count}, %s, etc.

# Input Format
Either:
1. Array: ["text1", "text2"]
2. Object: {{"items": [{{"text": "Run", "context": "in <button>"}}]}}

# Output Format
Return ONLY: {{"translations": ["translated1", "translated2"]}}"""

        if excluded_terms:
            terms = "\n".join(f"- {term}" for term in excluded_terms)
            prompt += f"\n\n# Exclusions\nDo NOT translate:\n{terms}"
        
        return prompt
    
    def _build_user_message(
        self,
        texts: list[str],
        text_contexts: list[str] | None,
    ) -> str:
        if not text_contexts or not any(text_contexts):
            return json.dumps(texts, ensure_ascii=False)
        
        items = []
        for i, text in enumerate(texts):
            ctx = text_contexts[i] if i < len(text_contexts) else ""
            if ctx:
                items.append({"text": text, "context": ctx})
            else:
                items.append({"text": text})
        
        return json.dumps({"items": items}, ensure_ascii=False)
    
    def _parse_response(self, response, expected_count: int) -> list[str]:
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Handle different response formats
        if isinstance(result, list):
            translations = result
        elif isinstance(result, dict):
            if "translations" in result:
                translations = result["translations"]
            else:
                # Fallback: first array value
                for value in result.values():
                    if isinstance(value, list):
                        translations = value
                        break
                else:
                    raise TranslationError("No translations array in response")
        else:
            raise TranslationError(f"Unexpected response type: {type(result)}")
        
        if len(translations) != expected_count:
            raise TranslationError(
                f"Expected {expected_count} translations, got {len(translations)}"
            )
        
        return translations
```

---

## Phase 3: Content Processor

### Interface

```python
from abc import ABC, abstractmethod
from typing import Any

class ContentProcessor(ABC):
    @abstractmethod
    def extract(self, content: str) -> tuple[Any, list[TextNode]]:
        """
        Extract translatable text nodes from content.
        
        Args:
            content: Raw content (HTML, Python source, etc.)
        
        Returns:
            Tuple of (parsed_content, list_of_text_nodes)
            parsed_content is implementation-specific (DOM, AST, etc.)
        """
        pass
    
    @abstractmethod
    def apply(
        self,
        parsed: Any,
        nodes: list[TextNode],
        translations: dict[str, str],  # hash → translated_text
    ) -> str:
        """
        Apply translations back to the parsed content.
        
        Args:
            parsed: The parsed content from extract()
            nodes: The text nodes from extract()
            translations: Map of hash → translated text
        
        Returns:
            Reconstructed content with translations applied
        """
        pass
    
    @abstractmethod
    def get_content_type(self) -> str:
        """Return content type identifier (e.g., 'html', 'python')."""
        pass
```

### HTML Processor

```python
import hashlib
import uuid
from bs4 import BeautifulSoup, NavigableString

class HTMLProcessor(ContentProcessor):
    IGNORED_TAGS = {"script", "style", "code", "pre", "textarea", "noscript"}
    
    def __init__(self, ignored_tags: set[str] | None = None):
        self._ignored_tags = ignored_tags or self.IGNORED_TAGS
    
    def extract(self, content: str) -> tuple[BeautifulSoup, list[TextNode]]:
        soup = BeautifulSoup(content, "html.parser")
        text_nodes: list[TextNode] = []
        node_map: dict[str, NavigableString] = {}
        
        for element in soup.find_all(string=True):
            # Skip ignored tags
            if element.parent and element.parent.name in self._ignored_tags:
                continue
            
            # Skip data-no-translate
            if element.parent and element.parent.has_attr("data-no-translate"):
                continue
            
            # Skip whitespace-only
            text = str(element)
            if not text.strip():
                continue
            
            node_id = str(uuid.uuid4())
            text_hash = hashlib.sha256(text.strip().encode()).hexdigest()
            context = self._build_context(element)
            
            text_node = TextNode(
                id=node_id,
                text=text.strip(),
                hash=text_hash,
                node_type="html_text",
                context=context,
                metadata={"parent_tag": element.parent.name if element.parent else None},
            )
            text_nodes.append(text_node)
            node_map[node_id] = element
        
        # Store node_map for apply()
        soup._node_map = node_map
        return soup, text_nodes
    
    def apply(
        self,
        parsed: BeautifulSoup,
        nodes: list[TextNode],
        translations: dict[str, str],
    ) -> str:
        node_map = parsed._node_map
        
        for node in nodes:
            if node.hash not in translations:
                continue
            
            element = node_map.get(node.id)
            if element is None:
                continue
            
            translated = translations[node.hash]
            
            # Preserve original whitespace
            original = str(element)
            leading_ws = original[:len(original) - len(original.lstrip())]
            trailing_ws = original[len(original.rstrip()):]
            
            element.replace_with(leading_ws + translated + trailing_ws)
        
        return str(parsed)
    
    def _build_context(self, element: NavigableString) -> str:
        """Build context string for disambiguation."""
        context_parts: list[str] = []
        
        parent = element.parent
        if parent:
            # Parent tag with useful attributes
            tag_info = f"<{parent.name}>"
            if parent.get("class"):
                classes = " ".join(parent.get("class", []))
                tag_info = f"<{parent.name} class=\"{classes}\">"
            elif parent.get("id"):
                tag_info = f"<{parent.name} id=\"{parent.get('id')}\">"
            
            context_parts.append(f"in {tag_info}")
            
            # Sibling text (up to 3)
            siblings = []
            for sib in parent.children:
                if sib == element:
                    continue
                sib_text = sib.get_text(strip=True) if hasattr(sib, "get_text") else str(sib).strip()
                if sib_text and len(sib_text) < 100:
                    siblings.append(sib_text)
            if siblings:
                context_parts.append(f"with: {', '.join(siblings[:3])}")
            
            # Ancestor path (up to 3 levels)
            ancestors = []
            for i, ancestor in enumerate(parent.parents):
                if i >= 3:
                    break
                if ancestor.name and ancestor.name not in ("html", "body", "[document]"):
                    ancestors.append(ancestor.name)
            if ancestors:
                context_parts.append(f"inside: {' > '.join(reversed(ancestors))}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def get_content_type(self) -> str:
        return "html"
```

---

## Phase 4: Main Translator

```python
import hashlib
from typing import Literal

class Translator:
    RTL_LANGUAGES = {"ar", "he", "fa", "ur", "ps", "sd", "ug"}
    
    def __init__(
        self,
        target_lang: str,
        provider: AIProvider | None = None,
        cache: TranslationCache | None = None,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
    ):
        self.target_lang = target_lang
        self.source_lang = source_lang
        self._provider = provider
        # IMPORTANT: Use 'is not None' to allow empty caches
        self._cache = cache if cache is not None else InMemoryCache()
        self._excluded_terms = excluded_terms or []
        self._context = context
        
        # Register processors
        self._processors = {
            "html": HTMLProcessor(),
        }
    
    def process(
        self,
        content: str,
        content_type: Literal["html", "python"] | None = None,
    ) -> ProcessedContent:
        """Translate content and return result."""
        
        # Skip if source == target
        if self._is_source_lang():
            return ProcessedContent(
                content=content,
                translated_count=0,
                cached_count=0,
                total_nodes=0,
            )
        
        # Auto-detect or use specified type
        if content_type is None:
            content_type = self._detect_content_type(content)
        
        processor = self._processors.get(content_type)
        if processor is None:
            raise ValueError(f"Unknown content type: {content_type}")
        
        # Extract text nodes
        parsed, nodes = processor.extract(content)
        
        if not nodes:
            return ProcessedContent(
                content=content,
                translated_count=0,
                cached_count=0,
                total_nodes=0,
            )
        
        # Translate batch
        translations, cached_count, translated_count = self._translate_batch(nodes)
        
        # Apply translations
        result = processor.apply(parsed, nodes, translations)
        
        # Set HTML attributes if applicable
        if content_type == "html":
            result = self._set_html_attributes(result)
        
        return ProcessedContent(
            content=result,
            translated_count=translated_count,
            cached_count=cached_count,
            total_nodes=len(nodes),
        )
    
    def _translate_batch(
        self,
        nodes: list[TextNode],
    ) -> tuple[dict[str, str], int, int]:
        """Translate nodes, using cache where possible."""
        translations: dict[str, str] = {}
        cache_misses: list[TextNode] = []
        seen_hashes: set[str] = set()
        cached_count = 0
        
        # Check cache for each node
        for node in nodes:
            cache_key = f"{node.hash}:{self.target_lang}"
            cached = self._cache.get(cache_key)
            
            if cached is not None:
                translations[node.hash] = cached
                cached_count += 1
            elif node.hash not in seen_hashes:
                # Deduplicate
                cache_misses.append(node)
                seen_hashes.add(node.hash)
        
        # Translate cache misses via AI
        translated_count = 0
        if cache_misses and self._provider:
            texts = [node.text for node in cache_misses]
            text_contexts = [node.context for node in cache_misses]
            
            results = self._provider.translate(
                texts=texts,
                target_lang=self.target_lang,
                source_lang=self.source_lang,
                excluded_terms=self._excluded_terms,
                context=self._context,
                text_contexts=text_contexts if any(text_contexts) else None,
            )
            
            # Cache and store results
            for node, translated in zip(cache_misses, results):
                translations[node.hash] = translated
                cache_key = f"{node.hash}:{self.target_lang}"
                self._cache.set(cache_key, translated)
                translated_count += 1
        
        return translations, cached_count, translated_count
    
    def _is_source_lang(self) -> bool:
        """Check if target matches source (no translation needed)."""
        target = self.target_lang.split("_")[0].lower()
        source = self.source_lang.split("_")[0].lower()
        return target == source
    
    def _set_html_attributes(self, html: str) -> str:
        """Set lang and dir attributes on <html> tag."""
        soup = BeautifulSoup(html, "html.parser")
        html_tag = soup.find("html")
        
        if html_tag:
            html_tag["lang"] = self.target_lang.replace("_", "-")
            lang_code = self.target_lang.split("_")[0].lower()
            html_tag["dir"] = "rtl" if lang_code in self.RTL_LANGUAGES else "ltr"
        
        return str(soup)
    
    def _detect_content_type(self, content: str) -> str:
        """Auto-detect content type."""
        stripped = content.strip()
        
        if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html"):
            return "html"
        if "<" in stripped and ">" in stripped:
            return "html"
        
        raise ValueError("Cannot auto-detect content type")
```

---

## Phase 5: Error Handling

### Custom Exceptions

```python
class TranslationError(Exception):
    """Base exception for translation errors."""
    pass

class ProviderError(TranslationError):
    """AI provider error (API failure, rate limit, etc.)."""
    pass

class CacheError(TranslationError):
    """Cache operation error."""
    pass

class ProcessorError(TranslationError):
    """Content processing error (parse failure, etc.)."""
    pass
```

### Common Failure Modes

| Failure | Cause | Recovery |
|---------|-------|----------|
| API rate limit | Too many requests | Exponential backoff, retry |
| API timeout | Slow response | Retry with longer timeout |
| Invalid JSON | Malformed AI response | Retry, log warning |
| Count mismatch | AI returned wrong number | Raise error, don't cache |
| Parse error | Invalid HTML/content | Return original, log error |
| Cache unavailable | Redis down | Fall back to memory cache |

### Retry Logic

```python
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ProviderError, TimeoutError) as e:
                    last_error = e
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator
```

---

## Phase 6: Context Disambiguation

### Why Context Matters

| Word | Context | Correct Translation (Spanish) |
|------|---------|------------------------------|
| Run | `<button>` | Ejecutar |
| Run | sports article | Correr |
| Post | blog interface | Publicar |
| Post | mail context | Correo |
| Save | file dialog | Guardar |
| Save | banking app | Ahorrar |
| Check | checkbox | Marcar |
| Check | payment | Cheque |

### Context Building Strategies

**HTML Context:**
```python
def build_html_context(element) -> str:
    parts = []
    
    # 1. Parent tag with attributes
    if element.parent:
        tag = element.parent.name
        cls = element.parent.get("class", [])
        if cls:
            parts.append(f"in <{tag} class=\"{' '.join(cls)}\">")
        else:
            parts.append(f"in <{tag}>")
    
    # 2. Sibling text
    siblings = [sib.get_text(strip=True) for sib in element.parent.children 
                if sib != element and hasattr(sib, "get_text")]
    if siblings:
        parts.append(f"with: {', '.join(siblings[:3])}")
    
    # 3. Ancestor path
    ancestors = [a.name for a in element.parents if a.name not in ("html", "body")][:3]
    if ancestors:
        parts.append(f"inside: {' > '.join(reversed(ancestors))}")
    
    return " | ".join(parts)
```

**Python Docstring Context:**
```python
def build_docstring_context(node, node_type: str) -> str:
    parts = []
    
    if node_type == "function_docstring":
        parts.append(f"docstring for function '{node.name}'")
        args = [arg.arg for arg in node.args.args if arg.arg != "self"]
        if args:
            parts.append(f"parameters: {', '.join(args)}")
    elif node_type == "class_docstring":
        parts.append(f"docstring for class '{node.name}'")
    
    return " | ".join(parts)
```

---

## Phase 7: Testing

### Required Test Cases

```python
class TestTranslator:
    def test_basic_translation(self):
        """Single text node is translated."""
        translator = Translator("es_ES", provider=MockProvider())
        result = translator.process("<p>Hello</p>")
        assert "Hola" in result.content
        assert result.translated_count == 1
    
    def test_cache_hit(self):
        """Second call uses cache."""
        cache = InMemoryCache()
        translator = Translator("es_ES", provider=MockProvider(), cache=cache)
        
        translator.process("<p>Hello</p>")
        result = translator.process("<p>Hello</p>")
        
        assert result.cached_count == 1
        assert result.translated_count == 0
    
    def test_ignored_tags(self):
        """Script/style/code content is not translated."""
        translator = Translator("es_ES", provider=MockProvider())
        result = translator.process("<script>code</script><p>Hello</p>")
        
        assert "code" in result.content  # Unchanged
        assert result.total_nodes == 1   # Only <p> extracted
    
    def test_data_no_translate(self):
        """Elements with data-no-translate are skipped."""
        translator = Translator("es_ES", provider=MockProvider())
        result = translator.process('<p data-no-translate>Keep</p><p>Translate</p>')
        
        assert "Keep" in result.content
        assert result.total_nodes == 1
    
    def test_rtl_language(self):
        """RTL languages get dir='rtl' attribute."""
        translator = Translator("ar_SA", provider=MockProvider())
        result = translator.process("<html><body>Hello</body></html>")
        
        assert 'dir="rtl"' in result.content
        assert 'lang="ar-SA"' in result.content
    
    def test_whitespace_preserved(self):
        """Leading/trailing whitespace is preserved."""
        translator = Translator("es_ES", provider=MockProvider())
        result = translator.process("<p>  Hello  </p>")
        
        # Translation should have same whitespace pattern
        assert "  " in result.content
    
    def test_deduplication(self):
        """Identical texts are only translated once."""
        provider = MockProvider()
        translator = Translator("es_ES", provider=provider)
        
        translator.process("<p>Hello</p><p>Hello</p><p>Hello</p>")
        
        # Provider should only receive one "Hello"
        assert len(provider.last_texts) == 1
    
    def test_empty_content(self):
        """Empty content returns unchanged."""
        translator = Translator("es_ES", provider=MockProvider())
        result = translator.process("")
        
        assert result.content == ""
        assert result.total_nodes == 0
    
    def test_source_equals_target(self):
        """No translation when source == target."""
        provider = MockProvider()
        translator = Translator("en_US", provider=provider, source_lang="en")
        
        result = translator.process("<p>Hello</p>")
        
        assert provider.call_count == 0
        assert result.translated_count == 0
```

### Mock Provider

```python
class MockProvider(AIProvider):
    def __init__(self, translations: dict[str, str] | None = None):
        self.translations = translations or {"Hello": "Hola", "World": "Mundo"}
        self.call_count = 0
        self.last_texts: list[str] = []
    
    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "en",
        excluded_terms: list[str] | None = None,
        context: str | None = None,
        text_contexts: list[str] | None = None,
    ) -> list[str]:
        self.call_count += 1
        self.last_texts = texts
        return [self.translations.get(t, f"[{t}]") for t in texts]
```

---

## Phase 8: Extensions

### Export/Import

For offline mode, export translations to files:

```python
class TranslationExporter:
    def export_json(self, translations: dict, path: str, target_lang: str):
        data = {
            "meta": {
                "target_lang": target_lang,
                "generated": datetime.now().isoformat(),
            },
            "translations": {
                hash: {"source": src, "target": tgt}
                for hash, (src, tgt) in translations.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def export_po(self, translations: dict, path: str):
        with open(path, "w") as f:
            for hash, (source, target) in translations.items():
                f.write(f'#: {hash}\n')
                f.write(f'msgid "{source}"\n')
                f.write(f'msgstr "{target}"\n\n')
```

### CLI Interface

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Translation CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # translate command
    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument("input", help="Input file")
    translate_parser.add_argument("-o", "--output", help="Output file")
    translate_parser.add_argument("-l", "--lang", required=True, help="Target language")
    
    # extract command
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("input", help="Input file")
    extract_parser.add_argument("-o", "--output", help="Output JSON file")
    
    args = parser.parse_args()
    # ... handle commands
```

---

## Language-Specific Notes

### Python

- **HTML Parser**: Use BeautifulSoup4 with `html.parser` or `lxml`
- **Gotcha**: `NavigableString` is immutable; use `replace_with()`
- **Gotcha**: `bool([])` is `False`; use `is not None` for cache checks
- **AST**: Use `ast` module for Python source; `tokenize` for comments

### Go

- **HTML Parser**: Use `golang.org/x/net/html` or `goquery`
- **Concurrency**: Use goroutines for parallel cache lookups
- **Gotcha**: HTML parser doesn't preserve whitespace well

### PHP

- **HTML Parser**: Use `DOMDocument` with `loadHTML()`
- **Gotcha**: Use `@` to suppress HTML5 warnings
- **Gotcha**: `DOMDocument` may add DOCTYPE; use `LIBXML_HTML_NOIMPLIED`

### Ruby

- **HTML Parser**: Use Nokogiri
- **Gotcha**: Nokogiri modifies whitespace; may need custom handling

### Rust

- **HTML Parser**: Use `scraper` or `html5ever`
- **Gotcha**: Ownership makes in-place mutation complex

---

## Performance Optimization

### 1. Batch Translations

Always send multiple texts in one API call:
```python
# Bad: One call per text
for text in texts:
    result = provider.translate([text], lang)

# Good: One call for all texts
results = provider.translate(texts, lang)
```

### 2. Deduplicate Before API Call

```python
seen = set()
unique_texts = []
for text in texts:
    if text not in seen:
        unique_texts.append(text)
        seen.add(text)
```

### 3. Parallel Cache Lookups

```python
import asyncio

async def check_cache_parallel(keys: list[str]) -> dict[str, str]:
    tasks = [cache.get_async(key) for key in keys]
    results = await asyncio.gather(*tasks)
    return {k: v for k, v in zip(keys, results) if v is not None}
```

### 4. Connection Pooling

Reuse HTTP and Redis connections:
```python
# Create once, reuse
client = openai.OpenAI()
redis = redis.ConnectionPool(...)
```

### 5. Benchmarks

Target performance:
| Operation | Target |
|-----------|--------|
| Cache lookup (memory) | < 1ms |
| Cache lookup (Redis) | < 5ms |
| HTML parsing (1KB) | < 10ms |
| API call (10 texts) | < 2s |
| Full page (50 nodes) | < 5s |

---

## Conclusion

This guide provides a complete specification for implementing a translation engine. Key takeaways:

1. **Start with interfaces** - Define Cache, Provider, Processor ABCs first
2. **Test early** - Use mock providers for unit tests
3. **Handle edge cases** - Whitespace, errors, empty content
4. **Add context** - Disambiguation improves translation quality
5. **Cache aggressively** - Same text = same translation

For questions or contributions, see the main project repository.
