# Porting Guide: Implementing Tstlai in Other Languages

This document provides a complete specification for porting the Tstlai HTML translation engine to other programming languages (Python, PHP, Go, Ruby, etc.).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Structures](#data-structures)
4. [Algorithm Specifications](#algorithm-specifications)
5. [AI Provider Interface](#ai-provider-interface)
6. [Cache Interface](#cache-interface)
7. [HTML Processing](#html-processing)
8. [Language Reference](#language-reference)
9. [Implementation Checklist](#implementation-checklist)
10. [Example Implementations](#example-implementations)

---

## Architecture Overview

Tstlai is a content-aware HTML translation engine that:

1. **Parses HTML** and extracts translatable text nodes
2. **Hashes content** using SHA-256 for cache keys
3. **Checks cache** for existing translations
4. **Batches cache misses** and sends to an AI provider
5. **Caches new translations** for future use
6. **Reconstructs HTML** with translated content

```
┌─────────────────────────────────────────────────────────────────┐
│                         Tstlai Core                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  HTML    │───▶│    Text      │───▶│   Translation        │   │
│  │  Input   │    │  Extraction  │    │   Pipeline           │   │
│  └──────────┘    └──────────────┘    └──────────────────────┘   │
│                         │                      │                │
│                         ▼                      ▼                │
│                  ┌──────────────┐    ┌──────────────────────┐   │
│                  │   SHA-256    │    │   Cache Layer        │   │
│                  │   Hashing    │    │   (Memory/Redis)     │   │
│                  └──────────────┘    └──────────────────────┘   │
│                                               │                 │
│                                               ▼                 │
│                                      ┌──────────────────────┐   │
│                                      │   AI Provider        │   │
│                                      │   (OpenAI, etc.)     │   │
│                                      └──────────────────────┘   │
│                                               │                 │
│                                               ▼                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  HTML    │◀───│    Text      │◀───│   Translated         │   │
│  │  Output  │    │  Replacement │    │   Content            │   │
│  └──────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Main Translator Class (`Tstlai`)

The central orchestrator that:
- Holds configuration (target language, provider, cache settings)
- Manages the translation pipeline
- Coordinates between HTML processor, cache, and AI provider

### 2. HTML Processor

Responsible for:
- Parsing HTML into a DOM tree
- Extracting text nodes (skipping ignored tags)
- Applying translations back to the DOM
- Setting page attributes (`lang`, `dir`)

### 3. Cache Layer

Two implementations:
- **In-Memory Cache**: Simple key-value store with TTL
- **Redis Cache**: Distributed cache with TTL support

### 4. AI Provider

Interface for translation backends:
- **OpenAI Provider**: Uses GPT models with JSON mode
- Extensible for Anthropic, Google, or custom providers

---

## Data Structures

### TranslationConfig

```typescript
interface TranslationConfig {
  targetLang: string;           // e.g., "es_ES", "fr_FR"
  sourceLang?: string;          // Default: "en"
  provider: AIProviderConfig;
  cache?: CacheConfig;
  excludedTerms?: string[];     // Terms to never translate
  translationContext?: string;  // e.g., "Marketing site for B2B SaaS"
}
```

### AIProviderConfig

```typescript
interface AIProviderConfig {
  type: "openai" | "anthropic" | "google" | "custom";
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  timeout?: number;  // milliseconds
}
```

### CacheConfig

```typescript
interface CacheConfig {
  type: "memory" | "redis";
  ttl?: number;              // seconds, default: 3600
  connectionString?: string; // for Redis
  keyPrefix?: string;        // default: "tstlai:"
}
```

### TextNodeRef

Internal structure for tracking text nodes:

```typescript
interface TextNodeRef {
  id: string;        // Random ID for tracking
  text: string;      // Original text content
  hash: string;      // SHA-256 hash of trimmed text
  node: DOMNode;     // Reference to DOM node for mutation
}
```

### ProcessedPage (Return Value)

```typescript
interface ProcessedPage {
  html: string;           // Translated HTML
  translatedCount: number; // Number of newly translated items
  cachedCount: number;     // Number of cache hits
}
```

---

## Algorithm Specifications

### 1. Content Hashing

**Purpose**: Create a unique, deterministic key for each piece of content.

```
hash = SHA256(text.trim())
cacheKey = hash + ":" + targetLang
```

**Example**:
```
text = "Hello World"
hash = sha256("Hello World") = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
targetLang = "es_ES"
cacheKey = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e:es_ES"
```

### 2. Translation Pipeline

```python
def translate_batch(items: List[TextItem], target_lang: str) -> Dict[str, str]:
    translations = {}
    cache_misses = []
    
    # Step 1: Check cache for each item
    for item in items:
        cache_key = f"{item.hash}:{target_lang}"
        cached = cache.get(cache_key)
        if cached:
            translations[item.hash] = cached
        else:
            if item.hash not in [m.hash for m in cache_misses]:
                cache_misses.append(item)
    
    # Step 2: Translate cache misses via AI
    if cache_misses:
        texts = [m.text for m in cache_misses]
        translated = ai_provider.translate(texts, target_lang)
        
        # Step 3: Cache new translations
        for item, translation in zip(cache_misses, translated):
            translations[item.hash] = translation
            cache_key = f"{item.hash}:{target_lang}"
            cache.set(cache_key, translation)
    
    return translations
```

### 3. HTML Processing Algorithm

```python
IGNORED_TAGS = {"script", "style", "code", "pre", "textarea"}

def extract_text_nodes(root) -> List[TextNodeRef]:
    text_nodes = []
    
    def walk(node):
        # Skip ignored element tags
        if node.type == ELEMENT:
            if node.tag_name.lower() in IGNORED_TAGS:
                return
            if node.has_attribute("data-no-translate"):
                return
        
        # Collect text nodes
        if node.type == TEXT:
            text = node.text.strip()
            if len(text) > 0:
                text_nodes.append(TextNodeRef(
                    id=random_id(),
                    text=text,
                    hash=sha256(text),
                    node=node
                ))
        
        # Recurse into children
        for child in node.children:
            walk(child)
    
    walk(root)
    return text_nodes
```

### 4. RTL Language Detection

```python
RTL_LANGUAGES = {"ar", "he", "fa", "ur", "ps", "sd", "ug"}

def get_direction(target_lang: str) -> str:
    lang_code = target_lang.split("_")[0].lower()
    return "rtl" if lang_code in RTL_LANGUAGES else "ltr"
```

---

## AI Provider Interface

### Interface Definition

```typescript
interface AIProvider {
  translate(
    texts: string[],
    targetLang: string,
    excludedTerms?: string[],
    context?: string
  ): Promise<string[]>;
  
  getModelInfo(): { name: string; capabilities: string[] };
}
```

### OpenAI Implementation

**System Prompt Template**:

```
# Role
You are an expert native translator. You translate content to {TARGET_LANGUAGE} with the fluency and nuance of a highly educated native speaker.

# Context
{CONTEXT_IF_PROVIDED or "The content is general web content."}

# Task
Translate the provided texts into idiomatic {TARGET_LANGUAGE}.

# Style Guide
- **Natural Flow**: Avoid literal translations. Rephrase sentences to sound completely natural to a native speaker.
- **Vocabulary**: Use precise, culturally relevant terminology. Avoid awkward "translationese" or robotic phrasing.
- **Tone**: Maintain the original intent but adapt the wording to fit the target culture's expectations.
- **HTML Safety**: Do NOT translate HTML tags, class names, IDs, or attributes.
- **Interpolation**: Do NOT translate variables (e.g., {{name}}, {count}).

# Format
Return ONLY a JSON array of strings in the exact same order as the input.

# Exclusions (if any)
Do NOT translate the following terms. Keep them exactly as they appear in the source:
- {term1}
- {term2}
```

**API Call Parameters**:
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    { "role": "system", "content": "<system_prompt>" },
    { "role": "user", "content": "[\"Text 1\", \"Text 2\", \"Text 3\"]" }
  ],
  "temperature": 0.3,
  "response_format": { "type": "json_object" }
}
```

**Response Parsing**:
The API may return:
1. A direct JSON array: `["Translation 1", "Translation 2"]`
2. An object with a key: `{"translations": ["Translation 1", "Translation 2"]}`

Handle both cases by checking if the parsed result is an array or extracting the first array value from an object.

---

## Cache Interface

### Interface Definition

```typescript
interface TranslationCache {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
}
```

### In-Memory Implementation

```python
class InMemoryCache:
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[str]:
        timestamp = self.timestamps.get(key)
        if not timestamp:
            return None
        
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        self.cache[key] = value
        self.timestamps[key] = time.time()
```

### Redis Implementation

```python
class RedisCache:
    def __init__(self, connection_string=None, ttl=3600, key_prefix="tstlai:"):
        self.ttl = ttl
        self.prefix = key_prefix
        self.redis = Redis.from_url(connection_string or "redis://localhost:6379")
    
    def get(self, key: str) -> Optional[str]:
        return self.redis.get(self.prefix + key)
    
    def set(self, key: str, value: str):
        if self.ttl > 0:
            self.redis.setex(self.prefix + key, self.ttl, value)
        else:
            self.redis.set(self.prefix + key, value)
```

---

## HTML Processing

### Recommended Libraries by Language

| Language | HTML Parser Library |
|----------|---------------------|
| Python   | `lxml`, `BeautifulSoup4`, `html5lib` |
| PHP      | `DOMDocument`, `symfony/dom-crawler` |
| Go       | `golang.org/x/net/html`, `goquery` |
| Ruby     | `Nokogiri` |
| Java     | `Jsoup` |
| C#       | `HtmlAgilityPack`, `AngleSharp` |
| Rust     | `scraper`, `html5ever` |

### Key Requirements

1. **Parse HTML** into a mutable DOM tree
2. **Walk the tree** recursively
3. **Identify text nodes** (node type 3 in DOM spec)
4. **Mutate text content** in place
5. **Serialize back to HTML** string

### Ignored Tags

Never extract text from these elements:
- `<script>`
- `<style>`
- `<code>`
- `<pre>`
- `<textarea>`

Also skip any element with `data-no-translate` attribute.

---

## Language Reference

### Supported Locales

Use underscore format: `{language}_{REGION}` (e.g., `en_US`, `es_ES`)

**Tier 1 (High Quality)**:
`en_US`, `en_GB`, `de_DE`, `es_ES`, `es_MX`, `fr_FR`, `it_IT`, `ja_JP`, `pt_BR`, `pt_PT`, `zh_CN`, `zh_TW`

**Tier 2 (Good Quality)**:
`ar_SA`, `bn_BD`, `cs_CZ`, `da_DK`, `el_GR`, `fi_FI`, `he_IL`, `hi_IN`, `hu_HU`, `id_ID`, `ko_KR`, `nl_NL`, `nb_NO`, `pl_PL`, `ro_RO`, `ru_RU`, `sv_SE`, `th_TH`, `tr_TR`, `uk_UA`, `vi_VN`

**Tier 3 (Functional)**:
`bg_BG`, `ca_ES`, `fa_IR`, `hr_HR`, `lt_LT`, `lv_LV`, `ms_MY`, `sk_SK`, `sl_SI`, `sr_RS`, `sw_KE`, `tl_PH`, `ur_PK`

### Short Code Mapping

Map short codes to full locales:
```
en -> en_US    de -> de_DE    es -> es_ES    fr -> fr_FR
it -> it_IT    ja -> ja_JP    pt -> pt_BR    zh -> zh_CN
ko -> ko_KR    ru -> ru_RU    ar -> ar_SA    ...
```

### Language Name Mapping (for AI prompts)

```python
LANG_NAMES = {
    "en_US": "English (United States)",
    "es_ES": "Spanish (Spain)",
    "es_MX": "Spanish (Mexico)",
    "fr_FR": "French (France)",
    "de_DE": "German (Germany)",
    "ja_JP": "Japanese (Japan)",
    "zh_CN": "Chinese (Simplified)",
    "zh_TW": "Chinese (Traditional)",
    # ... etc
}
```

---

## Implementation Checklist

### Phase 1: Core

- [ ] **TranslationConfig** data structure
- [ ] **SHA-256 hashing** function
- [ ] **Cache interface** definition
- [ ] **InMemoryCache** implementation
- [ ] **AIProvider interface** definition
- [ ] **OpenAIProvider** implementation
- [ ] **translateBatch()** method

### Phase 2: HTML Processing

- [ ] **HTMLProcessor** class
- [ ] **parse()** method
- [ ] **extractTextNodes()** method with ignored tags
- [ ] **applyTranslations()** method
- [ ] **setPageAttributes()** for lang/dir

### Phase 3: Main Class

- [ ] **Tstlai** main class
- [ ] **process(html)** method
- [ ] **translateText(text)** single-text convenience method
- [ ] **isSourceLang()** bypass check
- [ ] RTL language detection

### Phase 4: Optional Enhancements

- [ ] **RedisCache** implementation
- [ ] **Request batching** with delay (50ms default)
- [ ] **Streaming translation** support
- [ ] **Framework integrations** (Express, FastAPI, etc.)

---

## Example Implementations

### Python Skeleton

```python
import hashlib
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import openai

class TranslationCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]: pass
    
    @abstractmethod
    def set(self, key: str, value: str) -> None: pass

class InMemoryCache(TranslationCache):
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, str] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[str]:
        import time
        ts = self._timestamps.get(key)
        if ts is None or time.time() - ts > self._ttl:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            return None
        return self._cache.get(key)
    
    def set(self, key: str, value: str) -> None:
        import time
        self._cache[key] = value
        self._timestamps[key] = time.time()

class AIProvider(ABC):
    @abstractmethod
    def translate(self, texts: List[str], target_lang: str,
                  excluded_terms: List[str] = None,
                  context: str = None) -> List[str]: pass

class OpenAIProvider(AIProvider):
    IGNORED_TAGS = {"script", "style", "code", "pre", "textarea"}
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def translate(self, texts: List[str], target_lang: str,
                  excluded_terms: List[str] = None,
                  context: str = None) -> List[str]:
        import json
        
        system_prompt = f"""# Role
You are an expert native translator. Translate to {target_lang}.

# Format
Return ONLY a JSON array of strings in the exact same order as input."""
        
        if excluded_terms:
            system_prompt += f"\n\n# Exclusions\nDo NOT translate: {', '.join(excluded_terms)}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(texts)}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        if isinstance(result, list):
            return result
        return result.get("translations", list(result.values())[0])

class Tstlai:
    RTL_LANGUAGES = {"ar", "he", "fa", "ur", "ps", "sd", "ug"}
    IGNORED_TAGS = {"script", "style", "code", "pre", "textarea"}
    
    def __init__(self, target_lang: str, provider: AIProvider,
                 cache: TranslationCache = None,
                 excluded_terms: List[str] = None,
                 context: str = None,
                 source_lang: str = "en"):
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.provider = provider
        self.cache = cache or InMemoryCache()
        self.excluded_terms = excluded_terms or []
        self.context = context
    
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode()).hexdigest()
    
    def _cache_key(self, hash: str, lang: str) -> str:
        return f"{hash}:{lang}"
    
    def process(self, html: str) -> Dict:
        soup = BeautifulSoup(html, "html.parser")
        text_nodes = []
        
        # Extract text nodes
        for element in soup.find_all(string=True):
            if element.parent.name in self.IGNORED_TAGS:
                continue
            if element.parent.has_attr("data-no-translate"):
                continue
            text = str(element).strip()
            if text:
                text_nodes.append({
                    "text": text,
                    "hash": self._hash(text),
                    "node": element
                })
        
        if not text_nodes:
            return {"html": html, "translated_count": 0, "cached_count": 0}
        
        # Translate batch
        translations, cached, translated = self._translate_batch(text_nodes)
        
        # Apply translations
        for ref in text_nodes:
            if ref["hash"] in translations:
                ref["node"].replace_with(translations[ref["hash"]])
        
        # Set page attributes
        html_tag = soup.find("html")
        if html_tag:
            html_tag["lang"] = self.target_lang
            lang_code = self.target_lang.split("_")[0].lower()
            html_tag["dir"] = "rtl" if lang_code in self.RTL_LANGUAGES else "ltr"
        
        return {
            "html": str(soup),
            "translated_count": translated,
            "cached_count": cached
        }
    
    def _translate_batch(self, items: List[Dict]) -> tuple:
        translations = {}
        cache_misses = []
        cached_count = 0
        
        # Check cache
        for item in items:
            key = self._cache_key(item["hash"], self.target_lang)
            cached = self.cache.get(key)
            if cached:
                translations[item["hash"]] = cached
                cached_count += 1
            elif item["hash"] not in [m["hash"] for m in cache_misses]:
                cache_misses.append(item)
        
        # Translate misses
        if cache_misses:
            texts = [m["text"] for m in cache_misses]
            results = self.provider.translate(
                texts, self.target_lang,
                self.excluded_terms, self.context
            )
            for item, result in zip(cache_misses, results):
                translations[item["hash"]] = result
                key = self._cache_key(item["hash"], self.target_lang)
                self.cache.set(key, result)
        
        return translations, cached_count, len(cache_misses)
```

### Go Skeleton

```go
package tstlai

import (
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "strings"
    "sync"
    "time"
    
    "golang.org/x/net/html"
)

// TranslationCache interface
type TranslationCache interface {
    Get(key string) (string, bool)
    Set(key string, value string)
}

// InMemoryCache implementation
type InMemoryCache struct {
    cache      map[string]string
    timestamps map[string]time.Time
    ttl        time.Duration
    mu         sync.RWMutex
}

func NewInMemoryCache(ttlSeconds int) *InMemoryCache {
    return &InMemoryCache{
        cache:      make(map[string]string),
        timestamps: make(map[string]time.Time),
        ttl:        time.Duration(ttlSeconds) * time.Second,
    }
}

func (c *InMemoryCache) Get(key string) (string, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    ts, ok := c.timestamps[key]
    if !ok || time.Since(ts) > c.ttl {
        return "", false
    }
    val, ok := c.cache[key]
    return val, ok
}

func (c *InMemoryCache) Set(key string, value string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.cache[key] = value
    c.timestamps[key] = time.Now()
}

// AIProvider interface
type AIProvider interface {
    Translate(texts []string, targetLang string, excludedTerms []string, context string) ([]string, error)
}

// TextNodeRef for tracking
type TextNodeRef struct {
    Text string
    Hash string
    Node *html.Node
}

// Tstlai main struct
type Tstlai struct {
    TargetLang    string
    SourceLang    string
    Provider      AIProvider
    Cache         TranslationCache
    ExcludedTerms []string
    Context       string
}

var ignoredTags = map[string]bool{
    "script": true, "style": true, "code": true, "pre": true, "textarea": true,
}

var rtlLanguages = map[string]bool{
    "ar": true, "he": true, "fa": true, "ur": true, "ps": true, "sd": true, "ug": true,
}

func hashText(text string) string {
    h := sha256.Sum256([]byte(strings.TrimSpace(text)))
    return hex.EncodeToString(h[:])
}

func (t *Tstlai) cacheKey(hash, lang string) string {
    return hash + ":" + lang
}

// Process translates HTML content
func (t *Tstlai) Process(htmlContent string) (string, int, int, error) {
    // Parse HTML, extract nodes, translate, reconstruct
    // Implementation follows the same pattern as TypeScript
    // ...
    return "", 0, 0, nil
}
```

### PHP Skeleton

```php
<?php

interface TranslationCache {
    public function get(string $key): ?string;
    public function set(string $key, string $value): void;
}

class InMemoryCache implements TranslationCache {
    private array $cache = [];
    private array $timestamps = [];
    private int $ttl;
    
    public function __construct(int $ttl = 3600) {
        $this->ttl = $ttl;
    }
    
    public function get(string $key): ?string {
        $ts = $this->timestamps[$key] ?? null;
        if ($ts === null || time() - $ts > $this->ttl) {
            unset($this->cache[$key], $this->timestamps[$key]);
            return null;
        }
        return $this->cache[$key] ?? null;
    }
    
    public function set(string $key, string $value): void {
        $this->cache[$key] = $value;
        $this->timestamps[$key] = time();
    }
}

interface AIProvider {
    public function translate(array $texts, string $targetLang, 
                              array $excludedTerms = [], ?string $context = null): array;
}

class Tstlai {
    private const IGNORED_TAGS = ['script', 'style', 'code', 'pre', 'textarea'];
    private const RTL_LANGUAGES = ['ar', 'he', 'fa', 'ur', 'ps', 'sd', 'ug'];
    
    private string $targetLang;
    private string $sourceLang;
    private AIProvider $provider;
    private TranslationCache $cache;
    private array $excludedTerms;
    private ?string $context;
    
    public function __construct(
        string $targetLang,
        AIProvider $provider,
        ?TranslationCache $cache = null,
        array $excludedTerms = [],
        ?string $context = null,
        string $sourceLang = 'en'
    ) {
        $this->targetLang = $targetLang;
        $this->sourceLang = $sourceLang;
        $this->provider = $provider;
        $this->cache = $cache ?? new InMemoryCache();
        $this->excludedTerms = $excludedTerms;
        $this->context = $context;
    }
    
    private function hash(string $text): string {
        return hash('sha256', trim($text));
    }
    
    private function cacheKey(string $hash, string $lang): string {
        return "{$hash}:{$lang}";
    }
    
    public function process(string $html): array {
        $dom = new DOMDocument();
        @$dom->loadHTML($html, LIBXML_HTML_NOIMPLIED | LIBXML_HTML_NODEFDTD);
        
        $textNodes = $this->extractTextNodes($dom);
        
        if (empty($textNodes)) {
            return ['html' => $html, 'translated_count' => 0, 'cached_count' => 0];
        }
        
        [$translations, $cachedCount, $translatedCount] = $this->translateBatch($textNodes);
        
        // Apply translations
        foreach ($textNodes as $ref) {
            if (isset($translations[$ref['hash']])) {
                $ref['node']->textContent = $translations[$ref['hash']];
            }
        }
        
        // Set page attributes
        $htmlTag = $dom->getElementsByTagName('html')->item(0);
        if ($htmlTag) {
            $htmlTag->setAttribute('lang', $this->targetLang);
            $langCode = explode('_', $this->targetLang)[0];
            $dir = in_array($langCode, self::RTL_LANGUAGES) ? 'rtl' : 'ltr';
            $htmlTag->setAttribute('dir', $dir);
        }
        
        return [
            'html' => $dom->saveHTML(),
            'translated_count' => $translatedCount,
            'cached_count' => $cachedCount
        ];
    }
    
    private function extractTextNodes(DOMDocument $dom): array {
        $xpath = new DOMXPath($dom);
        $textNodes = [];
        
        // Get all text nodes
        $nodes = $xpath->query('//text()');
        
        foreach ($nodes as $node) {
            // Skip ignored tags
            $parent = $node->parentNode;
            if ($parent && in_array(strtolower($parent->nodeName), self::IGNORED_TAGS)) {
                continue;
            }
            
            // Skip data-no-translate
            if ($parent instanceof DOMElement && $parent->hasAttribute('data-no-translate')) {
                continue;
            }
            
            $text = trim($node->textContent);
            if (!empty($text)) {
                $textNodes[] = [
                    'text' => $text,
                    'hash' => $this->hash($text),
                    'node' => $node
                ];
            }
        }
        
        return $textNodes;
    }
    
    private function translateBatch(array $items): array {
        $translations = [];
        $cacheMisses = [];
        $cachedCount = 0;
        $seenHashes = [];
        
        foreach ($items as $item) {
            $key = $this->cacheKey($item['hash'], $this->targetLang);
            $cached = $this->cache->get($key);
            
            if ($cached !== null) {
                $translations[$item['hash']] = $cached;
                $cachedCount++;
            } elseif (!isset($seenHashes[$item['hash']])) {
                $cacheMisses[] = $item;
                $seenHashes[$item['hash']] = true;
            }
        }
        
        if (!empty($cacheMisses)) {
            $texts = array_column($cacheMisses, 'text');
            $results = $this->provider->translate(
                $texts, 
                $this->targetLang, 
                $this->excludedTerms, 
                $this->context
            );
            
            foreach ($cacheMisses as $i => $item) {
                $translations[$item['hash']] = $results[$i];
                $key = $this->cacheKey($item['hash'], $this->targetLang);
                $this->cache->set($key, $results[$i]);
            }
        }
        
        return [$translations, $cachedCount, count($cacheMisses)];
    }
}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | Model to use | `gpt-3.5-turbo` |
| `OPENAI_BASE_URL` | API base URL | `https://api.openai.com/v1` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `TSTLAI_EXCLUDED_TEXT` | Comma-separated terms to exclude | - |
| `TSTLAI_DEBUG` | Enable debug logging | - |

---

## Testing Your Implementation

### Test Case 1: Basic Translation

```html
Input:
<div><h1>Hello World</h1><p>Welcome to our site.</p></div>

Expected behavior:
- Extract 2 text nodes: "Hello World", "Welcome to our site."
- Hash both, check cache
- Translate cache misses
- Return translated HTML
```

### Test Case 2: Ignored Tags

```html
Input:
<div>
  <p>Translate me</p>
  <script>doNotTranslate();</script>
  <code>const x = 1;</code>
</div>

Expected: Only "Translate me" is extracted and translated.
```

### Test Case 3: Cache Hit

```python
# First call
result1 = translator.process("<p>Hello</p>")  # API called

# Second call (same content)
result2 = translator.process("<p>Hello</p>")  # Cache hit, no API call
assert result2["cached_count"] == 1
```

### Test Case 4: RTL Language

```python
translator = Tstlai(target_lang="ar_SA", ...)
result = translator.process("<html><body>Hello</body></html>")
assert 'dir="rtl"' in result["html"]
assert 'lang="ar_SA"' in result["html"]
```

---

## Performance Considerations

1. **Batch translations** - Send multiple texts in one API call
2. **Deduplicate** - Don't translate the same text twice in a batch
3. **Cache aggressively** - Use Redis for distributed caching
4. **Async processing** - Use async/await or goroutines for cache lookups
5. **Connection pooling** - Reuse HTTP and Redis connections

---

## License

This porting guide is provided under the same license (MIT) as the Tstlai project.
