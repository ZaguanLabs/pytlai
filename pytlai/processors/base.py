"""Abstract base class for content processors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextNode:
    """A translatable text node extracted from content.

    Attributes:
        id: Unique identifier for this node.
        text: The original text content.
        hash: SHA-256 hash of the trimmed text.
        node_type: Type of node (e.g., 'text', 'docstring', 'comment').
        metadata: Additional metadata about the node (line number, etc.).
    """

    id: str
    text: str
    hash: str
    node_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedContent:
    """Result of processing content for translation.

    Attributes:
        content: The reconstructed content with translations applied.
        translated_count: Number of newly translated items.
        cached_count: Number of items retrieved from cache.
        total_nodes: Total number of translatable nodes found.
    """

    content: str
    translated_count: int = 0
    cached_count: int = 0
    total_nodes: int = 0


class ContentProcessor(ABC):
    """Abstract base class for content processors.

    Content processors are responsible for:
    1. Parsing content (HTML, Python, etc.)
    2. Extracting translatable text nodes
    3. Applying translations back to the content
    4. Reconstructing the final output

    Subclasses must implement extract() and apply() methods.
    """

    @abstractmethod
    def extract(self, content: str) -> tuple[Any, list[TextNode]]:
        """Extract translatable text nodes from content.

        Args:
            content: The raw content string (HTML, Python source, etc.).

        Returns:
            A tuple of (parsed_content, text_nodes) where:
            - parsed_content: The parsed representation (DOM, AST, etc.)
            - text_nodes: List of TextNode objects to translate
        """
        ...

    @abstractmethod
    def apply(
        self,
        parsed: Any,
        nodes: list[TextNode],
        translations: dict[str, str],
    ) -> str:
        """Apply translations to the parsed content.

        Args:
            parsed: The parsed content from extract().
            nodes: The text nodes from extract().
            translations: Dictionary mapping text hashes to translated text.

        Returns:
            The reconstructed content string with translations applied.
        """
        ...

    def get_content_type(self) -> str:
        """Get the content type this processor handles.

        Returns:
            Content type identifier (e.g., 'html', 'python').
        """
        return "unknown"
