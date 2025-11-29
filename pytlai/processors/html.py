"""HTML content processor for pytlai."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

from pytlai.languages import get_text_direction
from pytlai.processors.base import ContentProcessor, TextNode

if TYPE_CHECKING:
    from bs4 import BeautifulSoup, NavigableString


class HTMLProcessor(ContentProcessor):
    """Processor for HTML content.

    Extracts text nodes from HTML, skipping script, style, and other
    non-translatable elements. Reconstructs HTML with translations
    and sets appropriate lang/dir attributes.

    Attributes:
        ignored_tags: Set of tag names to skip during extraction.
    """

    IGNORED_TAGS: set[str] = {"script", "style", "code", "pre", "textarea", "svg"}

    def __init__(
        self,
        ignored_tags: set[str] | None = None,
        parser: str = "lxml",
    ) -> None:
        """Initialize the HTML processor.

        Args:
            ignored_tags: Additional tags to ignore. Merged with defaults.
            parser: BeautifulSoup parser to use. Default is 'lxml'.
        """
        self._ignored_tags = self.IGNORED_TAGS.copy()
        if ignored_tags:
            self._ignored_tags.update(ignored_tags)
        self._parser = parser

    def extract(self, content: str) -> tuple[BeautifulSoup, list[TextNode]]:
        """Extract translatable text nodes from HTML.

        Args:
            content: HTML string to process.

        Returns:
            Tuple of (BeautifulSoup object, list of TextNode objects).
        """
        from bs4 import BeautifulSoup, NavigableString

        soup = BeautifulSoup(content, self._parser)
        text_nodes: list[TextNode] = []
        node_map: dict[str, NavigableString] = {}

        # Find all text nodes
        for element in soup.find_all(string=True):
            # Skip if parent is an ignored tag
            if element.parent and element.parent.name in self._ignored_tags:
                continue

            # Skip if any ancestor has data-no-translate
            skip = False
            for parent in element.parents:
                if parent.get("data-no-translate") is not None:
                    skip = True
                    break
            if skip:
                continue

            # Skip empty or whitespace-only text
            text = str(element).strip()
            if not text:
                continue

            # Create text node
            node_id = str(uuid.uuid4())
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            text_node = TextNode(
                id=node_id,
                text=text,
                hash=text_hash,
                node_type="html_text",
                metadata={"parent_tag": element.parent.name if element.parent else None},
            )
            text_nodes.append(text_node)
            node_map[node_id] = element

        # Store node map in soup for apply()
        soup._pytlai_node_map = node_map  # type: ignore[attr-defined]

        return soup, text_nodes

    def apply(
        self,
        parsed: BeautifulSoup,
        nodes: list[TextNode],
        translations: dict[str, str],
        target_lang: str | None = None,
    ) -> str:
        """Apply translations to the HTML.

        Args:
            parsed: BeautifulSoup object from extract().
            nodes: Text nodes from extract().
            translations: Dictionary mapping text hashes to translated text.
            target_lang: Target language code for setting lang/dir attributes.

        Returns:
            Reconstructed HTML string with translations applied.
        """
        from bs4 import NavigableString

        node_map: dict[str, NavigableString] = getattr(parsed, "_pytlai_node_map", {})

        # Apply translations
        for text_node in nodes:
            if text_node.hash in translations:
                nav_string = node_map.get(text_node.id)
                if nav_string:
                    translated = translations[text_node.hash]
                    # Preserve leading/trailing whitespace from original
                    original = str(nav_string)
                    leading = original[: len(original) - len(original.lstrip())]
                    trailing = original[len(original.rstrip()) :]
                    nav_string.replace_with(NavigableString(leading + translated + trailing))

        # Set lang and dir attributes on <html> tag
        if target_lang:
            html_tag = parsed.find("html")
            if html_tag:
                html_tag["lang"] = target_lang.replace("_", "-")
                html_tag["dir"] = get_text_direction(target_lang)

        return str(parsed)

    def get_content_type(self) -> str:
        """Get the content type this processor handles.

        Returns:
            'html'
        """
        return "html"
