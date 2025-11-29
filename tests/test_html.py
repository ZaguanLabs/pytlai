"""Tests for HTML processor."""

import hashlib

import pytest

from pytlai.processors.html import HTMLProcessor


class TestHTMLProcessor:
    """Tests for HTMLProcessor."""

    def test_extract_simple_text(self) -> None:
        """Test extracting text from simple HTML."""
        processor = HTMLProcessor()
        html = "<p>Hello World</p>"

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 1
        assert nodes[0].text == "Hello World"
        assert nodes[0].node_type == "html_text"

    def test_extract_multiple_nodes(self) -> None:
        """Test extracting multiple text nodes."""
        processor = HTMLProcessor()
        html = "<div><h1>Title</h1><p>Paragraph</p></div>"

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 2
        texts = {n.text for n in nodes}
        assert texts == {"Title", "Paragraph"}

    def test_skip_script_tags(self) -> None:
        """Test that script content is not extracted."""
        processor = HTMLProcessor()
        html = "<div><p>Visible</p><script>doNotTranslate();</script></div>"

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 1
        assert nodes[0].text == "Visible"

    def test_skip_style_tags(self) -> None:
        """Test that style content is not extracted."""
        processor = HTMLProcessor()
        html = "<div><p>Visible</p><style>.class { color: red; }</style></div>"

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 1
        assert nodes[0].text == "Visible"

    def test_skip_code_tags(self) -> None:
        """Test that code content is not extracted."""
        processor = HTMLProcessor()
        html = "<div><p>Visible</p><code>const x = 1;</code></div>"

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 1
        assert nodes[0].text == "Visible"

    def test_skip_pre_tags(self) -> None:
        """Test that pre content is not extracted."""
        processor = HTMLProcessor()
        html = "<div><p>Visible</p><pre>Preformatted text</pre></div>"

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 1
        assert nodes[0].text == "Visible"

    def test_skip_data_no_translate(self) -> None:
        """Test that data-no-translate elements are skipped."""
        processor = HTMLProcessor()
        html = '<div><p>Translate me</p><p data-no-translate>Keep this</p></div>'

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 1
        assert nodes[0].text == "Translate me"

    def test_skip_nested_no_translate(self) -> None:
        """Test that nested elements under data-no-translate are skipped."""
        processor = HTMLProcessor()
        html = '<div data-no-translate><p>Skip this</p><span>And this</span></div>'

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 0

    def test_skip_whitespace_only(self) -> None:
        """Test that whitespace-only text nodes are skipped."""
        processor = HTMLProcessor()
        html = "<div>   \n\t   </div><p>Visible</p>"

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 1
        assert nodes[0].text == "Visible"

    def test_hash_generation(self) -> None:
        """Test that correct SHA-256 hash is generated."""
        processor = HTMLProcessor()
        html = "<p>Hello World</p>"

        parsed, nodes = processor.extract(html)

        expected_hash = hashlib.sha256("Hello World".encode()).hexdigest()
        assert nodes[0].hash == expected_hash

    def test_apply_translations(self) -> None:
        """Test applying translations to HTML."""
        processor = HTMLProcessor()
        html = "<p>Hello World</p>"

        parsed, nodes = processor.extract(html)
        translations = {nodes[0].hash: "Hola Mundo"}

        result = processor.apply(parsed, nodes, translations)

        assert "Hola Mundo" in result
        assert "Hello World" not in result

    def test_apply_preserves_structure(self) -> None:
        """Test that HTML structure is preserved after translation."""
        processor = HTMLProcessor()
        html = '<div class="container"><h1>Title</h1><p>Text</p></div>'

        parsed, nodes = processor.extract(html)
        translations = {n.hash: f"Translated_{n.text}" for n in nodes}

        result = processor.apply(parsed, nodes, translations)

        assert 'class="container"' in result
        assert "<h1>" in result
        assert "<p>" in result
        assert "Translated_Title" in result
        assert "Translated_Text" in result

    def test_apply_sets_lang_attribute(self) -> None:
        """Test that lang attribute is set on html tag."""
        processor = HTMLProcessor()
        html = "<html><body><p>Hello</p></body></html>"

        parsed, nodes = processor.extract(html)
        translations = {nodes[0].hash: "Hola"}

        result = processor.apply(parsed, nodes, translations, target_lang="es_ES")

        assert 'lang="es-ES"' in result

    def test_apply_sets_dir_attribute_ltr(self) -> None:
        """Test that dir attribute is set to ltr for LTR languages."""
        processor = HTMLProcessor()
        html = "<html><body><p>Hello</p></body></html>"

        parsed, nodes = processor.extract(html)
        translations = {nodes[0].hash: "Hola"}

        result = processor.apply(parsed, nodes, translations, target_lang="es_ES")

        assert 'dir="ltr"' in result

    def test_apply_sets_dir_attribute_rtl(self) -> None:
        """Test that dir attribute is set to rtl for RTL languages."""
        processor = HTMLProcessor()
        html = "<html><body><p>Hello</p></body></html>"

        parsed, nodes = processor.extract(html)
        translations = {nodes[0].hash: "مرحبا"}

        result = processor.apply(parsed, nodes, translations, target_lang="ar_SA")

        assert 'dir="rtl"' in result

    def test_custom_ignored_tags(self) -> None:
        """Test adding custom ignored tags."""
        processor = HTMLProcessor(ignored_tags={"custom-tag"})
        html = "<div><p>Visible</p><custom-tag>Ignored</custom-tag></div>"

        parsed, nodes = processor.extract(html)

        assert len(nodes) == 1
        assert nodes[0].text == "Visible"

    def test_preserve_whitespace(self) -> None:
        """Test that leading/trailing whitespace is preserved."""
        processor = HTMLProcessor()
        html = "<p>  Hello World  </p>"

        parsed, nodes = processor.extract(html)
        translations = {nodes[0].hash: "Hola Mundo"}

        result = processor.apply(parsed, nodes, translations)

        # The translation should have whitespace preserved
        assert "  Hola Mundo  " in result

    def test_get_content_type(self) -> None:
        """Test content type identifier."""
        processor = HTMLProcessor()
        assert processor.get_content_type() == "html"

    def test_metadata_includes_parent_tag(self) -> None:
        """Test that metadata includes parent tag name."""
        processor = HTMLProcessor()
        html = "<h1>Title</h1><p>Paragraph</p>"

        parsed, nodes = processor.extract(html)

        h1_node = next(n for n in nodes if n.text == "Title")
        p_node = next(n for n in nodes if n.text == "Paragraph")

        assert h1_node.metadata.get("parent_tag") == "h1"
        assert p_node.metadata.get("parent_tag") == "p"
