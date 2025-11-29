"""Tests for Python processor."""

import hashlib

import pytest

from pytlai.config import PythonOptions
from pytlai.processors.python import PythonProcessor


class TestPythonProcessor:
    """Tests for PythonProcessor."""

    def test_extract_module_docstring(self) -> None:
        """Test extracting module-level docstring."""
        processor = PythonProcessor()
        code = '''"""This is a module docstring."""

def foo():
    pass
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "This is a module docstring."
        assert nodes[0].node_type == "module_docstring"

    def test_extract_function_docstring(self) -> None:
        """Test extracting function docstring."""
        processor = PythonProcessor()
        code = '''def greet(name):
    """Return a greeting message."""
    return f"Hello, {name}!"
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Return a greeting message."
        assert nodes[0].node_type == "function_docstring"

    def test_extract_class_docstring(self) -> None:
        """Test extracting class docstring."""
        processor = PythonProcessor()
        code = '''class MyClass:
    """A sample class."""
    pass
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "A sample class."
        assert nodes[0].node_type == "class_docstring"

    def test_extract_method_docstring(self) -> None:
        """Test extracting method docstring."""
        processor = PythonProcessor()
        code = '''class MyClass:
    def method(self):
        """A method docstring."""
        pass
'''
        parsed, nodes = processor.extract(code)

        # Should get method docstring (class has no docstring)
        assert len(nodes) == 1
        assert nodes[0].text == "A method docstring."
        assert nodes[0].node_type == "function_docstring"

    def test_extract_async_function_docstring(self) -> None:
        """Test extracting async function docstring."""
        processor = PythonProcessor()
        code = '''async def fetch_data():
    """Fetch data asynchronously."""
    pass
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Fetch data asynchronously."
        assert nodes[0].node_type == "function_docstring"

    def test_extract_comments(self) -> None:
        """Test extracting comments."""
        processor = PythonProcessor()
        code = '''# This is a comment
x = 1
# Another comment
y = 2
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 2
        texts = {n.text for n in nodes}
        assert texts == {"This is a comment", "Another comment"}
        assert all(n.node_type == "comment" for n in nodes)

    def test_skip_shebang(self) -> None:
        """Test that shebang is not extracted."""
        processor = PythonProcessor()
        code = '''#!/usr/bin/env python3
# Real comment
x = 1
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Real comment"

    def test_skip_encoding_declaration(self) -> None:
        """Test that encoding declaration is not extracted."""
        processor = PythonProcessor()
        code = '''# -*- coding: utf-8 -*-
# Real comment
x = 1
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Real comment"

    def test_skip_type_comments(self) -> None:
        """Test that type comments are not extracted."""
        processor = PythonProcessor()
        code = '''# type: ignore
# Real comment
x = 1  # type: int
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Real comment"

    def test_skip_noqa_comments(self) -> None:
        """Test that noqa comments are not extracted."""
        processor = PythonProcessor()
        code = '''# noqa: E501
# Real comment
x = 1
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Real comment"

    def test_skip_pragma_comments(self) -> None:
        """Test that pragma comments are not extracted."""
        processor = PythonProcessor()
        code = '''# pragma: no cover
# Real comment
x = 1
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Real comment"

    def test_options_disable_docstrings(self) -> None:
        """Test disabling docstring extraction."""
        options = PythonOptions(translate_docstrings=False, translate_comments=True)
        processor = PythonProcessor(options)
        code = '''"""Module docstring."""
# A comment
def foo():
    """Function docstring."""
    pass
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "A comment"

    def test_options_disable_comments(self) -> None:
        """Test disabling comment extraction."""
        options = PythonOptions(translate_docstrings=True, translate_comments=False)
        processor = PythonProcessor(options)
        code = '''"""Module docstring."""
# A comment
x = 1
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Module docstring."

    def test_extract_marked_strings(self) -> None:
        """Test extracting _() marked strings."""
        options = PythonOptions(translate_strings=True)
        processor = PythonProcessor(options)
        code = '''message = _("Hello, World!")
'''
        parsed, nodes = processor.extract(code)

        # Should find the marked string
        marked = [n for n in nodes if n.node_type == "marked_string"]
        assert len(marked) == 1
        assert marked[0].text == "Hello, World!"

    def test_apply_comment_translation(self) -> None:
        """Test applying translation to a comment."""
        processor = PythonProcessor()
        code = '''# Original comment
x = 1
'''
        parsed, nodes = processor.extract(code)
        translations = {nodes[0].hash: "Translated comment"}

        result = processor.apply(parsed, nodes, translations)

        assert "# Translated comment" in result
        assert "Original comment" not in result
        assert "x = 1" in result

    def test_apply_docstring_translation(self) -> None:
        """Test applying translation to a docstring."""
        processor = PythonProcessor()
        code = '''def greet():
    """Original docstring."""
    pass
'''
        parsed, nodes = processor.extract(code)
        translations = {nodes[0].hash: "Translated docstring."}

        result = processor.apply(parsed, nodes, translations)

        assert "Translated docstring." in result
        assert "Original docstring." not in result
        assert "def greet():" in result

    def test_apply_preserves_indentation(self) -> None:
        """Test that indentation is preserved after translation."""
        processor = PythonProcessor()
        code = '''class MyClass:
    def method(self):
        # Indented comment
        pass
'''
        parsed, nodes = processor.extract(code)
        translations = {nodes[0].hash: "Translated comment"}

        result = processor.apply(parsed, nodes, translations)

        # Check indentation is preserved
        lines = result.split("\n")
        comment_line = next(l for l in lines if "Translated comment" in l)
        assert comment_line.startswith("        #")  # 8 spaces

    def test_hash_generation(self) -> None:
        """Test that correct SHA-256 hash is generated."""
        processor = PythonProcessor()
        code = '''# Hello World
'''
        parsed, nodes = processor.extract(code)

        expected_hash = hashlib.sha256("Hello World".encode()).hexdigest()
        assert nodes[0].hash == expected_hash

    def test_get_content_type(self) -> None:
        """Test content type identifier."""
        processor = PythonProcessor()
        assert processor.get_content_type() == "python"

    def test_multiple_docstrings_and_comments(self) -> None:
        """Test extracting multiple docstrings and comments."""
        processor = PythonProcessor()
        code = '''"""Module docstring."""

# Module comment

class MyClass:
    """Class docstring."""

    def method(self):
        """Method docstring."""
        # Method comment
        pass
'''
        parsed, nodes = processor.extract(code)

        docstrings = [n for n in nodes if "docstring" in n.node_type]
        comments = [n for n in nodes if n.node_type == "comment"]

        assert len(docstrings) == 3
        assert len(comments) == 2

    def test_syntax_error_handling(self) -> None:
        """Test handling of syntax errors in code."""
        processor = PythonProcessor()
        code = '''def broken(
    # Missing closing paren
'''
        # Should not raise, just return empty
        parsed, nodes = processor.extract(code)
        # May have partial results or empty, but shouldn't crash

    def test_empty_comment_skipped(self) -> None:
        """Test that empty comments are skipped."""
        processor = PythonProcessor()
        code = '''#
# Real comment
#
'''
        parsed, nodes = processor.extract(code)

        assert len(nodes) == 1
        assert nodes[0].text == "Real comment"
