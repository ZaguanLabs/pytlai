"""Python source code processor for pytlai."""

from __future__ import annotations

import ast
import hashlib
import re
import tokenize
import uuid
from dataclasses import dataclass
from io import StringIO
from typing import Any

from pytlai.config import PythonOptions
from pytlai.processors.base import ContentProcessor, TextNode


@dataclass
class SourceLocation:
    """Location of a translatable item in source code.

    Attributes:
        start_line: 1-indexed start line number.
        end_line: 1-indexed end line number.
        start_col: 0-indexed start column.
        end_col: 0-indexed end column.
        original: Original text including quotes/delimiters.
    """

    start_line: int
    end_line: int
    start_col: int
    end_col: int
    original: str


class PythonProcessor(ContentProcessor):
    """Processor for Python source code.

    Extracts docstrings and comments from Python source, preserving
    code structure and indentation. Optionally extracts marked string
    literals for translation.

    Attributes:
        options: Python-specific translation options.
    """

    # Patterns to skip in comments
    SKIP_COMMENT_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"^#!"),  # Shebang
        re.compile(r"^#.*coding[:=]"),  # Encoding declaration
        re.compile(r"^#\s*type:\s*"),  # Type comments
        re.compile(r"^#\s*noqa"),  # noqa comments
        re.compile(r"^#\s*pragma"),  # pragma comments
        re.compile(r"^#\s*pylint"),  # pylint comments
        re.compile(r"^#\s*mypy"),  # mypy comments
        re.compile(r"^#\s*ruff"),  # ruff comments
    ]

    def __init__(self, options: PythonOptions | None = None) -> None:
        """Initialize the Python processor.

        Args:
            options: Python-specific options. Uses defaults if None.
        """
        self._options = options or PythonOptions()

    def extract(self, content: str) -> tuple[dict[str, Any], list[TextNode]]:
        """Extract translatable content from Python source.

        Args:
            content: Python source code string.

        Returns:
            Tuple of (source_info dict, list of TextNode objects).
        """
        text_nodes: list[TextNode] = []
        locations: dict[str, SourceLocation] = {}
        lines = content.splitlines(keepends=True)

        # Extract docstrings using AST
        if self._options.translate_docstrings:
            docstring_nodes, docstring_locs = self._extract_docstrings(content)
            text_nodes.extend(docstring_nodes)
            locations.update(docstring_locs)

        # Extract comments using tokenize
        if self._options.translate_comments:
            comment_nodes, comment_locs = self._extract_comments(content)
            text_nodes.extend(comment_nodes)
            locations.update(comment_locs)

        # Extract marked strings
        if self._options.translate_strings:
            string_nodes, string_locs = self._extract_marked_strings(content)
            text_nodes.extend(string_nodes)
            locations.update(string_locs)

        source_info = {
            "lines": lines,
            "locations": locations,
            "original": content,
        }

        return source_info, text_nodes

    def _extract_docstrings(
        self, content: str
    ) -> tuple[list[TextNode], dict[str, SourceLocation]]:
        """Extract docstrings from Python source.

        Args:
            content: Python source code.

        Returns:
            Tuple of (text_nodes, locations).
        """
        text_nodes: list[TextNode] = []
        locations: dict[str, SourceLocation] = {}

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return text_nodes, locations

        for node in ast.walk(tree):
            docstring = None
            node_type = None

            # Module docstring
            if isinstance(node, ast.Module):
                docstring = ast.get_docstring(node)
                node_type = "module_docstring"
            # Class docstring
            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                node_type = "class_docstring"
            # Function/method docstring
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                node_type = "function_docstring"

            if docstring and node_type:
                # Find the actual string node for location
                if node.body and isinstance(node.body[0], ast.Expr):
                    expr = node.body[0]
                    if isinstance(expr.value, ast.Constant) and isinstance(
                        expr.value.value, str
                    ):
                        const = expr.value
                        node_id = str(uuid.uuid4())
                        text_hash = hashlib.sha256(docstring.encode()).hexdigest()

                        text_node = TextNode(
                            id=node_id,
                            text=docstring,
                            hash=text_hash,
                            node_type=node_type,
                            metadata={
                                "line": const.lineno,
                                "end_line": const.end_lineno or const.lineno,
                            },
                        )
                        text_nodes.append(text_node)

                        # Get original source for this docstring
                        lines = content.splitlines()
                        start_line = const.lineno - 1
                        end_line = (const.end_lineno or const.lineno) - 1
                        original_lines = lines[start_line : end_line + 1]
                        original = "\n".join(original_lines)

                        locations[node_id] = SourceLocation(
                            start_line=const.lineno,
                            end_line=const.end_lineno or const.lineno,
                            start_col=const.col_offset,
                            end_col=const.end_col_offset or 0,
                            original=original,
                        )

        return text_nodes, locations

    def _extract_comments(
        self, content: str
    ) -> tuple[list[TextNode], dict[str, SourceLocation]]:
        """Extract comments from Python source.

        Args:
            content: Python source code.

        Returns:
            Tuple of (text_nodes, locations).
        """
        text_nodes: list[TextNode] = []
        locations: dict[str, SourceLocation] = {}

        try:
            tokens = list(tokenize.generate_tokens(StringIO(content).readline))
        except tokenize.TokenError:
            return text_nodes, locations

        for token in tokens:
            if token.type == tokenize.COMMENT:
                comment = token.string

                # Skip special comments
                if any(pattern.match(comment) for pattern in self.SKIP_COMMENT_PATTERNS):
                    continue

                # Extract comment text (remove # and strip)
                comment_text = comment.lstrip("#").strip()
                if not comment_text:
                    continue

                node_id = str(uuid.uuid4())
                text_hash = hashlib.sha256(comment_text.encode()).hexdigest()

                text_node = TextNode(
                    id=node_id,
                    text=comment_text,
                    hash=text_hash,
                    node_type="comment",
                    metadata={"line": token.start[0]},
                )
                text_nodes.append(text_node)

                locations[node_id] = SourceLocation(
                    start_line=token.start[0],
                    end_line=token.end[0],
                    start_col=token.start[1],
                    end_col=token.end[1],
                    original=comment,
                )

        return text_nodes, locations

    def _extract_marked_strings(
        self, content: str
    ) -> tuple[list[TextNode], dict[str, SourceLocation]]:
        """Extract marked string literals from Python source.

        Looks for strings marked with _() or followed by # translate comment.

        Args:
            content: Python source code.

        Returns:
            Tuple of (text_nodes, locations).
        """
        text_nodes: list[TextNode] = []
        locations: dict[str, SourceLocation] = {}

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return text_nodes, locations

        lines = content.splitlines()

        for node in ast.walk(tree):
            # Look for _("string") calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "_":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        const = node.args[0]
                        if isinstance(const.value, str):
                            string_value = const.value
                            node_id = str(uuid.uuid4())
                            text_hash = hashlib.sha256(string_value.encode()).hexdigest()

                            text_node = TextNode(
                                id=node_id,
                                text=string_value,
                                hash=text_hash,
                                node_type="marked_string",
                                metadata={
                                    "line": const.lineno,
                                    "marker": "_()",
                                },
                            )
                            text_nodes.append(text_node)

                            # Get original source
                            line_idx = const.lineno - 1
                            original = lines[line_idx] if line_idx < len(lines) else ""

                            locations[node_id] = SourceLocation(
                                start_line=const.lineno,
                                end_line=const.end_lineno or const.lineno,
                                start_col=const.col_offset,
                                end_col=const.end_col_offset or 0,
                                original=original,
                            )

        return text_nodes, locations

    def apply(
        self,
        parsed: dict[str, Any],
        nodes: list[TextNode],
        translations: dict[str, str],
        target_lang: str | None = None,
    ) -> str:
        """Apply translations to Python source.

        Args:
            parsed: Source info dict from extract().
            nodes: Text nodes from extract().
            translations: Dictionary mapping text hashes to translated text.
            target_lang: Target language (unused for Python, but kept for API).

        Returns:
            Reconstructed Python source with translations applied.
        """
        lines: list[str] = parsed["lines"].copy()
        locations: dict[str, SourceLocation] = parsed["locations"]

        # Sort nodes by line number (descending) to apply from bottom up
        # This prevents line number shifts from affecting earlier replacements
        sorted_nodes = sorted(
            nodes,
            key=lambda n: locations.get(n.id, SourceLocation(0, 0, 0, 0, "")).start_line,
            reverse=True,
        )

        for text_node in sorted_nodes:
            if text_node.hash not in translations:
                continue

            location = locations.get(text_node.id)
            if not location:
                continue

            translated = translations[text_node.hash]

            if text_node.node_type == "comment":
                self._apply_comment_translation(lines, location, translated)
            elif text_node.node_type in (
                "module_docstring",
                "class_docstring",
                "function_docstring",
            ):
                self._apply_docstring_translation(lines, location, text_node, translated)
            elif text_node.node_type == "marked_string":
                self._apply_string_translation(lines, location, text_node, translated)

        return "".join(lines)

    def _apply_comment_translation(
        self,
        lines: list[str],
        location: SourceLocation,
        translated: str,
    ) -> None:
        """Apply translation to a comment."""
        line_idx = location.start_line - 1
        if line_idx >= len(lines):
            return

        line = lines[line_idx]
        # Find the comment start position
        comment_start = line.find("#")
        if comment_start == -1:
            return

        # Preserve indentation and # prefix
        prefix = line[: comment_start + 1]
        # Check if there was a space after #
        if len(line) > comment_start + 1 and line[comment_start + 1] == " ":
            prefix += " "

        # Preserve line ending
        line_end = ""
        if line.endswith("\r\n"):
            line_end = "\r\n"
        elif line.endswith("\n"):
            line_end = "\n"
        elif line.endswith("\r"):
            line_end = "\r"

        lines[line_idx] = prefix + translated + line_end

    def _apply_docstring_translation(
        self,
        lines: list[str],
        location: SourceLocation,
        text_node: TextNode,
        translated: str,
    ) -> None:
        """Apply translation to a docstring."""
        start_idx = location.start_line - 1
        end_idx = location.end_line - 1

        if start_idx >= len(lines):
            return

        # Get the original docstring line(s)
        original_first_line = lines[start_idx]

        # Detect quote style and indentation
        stripped = original_first_line.lstrip()
        indent = original_first_line[: len(original_first_line) - len(stripped)]

        # Detect quote style (""" or ''')
        quote_style = '"""'
        if stripped.startswith("'''"):
            quote_style = "'''"
        elif stripped.startswith('"""'):
            quote_style = '"""'
        elif stripped.startswith("'"):
            quote_style = "'"
        elif stripped.startswith('"'):
            quote_style = '"'

        # Preserve line ending from last line
        last_line = lines[end_idx] if end_idx < len(lines) else ""
        line_end = ""
        if last_line.endswith("\r\n"):
            line_end = "\r\n"
        elif last_line.endswith("\n"):
            line_end = "\n"
        elif last_line.endswith("\r"):
            line_end = "\r"

        # Build new docstring
        if "\n" in translated:
            # Multi-line docstring
            new_docstring = f"{indent}{quote_style}{translated}{quote_style}{line_end}"
        else:
            # Single-line docstring
            new_docstring = f"{indent}{quote_style}{translated}{quote_style}{line_end}"

        # Replace the lines
        lines[start_idx : end_idx + 1] = [new_docstring]

    def _apply_string_translation(
        self,
        lines: list[str],
        location: SourceLocation,
        text_node: TextNode,
        translated: str,
    ) -> None:
        """Apply translation to a marked string literal."""
        line_idx = location.start_line - 1
        if line_idx >= len(lines):
            return

        line = lines[line_idx]
        original_text = text_node.text

        # Escape the translated text for Python string
        escaped = translated.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

        # Replace the original string with translated
        # This is a simple replacement - may need refinement for complex cases
        line = line.replace(f'"{original_text}"', f'"{escaped}"')
        line = line.replace(f"'{original_text}'", f"'{escaped}'")

        lines[line_idx] = line

    def get_content_type(self) -> str:
        """Get the content type this processor handles.

        Returns:
            'python'
        """
        return "python"
