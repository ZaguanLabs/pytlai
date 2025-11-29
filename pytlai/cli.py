"""Command-line interface for pytlai."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pytlai import __version__


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="pytlai",
        description="Python Translation AI - Translate Python scripts or web pages to any language",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"pytlai {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # translate command
    translate_parser = subparsers.add_parser(
        "translate",
        help="Translate a file or text",
    )
    translate_parser.add_argument(
        "input",
        help="Input file path or '-' for stdin",
    )
    translate_parser.add_argument(
        "-l", "--lang",
        required=True,
        help="Target language code (e.g., es_ES, fr_FR, ja_JP)",
    )
    translate_parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)",
    )
    translate_parser.add_argument(
        "-t", "--type",
        choices=["html", "python"],
        help="Content type (auto-detected if not specified)",
    )
    translate_parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language code (default: en)",
    )
    translate_parser.add_argument(
        "--model",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    translate_parser.add_argument(
        "--cache",
        help="Path to cache file (JSON/YAML) for offline mode",
    )
    translate_parser.add_argument(
        "--context",
        help="Additional context for the AI translator",
    )
    translate_parser.add_argument(
        "--exclude",
        nargs="+",
        help="Terms to exclude from translation",
    )

    # extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract translatable strings from a file",
    )
    extract_parser.add_argument(
        "input",
        help="Input file path",
    )
    extract_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path (JSON/YAML/CSV)",
    )
    extract_parser.add_argument(
        "-t", "--type",
        choices=["html", "python"],
        help="Content type (auto-detected if not specified)",
    )

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export translations to a language file",
    )
    export_parser.add_argument(
        "input",
        help="Input cache file (JSON)",
    )
    export_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path (JSON/YAML/PO/CSV)",
    )
    export_parser.add_argument(
        "-f", "--format",
        choices=["json", "yaml", "po", "csv"],
        help="Output format (auto-detected from extension if not specified)",
    )
    export_parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language code",
    )
    export_parser.add_argument(
        "--target-lang",
        help="Target language code",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "translate":
            return cmd_translate(args)
        elif args.command == "extract":
            return cmd_extract(args)
        elif args.command == "export":
            return cmd_export(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    """Handle the translate command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    from pytlai import Pytlai
    from pytlai.cache.file import FileCache
    from pytlai.providers.openai import OpenAIProvider

    # Read input
    if args.input == "-":
        content = sys.stdin.read()
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            return 1
        content = input_path.read_text()

    # Set up cache
    cache = None
    if args.cache:
        cache_path = Path(args.cache)
        if cache_path.exists():
            cache = FileCache(cache_path, read_only=False)
        else:
            cache = FileCache(cache_path, read_only=False)

    # Set up provider (None if cache-only mode)
    provider = None
    if not args.cache or args.model:
        provider = OpenAIProvider(model=args.model) if args.model else OpenAIProvider()

    # Create translator
    translator = Pytlai(
        target_lang=args.lang,
        source_lang=args.source_lang,
        provider=provider,
        cache=cache,
        context=args.context,
        excluded_terms=args.exclude,
    )

    # Translate
    result = translator.process(content, content_type=args.type)

    # Output
    if args.output:
        Path(args.output).write_text(result.content)
        print(f"Translated {result.total_nodes} items ({result.cached_count} cached, {result.translated_count} new)")
    else:
        print(result.content)

    # Save cache if using file cache
    if cache and hasattr(cache, "save"):
        cache.save()

    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Handle the extract command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    import json

    from pytlai.processors.html import HTMLProcessor
    from pytlai.processors.python import PythonProcessor

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    content = input_path.read_text()

    # Determine content type
    content_type = args.type
    if not content_type:
        ext = input_path.suffix.lower()
        if ext in (".html", ".htm"):
            content_type = "html"
        elif ext == ".py":
            content_type = "python"
        else:
            print(f"Error: Cannot determine content type for {ext}", file=sys.stderr)
            return 1

    # Extract
    if content_type == "html":
        processor = HTMLProcessor()
    else:
        processor = PythonProcessor()

    _, nodes = processor.extract(content)

    # Build output
    strings = {}
    for node in nodes:
        strings[node.hash] = {
            "source": node.text,
            "target": "",
            "type": node.node_type,
        }

    output_path = Path(args.output)
    output_ext = output_path.suffix.lower()

    if output_ext == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"strings": strings}, f, ensure_ascii=False, indent=2)
    elif output_ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            print("Error: YAML support requires pyyaml. Install with: pip install pytlai[yaml]", file=sys.stderr)
            return 1
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump({"strings": strings}, f, allow_unicode=True)
    elif output_ext == ".csv":
        import csv
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hash", "source", "target", "type"])
            for hash_key, entry in strings.items():
                writer.writerow([hash_key, entry["source"], entry["target"], entry["type"]])
    else:
        print(f"Error: Unsupported output format: {output_ext}", file=sys.stderr)
        return 1

    print(f"Extracted {len(strings)} strings to {output_path}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Handle the export command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    import json

    from pytlai.export import TranslationExporter

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    # Load input
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Get translations
    translations = data.get("translations", data.get("strings", {}))

    # Export
    exporter = TranslationExporter()
    output_path = exporter.export(
        translations=translations,
        output_path=args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang or "unknown",
        file_format=args.format,
    )

    print(f"Exported {len(translations)} translations to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
