#!/usr/bin/env python3
"""Example: Translate HTML content to Spanish.

This example demonstrates how to use pytlai to translate HTML content.
It shows basic usage with the OpenAI provider.

Requirements:
    - Set OPENAI_API_KEY environment variable
    - pip install pytlai

Usage:
    python translate_html.py
"""

from pytlai import Pytlai
from pytlai.providers import OpenAIProvider


def main() -> None:
    """Translate sample HTML content."""
    # Sample HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Welcome Page</title>
    </head>
    <body>
        <header>
            <h1>Welcome to Our Website</h1>
            <nav>
                <a href="/">Home</a>
                <a href="/about">About Us</a>
                <a href="/contact">Contact</a>
            </nav>
        </header>
        <main>
            <section>
                <h2>Our Mission</h2>
                <p>We believe in making technology accessible to everyone.</p>
                <p>Our team works hard to deliver the best solutions.</p>
            </section>
            <section>
                <h2>Get Started</h2>
                <p>Sign up today and join thousands of happy customers.</p>
                <button>Sign Up Now</button>
            </section>
        </main>
        <footer>
            <p>© 2024 Our Company. All rights reserved.</p>
        </footer>
    </body>
    </html>
    """

    # Create translator with OpenAI provider
    translator = Pytlai(
        target_lang="es_ES",  # Spanish (Spain)
        provider=OpenAIProvider(),
        context="Marketing website for a technology company",
    )

    # Translate the HTML
    print("Translating HTML to Spanish...")
    result = translator.process(html_content)

    # Print results
    print("\n" + "=" * 60)
    print("TRANSLATION RESULTS")
    print("=" * 60)
    print(f"Total text nodes found: {result.total_nodes}")
    print(f"Newly translated: {result.translated_count}")
    print(f"From cache: {result.cached_count}")
    print("\n" + "-" * 60)
    print("TRANSLATED HTML:")
    print("-" * 60)
    print(result.content)

    # Demonstrate caching - translate again
    print("\n" + "=" * 60)
    print("TRANSLATING AGAIN (should use cache)")
    print("=" * 60)
    result2 = translator.process(html_content)
    print(f"Total text nodes: {result2.total_nodes}")
    print(f"Newly translated: {result2.translated_count}")
    print(f"From cache: {result2.cached_count}")


if __name__ == "__main__":
    main()
