#!/usr/bin/env python3
"""Example: Translate Python script docstrings and comments.

This example demonstrates how to use pytlai to translate Python source code.
It translates docstrings and comments while preserving code structure.

Requirements:
    - Set OPENAI_API_KEY environment variable
    - pip install pytlai

Usage:
    python translate_script.py
"""

from pytlai import Pytlai, PythonOptions
from pytlai.providers import OpenAIProvider


def main() -> None:
    """Translate sample Python code."""
    # Sample Python code to translate
    python_code = '''
"""User authentication module.

This module provides functions for user authentication,
including login, logout, and session management.
"""

import hashlib
from datetime import datetime


class User:
    """Represents a user in the system.

    Attributes:
        username: The user's unique identifier.
        email: The user's email address.
        created_at: When the user account was created.
    """

    def __init__(self, username: str, email: str):
        """Initialize a new user.

        Args:
            username: The unique username.
            email: The user's email address.
        """
        self.username = username
        self.email = email
        self.created_at = datetime.now()

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against a stored hash.

        Args:
            password: The password to verify.
            stored_hash: The stored password hash.

        Returns:
            True if the password matches, False otherwise.
        """
        # Hash the provided password
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        # Compare with stored hash
        return password_hash == stored_hash


def create_session(user: User) -> dict:
    """Create a new session for a user.

    This function generates a new session token and stores
    the session information in the database.

    Args:
        user: The user to create a session for.

    Returns:
        A dictionary containing session information.
    """
    # Generate a unique session token
    token = hashlib.sha256(f"{user.username}{datetime.now()}".encode()).hexdigest()

    # Return session data
    return {
        "token": token,
        "user": user.username,
        "created": datetime.now().isoformat(),
    }
'''

    # Create translator with Python-specific options
    translator = Pytlai(
        target_lang="ja_JP",  # Japanese
        provider=OpenAIProvider(),
        context="Technical documentation for a Python authentication library",
        python_options=PythonOptions(
            translate_docstrings=True,
            translate_comments=True,
            translate_strings=False,  # Don't translate string literals
        ),
    )

    # Translate the Python code
    print("Translating Python code to Japanese...")
    result = translator.process(python_code, content_type="python")

    # Print results
    print("\n" + "=" * 60)
    print("TRANSLATION RESULTS")
    print("=" * 60)
    print(f"Total translatable items: {result.total_nodes}")
    print(f"Newly translated: {result.translated_count}")
    print(f"From cache: {result.cached_count}")
    print("\n" + "-" * 60)
    print("TRANSLATED PYTHON CODE:")
    print("-" * 60)
    print(result.content)


if __name__ == "__main__":
    main()
