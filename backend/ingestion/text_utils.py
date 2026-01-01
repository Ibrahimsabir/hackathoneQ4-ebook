"""
Text utility functions for normalization and cleaning
"""
import re
import html
from typing import List, Dict, Any
import unicodedata
import logging
from logging_config import logger


class TextNormalizer:
    """
    Class containing utility functions for text normalization and cleaning
    """

    def __init__(self):
        self.logger = logger

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace, handling encoding issues, etc.

        Args:
            text (str): Input text to normalize

        Returns:
            str: Normalized text
        """
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)

        # Remove zero-width characters and other invisible characters
        text = re.sub(r'[\u200B-\u200D\uFEFF\u2060\u202A-\u202E]', '', text)

        # Replace multiple consecutive whitespace with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted patterns and standardizing format

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Remove special characters that might cause issues
        # Keep alphanumeric, basic punctuation, and common symbols
        text = re.sub(r'[^\w\s\-\.\!\?\,\;\:\'\"\(\)\[\]\{\}\/\\]', ' ', text)

        # Normalize whitespace again after cleaning
        text = re.sub(r'\s+', ' ', text)

        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)

        return text.strip()

    def remove_boilerplate(self, text: str, patterns: List[str] = None) -> str:
        """
        Remove common boilerplate text patterns from content

        Args:
            text (str): Input text to clean
            patterns (List[str]): Additional patterns to remove

        Returns:
            str: Text with boilerplate removed
        """
        if not text:
            return ""

        # Common boilerplate patterns to remove
        default_patterns = [
            r'Copyright\s+\d{4}.*?(?=\n|$)',  # Copyright notices
            r'All\s+rights\s+reserved.*?(?=\n|$)',  # Rights reserved notices
            r'Privacy\s+Policy.*?(?=\n|$)',  # Privacy policy links
            r'Terms\s+of\s+Use.*?(?=\n|$)',  # Terms of use links
            r'Contact\s+us.*?(?=\n|$)',  # Contact links
            r'Back\s+to\s+top.*?(?=\n|$)',  # Back to top links
            r'Next\s+page.*?(?=\n|$)',  # Next page links
            r'Previous\s+page.*?(?=\n|$)',  # Previous page links
            r'Was\s+this\s+helpful.*?(?=\n|$)',  # Feedback prompts
            r'Edit\s+this\s+page.*?(?=\n|$)',  # Edit page links
            r'Share\s+on.*?(?=\n|$)',  # Share buttons
            r'Follow\s+us.*?(?=\n|$)',  # Social media links
        ]

        # Add custom patterns if provided
        all_patterns = default_patterns + (patterns or [])

        for pattern in all_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up multiple consecutive empty lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    def preprocess_for_embedding(self, text: str) -> str:
        """
        Preprocess text specifically for embedding generation

        Args:
            text (str): Input text

        Returns:
            str: Preprocessed text ready for embedding
        """
        # Apply all cleaning steps
        text = self.normalize_text(text)
        text = self.clean_text(text)
        text = self.remove_boilerplate(text)

        return text

    def get_content_hash(self, content: str) -> str:
        """
        Generate a hash for content to detect changes

        Args:
            content (str): Content to hash

        Returns:
            str: Content hash
        """
        import hashlib

        if not content:
            return ""

        # Normalize content before hashing
        normalized_content = self.normalize_text(content)
        return hashlib.md5(normalized_content.encode('utf-8')).hexdigest()

    def compare_content_hashes(self, old_hash: str, new_hash: str) -> bool:
        """
        Compare two content hashes to detect changes

        Args:
            old_hash (str): Previous content hash
            new_hash (str): New content hash

        Returns:
            bool: True if content has changed, False otherwise
        """
        return old_hash != new_hash

    def detect_content_changes(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """
        Detect changes between old and new content

        Args:
            old_content (str): Previous content
            new_content (str): New content

        Returns:
            Dict[str, Any]: Change detection results
        """
        old_hash = self.get_content_hash(old_content)
        new_hash = self.get_content_hash(new_content)

        has_changed = self.compare_content_hashes(old_hash, new_hash)

        return {
            "has_changed": has_changed,
            "old_hash": old_hash,
            "new_hash": new_hash,
            "change_percentage": self._calculate_change_percentage(old_content, new_content) if has_changed else 0
        }

    def _calculate_change_percentage(self, old_content: str, new_content: str) -> float:
        """
        Calculate approximate percentage of content that changed

        Args:
            old_content (str): Previous content
            new_content (str): New content

        Returns:
            float: Approximate change percentage
        """
        try:
            # Simple approach: compare lengths and use basic similarity
            from difflib import SequenceMatcher

            similarity = SequenceMatcher(None, old_content, new_content).ratio()
            change_percentage = (1 - similarity) * 100

            return round(change_percentage, 2)
        except:
            # Fallback: just compare lengths
            old_len = len(old_content)
            new_len = len(new_content)
            max_len = max(old_len, new_len) if max(old_len, new_len) > 0 else 1
            length_diff = abs(old_len - new_len)
            return round((length_diff / max_len) * 100, 2)


def test_text_utils():
    """
    Test function to verify text utilities work
    """
    normalizer = TextNormalizer()

    # Test text with various issues
    test_text = """
    This   is  a    test   text with   multiple    spaces.
    It has some HTML entities like &amp; and &lt;.
    Copyright 2023 Some Company. All rights reserved.
    And it has some special characters: \u200B\u200C\u200D.
    """

    print("Original text:")
    print(repr(test_text))
    print("\nNormalized text:")
    normalized = normalizer.normalize_text(test_text)
    print(repr(normalized))
    print("\nCleaned text:")
    cleaned = normalizer.clean_text(normalized)
    print(repr(cleaned))
    print("\nBoilerplate removed:")
    cleaned_more = normalizer.remove_boilerplate(cleaned)
    print(repr(cleaned_more))
    print("\nPreprocessed for embedding:")
    preprocessed = normalizer.preprocess_for_embedding(test_text)
    print(repr(preprocessed))
    print(f"\nContent hash: {normalizer.get_content_hash(test_text)}")


if __name__ == "__main__":
    test_text_utils()