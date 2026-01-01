"""
Query entity model with validation rules.

This module defines the Query entity with validation rules to ensure
data integrity and proper handling of user queries.
"""
import re
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field
from .retrieval_entities import Query as BaseQuery


@dataclass
class Query:
    """
    Query entity with validation rules.

    This class extends the base Query with additional validation and utility methods.
    """
    text: str
    processed_text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    query_id: Optional[str] = None

    def __post_init__(self):
        """Validate the query after initialization."""
        self.validate()

    def validate(self) -> bool:
        """
        Validate the query according to business rules.

        Returns:
            True if query is valid

        Raises:
            ValueError: If query fails validation
        """
        # Validate text length
        if not self.text or len(self.text.strip()) == 0:
            raise ValueError("Query text cannot be empty or whitespace only")

        # Validate text length range (1-1000 characters as per spec)
        if len(self.text) < 1 or len(self.text) > 1000:
            raise ValueError(f"Query text must be between 1 and 1000 characters, got {len(self.text)}")

        # Validate that text is not only whitespace
        if not self.text.strip():
            raise ValueError("Query text cannot contain only whitespace")

        # Validate processed_text if provided
        if self.processed_text is not None:
            if len(self.processed_text) > 1000:
                raise ValueError("Processed text must not exceed 1000 characters")

        # Validate timestamp
        if self.timestamp is None:
            raise ValueError("Query timestamp cannot be None")

        # Validate query_id format if provided (alphanumeric, hyphens, underscores)
        if self.query_id is not None:
            if not re.match(r'^[a-zA-Z0-9_-]+$', self.query_id):
                raise ValueError("Query ID must contain only alphanumeric characters, hyphens, and underscores")

        return True

    def is_valid(self) -> bool:
        """
        Check if the query is valid without raising exceptions.

        Returns:
            True if query is valid, False otherwise
        """
        try:
            self.validate()
            return True
        except ValueError:
            return False

    def get_text_length(self) -> int:
        """
        Get the length of the query text.

        Returns:
            Length of the query text
        """
        return len(self.text)

    def get_processed_text(self) -> str:
        """
        Get the processed text, falling back to original text if not available.

        Returns:
            Processed text or original text
        """
        return self.processed_text if self.processed_text is not None else self.text

    def has_sufficient_content(self) -> bool:
        """
        Check if the query has sufficient content for processing.

        Returns:
            True if query has sufficient content
        """
        # Check for meaningful content (not just common words)
        processed = self.get_processed_text().lower()
        words = processed.split()

        # Filter out common stop words and check if we have meaningful content
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should'
        }

        meaningful_words = [word for word in words if word not in common_words and len(word) > 2]
        return len(meaningful_words) > 0

    def sanitize_text(self) -> str:
        """
        Sanitize the query text by removing potentially problematic characters.

        Returns:
            Sanitized text
        """
        # Remove or replace potentially problematic characters
        sanitized = re.sub(r'[^\w\s\-\.\,\!\?\;\:]', ' ', self.text)
        return ' '.join(sanitized.split())  # Normalize whitespace

    @classmethod
    def create_from_text(cls, text: str, query_id: Optional[str] = None) -> 'Query':
        """
        Create a Query instance from text with optional query ID.

        Args:
            text: Query text
            query_id: Optional query ID

        Returns:
            Query instance
        """
        return cls(text=text, query_id=query_id)