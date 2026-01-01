"""
Query validation utilities for the retrieval pipeline.

This module provides functions for validating user queries to ensure
they meet the requirements for the retrieval system.
"""
import re
from typing import Union, List
from ..models.query import Query as QueryModel


class QueryValidator:
    """Utility class for validating query objects and text."""

    @staticmethod
    def validate_text_length(text: str, min_length: int = 1, max_length: int = 1000) -> bool:
        """
        Validate the length of query text.

        Args:
            text: Query text to validate
            min_length: Minimum allowed length (default: 1)
            max_length: Maximum allowed length (default: 1000)

        Returns:
            True if text length is valid

        Raises:
            ValueError: If text length is outside the allowed range
        """
        if len(text) < min_length or len(text) > max_length:
            raise ValueError(
                f"Query text length {len(text)} is outside allowed range "
                f"[{min_length}, {max_length}]"
            )

        return True

    @staticmethod
    def validate_text_content(text: str) -> bool:
        """
        Validate that query text contains meaningful content.

        Args:
            text: Query text to validate

        Returns:
            True if text contains meaningful content

        Raises:
            ValueError: If text contains only whitespace or no meaningful content
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Query text cannot be empty or contain only whitespace")

        # Check if text contains only whitespace
        if text == text.strip() and len(text) == 0:
            raise ValueError("Query text cannot contain only whitespace")

        return True

    @staticmethod
    def validate_query_object(query: QueryModel) -> bool:
        """
        Validate a Query object.

        Args:
            query: Query object to validate

        Returns:
            True if query object is valid

        Raises:
            ValueError: If query object fails validation
        """
        # Validate the query using its own validation method
        return query.validate()

    @staticmethod
    def is_valid_query_text(text: str) -> bool:
        """
        Check if query text is valid without raising exceptions.

        Args:
            text: Query text to validate

        Returns:
            True if query text is valid, False otherwise
        """
        try:
            QueryValidator.validate_text_length(text)
            QueryValidator.validate_text_content(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_query_id(query_id: str) -> bool:
        """
        Validate query ID format.

        Args:
            query_id: Query ID to validate

        Returns:
            True if query ID format is valid

        Raises:
            ValueError: If query ID format is invalid
        """
        if not query_id:
            return True  # query_id is optional

        # Validate that query_id contains only allowed characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', query_id):
            raise ValueError(
                "Query ID must contain only alphanumeric characters, hyphens, and underscores"
            )

        return True

    @staticmethod
    def validate_multiple_queries(queries: List[Union[str, QueryModel]]) -> bool:
        """
        Validate multiple queries.

        Args:
            queries: List of query strings or Query objects to validate

        Returns:
            True if all queries are valid

        Raises:
            ValueError: If any query fails validation
        """
        for i, query in enumerate(queries):
            if isinstance(query, str):
                QueryValidator.validate_text_length(query)
                QueryValidator.validate_text_content(query)
            elif isinstance(query, QueryModel):
                QueryValidator.validate_query_object(query)
            else:
                raise ValueError(f"Query at index {i} is not a string or Query object")

        return True

    @staticmethod
    def sanitize_query_text(text: str) -> str:
        """
        Sanitize query text by removing potentially problematic characters.

        Args:
            text: Query text to sanitize

        Returns:
            Sanitized query text
        """
        # Remove or replace potentially problematic characters
        sanitized = re.sub(r'[^\w\s\-\.\,\!\?\;\:]', ' ', text)
        return ' '.join(sanitized.split())  # Normalize whitespace

    @staticmethod
    def validate_query_complexity(text: str, min_words: int = 1) -> bool:
        """
        Validate query complexity by checking minimum word count.

        Args:
            text: Query text to validate
            min_words: Minimum number of words required (default: 1)

        Returns:
            True if query has sufficient complexity

        Raises:
            ValueError: If query doesn't meet complexity requirements
        """
        words = text.split()
        if len(words) < min_words:
            raise ValueError(f"Query must contain at least {min_words} word(s), got {len(words)}")

        return True

    @staticmethod
    def validate_for_retrieval(text: str) -> bool:
        """
        Validate query specifically for retrieval purposes.

        Args:
            text: Query text to validate

        Returns:
            True if query is suitable for retrieval

        Raises:
            ValueError: If query is not suitable for retrieval
        """
        # Validate basic requirements
        QueryValidator.validate_text_length(text)
        QueryValidator.validate_text_content(text)

        # Validate complexity for effective retrieval
        QueryValidator.validate_query_complexity(text, min_words=1)

        return True