"""
Query preprocessing utilities for the retrieval pipeline.

This module provides functions for cleaning, normalizing, and preprocessing
user queries to improve retrieval effectiveness.
"""
import re
from typing import Union
from ..models.retrieval_entities import Query
from datetime import datetime


class QueryPreprocessor:
    """Utility class for query preprocessing operations."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize query text by applying standard transformations.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return text

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean query text by removing unwanted characters and patterns.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return text

        # Remove special characters but keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\-\.\,\!\?\;\:]', ' ', text)

        # Remove extra whitespace again after cleaning
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def preprocess_query(query: Union[str, Query]) -> Query:
        """
        Preprocess a query string or Query object.

        Args:
            query: Input query as string or Query object

        Returns:
            Processed Query object with cleaned and normalized text
        """
        if isinstance(query, str):
            original_text = query
            query_obj = Query(text=original_text)
        else:
            original_text = query.text
            query_obj = query

        # Clean and normalize the text
        cleaned_text = QueryPreprocessor.clean_text(original_text)
        normalized_text = QueryPreprocessor.normalize_text(cleaned_text)

        # Create or update the query object
        processed_query = Query(
            text=original_text,
            processed_text=normalized_text,
            timestamp=query_obj.timestamp or datetime.now(),
            query_id=query_obj.query_id
        )

        return processed_query

    @staticmethod
    def truncate_query(text: str, max_length: int = 1000) -> str:
        """
        Truncate query if it exceeds maximum length.

        Args:
            text: Input text to truncate
            max_length: Maximum allowed length (default: 1000)

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[:max_length]

    @staticmethod
    def remove_stopwords(text: str, stopwords: set = None) -> str:
        """
        Remove common stopwords from the query text.

        Args:
            text: Input text
            stopwords: Set of stopwords to remove (default: common English stopwords)

        Returns:
            Text with stopwords removed
        """
        if stopwords is None:
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'about', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out',
                'off', 'over', 'under', 'again', 'further', 'then', 'once', 'i',
                'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                'do', 'does', 'did', 'doing', 'will', 'would', 'should', 'can',
                'could', 'ought', 'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s',
                'we\'re', 'they\'re', 'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve',
                'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d', 'they\'d', 'i\'ll',
                'you\'ll', 'he\'ll', 'she\'ll', 'we\'ll', 'they\'ll', 'isn\'t',
                'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t',
                'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'shan\'t',
                'shouldn\'t', 'can\'t', 'cannot', 'couldn\'t', 'mustn\'t', 'let\'s',
                'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s', 'when\'s',
                'where\'s', 'why\'s', 'how\'s', 'a\'s', 'o\'clock', 'ma\'am',
                'tis', 't\'was', '\'cause'
            }

        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        return ' '.join(filtered_words)