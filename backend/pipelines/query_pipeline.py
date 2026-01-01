"""
Query preprocessing pipeline for the retrieval system.

This module provides a pipeline for processing user queries through multiple
stages of cleaning, normalization, and validation before embedding generation.
"""
from typing import Union
from ..models.query import Query
from ..utils.query_preprocessor import QueryPreprocessor
from ..utils.query_validator import QueryValidator
from ..utils.logging import retrieval_logger


class QueryPipeline:
    """Pipeline for processing user queries."""

    def __init__(self):
        """Initialize the query pipeline."""
        self.preprocessor = QueryPreprocessor()
        self.validator = QueryValidator()

    def process_query(self, query_input: Union[str, Query]) -> Query:
        """
        Process a query through the complete pipeline.

        Args:
            query_input: Input query as string or Query object

        Returns:
            Processed Query object ready for embedding generation

        Raises:
            ValueError: If query fails validation
        """
        # Convert input to Query object if needed
        if isinstance(query_input, str):
            query = Query(text=query_input)
        else:
            query = query_input

        # Log the initial query
        retrieval_logger.info(f"Processing query: {query.text[:100]}...")

        # Step 1: Validate the query
        self.validator.validate_query_object(query)

        # Step 2: Preprocess the query text
        processed_text = self.preprocessor.preprocess_query(query)
        query.processed_text = processed_text.processed_text

        # Step 3: Additional validation on processed text
        if query.processed_text:
            self.validator.validate_for_retrieval(query.processed_text)

        # Step 4: Sanitize the text
        sanitized_text = self.preprocessor.clean_text(query.processed_text or query.text)
        query.processed_text = sanitized_text

        # Log successful processing
        retrieval_logger.info(f"Query processed successfully: {query.query_id}")

        return query

    def process_batch(self, queries: list[Union[str, Query]]) -> list[Query]:
        """
        Process a batch of queries through the pipeline.

        Args:
            queries: List of input queries as strings or Query objects

        Returns:
            List of processed Query objects

        Raises:
            ValueError: If any query fails validation
        """
        processed_queries = []

        for i, query_input in enumerate(queries):
            try:
                processed_query = self.process_query(query_input)
                processed_queries.append(processed_query)
            except Exception as e:
                retrieval_logger.error(f"Failed to process query at index {i}: {str(e)}")
                raise e

        return processed_queries

    def validate_and_process(self, query_input: Union[str, Query]) -> Query:
        """
        Validate and process a query with additional safety checks.

        Args:
            query_input: Input query as string or Query object

        Returns:
            Processed Query object ready for embedding generation
        """
        # First validate the raw input
        if isinstance(query_input, str):
            self.validator.validate_for_retrieval(query_input)
            query = Query(text=query_input)
        else:
            query = query_input

        # Then process through the pipeline
        return self.process_query(query)

    def is_valid_for_processing(self, query_input: Union[str, Query]) -> bool:
        """
        Check if a query is valid for processing without raising exceptions.

        Args:
            query_input: Input query as string or Query object

        Returns:
            True if query is valid for processing, False otherwise
        """
        try:
            if isinstance(query_input, str):
                self.validator.validate_for_retrieval(query_input)
            else:
                self.validator.validate_query_object(query_input)
            return True
        except ValueError:
            return False

    def preprocess_for_embedding(self, text: str) -> str:
        """
        Preprocess text specifically for embedding generation.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text ready for embedding
        """
        # Clean and normalize the text
        cleaned_text = self.preprocessor.clean_text(text)
        normalized_text = self.preprocessor.normalize_text(cleaned_text)

        # Truncate if necessary
        truncated_text = self.preprocessor.truncate_query(normalized_text)

        return truncated_text