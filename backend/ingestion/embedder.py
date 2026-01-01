"""
Embedder module for generating embeddings using Cohere AI
"""
import cohere
from typing import List, Dict, Any
import logging
import time
from logging_config import logger
from config import COHERE_API_KEY, EMBEDDING_MODEL
from .retry_handler import RetryHandler


class CohereEmbedder:
    """
    Class to handle embedding generation using Cohere AI
    """

    def __init__(self, rate_limit_per_minute: int = 100):
        """
        Initialize the Cohere embedder

        Args:
            rate_limit_per_minute (int): Rate limit for API calls per minute
        """
        self.client = cohere.Client(COHERE_API_KEY)
        self.model = EMBEDDING_MODEL
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.logger = logger
        self.retry_handler = RetryHandler(max_retries=3, base_delay=1.0)

    def generate_embeddings(self, texts: List[str], batch_size: int = 96) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Cohere

        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Size of batches to process (Cohere free tier supports up to 96)

        Returns:
            List[List[float]]: List of embeddings, each embedding is a list of floats
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches to respect rate limits and API constraints
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Wait to respect rate limits
            self.rate_limiter.wait_if_needed(len(batch))

            try:
                self.logger.info(f"Generating embeddings for batch {i//batch_size + 1}, size: {len(batch)}")

                # Use retry logic for the API call
                response = self.retry_handler.retry_with_backoff(
                    self._call_cohere_api, batch
                )

                if response and hasattr(response, 'embeddings') and response.embeddings:
                    all_embeddings.extend(response.embeddings)
                    self.logger.debug(f"Generated {len(response.embeddings)} embeddings in batch")
                else:
                    self.logger.error(f"No embeddings returned for batch {i//batch_size + 1}")
                    # Add empty embeddings for failed items to maintain order
                    all_embeddings.extend([[]] * len(batch))

            except Exception as e:
                self.logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                # Add empty embeddings for failed items to maintain order
                all_embeddings.extend([[]] * len(batch))

        # Filter out any empty embeddings that might have been added due to errors
        valid_embeddings = [emb for emb in all_embeddings if emb]
        if len(valid_embeddings) != len(texts):
            self.logger.warning(f"Only {len(valid_embeddings)} out of {len(texts)} embeddings were successfully generated")

        return valid_embeddings

    def validate_embeddings(self, embeddings: List[List[float]], min_dimension: int = 100) -> bool:
        """
        Validate that embeddings meet quality requirements

        Args:
            embeddings (List[List[float]]): List of embeddings to validate
            min_dimension (int): Minimum required dimension for embeddings

        Returns:
            bool: True if embeddings are valid
        """
        if not embeddings:
            self.logger.error("No embeddings to validate")
            return False

        # Check that all embeddings have the same dimension
        first_dimension = len(embeddings[0])
        if first_dimension < min_dimension:
            self.logger.error(f"Embedding dimension {first_dimension} is below minimum {min_dimension}")
            return False

        for i, embedding in enumerate(embeddings):
            if len(embedding) != first_dimension:
                self.logger.error(f"Embedding {i} has inconsistent dimension: {len(embedding)} vs {first_dimension}")
                return False

            # Check for NaN or infinite values
            if any(not (float('-inf') < val < float('inf')) for val in embedding):
                self.logger.error(f"Embedding {i} contains invalid values (NaN or infinite)")
                return False

        self.logger.info(f"Validated {len(embeddings)} embeddings with dimension {first_dimension}")
        return True

    def _call_cohere_api(self, texts: List[str]):
        """
        Internal method to call Cohere API (used for retry logic)

        Args:
            texts (List[str]): Texts to embed

        Returns:
            Cohere API response
        """
        return self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"  # Optimize for search use case
        )

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings generated by this model

        Returns:
            int: Dimension of embeddings
        """
        # Test with a simple text to get the dimension
        try:
            test_response = self.client.embed(
                texts=["test"],
                model=self.model,
                input_type="search_document"
            )
            if test_response.embeddings and len(test_response.embeddings) > 0:
                return len(test_response.embeddings[0])
        except Exception as e:
            self.logger.error(f"Error getting embedding dimension: {e}")
            # Default to 1024 which is common for Cohere models
            return 1024

        return 1024  # Default fallback


class RateLimiter:
    """
    Simple rate limiter to respect API limits
    """

    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter

        Args:
            requests_per_minute (int): Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_time_between_requests = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.logger = logger

    def wait_if_needed(self, batch_size: int = 1):
        """
        Wait if needed to respect rate limits

        Args:
            batch_size (int): Number of requests in current batch
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        # Calculate required time based on batch size
        required_time = self.min_time_between_requests * batch_size

        if time_since_last_request < required_time:
            sleep_time = required_time - time_since_last_request
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


def test_embedder():
    """
    Test function to verify the embedder works
    """
    embedder = CohereEmbedder()

    # Test with sample texts
    test_texts = [
        "This is a test sentence for embedding.",
        "Another sentence to test the embedding functionality.",
        "The quick brown fox jumps over the lazy dog."
    ]

    try:
        embeddings = embedder.generate_embeddings(test_texts)
        print(f"Generated {len(embeddings)} embeddings")
        if embeddings:
            print(f"Embedding dimension: {len(embeddings[0])}")
            print("First embedding (first 10 values):", embeddings[0][:10])
    except Exception as e:
        print(f"Embedding test failed: {e}")


if __name__ == "__main__":
    test_embedder()