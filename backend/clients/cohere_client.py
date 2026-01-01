"""
Cohere client for the retrieval pipeline.

This module provides a wrapper around the Cohere API for generating embeddings,
with built-in rate limiting, retry logic, and error handling.
"""
import time
import asyncio
from typing import List, Optional
from functools import wraps
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ..config.retrieval_config import RetrievalConfig
from ..utils.logging import retrieval_logger


class RateLimiter:
    """Simple rate limiter to respect API limits."""

    def __init__(self, requests_per_minute: int):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # Time between requests
        self.last_request_time = 0

    def wait_if_needed(self):
        """Wait if needed to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.interval:
            sleep_time = self.interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for retrying with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        # Last attempt, raise the exception
                        raise e

                    # Calculate delay with exponential backoff
                    delay = base_delay * (2 ** attempt)
                    retrieval_logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)

            return None
        return wrapper
    return decorator


class CohereClient:
    """Client for interacting with Cohere API."""

    def __init__(self):
        """Initialize the Cohere client with configuration."""
        RetrievalConfig.validate_config()

        self.api_key = RetrievalConfig.COHERE_API_KEY
        self.model = RetrievalConfig.COHERE_MODEL
        self.dimension = RetrievalConfig.COHERE_EMBEDDING_DIMENSION
        self.timeout = RetrievalConfig.COHERE_TIMEOUT

        # Set up rate limiter based on configuration
        self.rate_limiter = RateLimiter(RetrievalConfig.COHERE_RPM_LIMIT)

        # Set up session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors

        Raises:
            Exception: If the API call fails
        """
        # Respect rate limits
        self.rate_limiter.wait_if_needed()

        url = "https://api.cohere.ai/v1/embed"

        payload = {
            "texts": texts,
            "model": self.model,
            "input_type": "search_query"  # Optimize for search queries
        }

        response = self.session.post(
            url,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise Exception(f"Cohere API request failed: {response.status_code} - {response.text}")

        result = response.json()

        # Validate embedding dimensions
        embeddings = result.get("embeddings", [])
        for embedding in embeddings:
            if len(embedding) != self.dimension:
                raise ValueError(f"Expected embedding dimension {self.dimension}, got {len(embedding)}")

        return embeddings

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    def validate_configuration(self) -> bool:
        """
        Validate that the Cohere client is properly configured.

        Returns:
            True if configuration is valid
        """
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set")

        if not self.model:
            raise ValueError("COHERE_MODEL is not set")

        return True