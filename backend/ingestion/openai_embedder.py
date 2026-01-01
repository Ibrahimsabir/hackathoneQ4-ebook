"""
Embedder module for generating embeddings using OpenAI
"""
from openai import OpenAI
from typing import List, Dict, Any
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAIEmbedder:
    """
    Class to handle embedding generation using OpenAI
    """

    def __init__(self, rate_limit_per_minute: int = 3000):
        """
        Initialize the OpenAI embedder

        Args:
            rate_limit_per_minute (int): Rate limit for API calls per minute
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"  # 1536 dimensions, cheap and fast
        self.logger = logging.getLogger(__name__)
        self.rate_limit_per_minute = rate_limit_per_minute
        self.last_call_time = 0
        self.calls_this_minute = 0

    def _wait_if_needed(self, num_calls: int = 1):
        """
        Wait if necessary to respect rate limits
        """
        current_time = time.time()

        # Reset counter if a minute has passed
        if current_time - self.last_call_time >= 60:
            self.calls_this_minute = 0
            self.last_call_time = current_time

        # Check if we need to wait
        if self.calls_this_minute + num_calls > self.rate_limit_per_minute:
            wait_time = 60 - (current_time - self.last_call_time)
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.calls_this_minute = 0
                self.last_call_time = time.time()

        self.calls_this_minute += num_calls

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI

        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Size of batches to process (OpenAI supports up to 2048)

        Returns:
            List[List[float]]: List of embeddings, each embedding is a list of floats
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Wait to respect rate limits
            self._wait_if_needed(len(batch))

            try:
                self.logger.info(f"Generating embeddings for batch {i//batch_size + 1}, size: {len(batch)}")

                # Call OpenAI API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float"
                )

                # Extract embeddings in correct order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                self.logger.info(f"Generated {len(batch_embeddings)} embeddings in batch")

            except Exception as e:
                self.logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                # Add empty embeddings for failed items to maintain order
                all_embeddings.extend([[]] * len(batch))

        # Filter out any empty embeddings
        valid_embeddings = [emb for emb in all_embeddings if emb]
        if len(valid_embeddings) != len(texts):
            self.logger.warning(f"Only {len(valid_embeddings)} out of {len(texts)} embeddings were successfully generated")

        return valid_embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model

        Returns:
            int: Embedding dimension (1536 for text-embedding-3-small)
        """
        return 1536
