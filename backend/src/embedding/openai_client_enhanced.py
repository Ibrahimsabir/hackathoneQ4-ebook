import asyncio
import time
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from src.utils.config import Config
from src.utils.logging_config import logger
from src.utils.validation import validate_embedding_dimensions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
import json
import os
from datetime import datetime, timedelta

class OpenAIClientEnhanced:
    """
    Enhanced OpenAI client for generating embeddings with better rate limit handling and caching.
    """
    def __init__(self):
        """
        Initialize the OpenAI client with API key from configuration.
        """
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_EMBEDDING_MODEL
        self.cache_file = "embedding_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to actual content
                    return {item['content']: item['embedding'] for item in data}
            except Exception as e:
                logger.warning(f"Could not load cache file: {e}")
        return {}

    def _save_cache(self):
        """Save embedding cache to file."""
        try:
            # Convert cache to list of objects for JSON serialization
            cache_list = [
                {"content": content, "embedding": embedding}
                for content, embedding in self.cache.items()
            ]
            with open(self.cache_file, 'w') as f:
                json.dump(cache_list, f)
        except Exception as e:
            logger.error(f"Could not save cache file: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    async def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts using OpenAI API with caching.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors, or None if failed after retries
        """
        try:
            # Check cache first for each text
            results = []
            texts_to_request = []
            text_indices = []  # Track which original indices need API calls

            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    results.append(self.cache[cache_key])
                    logger.info(f"Retrieved embedding from cache for text: {text[:50]}...")
                else:
                    results.append(None)  # Placeholder
                    texts_to_request.append(text)
                    text_indices.append(i)

            # If all texts were cached, return immediately
            if not texts_to_request:
                return results

            # Request embeddings for uncached texts
            if texts_to_request:
                logger.info(f"Requesting {len(texts_to_request)} embeddings from API...")
                response = await self.client.embeddings.create(
                    input=texts_to_request,
                    model=self.model
                )

                # Process the response and update cache
                for i, item in enumerate(response.data):
                    embedding = item.embedding
                    original_index = text_indices[i]
                    text = texts_to_request[i]

                    # Validate embedding dimensions
                    if validate_embedding_dimensions(embedding, expected_size=1536):
                        results[original_index] = embedding
                        # Add to cache
                        cache_key = self._get_cache_key(text)
                        self.cache[cache_key] = embedding
                    else:
                        logger.warning(f"Invalid embedding dimensions: {len(embedding)} for text: {text[:50]}...")
                        return None

                # Save cache after successful API call
                self._save_cache()

            return results

        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}")
            # Wait longer before retry
            await asyncio.sleep(60)
            raise
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise  # Re-raise to trigger retry

    async def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector, or None if failed
        """
        try:
            result = await self.generate_embeddings([text])
            if result and len(result) > 0:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Failed to generate single embedding: {str(e)}")
            return None

    async def batch_generate_embeddings(self, texts: List[str], batch_size: int = 20) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a large list of texts in batches to handle API limits.

        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch (reduced for quota management)

        Returns:
            List of embedding vectors, or None if failed
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                batch_embeddings = await self.generate_embeddings(batch)
                if batch_embeddings is None:
                    logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}")
                    return None
                all_embeddings.extend(batch_embeddings)

                # Add a delay between batches to respect rate limits
                await asyncio.sleep(Config.RATE_LIMIT_DELAY * 5)  # Increased delay

            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {str(e)}")
                # Wait longer before continuing
                await asyncio.sleep(60)
                return None

        return all_embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            "cached_items": len(self.cache),
            "cache_file": self.cache_file,
            "last_updated": datetime.now().isoformat()
        }