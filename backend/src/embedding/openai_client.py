import asyncio
import time
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from src.utils.config import Config
from src.utils.logging_config import logger
from src.utils.validation import validate_embedding_dimensions
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAIClient:
    """
    OpenAI client for generating embeddings using text-embedding models.
    """
    def __init__(self):
        """
        Initialize the OpenAI client with API key from configuration.
        """
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_EMBEDDING_MODEL

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts using OpenAI API.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors, or None if failed after retries
        """
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            embeddings = []
            for item in response.data:
                embedding = item.embedding
                # Validate embedding dimensions (should be 1536 for text-embedding-3-small)
                if validate_embedding_dimensions(embedding, expected_size=1536):
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Invalid embedding dimensions: {len(embedding)}")
                    return None

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise  # Re-raise to trigger retry

    async def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

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

    async def batch_generate_embeddings(self, texts: List[str], batch_size: int = 100) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a large list of texts in batches to handle API limits.

        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch

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

                # Add a small delay between batches to respect rate limits
                await asyncio.sleep(Config.RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {str(e)}")
                return None

        return all_embeddings