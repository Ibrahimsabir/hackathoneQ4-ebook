import asyncio
import cohere
from typing import List, Optional
from src.utils.config import Config
from src.utils.logging_config import logger
from src.utils.validation import validate_embedding_dimensions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

class CohereClient:
    """
    Cohere client for generating embeddings using Cohere's embedding models.
    """
    def __init__(self):
        """
        Initialize the Cohere client with API key from configuration.
        """
        self.client = cohere.Client(Config.COHERE_API_KEY)
        self.model = Config.COHERE_EMBEDDING_MODEL

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts using Cohere API.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors, or None if failed after retries
        """
        try:
            # Cohere's embed method is synchronous, but we'll wrap it for consistency
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"  # Using search_document for content chunks
            )

            embeddings = []
            for embedding in response.embeddings:
                # Cohere embeddings are typically 1024 dimensions for multilingual models
                # Validate embedding dimensions
                if validate_embedding_dimensions(embedding, expected_size=len(embedding)):
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Invalid embedding dimensions: {len(embedding)}")
                    return None

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings with Cohere: {str(e)}")
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
            logger.error(f"Failed to generate single embedding with Cohere: {str(e)}")
            return None

    async def batch_generate_embeddings(self, texts: List[str], batch_size: int = 96) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a large list of texts in batches to handle API limits.

        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch (Cohere supports up to 96)

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