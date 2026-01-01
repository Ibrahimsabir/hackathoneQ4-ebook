"""
Embedding service for the retrieval pipeline.

This module provides a service for generating embeddings from text queries,
with proper error handling, validation, and integration with the Cohere client.
"""
from typing import List, Optional
from ..clients.cohere_client import CohereClient
from ..models.retrieval_entities import Embedding
from ..utils.embedding_validator import EmbeddingValidator
from ..config.retrieval_config import RetrievalConfig
from ..utils.logging import retrieval_logger
from datetime import datetime


class EmbeddingService:
    """Service for generating embeddings from text queries."""

    def __init__(self):
        """Initialize the embedding service."""
        self.cohere_client = CohereClient()
        self.validator = EmbeddingValidator()

    def generate_embedding(self, text: str, query_id: Optional[str] = None) -> Embedding:
        """
        Generate an embedding for a single text query.

        Args:
            text: Text to generate embedding for
            query_id: Optional query ID for logging

        Returns:
            Embedding object with the generated vector

        Raises:
            Exception: If embedding generation fails
        """
        start_time = datetime.now()

        try:
            # Generate embedding using Cohere client
            embedding_vector = self.cohere_client.generate_single_embedding(text)

            # Create Embedding object
            embedding = Embedding(
                vector=embedding_vector,
                dimension=RetrievalConfig.COHERE_EMBEDDING_DIMENSION,
                model=RetrievalConfig.COHERE_MODEL,
                query_id=query_id
            )

            # Validate the embedding
            self.validator.validate_embedding(embedding)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms

            # Log success
            retrieval_logger.log_embedding_generation(
                query_id=query_id or "unknown",
                success=True,
                duration_ms=duration
            )

            return embedding

        except Exception as e:
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms

            # Log error
            retrieval_logger.log_embedding_generation(
                query_id=query_id or "unknown",
                success=False,
                duration_ms=duration
            )
            retrieval_logger.log_error(
                query_id=query_id or "unknown",
                error_type="EmbeddingGenerationError",
                error_message=str(e)
            )

            # Re-raise the exception
            raise e

    def generate_embeddings_batch(self, texts: List[str], query_ids: Optional[List[str]] = None) -> List[Embedding]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to generate embeddings for
            query_ids: Optional list of query IDs for logging

        Returns:
            List of Embedding objects

        Raises:
            Exception: If embedding generation fails
        """
        if query_ids and len(query_ids) != len(texts):
            raise ValueError("Number of query IDs must match number of texts")

        start_time = datetime.now()

        try:
            # Generate embeddings using Cohere client
            embedding_vectors = self.cohere_client.generate_embeddings(texts)

            embeddings = []
            for i, vector in enumerate(embedding_vectors):
                query_id = query_ids[i] if query_ids else f"batch_{i}"

                # Create Embedding object
                embedding = Embedding(
                    vector=vector,
                    dimension=RetrievalConfig.COHERE_EMBEDDING_DIMENSION,
                    model=RetrievalConfig.COHERE_MODEL,
                    query_id=query_id
                )

                # Validate the embedding
                self.validator.validate_embedding(embedding)
                embeddings.append(embedding)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms

            # Log success for each embedding
            for i, embedding in enumerate(embeddings):
                retrieval_logger.log_embedding_generation(
                    query_id=embedding.query_id,
                    success=True,
                    duration_ms=duration / len(embeddings)  # Average duration per embedding
                )

            return embeddings

        except Exception as e:
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms

            # Log error
            retrieval_logger.log_error(
                query_id="batch",
                error_type="BatchEmbeddingGenerationError",
                error_message=str(e)
            )

            # Re-raise the exception
            raise e

    def validate_embedding_service(self) -> bool:
        """
        Validate that the embedding service is properly configured and working.

        Returns:
            True if service is valid and working
        """
        try:
            # Test with a simple text
            test_text = "test query for validation"
            test_embedding = self.generate_embedding(test_text)

            # Validate the generated embedding
            self.validator.validate_embedding(test_embedding)

            return True
        except Exception as e:
            retrieval_logger.error(f"Embedding service validation failed: {str(e)}")
            return False