"""
Embedding validation utilities for the retrieval pipeline.

This module provides functions for validating embedding vectors to ensure
they meet the required specifications for the retrieval system.
"""
import math
from typing import List
from ..config.retrieval_config import RetrievalConfig
from ..models.retrieval_entities import Embedding


class EmbeddingValidator:
    """Utility class for validating embedding vectors."""

    @staticmethod
    def validate_dimension(embedding: Embedding) -> bool:
        """
        Validate that the embedding has the correct dimension.

        Args:
            embedding: Embedding object to validate

        Returns:
            True if dimension is correct
        """
        expected_dimension = RetrievalConfig.COHERE_EMBEDDING_DIMENSION
        actual_dimension = len(embedding.vector)

        if actual_dimension != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dimension}, "
                f"got {actual_dimension}"
            )

        return True

    @staticmethod
    def validate_finite_values(embedding: Embedding) -> bool:
        """
        Validate that all values in the embedding are finite (not NaN or infinity).

        Args:
            embedding: Embedding object to validate

        Returns:
            True if all values are finite
        """
        for i, value in enumerate(embedding.vector):
            if not math.isfinite(value):
                raise ValueError(f"Embedding contains non-finite value at index {i}: {value}")

        return True

    @staticmethod
    def validate_model(embedding: Embedding) -> bool:
        """
        Validate that the embedding model matches expected configuration.

        Args:
            embedding: Embedding object to validate

        Returns:
            True if model matches expected value
        """
        expected_model = RetrievalConfig.COHERE_MODEL

        if embedding.model != expected_model:
            raise ValueError(
                f"Embedding model mismatch: expected {expected_model}, "
                f"got {embedding.model}"
            )

        return True

    @staticmethod
    def validate_embedding(embedding: Embedding) -> bool:
        """
        Validate an embedding object against all requirements.

        Args:
            embedding: Embedding object to validate

        Returns:
            True if embedding is valid
        """
        # Validate dimension
        EmbeddingValidator.validate_dimension(embedding)

        # Validate finite values
        EmbeddingValidator.validate_finite_values(embedding)

        # Validate model
        EmbeddingValidator.validate_model(embedding)

        return True

    @staticmethod
    def validate_embedding_vector(vector: List[float], expected_dimension: int = None) -> bool:
        """
        Validate an embedding vector directly.

        Args:
            vector: Embedding vector to validate
            expected_dimension: Expected dimension (uses config default if None)

        Returns:
            True if vector is valid
        """
        if expected_dimension is None:
            expected_dimension = RetrievalConfig.COHERE_EMBEDDING_DIMENSION

        # Check dimension
        if len(vector) != expected_dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {expected_dimension}, "
                f"got {len(vector)}"
            )

        # Check for finite values
        for i, value in enumerate(vector):
            if not math.isfinite(value):
                raise ValueError(f"Vector contains non-finite value at index {i}: {value}")

        return True

    @staticmethod
    def normalize_embedding(embedding: Embedding) -> Embedding:
        """
        Normalize an embedding vector to unit length.

        Args:
            embedding: Embedding object to normalize

        Returns:
            Normalized embedding object
        """
        # Calculate magnitude
        magnitude = math.sqrt(sum(v * v for v in embedding.vector))

        # Avoid division by zero
        if magnitude == 0:
            return embedding

        # Normalize vector
        normalized_vector = [v / magnitude for v in embedding.vector]

        return Embedding(
            vector=normalized_vector,
            dimension=embedding.dimension,
            model=embedding.model,
            query_id=embedding.query_id
        )

    @staticmethod
    def are_embeddings_compatible(embedding1: Embedding, embedding2: Embedding) -> bool:
        """
        Check if two embeddings are compatible for similarity calculation.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            True if embeddings are compatible
        """
        # Check dimensions match
        if embedding1.dimension != embedding2.dimension:
            return False

        # Check models match
        if embedding1.model != embedding2.model:
            return False

        return True