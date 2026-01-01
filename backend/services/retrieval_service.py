"""
Retrieval service for the semantic search functionality.

This module provides the core retrieval functionality that performs
similarity search against the Qdrant vector database.
"""
from typing import List, Optional
from ..clients.qdrant_client import QdrantRetrievalClient
from ..models.retrieval_entities import Embedding, RetrievedChunk
from ..config.retrieval_config import RetrievalConfig
from ..utils.logging import retrieval_logger
from datetime import datetime


class RetrievalService:
    """Service for performing semantic similarity search."""

    def __init__(self):
        """Initialize the retrieval service."""
        self.qdrant_client = QdrantRetrievalClient()

    def retrieve(self, query_embedding: Embedding, top_k: Optional[int] = None,
                min_similarity_threshold: Optional[float] = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks based on query embedding.

        Args:
            query_embedding: Embedding vector to search for
            top_k: Number of results to return (uses config default if None)
            min_similarity_threshold: Minimum similarity threshold (uses config default if None)

        Returns:
            List of RetrievedChunk objects
        """
        start_time = datetime.now()

        try:
            # Validate the embedding
            if not query_embedding.vector or len(query_embedding.vector) != query_embedding.dimension:
                raise ValueError("Invalid embedding vector or dimension mismatch")

            # Perform similarity search
            retrieved_chunks = self.qdrant_client.search_vectors(
                query_vector=query_embedding.vector,
                top_k=top_k,
                min_similarity_threshold=min_similarity_threshold
            )

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms

            # Log the search results
            retrieval_logger.log_similarity_search(
                query_id=query_embedding.query_id or "unknown",
                num_results=len(retrieved_chunks),
                duration_ms=duration
            )

            return retrieved_chunks

        except Exception as e:
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms

            # Log error
            retrieval_logger.log_error(
                query_id=query_embedding.query_id or "unknown",
                error_type="RetrievalError",
                error_message=str(e)
            )

            # Re-raise the exception
            raise e

    def retrieve_with_filters(self, query_embedding: Embedding, filters: Optional[dict] = None,
                             top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks with additional filters.

        Args:
            query_embedding: Embedding vector to search for
            filters: Additional filters to apply to the search
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of RetrievedChunk objects
        """
        # For now, this is a simplified version that just calls retrieve
        # In a more complex implementation, we would apply filters before searching
        return self.retrieve(query_embedding, top_k)

    def validate_retrieval_service(self) -> bool:
        """
        Validate that the retrieval service is properly configured and working.

        Returns:
            True if service is valid and working
        """
        try:
            # Test connection to Qdrant
            if not self.qdrant_client.validate_connection():
                raise Exception("Qdrant connection validation failed")

            # Test embedding dimension compatibility
            if not self.qdrant_client.validate_embedding_dimension():
                raise Exception("Embedding dimension validation failed")

            return True
        except Exception as e:
            retrieval_logger.error(f"Retrieval service validation failed: {str(e)}")
            return False

    def get_total_documents(self) -> int:
        """
        Get the total number of documents in the collection.

        Returns:
            Total number of documents in the collection
        """
        return self.qdrant_client.get_total_points()

    def search_with_custom_threshold(self, query_embedding: Embedding, min_similarity: float,
                                    top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Perform search with a custom similarity threshold.

        Args:
            query_embedding: Embedding vector to search for
            min_similarity: Minimum similarity threshold
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of RetrievedChunk objects
        """
        return self.retrieve(query_embedding, top_k, min_similarity_threshold=min_similarity)

    def close(self):
        """Close the retrieval service and any open connections."""
        self.qdrant_client.close()