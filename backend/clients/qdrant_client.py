"""
Qdrant client for the retrieval pipeline.

This module provides a wrapper around the Qdrant API for performing similarity
search, with proper authentication and error handling.
"""
import time
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from ..config.retrieval_config import RetrievalConfig
from ..utils.logging import retrieval_logger
from ..models.retrieval_entities import RetrievedChunk


class QdrantRetrievalClient:
    """Client for interacting with Qdrant vector database."""

    def __init__(self):
        """Initialize the Qdrant client with configuration."""
        RetrievalConfig.validate_config()

        self.collection_name = RetrievalConfig.QDRANT_COLLECTION_NAME
        self.timeout = RetrievalConfig.QDRANT_TIMEOUT
        self.top_k = RetrievalConfig.TOP_K
        self.min_similarity_threshold = RetrievalConfig.MIN_SIMILARITY_THRESHOLD

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=RetrievalConfig.QDRANT_URL,
            api_key=RetrievalConfig.QDRANT_API_KEY,
            timeout=self.timeout
        )

    def search_vectors(self, query_vector: List[float], top_k: Optional[int] = None,
                      min_similarity_threshold: Optional[float] = None) -> List[RetrievedChunk]:
        """
        Perform similarity search in Qdrant collection.

        Args:
            query_vector: Query embedding vector to search for
            top_k: Number of results to return (uses config default if None)
            min_similarity_threshold: Minimum similarity threshold (uses config default if None)

        Returns:
            List of RetrievedChunk objects
        """
        if top_k is None:
            top_k = self.top_k

        if min_similarity_threshold is None:
            min_similarity_threshold = self.min_similarity_threshold

        # Perform search with cosine similarity
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=min_similarity_threshold,  # Filter by minimum similarity
            with_payload=True  # Include payload data
        )

        # Convert search results to RetrievedChunk objects
        retrieved_chunks = []
        for result in search_results:
            # Extract payload data
            payload = result.payload

            # Create RetrievedChunk object
            chunk = RetrievedChunk(
                content=payload.get('content', ''),
                url=payload.get('url', ''),
                page_title=payload.get('page_title', ''),
                section_heading=payload.get('section_heading', ''),
                chunk_id=payload.get('chunk_id', ''),
                similarity_score=result.score,
                content_hash=payload.get('content_hash', ''),
                position=payload.get('position', 0)
            )

            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def validate_connection(self) -> bool:
        """
        Validate that the connection to Qdrant is working.

        Returns:
            True if connection is valid
        """
        try:
            # Try to get collection info to validate connection
            collection_info = self.client.get_collection(self.collection_name)
            return True
        except Exception as e:
            retrieval_logger.error(f"Qdrant connection validation failed: {str(e)}")
            return False

    def get_total_points(self) -> int:
        """
        Get the total number of points in the collection.

        Returns:
            Total number of points in the collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            retrieval_logger.error(f"Failed to get total points: {str(e)}")
            return 0

    def validate_embedding_dimension(self) -> bool:
        """
        Validate that the embedding dimension matches expected value.

        Returns:
            True if dimension matches expected value
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            vector_params = collection_info.config.params.vectors
            actual_dimension = vector_params.size if hasattr(vector_params, 'size') else vector_params['size']
            expected_dimension = RetrievalConfig.COHERE_EMBEDDING_DIMENSION

            if actual_dimension != expected_dimension:
                raise ValueError(f"Qdrant collection dimension {actual_dimension} doesn't match "
                               f"expected dimension {expected_dimension}")

            return True
        except Exception as e:
            retrieval_logger.error(f"Failed to validate embedding dimension: {str(e)}")
            return False

    def close(self):
        """Close the Qdrant client connection."""
        if hasattr(self.client, 'close'):
            self.client.close()