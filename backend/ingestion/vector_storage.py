"""
Vector storage module for storing embeddings in Qdrant
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import logging
from logging_config import logger
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, VECTOR_DIMENSION
import uuid


class QdrantVectorStorage:
    """
    Class to handle vector storage in Qdrant
    """

    def __init__(self):
        """
        Initialize the Qdrant client and connect to the cluster
        """
        self.client = QdrantClient(
            url=QDRANT_URL.replace("https://", "").replace(":6333", ""),  # Extract just the host for cloud
            api_key=QDRANT_API_KEY,
            https=True
        )
        self.collection_name = QDRANT_COLLECTION_NAME
        self.vector_dimension = VECTOR_DIMENSION
        self.logger = logger

        # Ensure the collection exists
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Ensure the collection exists in Qdrant with the correct configuration
        """
        try:
            # Try to get collection info to see if it exists
            self.client.get_collection(self.collection_name)
            self.logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            # Collection doesn't exist, create it
            self.logger.info(f"Creating collection '{self.collection_name}'")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dimension,
                    distance=models.Distance.COSINE  # Use cosine similarity as specified
                )
            )

            # Create payload index for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_title",
                field_schema=models.PayloadSchemaType.TEXT
            )

            self.logger.info(f"Collection '{self.collection_name}' created successfully")

    def store_embeddings(
        self,
        embeddings: List[List[float]],
        chunk_data_list: List[Dict[str, str]],
        batch_size: int = 64
    ) -> bool:
        """
        Store embeddings in Qdrant with metadata

        Args:
            embeddings (List[List[float]]): List of embeddings to store
            chunk_data_list (List[Dict[str, str]]): List of chunk data with metadata
            batch_size (int): Size of batches to store

        Returns:
            bool: True if storage was successful
        """
        if len(embeddings) != len(chunk_data_list):
            self.logger.error("Mismatch between number of embeddings and chunk data")
            return False

        if not embeddings:
            self.logger.warning("No embeddings to store")
            return True

        try:
            # Process in batches to handle large amounts of data efficiently
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                batch_chunks = chunk_data_list[i:i + batch_size]

                # Prepare points for insertion
                points = []
                for embedding, chunk_data in zip(batch_embeddings, batch_chunks):
                    # Create a unique ID for the point
                    point_id = str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{chunk_data['url']}_{chunk_data['chunk_id']}"
                    ))

                    # Prepare payload with metadata
                    payload = {
                        "url": chunk_data.get("url", ""),
                        "page_title": chunk_data.get("page_title", ""),
                        "section_heading": chunk_data.get("section_heading", ""),
                        "chunk_id": chunk_data.get("chunk_id", ""),
                        "content": chunk_data.get("content", "")[:1000],  # Store first 1000 chars as preview
                        "content_length": len(chunk_data.get("content", ""))
                    }

                    # Add content hash if available
                    if "content_hash" in chunk_data:
                        payload["content_hash"] = chunk_data["content_hash"]

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    )

                # Upsert the points (this ensures idempotent operations)
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Wait for operation to complete
                )

                self.logger.info(f"Stored batch {i//batch_size + 1} with {len(points)} vectors")

            self.logger.info(f"Successfully stored {len(embeddings)} embeddings in Qdrant")
            return True

        except Exception as e:
            self.logger.error(f"Error storing embeddings in Qdrant: {e}")
            return False

    def idempotent_store_embeddings(
        self,
        embeddings: List[List[float]],
        chunk_data_list: List[Dict[str, str]],
        batch_size: int = 64
    ) -> bool:
        """
        Store embeddings in Qdrant with idempotent operations to prevent duplicates

        Args:
            embeddings (List[List[float]]): List of embeddings to store
            chunk_data_list (List[Dict[str, str]]): List of chunk data with metadata
            batch_size (int): Size of batches to store

        Returns:
            bool: True if storage was successful
        """
        if len(embeddings) != len(chunk_data_list):
            self.logger.error("Mismatch between number of embeddings and chunk data")
            return False

        if not embeddings:
            self.logger.warning("No embeddings to store")
            return True

        try:
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                batch_chunks = chunk_data_list[i:i + batch_size]

                # Prepare points for insertion
                points = []
                for embedding, chunk_data in zip(batch_embeddings, batch_chunks):
                    # Create a deterministic ID based on URL and chunk_id to ensure idempotency
                    point_id = str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{chunk_data['url']}_{chunk_data['chunk_id']}"
                    ))

                    # Prepare payload with metadata
                    payload = {
                        "url": chunk_data.get("url", ""),
                        "page_title": chunk_data.get("page_title", ""),
                        "section_heading": chunk_data.get("section_heading", ""),
                        "chunk_id": chunk_data.get("chunk_id", ""),
                        "content": chunk_data.get("content", "")[:1000],  # Store first 1000 chars as preview
                        "content_length": len(chunk_data.get("content", ""))
                    }

                    # Add content hash if available for change detection
                    if "content_hash" in chunk_data:
                        payload["content_hash"] = chunk_data["content_hash"]

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    )

                # Use upsert to ensure idempotent operations (will update if exists, create if not)
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Wait for operation to complete
                )

                self.logger.info(f"Upserted batch {i//batch_size + 1} with {len(points)} vectors (idempotent)")

            self.logger.info(f"Successfully upserted {len(embeddings)} embeddings in Qdrant (idempotent)")
            return True

        except Exception as e:
            self.logger.error(f"Error during idempotent storage in Qdrant: {e}")
            return False

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant

        Args:
            query_embedding (List[float]): Query embedding to search for
            top_k (int): Number of similar vectors to return
            filters (Optional[Dict[str, Any]]): Optional filters for search

        Returns:
            List[Dict[str, Any]]: List of similar vectors with metadata
        """
        try:
            # Prepare filters if provided
            qdrant_filters = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )

                if filter_conditions:
                    qdrant_filters = models.Filter(
                        must=filter_conditions
                    )

            # Perform search - using the newer query_points API
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qdrant_filters
            )

            # Format results - newer API returns QueryResponse object with points
            results = []
            for point in search_results.points:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })

            self.logger.debug(f"Found {len(results)} similar vectors")
            return results

        except Exception as e:
            self.logger.error(f"Error searching in Qdrant: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection

        Returns:
            Dict[str, Any]: Collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "point_count": collection_info.points_count
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_collection(self):
        """
        Delete the collection (useful for testing/reinitialization)
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")


def test_vector_storage():
    """
    Test function to verify the vector storage works
    """
    storage = QdrantVectorStorage()

    # Get collection info
    info = storage.get_collection_info()
    print(f"Collection info: {info}")

    # Test with dummy embeddings (for testing purposes)
    # In real usage, these would come from the embedder
    dummy_embeddings = [
        [0.1, 0.2, 0.3] + [0.0] * (VECTOR_DIMENSION - 3),  # Pad to correct dimension
        [0.4, 0.5, 0.6] + [0.0] * (VECTOR_DIMENSION - 3)
    ]

    dummy_chunks = [
        {
            "url": "https://example.com/test1",
            "page_title": "Test Page 1",
            "section_heading": "Introduction",
            "chunk_id": "1",
            "content": "This is test content 1"
        },
        {
            "url": "https://example.com/test2",
            "page_title": "Test Page 2",
            "section_heading": "Conclusion",
            "chunk_id": "2",
            "content": "This is test content 2"
        }
    ]

    # Store test embeddings
    success = storage.idempotent_store_embeddings(dummy_embeddings, dummy_chunks)
    print(f"Storage test {'succeeded' if success else 'failed'}")

    # Test search if we have embeddings
    if success and dummy_embeddings:
        results = storage.search_similar(dummy_embeddings[0], top_k=2)
        print(f"Search results: {len(results)} items found")


if __name__ == "__main__":
    test_vector_storage()