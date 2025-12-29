from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from src.utils.config import Config
from src.utils.logging_config import logger
import uuid

class QdrantStorage:
    """
    Qdrant storage client for managing document chunks with embeddings.
    """
    def __init__(self):
        """
        Initialize the Qdrant client with configuration from environment variables.
        """
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
        )
        self.collection_name = Config.QDRANT_COLLECTION_NAME

    def initialize_collection(self, vector_size: int = 1536, distance: str = "Cosine") -> bool:
        """
        Initialize the Qdrant collection with the specified vector size and distance metric.

        Args:
            vector_size: Size of the embedding vectors (default 1536 for OpenAI)
            distance: Distance metric for similarity search (default "Cosine")

        Returns:
            True if collection was created or already exists, False otherwise
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance[distance.upper()]
                    )
                )
                logger.info(f"Created new Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")

            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="section_title",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chapter",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            logger.info("Created payload indexes for efficient filtering")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {str(e)}")
            return False

    def store_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        Store document chunks with embeddings in Qdrant.

        Args:
            chunks: List of document chunks with embeddings and metadata
            batch_size: Number of chunks to process in each batch

        Returns:
            True if all chunks were stored successfully, False otherwise
        """
        try:
            points = []
            for chunk in chunks:
                # Create a PointStruct for each chunk
                point = models.PointStruct(
                    id=chunk.get('id', str(uuid.uuid4())),
                    vector=chunk['embedding'],
                    payload=chunk['payload']
                )
                points.append(point)

            # Batch upload to Qdrant
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch of {len(batch)} points to Qdrant")

            logger.info(f"Successfully stored {len(chunks)} document chunks in Qdrant")
            return True

        except Exception as e:
            logger.error(f"Failed to store embeddings in Qdrant: {str(e)}")
            return False

    def search_similar(self, query_vector: List[float], top_k: int = 5,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks using the query vector.

        Args:
            query_vector: The embedding vector to search for similar items
            top_k: Number of similar items to return
            filters: Optional filters for search (e.g., by source_url, section_title)

        Returns:
            List of similar document chunks with their scores
        """
        try:
            # Build filter conditions if provided
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

            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filters
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'content': result.payload.get('content', ''),
                    'source_url': result.payload.get('source_url', ''),
                    'section_title': result.payload.get('section_title', ''),
                    'chapter': result.payload.get('chapter', ''),
                    'score': result.score,
                    'payload': result.payload
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search similar documents in Qdrant: {str(e)}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection including size and configuration.

        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': collection_info.config.params.vectors.size,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance,
                'point_count': collection_info.points_count,
                'indexed_point_count': collection_info.indexed_vectors_count
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {}

    def delete_collection(self) -> bool:
        """
        Delete the entire collection (use with caution).

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            return False