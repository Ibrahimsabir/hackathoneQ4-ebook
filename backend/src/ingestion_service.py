import asyncio
from typing import List, Dict, Any, Optional
from src.crawler.content_crawler import ContentCrawler
from src.embedding.openai_client import OpenAIClient
from src.storage.qdrant_client import QdrantStorage
from src.models.document_chunk import DocumentChunk
from src.utils.config import Config
from src.utils.logging_config import logger
from src.utils.validation import validate_embedding_dimensions, validate_content_length
import time
from tqdm import tqdm


class IngestionService:
    """
    Main service to orchestrate the entire ingestion pipeline:
    crawl -> chunk -> embed -> store
    """
    def __init__(self):
        """
        Initialize all required components.
        """
        self.crawler = ContentCrawler()
        self.embedding_client = OpenAIClient()
        self.qdrant_storage = QdrantStorage()
        self.qdrant_storage.initialize_collection()

    async def run_ingestion_pipeline(self) -> bool:
        """
        Execute the complete ingestion pipeline:
        1. Crawl the book content from sitemap
        2. Extract and chunk the content
        3. Generate embeddings for chunks
        4. Store embeddings in Qdrant

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting ingestion pipeline...")

            # Step 1: Fetch URLs from sitemap
            logger.info("Fetching URLs from sitemap...")
            urls = self.crawler.fetch_sitemap()
            if not urls:
                logger.error("Failed to fetch URLs from sitemap")
                return False

            logger.info(f"Found {len(urls)} URLs to process")

            # Step 2: Crawl all content
            logger.info("Starting content crawling...")
            content_data_list = await self.crawler.crawl_all_content(urls)
            if not content_data_list:
                logger.error("No content was extracted from URLs")
                return False

            # Step 3: Chunk content
            logger.info("Chunking content...")
            all_chunks = []
            for content_data in content_data_list:
                chunks = self.crawler.chunk_content(
                    content_data,
                    max_chunk_size=Config.CHUNK_SIZE,
                    overlap=Config.CHUNK_OVERLAP
                )
                all_chunks.extend(chunks)

            logger.info(f"Created {len(all_chunks)} content chunks")

            # Step 4: Prepare content for embedding
            logger.info("Preparing content for embedding...")
            texts_to_embed = []
            chunk_mapping = []  # To keep track of which text corresponds to which chunk

            for i, chunk in enumerate(all_chunks):
                # Validate content length before embedding
                if validate_content_length(chunk.content):
                    texts_to_embed.append(chunk.content)
                    chunk_mapping.append((i, chunk))
                else:
                    logger.warning(f"Skipping chunk {i} due to content length validation failure")

            logger.info(f"Preparing to embed {len(texts_to_embed)} chunks...")

            # Step 5: Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = await self.embedding_client.batch_generate_embeddings(
                texts_to_embed,
                batch_size=Config.BATCH_SIZE
            )

            if not embeddings or len(embeddings) != len(texts_to_embed):
                logger.error("Failed to generate embeddings for all chunks")
                return False

            # Step 6: Associate embeddings with chunks
            logger.info("Associating embeddings with chunks...")
            chunks_with_embeddings = []
            for i, (chunk_idx, chunk) in enumerate(chunk_mapping):
                # Create a new chunk with embedding
                chunk_with_embedding = DocumentChunk(
                    id=chunk.id,
                    content=chunk.content,
                    embedding=embeddings[i],
                    source_url=chunk.source_url,
                    section_title=chunk.section_title,
                    chapter=chunk.chapter,
                    position=chunk.position,
                    metadata=chunk.metadata,
                    created_at=chunk.created_at,
                    updated_at=chunk.updated_at
                )
                chunks_with_embeddings.append(chunk_with_embedding)

            logger.info(f"Associated embeddings with {len(chunks_with_embeddings)} chunks")

            # Step 7: Prepare data for storage in Qdrant
            logger.info("Preparing data for Qdrant storage...")
            qdrant_points = []
            for chunk in chunks_with_embeddings:
                # Validate embedding dimensions
                if not validate_embedding_dimensions(chunk.embedding):
                    logger.warning(f"Invalid embedding dimensions for chunk {chunk.id}")
                    continue

                point = {
                    'id': chunk.id,
                    'embedding': chunk.embedding,
                    'payload': chunk.to_payload()
                }
                qdrant_points.append(point)

            logger.info(f"Prepared {len(qdrant_points)} points for storage")

            # Step 8: Store in Qdrant
            logger.info("Storing embeddings in Qdrant...")
            success = self.qdrant_storage.store_embeddings(
                qdrant_points,
                batch_size=Config.BATCH_SIZE
            )

            if success:
                logger.info("Ingestion pipeline completed successfully!")
                return True
            else:
                logger.error("Failed to store embeddings in Qdrant")
                return False

        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {str(e)}")
            return False

    async def validate_storage(self) -> bool:
        """
        Validate that the embeddings were stored correctly in Qdrant.

        Returns:
            True if validation passes, False otherwise
        """
        try:
            collection_info = self.qdrant_storage.get_collection_info()
            if not collection_info:
                logger.error("Failed to get collection info for validation")
                return False

            if collection_info.get('point_count', 0) == 0:
                logger.warning("Collection is empty - no points stored")
                return False

            logger.info(f"Collection validation passed: {collection_info['point_count']} points stored")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False

    async def test_similarity_search(self, query: str = "What is Physical AI?") -> List[Dict[str, Any]]:
        """
        Test similarity search to verify the embeddings work correctly.

        Args:
            query: Test query for similarity search

        Returns:
            List of similar document chunks
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_client.generate_single_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate embedding for test query")
                return []

            # Search for similar documents
            results = self.qdrant_storage.search_similar(query_embedding, top_k=3)
            logger.info(f"Test search returned {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Test search failed: {str(e)}")
            return []

    def get_ingestion_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the ingestion process.

        Returns:
            Dictionary with ingestion metrics
        """
        collection_info = self.qdrant_storage.get_collection_info()
        return {
            'collection_name': Config.QDRANT_COLLECTION_NAME,
            'vector_size': collection_info.get('vector_size', 0),
            'point_count': collection_info.get('point_count', 0),
            'indexed_point_count': collection_info.get('indexed_point_count', 0),
            'embedding_model': Config.OPENAI_EMBEDDING_MODEL
        }


# Main execution function
async def main():
    """
    Main function to run the ingestion pipeline.
    """
    # Create ingestion service
    ingestion_service = IngestionService()

    # Run the ingestion pipeline
    success = await ingestion_service.run_ingestion_pipeline()

    if success:
        logger.info("Ingestion completed successfully!")

        # Validate storage
        validation_success = await ingestion_service.validate_storage()
        if validation_success:
            logger.info("Storage validation passed!")

            # Run a test search
            test_results = await ingestion_service.test_similarity_search()
            logger.info(f"Test search completed with {len(test_results)} results")

            # Print metrics
            metrics = ingestion_service.get_ingestion_metrics()
            logger.info(f"Ingestion metrics: {metrics}")
        else:
            logger.error("Storage validation failed!")
    else:
        logger.error("Ingestion failed!")


if __name__ == "__main__":
    asyncio.run(main())