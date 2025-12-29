import asyncio
from src.ingestion_service import IngestionService
from src.utils.config import Config
from src.utils.logging_config import logger
import os

async def test_ingestion_pipeline():
    """
    Test the ingestion pipeline with basic validation.
    """
    print("Testing Ingestion Pipeline...")

    # Check that required environment variables are set
    if not Config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return False

    if not Config.QDRANT_URL:
        print("ERROR: QDRANT_URL environment variable not set")
        return False

    if not Config.QDRANT_API_KEY:
        print("ERROR: QDRANT_API_KEY environment variable not set")
        return False

    try:
        # Create ingestion service
        ingestion_service = IngestionService()

        # Test configuration validation
        config_errors = Config.validate()
        if config_errors:
            print(f"Configuration errors: {config_errors}")
            return False
        print("✓ Configuration validation passed")

        # Test Qdrant connection
        collection_info = ingestion_service.qdrant_storage.get_collection_info()
        if collection_info:
            print("✓ Qdrant connection successful")
            print(f"  Collection info: {collection_info}")
        else:
            print("✗ Qdrant connection failed")
            return False

        # Test a simple embedding generation
        test_text = "This is a test document for the RAG system."
        embedding = await ingestion_service.embedding_client.generate_single_embedding(test_text)
        if embedding and len(embedding) == 1536:  # text-embedding-3-small has 1536 dimensions
            print("✓ Embedding generation successful")
            print(f"  Embedding dimensions: {len(embedding)}")
        else:
            print("✗ Embedding generation failed")
            return False

        # Test sitemap fetching
        urls = ingestion_service.crawler.fetch_sitemap()
        if urls and len(urls) > 0:
            print(f"✓ Sitemap fetch successful: {len(urls)} URLs found")
        else:
            print("✗ Sitemap fetch failed or no URLs found")
            return False

        # Test content extraction (try first URL)
        if urls:
            content_data = ingestion_service.crawler.extract_content_from_url(urls[0])
            if content_data:
                print("✓ Content extraction successful")
                print(f"  Content length: {len(content_data['content'])} characters")
            else:
                print("✗ Content extraction failed")
                return False

        print("\nAll tests passed! The ingestion pipeline is ready.")
        return True

    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ingestion_pipeline())
    if not success:
        exit(1)