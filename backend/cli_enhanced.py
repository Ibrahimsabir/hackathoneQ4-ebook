import asyncio
import argparse
from src.ingestion_service_enhanced import IngestionServiceEnhanced
from src.utils.logging_config import logger
from src.utils.config import Config
import sys

def main():
    """
    Enhanced command-line interface for the ingestion pipeline with better error handling.
    """
    parser = argparse.ArgumentParser(description="RAG Chatbot Content Ingestion Pipeline (Enhanced)")
    parser.add_argument(
        "command",
        choices=["ingest", "validate", "search", "metrics", "reindex", "resume"],
        help="Command to execute"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query for testing search functionality"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for embedding generation (default: 20, smaller for quota management)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Validate configuration
    config_errors = Config.validate()
    if config_errors:
        print(f"Configuration errors: {', '.join(config_errors)}")
        sys.exit(1)

    # Create ingestion service
    ingestion_service = IngestionServiceEnhanced()

    # Execute the requested command
    if args.command == "ingest":
        print("Starting ingestion pipeline with enhanced error handling...")
        success = asyncio.run(ingestion_service.run_ingestion_pipeline())
        if success:
            print("Ingestion completed successfully!")
        else:
            print("Ingestion failed!")
            sys.exit(1)

    elif args.command == "resume":
        print("Resuming ingestion pipeline from last state...")
        success = asyncio.run(ingestion_service.run_ingestion_pipeline_with_resume())
        if success:
            print("Resume completed successfully!")
        else:
            print("Resume failed!")
            sys.exit(1)

    elif args.command == "validate":
        print("Validating storage...")
        success = asyncio.run(ingestion_service.validate_storage())
        if success:
            print("Validation passed!")
        else:
            print("Validation failed!")
            sys.exit(1)

    elif args.command == "search":
        query = args.query or "What is Physical AI?"
        print(f"Testing search with query: '{query}'")
        results = asyncio.run(ingestion_service.test_similarity_search(query))
        if results:
            print(f"Found {len(results)} similar documents:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     URL: {result['source_url']}")
                print(f"     Content preview: {result['content'][:100]}...")
                print()
        else:
            print("No results found or search failed!")

    elif args.command == "metrics":
        metrics = ingestion_service.get_ingestion_metrics()
        print("Ingestion Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    elif args.command == "reindex":
        print("Re-indexing content...")
        print("Warning: This will recreate the collection and re-ingest all content.")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            # Delete existing collection
            print("Deleting existing collection...")
            ingestion_service.qdrant_storage.delete_collection()

            # Re-initialize collection
            ingestion_service.qdrant_storage.initialize_collection()

            # Run ingestion pipeline
            success = asyncio.run(ingestion_service.run_ingestion_pipeline())
            if success:
                print("Re-indexing completed successfully!")
            else:
                print("Re-indexing failed!")
                sys.exit(1)
        else:
            print("Re-indexing cancelled.")


if __name__ == "__main__":
    main()