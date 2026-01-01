"""
Ingestion orchestrator module to manage the complete ingestion pipeline
"""
from typing import List, Dict, Any, Tuple
import logging
import time
from datetime import datetime
from logging_config import logger
from .sitemap_parser import SitemapParser
from .content_extractor import ContentExtractor
from .chunker import ContentChunker
from .openai_embedder import OpenAIEmbedder  # Changed from CohereEmbedder
from .vector_storage import QdrantVectorStorage
from .text_utils import TextNormalizer
import json


class IngestionOrchestrator:
    """
    Class to orchestrate the complete ingestion pipeline
    """

    def __init__(self):
        """
        Initialize all components of the ingestion pipeline
        """
        from config import SITEMAP_URL
        self.sitemap_parser = SitemapParser(SITEMAP_URL)
        self.content_extractor = ContentExtractor()
        self.chunker = ContentChunker()
        self.embedder = OpenAIEmbedder()  # Changed from CohereEmbedder
        self.vector_storage = QdrantVectorStorage()
        self.text_normalizer = TextNormalizer()
        self.logger = logger

    def run_ingestion_pipeline(self, sitemap_url: str) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline from sitemap to vector storage

        Args:
            sitemap_url (str): URL to the sitemap.xml

        Returns:
            Dict[str, Any]: Ingestion results and metrics
        """
        start_time = time.time()
        self.logger.info(f"Starting ingestion pipeline for: {sitemap_url}")

        # Initialize metrics
        metrics = {
            "start_time": datetime.now().isoformat(),
            "sitemap_url": sitemap_url,
            "total_urls": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "total_content_extracted": 0,
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "successful_storages": 0,
            "failed_storages": 0,
            "total_content_length": 0,
            "end_time": None,
            "duration_seconds": None,
            "errors": []
        }

        try:
            # Step 1: Parse sitemap
            self.logger.info("Step 1: Parsing sitemap")
            urls = self.sitemap_parser.get_unique_urls()
            metrics["total_urls"] = len(urls)
            self.logger.info(f"Found {len(urls)} URLs in sitemap")

            # Step 2: Extract content from all URLs
            self.logger.info("Step 2: Extracting content from URLs")
            content_items = []
            for i, url in enumerate(urls, 1):
                self.logger.info(f"Processing {i}/{len(urls)}: {url}")
                content_data = self.content_extractor.extract_content(url)

                if content_data['content']:
                    content_data['url'] = url
                    content_data['content_hash'] = self.text_normalizer.get_content_hash(content_data['content'])
                    content_items.append(content_data)
                    metrics["successful_crawls"] += 1
                    metrics["total_content_length"] += len(content_data['content'])
                else:
                    metrics["failed_crawls"] += 1
                    metrics["errors"].append(f"Failed to extract content from {url}")

            metrics["total_content_extracted"] = len(content_items)
            self.logger.info(f"Successfully extracted content from {len(content_items)} pages")

            # Step 3: Chunk content
            self.logger.info("Step 3: Chunking content")
            all_chunks = []
            for content_item in content_items:
                chunks = self.chunker.chunk_by_headings(
                    content_item['content'],
                    content_item['url'],
                    content_item['title']
                )

                # Add content hash to each chunk
                for chunk in chunks:
                    chunk['content_hash'] = content_item['content_hash']

                all_chunks.extend(chunks)

            metrics["total_chunks"] = len(all_chunks)
            self.logger.info(f"Created {len(all_chunks)} content chunks")

            # Step 4: Generate embeddings
            self.logger.info("Step 4: Generating embeddings")
            if all_chunks:
                # Extract just the content text for embedding
                texts_to_embed = [chunk['content'] for chunk in all_chunks]

                embeddings = self.embedder.generate_embeddings(texts_to_embed)

                # Validate embeddings
                if embeddings and self.embedder.validate_embeddings(embeddings):
                    metrics["successful_embeddings"] = len(embeddings)
                    self.logger.info(f"Successfully generated {len(embeddings)} embeddings")

                    # Step 5: Store embeddings in Qdrant
                    self.logger.info("Step 5: Storing embeddings in Qdrant")
                    success = self.vector_storage.idempotent_store_embeddings(embeddings, all_chunks)

                    if success:
                        metrics["successful_storages"] = len(embeddings)
                        self.logger.info("Successfully stored embeddings in Qdrant")
                    else:
                        metrics["failed_storages"] = len(embeddings)
                        metrics["errors"].append("Failed to store embeddings in Qdrant")
                else:
                    metrics["failed_embeddings"] = len(texts_to_embed)
                    metrics["errors"].append("Embeddings validation failed")
            else:
                self.logger.warning("No chunks to process for embeddings")

        except Exception as e:
            error_msg = f"Ingestion pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            metrics["errors"].append(error_msg)

        finally:
            # Finalize metrics
            metrics["end_time"] = datetime.now().isoformat()
            metrics["duration_seconds"] = time.time() - start_time

            # Log summary
            self.logger.info(f"Ingestion pipeline completed in {metrics['duration_seconds']:.2f} seconds")
            self.logger.info(f"Summary: {metrics['successful_crawls']}/{metrics['total_urls']} URLs processed successfully")
            self.logger.info(f"Chunks created: {metrics['total_chunks']}")
            self.logger.info(f"Embeddings generated: {metrics['successful_embeddings']}")
            self.logger.info(f"Items stored: {metrics['successful_storages']}")

        return metrics

    def validate_ingestion(self, metrics: Dict[str, Any]) -> bool:
        """
        Validate the ingestion process based on success metrics

        Args:
            metrics (Dict[str, Any]): Ingestion metrics

        Returns:
            bool: True if ingestion is considered successful
        """
        # Define success criteria
        success_thresholds = {
            "crawl_success_rate": 0.8,  # 80% of URLs should be crawled successfully
            "min_embeddings_generated": 1,  # At least 1 embedding should be generated
            "storage_success_rate": 0.95  # 95% of embeddings should be stored successfully
        }

        # Calculate success rates
        if metrics["total_urls"] > 0:
            crawl_success_rate = metrics["successful_crawls"] / metrics["total_urls"]
        else:
            crawl_success_rate = 0

        if metrics["successful_embeddings"] > 0:
            storage_success_rate = metrics["successful_storages"] / metrics["successful_embeddings"]
        else:
            storage_success_rate = 1.0 if metrics["successful_embeddings"] == 0 else 0

        # Check if all thresholds are met
        success = (
            crawl_success_rate >= success_thresholds["crawl_success_rate"] and
            metrics["successful_embeddings"] >= success_thresholds["min_embeddings_generated"] and
            storage_success_rate >= success_thresholds["storage_success_rate"]
        )

        validation_report = {
            "success": success,
            "thresholds": success_thresholds,
            "actual_values": {
                "crawl_success_rate": crawl_success_rate,
                "successful_embeddings": metrics["successful_embeddings"],
                "storage_success_rate": storage_success_rate
            },
            "metrics": metrics
        }

        if success:
            self.logger.info("Ingestion validation PASSED")
        else:
            self.logger.warning("Ingestion validation FAILED")
            self.logger.info(f"Validation report: {validation_report}")

        return success

    def run_incremental_ingestion(self, sitemap_url: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Run incremental ingestion that detects changes and only processes updated content

        Args:
            sitemap_url (str): URL to the sitemap.xml
            force_reindex (bool): If True, reindex all content regardless of changes

        Returns:
            Dict[str, Any]: Ingestion results and metrics
        """
        start_time = time.time()
        self.logger.info(f"Starting incremental ingestion for: {sitemap_url}, force_reindex: {force_reindex}")

        # Initialize metrics
        metrics = {
            "start_time": datetime.now().isoformat(),
            "sitemap_url": sitemap_url,
            "total_urls": 0,
            "urls_checked": 0,
            "urls_with_changes": 0,
            "urls_unchanged": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "total_content_extracted": 0,
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "successful_storages": 0,
            "failed_storages": 0,
            "total_content_length": 0,
            "end_time": None,
            "duration_seconds": None,
            "errors": []
        }

        try:
            # Step 1: Parse sitemap
            self.logger.info("Step 1: Parsing sitemap")
            urls = self.sitemap_parser.get_unique_urls()
            metrics["total_urls"] = len(urls)
            self.logger.info(f"Found {len(urls)} URLs in sitemap")

            # Step 2: Check for changes and extract only changed content
            self.logger.info("Step 2: Checking for content changes and extracting content")
            content_items = []

            for i, url in enumerate(urls, 1):
                self.logger.info(f"Processing {i}/{len(urls)}: {url}")

                # Extract content
                content_data = self.content_extractor.extract_content(url)
                metrics["urls_checked"] += 1

                if content_data['content']:
                    content_data['url'] = url
                    current_content_hash = self.text_normalizer.get_content_hash(content_data['content'])
                    content_data['content_hash'] = current_content_hash

                    # Check if content has changed (if not forcing reindex)
                    content_changed = True  # Assume changed if forcing reindex
                    if not force_reindex:
                        # In a real implementation, we'd look up the previous hash from storage
                        # For now, we'll simulate by checking if we can find similar content in Qdrant
                        from qdrant_client.http import models

                        existing_items = self.vector_storage.client.scroll(
                            collection_name=self.vector_storage.collection_name,
                            scroll_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="url",
                                        match=models.MatchValue(value=url)
                                    )
                                ]
                            ),
                            limit=1
                        )

                        if existing_items[0]:  # If we found existing content
                            # Compare with existing content hash if available in payload
                            existing_payload = existing_items[0].payload
                            if 'content_hash' in existing_payload:
                                previous_hash = existing_payload['content_hash']
                                content_changed = self.text_normalizer.compare_content_hashes(previous_hash, current_content_hash)
                            else:
                                content_changed = True  # If no previous hash, assume changed
                        else:
                            content_changed = True  # If no existing content, it's new

                    if content_changed or force_reindex:
                        content_items.append(content_data)
                        metrics["urls_with_changes"] += 1
                        metrics["total_content_length"] += len(content_data['content'])
                        metrics["successful_crawls"] += 1
                        self.logger.info(f"Content changed for {url}, will process")
                    else:
                        metrics["urls_unchanged"] += 1
                        self.logger.info(f"Content unchanged for {url}, skipping")
                else:
                    metrics["failed_crawls"] += 1
                    metrics["errors"].append(f"Failed to extract content from {url}")

            metrics["total_content_extracted"] = len(content_items)
            self.logger.info(f"Successfully extracted content from {len(content_items)} pages that changed")

            # Step 3: Chunk changed content
            self.logger.info("Step 3: Chunking changed content")
            all_chunks = []
            for content_item in content_items:
                chunks = self.chunker.chunk_by_headings(
                    content_item['content'],
                    content_item['url'],
                    content_item['title']
                )

                # Add content hash to each chunk
                for chunk in chunks:
                    chunk['content_hash'] = content_item['content_hash']

                all_chunks.extend(chunks)

            metrics["total_chunks"] = len(all_chunks)
            self.logger.info(f"Created {len(all_chunks)} content chunks from changed pages")

            # Step 4: Generate embeddings for changed content
            self.logger.info("Step 4: Generating embeddings for changed content")
            if all_chunks:
                # Extract just the content text for embedding
                texts_to_embed = [chunk['content'] for chunk in all_chunks]

                embeddings = self.embedder.generate_embeddings(texts_to_embed)

                # Validate embeddings
                if embeddings and self.embedder.validate_embeddings(embeddings):
                    metrics["successful_embeddings"] = len(embeddings)
                    self.logger.info(f"Successfully generated {len(embeddings)} embeddings")

                    # Step 5: Store embeddings in Qdrant (using idempotent storage)
                    self.logger.info("Step 5: Storing embeddings in Qdrant")
                    success = self.vector_storage.idempotent_store_embeddings(embeddings, all_chunks)

                    if success:
                        metrics["successful_storages"] = len(embeddings)
                        self.logger.info("Successfully stored embeddings in Qdrant")
                    else:
                        metrics["failed_storages"] = len(embeddings)
                        metrics["errors"].append("Failed to store embeddings in Qdrant")
                else:
                    metrics["failed_embeddings"] = len(texts_to_embed)
                    metrics["errors"].append("Embeddings validation failed")
            else:
                self.logger.info("No changed content to process for embeddings")

        except Exception as e:
            error_msg = f"Incremental ingestion pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            metrics["errors"].append(error_msg)

        finally:
            # Finalize metrics
            metrics["end_time"] = datetime.now().isoformat()
            metrics["duration_seconds"] = time.time() - start_time

            # Log summary
            self.logger.info(f"Incremental ingestion pipeline completed in {metrics['duration_seconds']:.2f} seconds")
            self.logger.info(f"Summary: {metrics['successful_crawls']}/{metrics['urls_checked']} URLs processed")
            self.logger.info(f"URLs with changes: {metrics['urls_with_changes']}")
            self.logger.info(f"URLs unchanged: {metrics['urls_unchanged']}")
            self.logger.info(f"Chunks created: {metrics['total_chunks']}")
            self.logger.info(f"Embeddings generated: {metrics['successful_embeddings']}")
            self.logger.info(f"Items stored: {metrics['successful_storages']}")

        return metrics

    def _get_lock_key(self, url: str) -> str:
        """
        Generate a lock key for a URL to prevent concurrent processing

        Args:
            url (str): URL to generate lock key for

        Returns:
            str: Lock key
        """
        import hashlib
        return f"ingestion_lock_{hashlib.md5(url.encode()).hexdigest()[:12]}"

    def create_ingestion_report(self, metrics: Dict[str, Any]) -> str:
        """
        Create a comprehensive ingestion report

        Args:
            metrics (Dict[str, Any]): Ingestion metrics

        Returns:
            str: Formatted ingestion report
        """
        report = f"""
INGESTION PIPELINE REPORT
========================

Pipeline Summary:
- Start Time: {metrics['start_time']}
- End Time: {metrics['end_time']}
- Duration: {metrics['duration_seconds']:.2f} seconds

Content Statistics:
- Total URLs processed: {metrics['total_urls']}
- Successful crawls: {metrics['successful_crawls']}
- Failed crawls: {metrics['failed_crawls']}
- Total content extracted: {metrics['total_content_extracted']} pages
- Total content length: {metrics['total_content_length']} characters
- Total chunks created: {metrics['total_chunks']}

Embedding Statistics:
- Successful embeddings: {metrics['successful_embeddings']}
- Failed embeddings: {metrics['failed_embeddings']}

Storage Statistics:
- Successful storages: {metrics['successful_storages']}
- Failed storages: {metrics['failed_storages']}

Success Rates:
- Crawl Success Rate: {(metrics['successful_crawls']/metrics['total_urls']*100) if metrics['total_urls'] > 0 else 0:.2f}%
- Storage Success Rate: {(metrics['successful_storages']/metrics['successful_embeddings']*100) if metrics['successful_embeddings'] > 0 else 0:.2f}%

Errors:
"""
        if metrics['errors']:
            for error in metrics['errors']:
                report += f"- {error}\n"
        else:
            report += "- None\n"

        return report


def main():
    """
    Main function to run the ingestion pipeline
    """
    from config import SITEMAP_URL

    orchestrator = IngestionOrchestrator()

    print("Starting ingestion pipeline...")
    metrics = orchestrator.run_ingestion_pipeline(SITEMAP_URL)

    print(orchestrator.create_ingestion_report(metrics))

    is_valid = orchestrator.validate_ingestion(metrics)
    print(f"\nIngestion validation: {'PASSED' if is_valid else 'FAILED'}")

    print("\n" + "="*50)
    print("Starting incremental ingestion...")
    incremental_metrics = orchestrator.run_incremental_ingestion(SITEMAP_URL)

    print(orchestrator.create_ingestion_report(incremental_metrics))

    is_valid = orchestrator.validate_ingestion(incremental_metrics)
    print(f"\nIncremental ingestion validation: {'PASSED' if is_valid else 'FAILED'}")


if __name__ == "__main__":
    main()