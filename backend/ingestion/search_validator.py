"""
Search validator module for validating ingestion by running similarity queries
"""
from typing import List, Dict, Any
import logging
from logging_config import logger
from .vector_storage import QdrantVectorStorage
from .embedder import CohereEmbedder


class SearchValidator:
    """
    Class to validate ingestion by running sample similarity queries
    """

    def __init__(self):
        """
        Initialize the search validator
        """
        self.vector_storage = QdrantVectorStorage()
        self.embedder = CohereEmbedder()
        self.logger = logger

    def run_sample_similarity_search(self, query_texts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Run sample similarity searches to validate ingestion

        Args:
            query_texts (List[str]): List of query texts to search for

        Returns:
            List[Dict[str, Any]]: List of search results
        """
        if not query_texts:
            # Default sample queries for validation
            query_texts = [
                "What is the main topic of this book?",
                "Give me an overview of the content",
                "What are the key concepts discussed?",
                "How is this topic relevant?",
                "What are the main sections covered?"
            ]

        results = []
        self.logger.info(f"Running sample similarity search with {len(query_texts)} queries")

        for i, query_text in enumerate(query_texts, 1):
            self.logger.info(f"Processing query {i}/{len(query_texts)}: {query_text[:50]}...")

            try:
                # Generate embedding for the query
                query_embedding = self.embedder.generate_embeddings([query_text])

                if not query_embedding or not query_embedding[0]:
                    self.logger.error(f"Failed to generate embedding for query: {query_text}")
                    continue

                # Search in Qdrant
                search_results = self.vector_storage.search_similar(
                    query_embedding=query_embedding[0],
                    top_k=3  # Get top 3 results
                )

                result = {
                    "query": query_text,
                    "query_embedding": query_embedding[0],
                    "results": search_results,
                    "result_count": len(search_results)
                }

                results.append(result)

                self.logger.debug(f"Query '{query_text[:30]}...' returned {len(search_results)} results")

            except Exception as e:
                self.logger.error(f"Error processing query '{query_text}': {e}")
                # Add error result
                results.append({
                    "query": query_text,
                    "error": str(e),
                    "results": [],
                    "result_count": 0
                })

        return results

    def validate_ingestion_quality(self, sample_queries: List[str] = None) -> Dict[str, Any]:
        """
        Validate the quality of ingestion by running sample searches

        Args:
            sample_queries (List[str]): Sample queries to test

        Returns:
            Dict[str, Any]: Validation results
        """
        if not sample_queries:
            sample_queries = [
                "What is this book about?",
                "Give me an overview",
                "Key concepts",
                "Main topics",
                "How does this work?"
            ]

        self.logger.info("Starting ingestion quality validation")

        # Run sample searches
        search_results = self.run_sample_similarity_search(sample_queries)

        # Calculate metrics
        total_queries = len(search_results)
        successful_queries = sum(1 for result in search_results if "error" not in result)
        total_results = sum(result.get("result_count", 0) for result in search_results)
        avg_results_per_query = total_results / total_queries if total_queries > 0 else 0

        # Check if we're getting reasonable results (at least some results for most queries)
        queries_with_results = sum(1 for result in search_results if result.get("result_count", 0) > 0)
        success_rate = queries_with_results / total_queries if total_queries > 0 else 0

        # Determine if quality is acceptable
        quality_thresholds = {
            "min_success_rate": 0.6,  # At least 60% of queries should succeed
            "min_avg_results": 1.0,   # Average at least 1 result per query
            "min_queries_with_results": 0.5  # At least 50% of queries should have results
        }

        quality_passed = (
            success_rate >= quality_thresholds["min_success_rate"] and
            avg_results_per_query >= quality_thresholds["min_avg_results"] and
            queries_with_results / total_queries >= quality_thresholds["min_queries_with_results"]
        )

        validation_report = {
            "quality_passed": quality_passed,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "total_results": total_results,
            "avg_results_per_query": avg_results_per_query,
            "queries_with_results": queries_with_results,
            "success_rate": success_rate,
            "quality_thresholds": quality_thresholds,
            "detailed_results": search_results
        }

        if quality_passed:
            self.logger.info("Ingestion quality validation PASSED")
        else:
            self.logger.warning("Ingestion quality validation FAILED")
            self.logger.info(f"Validation report: {validation_report}")

        return validation_report

    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the stored vectors

        Returns:
            Dict[str, Any]: Collection statistics
        """
        try:
            collection_info = self.vector_storage.get_collection_info()

            # Get a sample of stored items to analyze content
            sample_items = self.vector_storage.client.scroll(
                collection_name=self.vector_storage.collection_name,
                limit=5  # Just get a few samples
            )

            sample_data = []
            for item in sample_items[0]:
                sample_data.append({
                    "id": item.id,
                    "payload_keys": list(item.payload.keys()) if item.payload else [],
                    "url": item.payload.get("url", "N/A") if item.payload else "N/A"
                })

            stats = {
                "collection_info": collection_info,
                "sample_data": sample_data,
                "has_data": collection_info.get("point_count", 0) > 0
            }

            self.logger.info(f"Collection statistics: {collection_info}")
            return stats

        except Exception as e:
            self.logger.error(f"Error getting collection statistics: {e}")
            return {"error": str(e)}


def test_search_validation():
    """
    Test function to verify the search validation works
    """
    validator = SearchValidator()

    print("Testing search validation...")

    # Test collection statistics
    stats = validator.get_collection_statistics()
    print(f"Collection stats: {stats}")

    # Test sample searches
    sample_queries = [
        "What is this book about?",
        "Overview of content",
        "Key topics"
    ]

    results = validator.run_sample_similarity_search(sample_queries)
    print(f"Sample search results: {len(results)} queries processed")

    for i, result in enumerate(results):
        print(f"Query {i+1}: {result['query']}")
        print(f"  Results: {result['result_count']}")
        if result['result_count'] > 0:
            print(f"  Top result score: {result['results'][0]['score']:.3f}")

    # Run quality validation
    quality_report = validator.validate_ingestion_quality(sample_queries)
    print(f"\nQuality validation: {'PASSED' if quality_report['quality_passed'] else 'FAILED'}")
    print(f"Success rate: {quality_report['success_rate']:.2%}")
    print(f"Avg results per query: {quality_report['avg_results_per_query']:.2f}")


if __name__ == "__main__":
    test_search_validation()