"""
Result validation and quality checks for the retrieval pipeline.

This module provides functions for validating the quality and relevance
of retrieved chunks to ensure they meet the requirements for downstream processing.
"""
from typing import List, Optional
from ..models.retrieval_entities import RetrievedChunk
from .ranking_utils import RankingUtils


class ResultValidator:
    """Utility class for validating retrieval results and quality checks."""

    @staticmethod
    def validate_chunk_content(chunk: RetrievedChunk) -> bool:
        """
        Validate that a retrieved chunk has meaningful content.

        Args:
            chunk: RetrievedChunk to validate

        Returns:
            True if chunk has meaningful content
        """
        if not chunk.content or len(chunk.content.strip()) == 0:
            return False

        if len(chunk.content.strip()) < 10:  # Minimum content length
            return False

        return True

    @staticmethod
    def validate_chunk_metadata(chunk: RetrievedChunk) -> bool:
        """
        Validate that a retrieved chunk has complete metadata.

        Args:
            chunk: RetrievedChunk to validate

        Returns:
            True if chunk has complete metadata
        """
        # Check required metadata fields
        if not chunk.url:
            return False

        if not chunk.page_title:
            return False

        if not chunk.chunk_id:
            return False

        if not chunk.content_hash:
            return False

        return True

    @staticmethod
    def validate_chunk_similarity_score(chunk: RetrievedChunk, min_score: float = 0.0) -> bool:
        """
        Validate that a chunk's similarity score is within acceptable range.

        Args:
            chunk: RetrievedChunk to validate
            min_score: Minimum acceptable similarity score

        Returns:
            True if similarity score is valid
        """
        if chunk.similarity_score < min_score or chunk.similarity_score > 1.0:
            return False

        return True

    @staticmethod
    def validate_retrieved_chunks(chunks: List[RetrievedChunk], min_similarity: float = 0.3) -> bool:
        """
        Validate a list of retrieved chunks for quality and completeness.

        Args:
            chunks: List of RetrievedChunk objects to validate
            min_similarity: Minimum similarity threshold

        Returns:
            True if all chunks are valid
        """
        for chunk in chunks:
            if not ResultValidator.validate_chunk_content(chunk):
                return False

            if not ResultValidator.validate_chunk_metadata(chunk):
                return False

            if not ResultValidator.validate_chunk_similarity_score(chunk, min_similarity):
                return False

        return True

    @staticmethod
    def check_result_quality(chunks: List[RetrievedChunk], min_avg_similarity: float = 0.3,
                           min_chunk_count: int = 1) -> dict:
        """
        Check the overall quality of retrieval results.

        Args:
            chunks: List of RetrievedChunk objects
            min_avg_similarity: Minimum average similarity threshold
            min_chunk_count: Minimum number of chunks required

        Returns:
            Dictionary with quality metrics and pass/fail status
        """
        quality_report = {
            'total_chunks': len(chunks),
            'has_min_chunks': len(chunks) >= min_chunk_count,
            'avg_similarity': 0.0,
            'max_similarity': 0.0,
            'min_similarity': 1.0,
            'has_min_avg_similarity': False,
            'valid_chunks_count': 0,
            'quality_score': 0.0,
            'is_quality_pass': False
        }

        if not chunks:
            return quality_report

        # Calculate similarity metrics
        similarities = [chunk.similarity_score for chunk in chunks]
        quality_report['avg_similarity'] = sum(similarities) / len(similarities)
        quality_report['max_similarity'] = max(similarities)
        quality_report['min_similarity'] = min(similarities)

        # Check if average similarity meets threshold
        quality_report['has_min_avg_similarity'] = quality_report['avg_similarity'] >= min_avg_similarity

        # Count valid chunks
        valid_chunks = 0
        for chunk in chunks:
            if (ResultValidator.validate_chunk_content(chunk) and
                ResultValidator.validate_chunk_metadata(chunk) and
                ResultValidator.validate_chunk_similarity_score(chunk, 0.0)):
                valid_chunks += 1

        quality_report['valid_chunks_count'] = valid_chunks

        # Calculate overall quality score (0-1 scale)
        chunk_count_score = min(1.0, len(chunks) / min_chunk_count) if min_chunk_count > 0 else 1.0
        avg_similarity_score = min(1.0, quality_report['avg_similarity'] / min_avg_similarity) if min_avg_similarity > 0 else 1.0
        valid_chunk_ratio = valid_chunks / len(chunks) if chunks else 0.0

        quality_report['quality_score'] = (
            0.4 * chunk_count_score +
            0.4 * avg_similarity_score +
            0.2 * valid_chunk_ratio
        )

        # Overall quality pass/fail
        quality_report['is_quality_pass'] = (
            quality_report['has_min_chunks'] and
            quality_report['has_min_avg_similarity'] and
            quality_report['valid_chunks_count'] == len(chunks)
        )

        return quality_report

    @staticmethod
    def validate_content_relevance(query: str, chunks: List[RetrievedChunk]) -> float:
        """
        Validate content relevance by checking for query terms in retrieved chunks.

        Args:
            query: Original query text
            chunks: List of RetrievedChunk objects

        Returns:
            Relevance score (0-1) based on query term matching
        """
        if not query or not chunks:
            return 0.0

        # Simple keyword matching approach
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0

        total_relevance_score = 0.0
        valid_chunks = 0

        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            if not chunk_words:
                continue

            # Calculate overlap between query and chunk content
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                relevance = overlap / len(query_words)  # Ratio of matched query terms
                total_relevance_score += relevance * chunk.similarity_score  # Weight by similarity
                valid_chunks += 1

        if valid_chunks == 0:
            return 0.0

        # Average relevance score weighted by similarity
        return total_relevance_score / valid_chunks

    @staticmethod
    def detect_low_quality_results(chunks: List[RetrievedChunk], query: str = "") -> dict:
        """
        Detect potential low-quality results and provide reasons.

        Args:
            chunks: List of RetrievedChunk objects
            query: Original query text (optional)

        Returns:
            Dictionary with low-quality detection results
        """
        issues = {
            'empty_results': len(chunks) == 0,
            'low_similarity_results': False,
            'low_content_relevance': False,
            'duplicate_content': False,
            'invalid_metadata': False,
            'summary': []
        }

        if not chunks:
            issues['summary'].append("No results retrieved")
            return issues

        # Check for low similarity
        avg_similarity = RankingUtils.calculate_average_similarity(chunks)
        if avg_similarity < 0.3:
            issues['low_similarity_results'] = True
            issues['summary'].append(f"Low average similarity: {avg_similarity:.3f}")

        # Check for content relevance if query is provided
        if query:
            relevance_score = ResultValidator.validate_content_relevance(query, chunks)
            if relevance_score < 0.1:  # Very low relevance
                issues['low_content_relevance'] = True
                issues['summary'].append(f"Low content relevance: {relevance_score:.3f}")

        # Check for duplicates
        unique_hashes = set(chunk.content_hash for chunk in chunks)
        if len(unique_hashes) < len(chunks):
            issues['duplicate_content'] = True
            issues['summary'].append("Duplicate content detected")

        # Check for invalid metadata
        invalid_metadata_count = 0
        for chunk in chunks:
            if not ResultValidator.validate_chunk_metadata(chunk):
                invalid_metadata_count += 1

        if invalid_metadata_count > 0:
            issues['invalid_metadata'] = True
            issues['summary'].append(f"Invalid metadata in {invalid_metadata_count} chunks")

        return issues

    @staticmethod
    def validate_for_downstream_processing(chunks: List[RetrievedChunk], query: str = "") -> dict:
        """
        Validate results specifically for downstream processing.

        Args:
            chunks: List of RetrievedChunk objects
            query: Original query text (optional)

        Returns:
            Validation report suitable for downstream consumption
        """
        validation_report = {
            'is_valid': False,
            'quality_metrics': {},
            'relevance_score': 0.0,
            'issues': [],
            'recommendation': 'proceed'
        }

        # Check basic validity
        validation_report['is_valid'] = ResultValidator.validate_retrieved_chunks(chunks)

        # Calculate quality metrics
        validation_report['quality_metrics'] = ResultValidator.check_result_quality(chunks)

        # Calculate relevance if query provided
        if query:
            validation_report['relevance_score'] = ResultValidator.validate_content_relevance(query, chunks)

        # Detect issues
        issues = ResultValidator.detect_low_quality_results(chunks, query)
        validation_report['issues'] = issues['summary']

        # Make recommendation
        if not validation_report['is_valid']:
            validation_report['recommendation'] = 'reject'
        elif issues['summary']:
            validation_report['recommendation'] = 'review'
        else:
            validation_report['recommendation'] = 'proceed'

        return validation_report