"""
Result filtering and ranking utilities for the retrieval pipeline.

This module provides functions for ranking and filtering retrieved chunks
based on similarity scores and other metadata.
"""
from typing import List, Optional
from ..models.retrieval_entities import RetrievedChunk


class RankingUtils:
    """Utility class for ranking and filtering retrieved results."""

    @staticmethod
    def rank_by_similarity(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rank chunks by similarity score in descending order.

        Args:
            chunks: List of RetrievedChunk objects to rank

        Returns:
            List of chunks ranked by similarity score (highest first)
        """
        return sorted(chunks, key=lambda chunk: chunk.similarity_score, reverse=True)

    @staticmethod
    def filter_by_similarity_threshold(chunks: List[RetrievedChunk], threshold: float) -> List[RetrievedChunk]:
        """
        Filter chunks based on similarity threshold.

        Args:
            chunks: List of RetrievedChunk objects to filter
            threshold: Minimum similarity score threshold

        Returns:
            List of chunks with similarity scores above threshold
        """
        return [chunk for chunk in chunks if chunk.similarity_score >= threshold]

    @staticmethod
    def filter_by_content_length(chunks: List[RetrievedChunk], min_length: int = 50) -> List[RetrievedChunk]:
        """
        Filter chunks based on content length.

        Args:
            chunks: List of RetrievedChunk objects to filter
            min_length: Minimum content length in characters (default: 50)

        Returns:
            List of chunks with content length above minimum
        """
        return [chunk for chunk in chunks if len(chunk.content) >= min_length]

    @staticmethod
    def remove_duplicates(chunks: List[RetrievedChunk], threshold: float = 0.95) -> List[RetrievedChunk]:
        """
        Remove duplicate or near-duplicate chunks based on content hash.

        Args:
            chunks: List of RetrievedChunk objects to deduplicate
            threshold: Similarity threshold for considering chunks as duplicates

        Returns:
            List of chunks with duplicates removed
        """
        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            if chunk.content_hash not in seen_hashes:
                seen_hashes.add(chunk.content_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    @staticmethod
    def rank_by_position(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rank chunks by their position in the original document.

        Args:
            chunks: List of RetrievedChunk objects to rank

        Returns:
            List of chunks ranked by position (earliest first)
        """
        return sorted(chunks, key=lambda chunk: chunk.position)

    @staticmethod
    def rank_by_multiple_criteria(chunks: List[RetrievedChunk],
                                similarity_weight: float = 0.7,
                                position_weight: float = 0.3) -> List[RetrievedChunk]:
        """
        Rank chunks by multiple criteria using weighted scoring.

        Args:
            chunks: List of RetrievedChunk objects to rank
            similarity_weight: Weight for similarity score (0-1)
            position_weight: Weight for position (0-1)

        Returns:
            List of chunks ranked by weighted scoring
        """
        # Normalize similarity scores to 0-1 range
        if not chunks:
            return chunks

        max_similarity = max(chunk.similarity_score for chunk in chunks)
        min_similarity = min(chunk.similarity_score for chunk in chunks)

        # Avoid division by zero
        if max_similarity == min_similarity:
            similarity_range = 1
        else:
            similarity_range = max_similarity - min_similarity

        # Calculate weighted scores
        scored_chunks = []
        for chunk in chunks:
            normalized_similarity = 0.0
            if similarity_range > 0:
                normalized_similarity = (chunk.similarity_score - min_similarity) / similarity_range

            # Position scoring (earlier positions are better)
            max_position = max(chunk.position for chunk in chunks)
            normalized_position = 0.0
            if max_position > 0:
                normalized_position = 1.0 - (chunk.position / max_position)

            # Calculate weighted score
            weighted_score = (normalized_similarity * similarity_weight +
                            normalized_position * position_weight)

            scored_chunks.append((chunk, weighted_score))

        # Sort by weighted score (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return just the chunks
        return [chunk for chunk, score in scored_chunks]

    @staticmethod
    def filter_by_metadata(chunks: List[RetrievedChunk],
                         metadata_filters: Optional[dict] = None) -> List[RetrievedChunk]:
        """
        Filter chunks based on metadata criteria.

        Args:
            chunks: List of RetrievedChunk objects to filter
            metadata_filters: Dictionary of metadata field-value pairs to filter by

        Returns:
            List of chunks matching the metadata filters
        """
        if not metadata_filters:
            return chunks

        filtered_chunks = []
        for chunk in chunks:
            match = True
            for field, value in metadata_filters.items():
                if hasattr(chunk, field):
                    chunk_value = getattr(chunk, field)
                    if chunk_value != value:
                        match = False
                        break
                else:
                    match = False
                    break

            if match:
                filtered_chunks.append(chunk)

        return filtered_chunks

    @staticmethod
    def calculate_average_similarity(chunks: List[RetrievedChunk]) -> float:
        """
        Calculate the average similarity score across all chunks.

        Args:
            chunks: List of RetrievedChunk objects

        Returns:
            Average similarity score
        """
        if not chunks:
            return 0.0

        total_similarity = sum(chunk.similarity_score for chunk in chunks)
        return total_similarity / len(chunks)

    @staticmethod
    def get_top_k_chunks(chunks: List[RetrievedChunk], k: int) -> List[RetrievedChunk]:
        """
        Get the top K chunks after ranking by similarity.

        Args:
            chunks: List of RetrievedChunk objects
            k: Number of top chunks to return

        Returns:
            List of top K chunks ranked by similarity
        """
        ranked_chunks = RankingUtils.rank_by_similarity(chunks)
        return ranked_chunks[:min(k, len(ranked_chunks))]

    @staticmethod
    def apply_all_filters(chunks: List[RetrievedChunk],
                         min_similarity: float = 0.3,
                         min_content_length: int = 50,
                         deduplicate: bool = True) -> List[RetrievedChunk]:
        """
        Apply all common filters to the chunks.

        Args:
            chunks: List of RetrievedChunk objects to filter
            min_similarity: Minimum similarity threshold
            min_content_length: Minimum content length
            deduplicate: Whether to remove duplicates

        Returns:
            List of chunks after applying all filters
        """
        # Filter by similarity
        filtered_chunks = RankingUtils.filter_by_similarity_threshold(chunks, min_similarity)

        # Filter by content length
        filtered_chunks = RankingUtils.filter_by_content_length(filtered_chunks, min_content_length)

        # Remove duplicates if requested
        if deduplicate:
            filtered_chunks = RankingUtils.remove_duplicates(filtered_chunks)

        return filtered_chunks