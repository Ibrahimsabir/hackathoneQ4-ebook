"""
Citation extraction and management for the RAG agent.
"""
from typing import List, Dict, Any
import re


class CitationExtractor:
    """
    Extracts citations from agent responses and maps them to source chunks.
    """

    def __init__(self):
        pass

    def extract_citations(self, response: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citations from the response and map them to source chunks.

        Args:
            response: The agent's response
            context_chunks: The context chunks used in the response

        Returns:
            List of citation objects
        """
        citations = []

        # Look for patterns that might indicate citations
        # This could be improved based on how the context is formatted
        for i, chunk in enumerate(context_chunks):
            # Check if any part of the chunk content appears in the response
            # This is a basic approach - could be enhanced with more sophisticated NLP
            if self._chunk_referenced_in_response(response, chunk):
                citation = {
                    "chunk_id": chunk.id,
                    "source_url": chunk.source_url,
                    "title": chunk.title,
                    "excerpt": self._extract_relevant_excerpt(response, chunk.content),
                    "confidence": chunk.score
                }
                citations.append(citation)

        return citations

    def _chunk_referenced_in_response(self, response: str, chunk: Dict[str, Any]) -> bool:
        """
        Determine if a context chunk is referenced in the response.

        Args:
            response: The agent's response
            chunk: The context chunk

        Returns:
            True if the chunk is likely referenced in the response
        """
        # Convert to lowercase for comparison
        response_lower = response.lower()
        content_lower = chunk.content.lower()

        # Check if a significant portion of the chunk content appears in the response
        # This is a simple heuristic - could be enhanced
        if len(content_lower) > 0:
            # Look for at least 10% of the chunk content in the response
            threshold = max(10, len(content_lower) // 10)
            common_words = set(response_lower.split()) & set(content_lower.split())
            if len(common_words) >= threshold:
                return True

        # Also check if the title or heading appears in the response
        if chunk.title and chunk.title.lower() in response_lower:
            return True
        if chunk.heading and chunk.heading.lower() in response_lower:
            return True

        return False

    def _extract_relevant_excerpt(self, response: str, chunk_content: str) -> str:
        """
        Extract the relevant excerpt from the chunk that was used in the response.

        Args:
            response: The agent's response
            chunk_content: The original chunk content

        Returns:
            Relevant excerpt from the chunk
        """
        # For now, return the first 200 characters of the chunk content
        # This could be improved to find the most relevant part
        return chunk_content[:200] if len(chunk_content) > 200 else chunk_content