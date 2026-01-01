"""
Fallback handling for the RAG agent.
"""
from typing import List, Dict, Any


class FallbackHandler:
    """
    Handles fallback scenarios when the agent cannot generate a satisfactory answer.
    """

    def __init__(self):
        self.fallback_phrases = [
            "i don't know",
            "not in the provided context",
            "not mentioned in the context",
            "not found in the provided materials",
            "information not available"
        ]

    def handle_insufficient_context(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a fallback response when context is insufficient.

        Args:
            query: The original query
            context_chunks: The context chunks provided

        Returns:
            Fallback response
        """
        return (
            f"I cannot answer the question '{query}' based on the provided context. "
            f"The information needed to answer this question is not available in the provided materials."
        )

    def is_fallback_response(self, response: str) -> bool:
        """
        Determine if a response is a fallback response.

        Args:
            response: The agent's response

        Returns:
            True if the response is a fallback response
        """
        response_lower = response.lower().strip()
        for phrase in self.fallback_phrases:
            if phrase in response_lower:
                return True
        return False