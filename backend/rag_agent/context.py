"""
Context formatting and management for the RAG agent.
"""
from typing import List, Dict, Any
import tiktoken


class ContextFormatter:
    """
    Handles formatting and truncation of context for the RAG agent.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the context formatter.

        Args:
            model_name: The model name to use for tokenization
        """
        self.model_name = model_name
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to a common tokenizer if model-specific one is not available
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks for inclusion in the agent prompt.

        Args:
            context_chunks: List of context chunks to format

        Returns:
            Formatted string of context chunks
        """
        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, 1):
            chunk_text = (
                f"Source {i}:\n"
                f"URL: {chunk.source_url}\n"
                f"Title: {chunk.title}\n"
                f"Heading: {chunk.heading}\n"
                f"Content: {chunk.content}\n"
                f"Relevance Score: {chunk.score}\n"
                f"---\n"
            )
            formatted_chunks.append(chunk_text)

        return "\n".join(formatted_chunks)

    def truncate_context_by_tokens(self, context: str, max_tokens: int) -> str:
        """
        Truncate context to fit within the token limit.

        Args:
            context: The context string to truncate
            max_tokens: Maximum number of tokens allowed

        Returns:
            Truncated context string
        """
        tokens = self.tokenizer.encode(context)

        if len(tokens) <= max_tokens:
            return context

        # Truncate tokens
        truncated_tokens = tokens[:max_tokens]

        # Decode back to text
        truncated_context = self.tokenizer.decode(truncated_tokens)

        return truncated_context

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))