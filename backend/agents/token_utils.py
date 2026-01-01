"""
Token utilities for the RAG chatbot system.

This module provides utilities for token counting and context management
to ensure efficient usage of the OpenAI API within token limits.
"""
import tiktoken
from typing import List, Union
from ..models.agent_inputs import ContextChunk


class TokenCounter:
    """
    Utility class for counting tokens in text content.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the token counter with a specific model tokenizer.

        Args:
            model_name: The name of the OpenAI model to use for tokenization
        """
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to a common tokenizer if model-specific one is not available
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens in the text
        """
        return len(self.tokenizer.encode(text))

    def count_tokens_in_context_chunks(self, context_chunks: List[ContextChunk]) -> int:
        """
        Count the total number of tokens in a list of context chunks.

        Args:
            context_chunks: List of context chunks to count tokens for

        Returns:
            Total number of tokens across all context chunks
        """
        total_tokens = 0
        for chunk in context_chunks:
            total_tokens += self.count_tokens(chunk.content)
        return total_tokens

    def truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a token limit.

        Args:
            text: The text to truncate
            max_tokens: The maximum number of tokens allowed

        Returns:
            Truncated text that fits within the token limit
        """
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Truncate tokens to the limit
        truncated_tokens = tokens[:max_tokens]

        # Decode back to text
        truncated_text = self.tokenizer.decode(truncated_tokens)

        # Add ellipsis to indicate truncation
        if truncated_text != text:
            truncated_text += "..."

        return truncated_text


def select_context_chunks_by_token_limit(
    context_chunks: List[ContextChunk],
    max_tokens: int,
    token_counter: TokenCounter
) -> List[ContextChunk]:
    """
    Select context chunks based on token limits, prioritizing by relevance score.

    Args:
        context_chunks: List of context chunks to select from (should be sorted by relevance)
        max_tokens: Maximum number of tokens allowed for context
        token_counter: TokenCounter instance for counting tokens

    Returns:
        List of context chunks that fit within the token limit
    """
    selected_chunks = []
    current_token_count = 0

    # Sort chunks by score in descending order (highest relevance first)
    sorted_chunks = sorted(context_chunks, key=lambda x: x.score, reverse=True)

    for chunk in sorted_chunks:
        chunk_tokens = token_counter.count_tokens(chunk.content)

        # Check if adding this chunk would exceed the limit
        if current_token_count + chunk_tokens <= max_tokens:
            selected_chunks.append(chunk)
            current_token_count += chunk_tokens
        else:
            # If this chunk alone exceeds the limit, truncate it
            remaining_tokens = max_tokens - current_token_count
            if remaining_tokens > 0:
                truncated_content = token_counter.truncate_text_to_token_limit(
                    chunk.content,
                    remaining_tokens
                )

                # Create a new chunk with truncated content
                truncated_chunk = ContextChunk(
                    id=chunk.id,
                    content=truncated_content,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    heading=chunk.heading,
                    score=chunk.score,
                    content_hash=chunk.content_hash
                )

                selected_chunks.append(truncated_chunk)
                current_token_count = max_tokens  # We've reached the limit
                break
            else:
                # No more tokens available
                break

    return selected_chunks


def truncate_context_chunks(context_chunks: List[ContextChunk], max_tokens_per_chunk: int) -> List[ContextChunk]:
    """
    Truncate individual context chunks if they exceed a token limit.

    Args:
        context_chunks: List of context chunks to potentially truncate
        max_tokens_per_chunk: Maximum number of tokens allowed per chunk

    Returns:
        List of context chunks, potentially with some content truncated
    """
    token_counter = TokenCounter()
    truncated_chunks = []

    for chunk in context_chunks:
        if token_counter.count_tokens(chunk.content) > max_tokens_per_chunk:
            truncated_content = token_counter.truncate_text_to_token_limit(
                chunk.content,
                max_tokens_per_chunk
            )

            # Create a new chunk with truncated content
            truncated_chunk = ContextChunk(
                id=chunk.id,
                content=truncated_content,
                source_url=chunk.source_url,
                title=chunk.title,
                heading=chunk.heading,
                score=chunk.score,
                content_hash=chunk.content_hash
            )
            truncated_chunks.append(truncated_chunk)
        else:
            truncated_chunks.append(chunk)

    return truncated_chunks