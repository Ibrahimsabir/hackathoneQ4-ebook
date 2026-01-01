"""
RAG Agent implementation using OpenAI Agents SDK.

This agent consumes retrieved book content and generates grounded answers
to user questions using OpenAI's API while enforcing strict grounding rules
to prevent hallucinations.
"""
from typing import List, Optional
import openai
import os
import logging
import time
from pydantic import BaseModel
import tiktoken
from .models import QueryWithContext, GeneratedAnswer, AgentConfig


class RAGAgent:
    """
    RAG Agent that generates grounded answers using OpenAI's API.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the RAG Agent.

        Args:
            config: Configuration for the agent. If None, uses default configuration.
        """
        self.config = config or AgentConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client with OpenRouter
        from src.api.config import settings
        api_key = settings.OPENROUTER_API_KEY
        base_url = settings.OPENROUTER_BASE_URL
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        openai.api_key = api_key
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url  # Routes to OpenRouter instead of OpenAI
        )

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)
        except KeyError:
            # Fallback to a common tokenizer if model-specific one is not available
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate_answer(self, query_with_context: QueryWithContext) -> GeneratedAnswer:
        """
        Generate a grounded answer based on the user query and retrieved context.

        Args:
            query_with_context: The user query with retrieved context chunks

        Returns:
            GeneratedAnswer: The generated answer with citations and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting RAG agent for query: {query_with_context.query[:100]}...")

        # Log input for debugging
        self.logger.debug(f"Input context chunks: {len(query_with_context.context_chunks)}")

        # Validate input
        self._validate_input(query_with_context)

        # Preprocess context
        processed_context = self._preprocess_context(query_with_context)
        self.logger.debug(f"Processed context: {len(processed_context)} chunks selected")

        # Format context for the agent
        formatted_context = self._format_context(processed_context)

        # Generate answer using OpenAI API
        raw_response = self._call_openai_api(
            context=formatted_context,
            query=query_with_context.query
        )
        self.logger.debug(f"Raw response generated, length: {len(raw_response)} characters")

        # Calculate confidence based on response characteristics
        confidence_score = self._calculate_confidence_score(
            raw_response,
            query_with_context.context_chunks
        )

        # Calculate tokens used
        tokens_used = len(self.tokenizer.encode(raw_response))

        # Calculate processing time
        processing_time = time.time() - start_time

        # Log the agent execution
        self.logger.info(
            f"RAG agent completed in {processing_time:.2f}s, "
            f"tokens used: {tokens_used}, "
            f"confidence: {confidence_score:.2f}"
        )

        return GeneratedAnswer(
            answer=raw_response,
            citations=[],  # In a real implementation, citations would be extracted
            confidence_score=confidence_score,
            tokens_used=tokens_used,
            processing_time=processing_time
        )

    def _validate_input(self, query_with_context: QueryWithContext) -> None:
        """
        Validate the input before processing.

        Args:
            query_with_context: The input to validate
        """
        # Basic validation
        if len(query_with_context.context_chunks) == 0:
            raise ValueError("At least one context chunk must be provided")

        # Check if context chunks have sufficient content
        for chunk in query_with_context.context_chunks:
            if not chunk.content.strip():
                raise ValueError(f"Context chunk with ID {chunk.id} has no content")

    def _preprocess_context(self, query_with_context: QueryWithContext) -> List:
        """
        Preprocess and validate context chunks with token management.

        Args:
            query_with_context: The query with context to preprocess

        Returns:
            List of validated context chunks
        """
        # Sort context chunks by score in descending order (highest relevance first)
        sorted_chunks = sorted(
            query_with_context.context_chunks,
            key=lambda x: x.score,
            reverse=True
        )

        # Limit to max_context_chunks if specified in config
        if self.config.max_context_chunks > 0:
            sorted_chunks = sorted_chunks[:self.config.max_context_chunks]

        return sorted_chunks

    def _format_context(self, context_chunks: List) -> str:
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

    def _call_openai_api(self, context: str, query: str) -> str:
        """
        Call the OpenAI API to generate a response.

        Args:
            context: The formatted context to provide to the assistant
            query: The user's query

        Returns:
            The generated response text
        """
        try:
            # Construct the prompt with system instructions
            system_prompt = (
                "You are an educational assistant for a technical book. "
                "Your responses must be based ONLY on the provided context. "
                "Do not use any external knowledge or information beyond what's provided in the context. "
                "If the answer is not available in the context, clearly state that the information is not in the provided context. "
                "Always cite the source of information from the context when possible. "
                "Maintain an educational and neutral tone."
            )

            # Construct the user message with context
            user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )

            return response.choices[0].message.content.strip()

        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during API call: {e}")
            raise

    def _calculate_confidence_score(self, response: str, context_chunks: List) -> float:
        """
        Calculate a confidence score based on the response and context.

        Args:
            response: The generated response
            context_chunks: The context chunks used

        Returns:
            Confidence score between 0 and 1
        """
        if not response or response.lower().startswith("i don't know") or "not in the provided context" in response.lower():
            return 0.1  # Low confidence for fallback responses

        # Base confidence on context utilization
        base_confidence = min(len(context_chunks) * 0.2, 0.8)  # Max 0.8 for context utilization

        # If response is substantial, boost confidence
        if len(response) > 100:
            base_confidence += 0.2  # Add 0.2 for substantial responses

        # Ensure confidence is between 0.1 and 1.0
        return max(0.1, min(1.0, base_confidence))