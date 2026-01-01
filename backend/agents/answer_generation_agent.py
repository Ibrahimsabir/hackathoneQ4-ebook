"""
Answer Generation Agent for the RAG chatbot system.

This agent consumes retrieved book content from the RAG pipeline and generates
accurate, grounded answers to user questions using OpenAI's API while enforcing
strict grounding rules to prevent hallucinations.
"""
from typing import List, Optional, Dict, Any
import openai
import tiktoken
from pydantic import BaseModel
import time
import logging
import os
from ..models.agent_inputs import QueryWithContext, ContextChunk
from ..models.agent_outputs import GeneratedAnswer, Citation, AgentConfig
from ..services.validation_service import ValidationService
from .token_utils import TokenCounter, select_context_chunks_by_token_limit
from ..utils.logging_config import log_answer_generation_event, log_validation_result


class AnswerGenerationAgent:
    """
    Main agent class that orchestrates the answer generation process.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Answer Generation Agent.

        Args:
            config: Configuration for the agent. If None, uses default configuration.
        """
        self.config = config or AgentConfig()
        self.logger = logging.getLogger(__name__)
        self.validation_service = ValidationService()

        # Initialize OpenAI client with OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        openai.api_key = api_key
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url  # Routes to OpenRouter
        )

        # Create a persistent assistant
        self.assistant = self.client.beta.assistants.create(
            name="RAG Educational Assistant",
            description="An assistant that answers questions based only on provided context from educational materials",
            model=self.config.model_name,
            instructions=(
                "You are an educational assistant for a technical book. "
                "Your responses must be based ONLY on the provided context. "
                "Do not use any external knowledge or information beyond what's provided in the context. "
                "If the answer is not available in the context, clearly state that the information is not in the provided context. "
                "Always cite the source of information from the context when possible. "
                "Maintain an educational and neutral tone."
            )
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
        self.logger.info(f"Starting answer generation for query: {query_with_context.query[:100]}...")

        # Validate input
        self._validate_input(query_with_context)

        # Preprocess context
        processed_context = self._preprocess_context(query_with_context)
        self.logger.debug(f"Preprocessed context: {len(processed_context)} chunks selected")

        # Construct prompt
        prompt = self._construct_prompt(query_with_context.query, processed_context, query_with_context.question_type)

        # Generate answer using OpenAI API
        raw_response = self._call_openai_api(prompt)
        self.logger.debug(f"Raw response generated, length: {len(raw_response)} characters")

        # Validate grounding of the answer
        validation_result = self.validation_service.validate_answer_grounding(raw_response, query_with_context.context_chunks)

        # Log validation results
        log_validation_result(
            self.logger,
            validation_result["grounding_score"],
            validation_result["consistency_score"],
            validation_result["is_acceptable"],
            len(validation_result["grounding_issues"])
        )

        # Check if the answer is acceptable
        if not validation_result["is_acceptable"]:
            self.logger.warning("Answer validation failed, using fallback response")
            # If answer is not acceptable, handle insufficient context
            fallback_response = self.validation_service.handle_insufficient_context(
                query_with_context.query,
                query_with_context.context_chunks
            )

            # For fallback responses, create minimal citations and use low confidence
            citations = []
            confidence_score = 0.1  # Low confidence for fallback responses
            raw_response = fallback_response
        else:
            self.logger.debug("Answer validation passed")
            # Validate grounding and extract citations
            citations = self.validation_service.extract_citations(raw_response, query_with_context.context_chunks)
            confidence_score = self.validation_service.calculate_confidence_score(raw_response, query_with_context.context_chunks)

        # Calculate tokens used
        tokens_used = len(self.tokenizer.encode(raw_response))

        # Calculate processing time
        processing_time = time.time() - start_time

        # Determine answer format based on question type
        answer_format = self._determine_answer_format(query_with_context.question_type)

        # Log the answer generation event
        log_answer_generation_event(
            self.logger,
            query_with_context.query,
            len(query_with_context.context_chunks),
            len(raw_response),
            confidence_score,
            processing_time,
            query_with_context.session_id
        )

        return GeneratedAnswer(
            answer=raw_response,
            citations=citations,
            confidence_score=confidence_score,
            answer_format=answer_format,
            tokens_used=tokens_used,
            processing_time=processing_time
        )

    def _validate_input(self, query_with_context: QueryWithContext) -> None:
        """
        Validate the input before processing.

        Args:
            query_with_context: The input to validate
        """
        # This will be enhanced in later tasks, for now just basic validation
        if len(query_with_context.context_chunks) == 0:
            raise ValueError("At least one context chunk must be provided")

        # Check if context chunks have sufficient content
        for chunk in query_with_context.context_chunks:
            if not chunk.content.strip():
                raise ValueError(f"Context chunk with ID {chunk.id} has no content")

    def _preprocess_context(self, query_with_context: QueryWithContext) -> List[ContextChunk]:
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

        # Apply token-based context selection if needed
        token_counter = TokenCounter(self.config.model_name)

        # Calculate available tokens for context (reserve some for prompt and response)
        reserved_tokens = 500  # Reserve tokens for prompt and response buffer
        available_context_tokens = self.config.max_tokens - reserved_tokens

        if available_context_tokens > 0:
            sorted_chunks = select_context_chunks_by_token_limit(
                sorted_chunks,
                available_context_tokens,
                token_counter
            )

        return sorted_chunks

    def _construct_prompt(self, query: str, context_chunks: List[ContextChunk], question_type: str) -> str:
        """
        Construct a structured prompt with grounding instructions.

        Args:
            query: The user's query
            context_chunks: The context chunks to include in the prompt
            question_type: The type of question (factual, explanatory, comparative)

        Returns:
            The constructed prompt string
        """
        # System instructions enforcing grounding rules
        system_instructions = (
            "You are an educational assistant for a technical book. "
            "Your responses must be based ONLY on the provided context. "
            "Do not use any external knowledge or information beyond what's provided in the context. "
            "If the answer is not available in the context, clearly state that the information is not in the provided context. "
            "Always cite the source of information from the context when possible. "
            "Maintain an educational and neutral tone."
        )

        # Format context chunks
        formatted_context = self._format_context_chunks(context_chunks)

        # Format the question based on type
        question_format = self._format_question_by_type(query, question_type)

        # Construct the full prompt
        prompt = (
            f"{system_instructions}\n\n"
            f"CONTEXT:\n{formatted_context}\n\n"
            f"QUESTION: {question_format}\n\n"
            f"ANSWER: Provide a clear, accurate answer based only on the provided context. "
            f"Include relevant citations to the source material. "
            f"If the information is not in the context, clearly state this."
        )

        return prompt

    def _format_context_chunks(self, context_chunks: List[ContextChunk]) -> str:
        """
        Format context chunks for inclusion in the prompt.

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

    def _format_question_by_type(self, query: str, question_type: str) -> str:
        """
        Format the question based on its type.

        Args:
            query: The original query
            question_type: The type of question

        Returns:
            Formatted question string
        """
        if question_type == "factual":
            return f"Provide a factual answer to: {query}"
        elif question_type == "explanatory":
            return f"Provide a detailed explanation for: {query}"
        elif question_type == "comparative":
            return f"Compare and contrast as it relates to: {query}"
        else:
            return query  # Default to original query

    def _call_openai_api(self, prompt: str) -> str:
        """
        Call the OpenAI API to generate a response using the Assistants API.

        Args:
            prompt: The prompt to send to the API

        Returns:
            The generated response text
        """
        try:
            # Create an assistant for this specific task
            assistant = self.client.beta.assistants.create(
                name="RAG Educational Assistant",
                description="An assistant that answers questions based only on provided context from educational materials",
                model=self.config.model_name,
                instructions=(
                    "You are an educational assistant for a technical book. "
                    "Your responses must be based ONLY on the provided context. "
                    "Do not use any external knowledge or information beyond what's provided in the context. "
                    "If the answer is not available in the context, clearly state that the information is not in the provided context. "
                    "Always cite the source of information from the context when possible. "
                    "Maintain an educational and neutral tone."
                )
            )

            # Create a thread with the user's message
            thread = self.client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Run the assistant on the thread
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
                max_completion_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )

            # Wait for the run to complete
            import time
            while run.status in ['queued', 'in_progress', 'cancelling']:
                time.sleep(1)  # Wait for 1 second
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            # Get the messages from the thread
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                order="asc"
            )

            # Extract the assistant's response
            assistant_response = ""
            for message in messages.data:
                if message.role == "assistant":
                    for content in message.content:
                        if content.type == "text":
                            assistant_response = content.text.value
                            break
                    break

            # Clean up by deleting the assistant and thread
            try:
                self.client.beta.assistants.delete(assistant.id)
            except:
                pass  # Ignore cleanup errors

            return assistant_response.strip()

        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during API call: {e}")
            raise

    def _determine_answer_format(self, question_type: str) -> str:
        """
        Determine the answer format based on question type.

        Args:
            question_type: The type of question

        Returns:
            The corresponding answer format
        """
        format_mapping = {
            "factual": "factual",
            "explanatory": "explanatory",
            "comparative": "summary",  # Comparative questions often result in summary-style answers
            "summary": "summary"  # Explicit summary requests
        }
        return format_mapping.get(question_type, "explanatory")

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))