"""
Prompt Templates for the RAG chatbot system.

This module provides utilities for constructing structured prompts with grounding enforcement.
"""
from typing import List
from ..models.agent_inputs import ContextChunk


class PromptTemplateManager:
    """
    Manager class for handling different prompt templates.
    """

    @staticmethod
    def create_basic_query_prompt(query: str, context_chunks: List[ContextChunk]) -> str:
        """
        Create a basic query prompt with grounding instructions.

        Args:
            query: The user's query
            context_chunks: The context chunks to include in the prompt

        Returns:
            The constructed prompt string
        """
        system_instructions = (
            "You are an educational assistant for a technical book. "
            "Your responses must be based ONLY on the provided context. "
            "Do not use any external knowledge or information beyond what's provided in the context. "
            "If the answer is not available in the context, clearly state that the information is not in the provided context. "
            "Always cite the source of information from the context when possible. "
            "Maintain an educational and neutral tone."
        )

        formatted_context = PromptTemplateManager._format_context_chunks(context_chunks)

        prompt = (
            f"{system_instructions}\n\n"
            f"CONTEXT:\n{formatted_context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER: Provide a clear, accurate answer based only on the provided context. "
            f"Include relevant citations to the source material. "
            f"If the information is not in the context, clearly state this."
        )

        return prompt

    @staticmethod
    def create_factual_query_prompt(query: str, context_chunks: List[ContextChunk]) -> str:
        """
        Create a prompt optimized for factual queries.

        Args:
            query: The user's factual query
            context_chunks: The context chunks to include in the prompt

        Returns:
            The constructed prompt string
        """
        system_instructions = (
            "You are an educational assistant for a technical book. "
            "Provide a concise, factual answer based ONLY on the provided context. "
            "Do not use any external knowledge. "
            "If the answer is not in the context, clearly state this. "
            "Cite the source when possible."
        )

        formatted_context = PromptTemplateManager._format_context_chunks(context_chunks)

        prompt = (
            f"{system_instructions}\n\n"
            f"CONTEXT:\n{formatted_context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER: Provide a concise, factual answer based only on the provided context. "
            f"Cite the source of information if available in the context."
        )

        return prompt

    @staticmethod
    def create_explanatory_query_prompt(query: str, context_chunks: List[ContextChunk]) -> str:
        """
        Create a prompt optimized for explanatory queries.

        Args:
            query: The user's explanatory query
            context_chunks: The context chunks to include in the prompt

        Returns:
            The constructed prompt string
        """
        system_instructions = (
            "You are an educational assistant for a technical book. "
            "Provide a detailed explanation based ONLY on the provided context. "
            "Do not use any external knowledge. "
            "If the information is not in the context, clearly state this. "
            "Cite the source when possible."
        )

        formatted_context = PromptTemplateManager._format_context_chunks(context_chunks)

        prompt = (
            f"{system_instructions}\n\n"
            f"CONTEXT:\n{formatted_context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER: Provide a detailed explanation based only on the provided context. "
            f"Include relevant citations to the source material."
        )

        return prompt

    @staticmethod
    def _format_context_chunks(context_chunks: List[ContextChunk]) -> str:
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


class CitationFormatter:
    """
    Class for formatting citations in various formats.
    """

    @staticmethod
    def format_citations_markdown(citations: List[dict]) -> str:
        """
        Format citations in markdown format.

        Args:
            citations: List of citation dictionaries

        Returns:
            Formatted citations string in markdown
        """
        if not citations:
            return ""

        formatted_citations = ["## Sources Referenced:"]
        for i, citation in enumerate(citations, 1):
            formatted_citations.append(
                f"{i}. [{citation.get('title', 'Untitled')}]({citation.get('source_url', '#')})"
            )

        return "\n".join(formatted_citations)

    @staticmethod
    def format_citations_text(citations: List[dict]) -> str:
        """
        Format citations in plain text format.

        Args:
            citations: List of citation dictionaries

        Returns:
            Formatted citations string in plain text
        """
        if not citations:
            return ""

        formatted_citations = ["Sources Referenced:"]
        for i, citation in enumerate(citations, 1):
            formatted_citations.append(
                f"{i}. {citation.get('title', 'Untitled')} - {citation.get('source_url', '')}"
            )

        return "\n".join(formatted_citations)