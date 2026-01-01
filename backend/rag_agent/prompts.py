"""
System prompts and templates for the RAG agent.
"""
from typing import Dict, Any


class SystemPromptBuilder:
    """
    Builds system prompts for the RAG agent with grounding instructions.
    """

    def __init__(self):
        self.grounding_rules = [
            "You are an educational assistant for a technical book.",
            "Your responses must be based ONLY on the provided context.",
            "Do not use any external knowledge or information beyond what's provided in the context.",
            "If the answer is not available in the context, clearly state that the information is not in the provided context.",
            "Always cite the source of information from the context when possible.",
            "Preserve original terminology from the book.",
            "Maintain a clear, educational tone.",
            "Provide concise, accurate answers without unnecessary elaboration."
        ]

    def build_system_prompt(self) -> str:
        """
        Build the system prompt with grounding instructions.

        Returns:
            Formatted system prompt string
        """
        return "\n".join(self.grounding_rules)

    def build_context_template(self) -> str:
        """
        Build the template for formatting context chunks.

        Returns:
            Context template string
        """
        return (
            "Source {index}:\n"
            "URL: {source_url}\n"
            "Title: {title}\n"
            "Heading: {heading}\n"
            "Content: {content}\n"
            "Relevance Score: {score}\n"
            "---\n"
        )

    def build_query_template(self) -> str:
        """
        Build the template for formatting user queries.

        Returns:
            Query template string
        """
        return "Question: {query}\n\nAnswer:"