"""
Answer Generation Service for the RAG chatbot system.

This service orchestrates the answer generation process using the OpenAI API
and ensures answers are properly grounded in the provided context.
"""
from typing import Optional
from ..models.agent_inputs import QueryWithContext
from ..models.agent_outputs import GeneratedAnswer, AgentConfig
from ..agents.answer_generation_agent import AnswerGenerationAgent
import logging


class AnswerGenerationService:
    """
    Service class that handles the core answer generation functionality.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Answer Generation Service.

        Args:
            config: Configuration for the agent. If None, uses default configuration.
        """
        self.config = config or AgentConfig()
        self.logger = logging.getLogger(__name__)
        self.agent = AnswerGenerationAgent(self.config)

    def generate_answer(self, query_with_context: QueryWithContext) -> GeneratedAnswer:
        """
        Generate a grounded answer based on the user query and retrieved context.

        Args:
            query_with_context: The user query with retrieved context chunks

        Returns:
            GeneratedAnswer: The generated answer with citations and metadata
        """
        return self.agent.generate_answer(query_with_context)

    def validate_input(self, query_with_context: QueryWithContext) -> bool:
        """
        Validate the input before processing.

        Args:
            query_with_context: The input to validate

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            # This will trigger Pydantic validation
            query_with_context.validate()
            return True
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False