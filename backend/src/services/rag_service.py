"""
RAG orchestration service for the chatbot system.
"""
import time
import logging
from typing import List, Optional
from ..models.citation import Citation
from ..api.config import settings
from rag_agent import RAGAgent, QueryWithContext, AgentConfig, create_rag_agent, ContextChunk


class RAGService:
    """
    Service to orchestrate the RAG flow: retrieval + generation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize the RAG agent from Spec 3
        agent_config = AgentConfig(
            model_name=settings.model_name,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_context_chunks=settings.max_context_chunks,
            confidence_threshold=settings.confidence_threshold
        )
        self.agent = create_rag_agent(agent_config)

    async def process_query(
        self,
        query: str,
        selected_text: Optional[str] = None,
        max_chunks: int = 5
    ) -> tuple[str, List[Citation], float, float]:
        """
        Process a query through the RAG pipeline.

        Args:
            query: The user's question
            selected_text: Optional selected text for contextual querying
            max_chunks: Maximum number of context chunks to retrieve

        Returns:
            Tuple of (answer, citations, confidence, processing_time)
        """
        start_time = time.time()

        # Import the retrieval service from Spec 2
        try:
            # Retrieve chunks from vector store
            from .retrieval_service import retrieve_context_chunks
            retrieved_chunks = await retrieve_context_chunks(query, max_chunks)

            # Convert from API ContextChunk model to RAG agent ContextChunk model
            context_chunks = []
            for chunk in retrieved_chunks:
                rag_chunk = ContextChunk(
                    id=chunk.id,
                    content=chunk.content,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    heading=chunk.heading,
                    score=chunk.score,
                    content_hash=chunk.content_hash
                )
                context_chunks.append(rag_chunk)
        except ImportError:
            # Fallback to mock implementation if retrieval service not available
            context_chunks = self._get_mock_context_chunks(query, selected_text)

        # If selected text is provided, add it as a high-priority context chunk
        if selected_text:
            selected_chunk = ContextChunk(
                id="selected_text",
                content=selected_text,
                source_url="selected_text",
                title="Selected Text",
                heading="User Selection",
                score=1.0,
                content_hash=str(hash(selected_text) % (10 ** 8))
            )
            # Insert selected text at the beginning for higher priority
            context_chunks.insert(0, selected_chunk)

        # Prepare input for the RAG agent from Spec 3
        query_with_context = QueryWithContext(
            query=query,
            context_chunks=context_chunks,
            question_type="factual"  # Default, could be inferred from query
        )

        # Generate answer using the RAG agent
        try:
            result = self.agent.generate_answer(query_with_context)

            # Convert agent citations to API format
            citations = []
            for citation in result.citations:
                api_citation = Citation(
                    chunk_id=citation["chunk_id"],
                    source_url=citation["source_url"],
                    title=citation["title"],
                    excerpt=citation["excerpt"],
                    confidence=citation["confidence"]
                )
                citations.append(api_citation)

            processing_time = time.time() - start_time
            return result.answer, citations, result.confidence_score, processing_time

        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            processing_time = time.time() - start_time
            fallback_answer = f"I'm sorry, I couldn't generate an answer for your question: '{query}'. Please try rephrasing your question."
            return fallback_answer, [], 0.1, processing_time

    def _get_mock_context_chunks(self, query: str, selected_text: Optional[str] = None) -> List[ContextChunk]:
        """
        Mock implementation of context retrieval for demonstration purposes.
        In a real implementation, this would call the actual retrieval pipeline from Spec 2.
        """
        # This is a mock implementation - in reality, this would connect to your retrieval service
        mock_chunks = [
            ContextChunk(
                id="mock_chunk_1",
                content=f"This is mock content related to your query: '{query}'. In a real implementation, this would come from the retrieval pipeline.",
                source_url="https://example.com/mock",
                title="Mock Content",
                heading="Mock Section",
                score=0.8,
                content_hash=str(hash(query) % (10 ** 8))
            )
        ]

        if selected_text:
            mock_chunks.append(
                ContextChunk(
                    id="selected_text",
                    content=selected_text,
                    source_url="selected_text",
                    title="Selected Text",
                    heading="User Selection",
                    score=1.0,
                    content_hash=str(hash(selected_text) % (10 ** 8))
                )
            )

        return mock_chunks


# Initialize the service
rag_service = RAGService()