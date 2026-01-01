"""
RAG Agent module initialization
"""
from .agent import RAGAgent
from .models import QueryWithContext, GeneratedAnswer, AgentConfig, ContextChunk, Citation


def create_rag_agent(config=None):
    """
    Create and return a RAG agent instance.
    """
    return RAGAgent(config)


__all__ = ['RAGAgent', 'QueryWithContext', 'GeneratedAnswer', 'AgentConfig', 'ContextChunk', 'Citation', 'create_rag_agent']