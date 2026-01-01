"""
Models package for the RAG chatbot answer generation system.
Contains data models for input/output validation and configuration.
"""
from .agent_inputs import QueryWithContext, ContextChunk
from .agent_outputs import GeneratedAnswer, Citation, AgentConfig

__all__ = [
    'QueryWithContext',
    'ContextChunk',
    'GeneratedAnswer',
    'Citation',
    'AgentConfig'
]