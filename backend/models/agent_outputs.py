"""
Data models for agent output validation in the RAG chatbot system.

Contains Pydantic models for validating generated answers and citations.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class Citation(BaseModel):
    """
    Reference to source content used in the answer.

    Attributes:
        chunk_id: ID of the source chunk (required)
        source_url: URL of the source (URL format)
        title: Title of the source (required)
        excerpt: Relevant excerpt from the source (20-200 chars)
        confidence: Confidence in the citation relevance (0.0-1.0)
    """
    chunk_id: str = Field(..., description="ID of the source chunk")
    source_url: str = Field(
        ...,
        regex=r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
        description="URL of the source"
    )
    title: str = Field(..., description="Title of the source")
    excerpt: str = Field(
        ...,
        min_length=20,
        max_length=200,
        description="Relevant excerpt from the source"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the citation relevance")

    class Config:
        # Allow extra fields to support future extensions
        extra = "allow"


class GeneratedAnswer(BaseModel):
    """
    Output model containing the generated response.

    Attributes:
        answer: Generated answer text (50-2000 chars)
        citations: Source citations for the answer (0-10 items)
        confidence_score: Confidence level of the answer (0.0-1.0)
        answer_format: Format of the generated answer
        tokens_used: Number of tokens used in generation (required)
        processing_time: Time taken to generate answer in seconds (required)
    """
    answer: str = Field(
        ...,
        min_length=50,
        max_length=2000,
        description="Generated answer text"
    )
    citations: List[Citation] = Field(
        ...,
        max_items=10,
        description="Source citations for the answer"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence level of the answer")
    answer_format: str = Field(
        ...,
        regex=r'^(factual|explanatory|summary)$',
        description="Format of the generated answer"
    )
    tokens_used: int = Field(..., ge=1, description="Number of tokens used in generation")
    processing_time: float = Field(..., ge=0.0, description="Time taken to generate answer in seconds")

    class Config:
        # Allow extra fields to support future extensions
        extra = "allow"


class AgentConfig(BaseModel):
    """
    Configuration parameters for the answer generation agent.

    Attributes:
        model_name: OpenAI model to use (default: "gpt-3.5-turbo")
        max_tokens: Maximum tokens for response (default: 1000)
        temperature: Creativity control (lower for more factual) (default: 0.3)
        top_p: Alternative creativity control (default: 0.9)
        max_context_chunks: Maximum chunks to include in context (default: 5)
        grounding_enforcement: Whether to enforce strict grounding (default: True)
        citation_format: Format for citations (default: "markdown")
        confidence_threshold: Minimum confidence for positive answers (default: 0.6)
    """
    model_name: str = Field("gpt-3.5-turbo", description="OpenAI model to use")
    max_tokens: int = Field(1000, description="Maximum tokens for response")
    temperature: float = Field(0.3, description="Creativity control (lower for more factual)")
    top_p: float = Field(0.9, description="Alternative creativity control")
    max_context_chunks: int = Field(5, description="Maximum chunks to include in context")
    grounding_enforcement: bool = Field(True, description="Whether to enforce strict grounding")
    citation_format: str = Field("markdown", description="Format for citations")
    confidence_threshold: float = Field(0.6, description="Minimum confidence for positive answers")

    class Config:
        # Allow extra fields to support future extensions
        extra = "allow"