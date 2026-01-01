"""
Data models for the RAG agent.
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class ContextChunk(BaseModel):
    """
    Individual content chunk retrieved from vector database.
    """
    id: str = Field(..., description="Unique identifier from vector database")
    content: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Content text from the book"
    )
    source_url: str = Field(
        ...,
        pattern=r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
        description="Original URL of the content"
    )
    title: str = Field(..., description="Title of the source page")
    heading: str = Field(..., description="Section heading")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score from retrieval")
    content_hash: Union[str, int] = Field(..., description="Hash for change detection")

    @field_validator('content_hash')
    @classmethod
    def convert_hash_to_string(cls, v):
        """Convert content_hash to string if it's an integer"""
        return str(v)


class QueryWithContext(BaseModel):
    """
    Input model containing user query and retrieved context.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's natural language question"
    )
    context_chunks: List[ContextChunk] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="Retrieved content chunks from RAG pipeline"
    )
    session_id: Optional[str] = Field(
        None,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        description="Optional session identifier for multi-turn conversations"
    )
    question_type: str = Field(
        "explanatory",
        pattern=r'^(factual|explanatory|comparative)$',
        description="Type of question to guide response format"
    )


class Citation(BaseModel):
    """
    Reference to source content used in the answer.
    """
    chunk_id: str = Field(..., description="ID of the source chunk")
    source_url: str = Field(
        ...,
        pattern=r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
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


class GeneratedAnswer(BaseModel):
    """
    Output model containing the generated response.
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
    tokens_used: int = Field(..., ge=1, description="Number of tokens used in generation")
    processing_time: float = Field(..., ge=0.0, description="Time taken to generate answer in seconds")


class AgentConfig(BaseModel):
    """
    Configuration parameters for the RAG agent.
    """
    model_name: str = Field("gpt-3.5-turbo", description="OpenAI model to use")
    max_tokens: int = Field(1000, description="Maximum tokens for response")
    temperature: float = Field(0.3, description="Creativity control (lower for more factual)")
    top_p: float = Field(0.9, description="Alternative creativity control")
    max_context_chunks: int = Field(5, description="Maximum chunks to include in context")
    grounding_enforcement: bool = Field(True, description="Whether to enforce strict grounding")
    citation_format: str = Field("markdown", description="Format for citations")
    confidence_threshold: float = Field(0.6, description="Minimum confidence for positive answers")