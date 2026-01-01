"""
Data models for agent input validation in the RAG chatbot system.

Contains Pydantic models for validating user queries and retrieved context.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import uuid


class ContextChunk(BaseModel):
    """
    Individual content chunk retrieved from vector database.

    Attributes:
        id: Unique identifier from vector database
        content: Content text from the book (10-2000 chars)
        source_url: Original URL of the content (URL format)
        title: Title of the source page (required)
        heading: Section heading (required)
        score: Similarity score from retrieval (0.0-1.0)
        content_hash: Hash for change detection (required)
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
        regex=r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
        description="Original URL of the content"
    )
    title: str = Field(..., description="Title of the source page")
    heading: str = Field(..., description="Section heading")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score from retrieval")
    content_hash: str = Field(..., description="Hash for change detection")

    @validator('source_url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class QueryWithContext(BaseModel):
    """
    Input model containing user query and retrieved context.

    Attributes:
        query: User's natural language question (1-1000 chars)
        context_chunks: Retrieved content chunks from RAG pipeline (1-20 items)
        session_id: Optional session identifier for multi-turn conversations (UUID format)
        question_type: Type of question to guide response format
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
        regex=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        description="Optional session identifier for multi-turn conversations"
    )
    question_type: str = Field(
        "explanatory",
        regex=r'^(factual|explanatory|comparative)$',
        description="Type of question to guide response format"
    )

    @validator('query')
    def validate_query_length(cls, v):
        if len(v) < 5:
            raise ValueError('Query must be at least 5 characters long for meaningful processing')
        return v

    @validator('session_id', pre=True)
    def validate_session_id_format(cls, v):
        if v is None:
            return v
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('session_id must be a valid UUID')