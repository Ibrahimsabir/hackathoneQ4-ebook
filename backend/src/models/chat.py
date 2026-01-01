"""
Request and response models for the chat system.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from .citation import Citation
from .context import ContextChunk


class ChatRequest(BaseModel):
    """
    Request model for general chat queries.
    """
    query: str = Field(..., min_length=1, max_length=1000, description="The user's question")
    session_id: Optional[str] = Field(
        None,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        description="Session identifier for conversation continuity"
    )


class ChatWithContextRequest(BaseModel):
    """
    Request model for chat queries with selected text context.
    """
    query: str = Field(..., min_length=1, max_length=1000, description="The user's question")
    selected_text: Optional[str] = Field(
        None,
        max_length=2000,
        description="Text selected by the user for contextual questioning"
    )
    session_id: Optional[str] = Field(
        None,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        description="Session identifier for conversation continuity"
    )


class ChatResponse(BaseModel):
    """
    Response model for chat queries.
    """
    answer: str = Field(..., min_length=1, description="The generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Citations for the answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the answer")
    processing_time: float = Field(..., ge=0.0, description="Time taken to process the request in seconds")
    session_id: str = Field(..., description="Session identifier")


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Overall health status")
    vector_store: str = Field(..., description="Vector store connection status")
    llm_service: str = Field(..., description="LLM service availability status")
    timestamp: str = Field(..., description="ISO 8601 timestamp")