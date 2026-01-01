"""
Citation model for the RAG chatbot system.
"""
from pydantic import BaseModel, Field
from typing import Optional


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