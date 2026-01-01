"""
Context chunk model for the RAG chatbot system.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union


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