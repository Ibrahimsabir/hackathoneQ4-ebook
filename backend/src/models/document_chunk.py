from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

class DocumentChunk(BaseModel):
    """
    A semantically coherent segment of book content with its associated embedding and metadata.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1)
    embedding: Optional[List[float]] = Field(None, description="1536-dimensional vector representation")
    source_url: str
    section_title: str = ""
    chapter: str = ""
    position: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        # Allow extra fields for flexibility
        extra = "allow"
        # Enable validation on assignment
        validate_assignment = True

    @validator('content')
    def validate_content_length(cls, v):
        """Validate content length is within OpenAI limits."""
        if len(v) > 8191:  # OpenAI's token limit is 8191
            raise ValueError('Content exceeds OpenAI token limit')
        if len(v) < 1:
            raise ValueError('Content cannot be empty')
        return v

    @validator('embedding')
    def validate_embedding_dimensions(cls, v):
        """Validate embedding has correct dimensions (typically 1024 for Cohere models)."""
        if v is not None:
            # Accept common embedding sizes: Cohere (1024, 384, 768), OpenAI (1536, 1535)
            valid_sizes = [384, 768, 1024, 1535, 1536]
            if len(v) not in valid_sizes:
                raise ValueError(f'Embedding must have a valid dimension size (one of {valid_sizes}), got {len(v)}')
        return v

    @validator('source_url')
    def validate_url_format(cls, v):
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

    def to_payload(self) -> Dict[str, Any]:
        """
        Convert the document chunk to a payload format suitable for Qdrant storage.
        """
        return {
            "content": self.content,
            "source_url": self.source_url,
            "section_title": self.section_title,
            "chapter": self.chapter,
            "position": self.position,
            "heading_hierarchy": self.metadata.get("heading_hierarchy", []),
            "content_type": self.metadata.get("content_type", "text"),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], vector: List[float]) -> 'DocumentChunk':
        """
        Create a DocumentChunk from Qdrant payload and vector.
        """
        return cls(
            id=payload.get("id", str(uuid.uuid4())),
            content=payload["content"],
            embedding=vector,
            source_url=payload["source_url"],
            section_title=payload.get("section_title", ""),
            chapter=payload.get("chapter", ""),
            position=payload.get("position", 0),
            metadata={
                "heading_hierarchy": payload.get("heading_hierarchy", []),
                "content_type": payload.get("content_type", "text")
            },
            created_at=datetime.fromisoformat(payload.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(payload.get("updated_at", datetime.utcnow().isoformat()))
        )