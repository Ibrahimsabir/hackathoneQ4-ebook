"""
Base classes and interfaces for retrieval pipeline entities.

This module defines the core data structures and interfaces used in the
retrieval pipeline, providing a foundation for consistent data handling
across all components.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Query:
    """
    Represents a user query for retrieval.

    Attributes:
        text: The raw query text
        processed_text: Cleaned and normalized query text
        timestamp: When the query was received
        query_id: Unique identifier for the query
    """
    text: str
    processed_text: Optional[str] = None
    timestamp: Optional[datetime] = None
    query_id: Optional[str] = None


@dataclass
class Embedding:
    """
    Represents a vector embedding of query text.

    Attributes:
        vector: The embedding vector
        dimension: Number of dimensions in the vector
        model: The model used to generate the embedding
        query_id: Reference to the original query
    """
    vector: List[float]
    dimension: int
    model: str
    query_id: Optional[str] = None


@dataclass
class RetrievedChunk:
    """
    Represents a content chunk retrieved from the vector database.

    Attributes:
        content: The actual content text
        url: Source URL of the content
        page_title: Title of the source page
        section_heading: Specific heading under which the content falls
        chunk_id: Unique identifier for the chunk
        similarity_score: Cosine similarity score to the query
        content_hash: Hash of the content for change detection
        position: Position of this chunk within the original document
    """
    content: str
    url: str
    page_title: str
    section_heading: str
    chunk_id: str
    similarity_score: float
    content_hash: str
    position: int


@dataclass
class ContextBlock:
    """
    Represents a structured assembly of retrieved chunks with metadata.

    Attributes:
        chunks: List of retrieved content chunks
        query: Original query that generated this context
        total_chunks: Number of chunks in the context
        avg_similarity: Average similarity score across all chunks
        retrieval_time_ms: Time taken to retrieve the context
        query_embedding: The embedding used for retrieval
    """
    chunks: List[RetrievedChunk]
    query: str
    total_chunks: int
    avg_similarity: float
    retrieval_time_ms: int
    query_embedding: Optional[Embedding] = None


@dataclass
class ValidationResult:
    """
    Represents validation information for a retrieval result.

    Attributes:
        query: The original query
        retrieved_chunks: Chunks that were retrieved
        expected_results: Expected results for validation
        accuracy_score: Validation accuracy score
        validation_timestamp: When validation was performed
        validation_notes: Additional notes about the validation
    """
    query: str
    retrieved_chunks: List[RetrievedChunk]
    expected_results: List[str]
    accuracy_score: float
    validation_timestamp: datetime
    validation_notes: str = ""


class IQueryProcessor(ABC):
    """Interface for query processing components."""

    @abstractmethod
    def process(self, query: Query) -> Query:
        """Process a query and return the processed version."""
        pass


class IEmbeddingGenerator(ABC):
    """Interface for embedding generation components."""

    @abstractmethod
    def generate(self, text: str) -> Embedding:
        """Generate an embedding for the given text."""
        pass


class IRetrievalService(ABC):
    """Interface for retrieval service components."""

    @abstractmethod
    def retrieve(self, query_embedding: Embedding) -> List[RetrievedChunk]:
        """Retrieve chunks based on the query embedding."""
        pass


class IContextAssembler(ABC):
    """Interface for context assembly components."""

    @abstractmethod
    def assemble(self, chunks: List[RetrievedChunk], query: str, retrieval_time_ms: int) -> ContextBlock:
        """Assemble chunks into a context block."""
        pass


class IValidationService(ABC):
    """Interface for validation service components."""

    @abstractmethod
    def validate(self, context_block: ContextBlock, expected_results: List[str]) -> ValidationResult:
        """Validate the retrieval results."""
        pass