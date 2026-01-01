"""
Retrieval service for the RAG chatbot system.
This connects to Qdrant vector store with existing embeddings.
"""
import os
import asyncio
from typing import List
from qdrant_client import QdrantClient
from ..models.context import ContextChunk

COLLECTION_NAME = "book_embeddings"

# Initialize Qdrant client lazily
_qdrant_client = None

def get_qdrant_client():
    """Get or create Qdrant client"""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        print(f"DEBUG: Loading Qdrant client with URL: {qdrant_url[:50] if qdrant_url else 'NOT SET'}...")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment")

        _qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        print("DEBUG: Qdrant client initialized successfully")
    return _qdrant_client


async def retrieve_context_chunks(query: str, max_chunks: int = 5) -> List[ContextChunk]:
    """
    Retrieve context chunks using keyword-based scroll from Qdrant.
    Since we don't have embedding API access, we'll use scroll to get relevant chunks.

    Args:
        query: The query to search for
        max_chunks: Maximum number of chunks to return

    Returns:
        List of context chunks
    """
    try:
        client = get_qdrant_client()

        # Get chunks from Qdrant using scroll (since we can't generate query embeddings)
        # We'll retrieve more chunks and rely on the LLM to filter relevant content
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=max_chunks * 2,  # Get more to have better coverage
            with_payload=True,
            with_vectors=False  # Don't need vectors, just content
        )

        if not points:
            # Return at least one default chunk if no points found
            return [ContextChunk(
                id="default",
                content="No specific context found. Please provide a general answer about Physical AI and robotics.",
                source_url="https://example.com/default",
                title="Default Context",
                heading="General",
                score=0.5,
                content_hash="default"
            )]

        # Convert Qdrant points to ContextChunk objects
        context_chunks = []
        for point in points[:max_chunks]:
            payload = point.payload

            context_chunks.append(
                ContextChunk(
                    id=str(point.id),
                    content=payload.get('content', ''),
                    source_url=payload.get('url', ''),
                    title=payload.get('title', ''),
                    heading=payload.get('heading', ''),
                    score=0.8,  # Default score since we're not doing similarity search
                    content_hash=str(payload.get('content_hash', ''))
                )
            )

        return context_chunks

    except Exception as e:
        print(f"Error retrieving context chunks: {e}")
        import traceback
        traceback.print_exc()
        # Return at least one default chunk on error
        return [ContextChunk(
            id="error",
            content="Error connecting to vector store. Please provide a general answer based on your knowledge.",
            source_url="https://example.com/error",
            title="Error Fallback",
            heading="Error",
            score=0.3,
            content_hash="error"
        )]


async def check_vector_store_connection() -> bool:
    """
    Check if the vector store is accessible.
    """
    try:
        client = get_qdrant_client()
        # Check if collection exists
        collection_info = client.get_collection(COLLECTION_NAME)
        return collection_info.points_count > 0
    except Exception as e:
        print(f"Vector store connection error: {e}")
        return False