"""
Chat API router for the RAG chatbot system.
"""
import uuid
import logging
from fastapi import APIRouter, HTTPException
from typing import Optional
from ...models.chat import ChatRequest, ChatWithContextRequest, ChatResponse
from ...services.rag_service import rag_service


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=ChatResponse)
async def ask_general_question(request: ChatRequest):
    """
    Handle general book questions without specific context.
    """
    try:
        # Validate query length
        if len(request.query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long, maximum 1000 characters")

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process the query through the RAG pipeline
        answer, citations, confidence, processing_time = await rag_service.process_query(
            query=request.query
        )

        # Create response
        response = ChatResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            processing_time=processing_time,
            session_id=session_id
        )

        logger.info(f"Processed general query: '{request.query[:50]}...' in {processing_time:.2f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing general query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing query")


@router.post("/ask-with-context", response_model=ChatResponse)
async def ask_with_context(request: ChatWithContextRequest):
    """
    Handle questions with selected text context.
    """
    try:
        # Validate query length
        if len(request.query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long, maximum 1000 characters")

        # Validate selected text length if provided
        if request.selected_text and len(request.selected_text) > 2000:
            raise HTTPException(status_code=400, detail="Selected text too long, maximum 2000 characters")

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process the query through the RAG pipeline with selected text context
        answer, citations, confidence, processing_time = await rag_service.process_query(
            query=request.query,
            selected_text=request.selected_text
        )

        # Create response
        response = ChatResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            processing_time=processing_time,
            session_id=session_id
        )

        logger.info(f"Processed contextual query: '{request.query[:50]}...' in {processing_time:.2f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing contextual query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing query")