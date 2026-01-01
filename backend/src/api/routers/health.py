"""
Health check API router for the RAG chatbot system.
"""
import time
from fastapi import APIRouter
from ...models.chat import HealthCheckResponse
from ...services.retrieval_service import check_vector_store_connection


router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify system status.
    """
    start_time = time.time()

    # Check vector store connection
    try:
        vector_store_connected = await check_vector_store_connection()
        vector_store_status = "connected" if vector_store_connected else "disconnected"
    except Exception:
        vector_store_status = "error"

    # Check LLM service (mock check - in reality would check OpenAI API)
    try:
        # In a real implementation, this would check the OpenAI API connection
        llm_service_status = "available"
    except Exception:
        llm_service_status = "unavailable"

    # Overall status
    overall_status = "healthy" if vector_store_status == "connected" and llm_service_status == "available" else "degraded"

    response = HealthCheckResponse(
        status=overall_status,
        vector_store=vector_store_status,
        llm_service=llm_service_status,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    )

    return response