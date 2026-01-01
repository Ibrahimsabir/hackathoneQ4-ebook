"""
Structured logging utilities for the RAG chatbot system.
"""
import logging
import sys
from datetime import datetime
from typing import Any, Dict


def setup_logging(log_level: str = "INFO", log_format: str = None) -> logging.Logger:
    """
    Set up structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = (
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("rag_chatbot.log")
        ]
    )

    # Create and return a logger for the application
    logger = logging.getLogger("rag_chatbot")
    return logger


def log_api_call(
    logger: logging.Logger,
    endpoint: str,
    method: str,
    query: str,
    processing_time: float,
    status_code: int = 200
) -> None:
    """
    Log an API call with structured information.

    Args:
        logger: Logger instance
        endpoint: API endpoint that was called
        method: HTTP method used
        query: User query (truncated if too long)
        processing_time: Time taken to process the request
        status_code: HTTP status code returned
    """
    query_preview = query[:100] + "..." if len(query) > 100 else query
    logger.info(
        f"API_CALL endpoint={endpoint} method={method} "
        f"query='{query_preview}' processing_time={processing_time:.3f}s "
        f"status={status_code}"
    )


def log_retrieval_result(
    logger: logging.Logger,
    query: str,
    num_chunks: int,
    retrieval_time: float,
    success: bool = True
) -> None:
    """
    Log a retrieval operation result.

    Args:
        logger: Logger instance
        query: Query that was used for retrieval
        num_chunks: Number of chunks retrieved
        retrieval_time: Time taken for retrieval
        success: Whether the retrieval was successful
    """
    status = "SUCCESS" if success else "FAILED"
    logger.info(
        f"RETRIEVAL query='{query[:50]}...' chunks={num_chunks} "
        f"time={retrieval_time:.3f}s status={status}"
    )


def log_generation_result(
    logger: logging.Logger,
    query: str,
    tokens_used: int,
    generation_time: float,
    success: bool = True
) -> None:
    """
    Log a generation operation result.

    Args:
        logger: Logger instance
        query: Query that was used for generation
        tokens_used: Number of tokens used in generation
        generation_time: Time taken for generation
        success: Whether the generation was successful
    """
    status = "SUCCESS" if success else "FAILED"
    logger.info(
        f"GENERATION query='{query[:50]}...' tokens={tokens_used} "
        f"time={generation_time:.3f}s status={status}"
    )


def log_error(
    logger: logging.Logger,
    error_type: str,
    error_message: str,
    context: Dict[str, Any] = None
) -> None:
    """
    Log an error with context information.

    Args:
        logger: Logger instance
        error_type: Type/classification of the error
        error_message: Detailed error message
        context: Additional context information about the error
    """
    context_str = f" context={context}" if context else ""
    logger.error(f"ERROR type={error_type} message='{error_message}'{context_str}")


# Initialize the main logger
main_logger = setup_logging()