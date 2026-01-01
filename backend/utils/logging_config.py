"""
Logging configuration for the RAG chatbot answer generation system.

This module provides structured logging utilities for debugging and monitoring.
"""
import logging
import sys
from typing import Optional


def setup_structured_logging(
    name: str = "rag_answer_generation",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up structured logging for the RAG answer generation system.

    Args:
        name: Name for the logger
        level: Logging level (default: INFO)
        log_file: Optional file path to log to

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already has handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_answer_generation_event(
    logger: logging.Logger,
    query: str,
    context_chunk_count: int,
    answer_length: int,
    confidence_score: float,
    processing_time: float,
    session_id: Optional[str] = None
):
    """
    Log an answer generation event with structured information.

    Args:
        logger: The logger instance to use
        query: The original query
        context_chunk_count: Number of context chunks used
        answer_length: Length of the generated answer
        confidence_score: Confidence score of the answer
        processing_time: Time taken to generate the answer
        session_id: Optional session ID
    """
    logger.info(
        "Answer Generated - Query: %s | Context Chunks: %d | Answer Length: %d | "
        "Confidence: %.2f | Time: %.2fs | Session: %s",
        query[:50] + "..." if len(query) > 50 else query,  # Truncate long queries
        context_chunk_count,
        answer_length,
        confidence_score,
        processing_time,
        session_id or "N/A"
    )


def log_validation_result(
    logger: logging.Logger,
    grounding_score: float,
    consistency_score: float,
    is_acceptable: bool,
    issues_count: int
):
    """
    Log validation results for an answer.

    Args:
        logger: The logger instance to use
        grounding_score: Grounding score of the answer
        consistency_score: Consistency score of the answer
        is_acceptable: Whether the answer meets quality standards
        issues_count: Number of issues detected
    """
    logger.debug(
        "Validation Result - Grounding: %.2f | Consistency: %.2f | Acceptable: %s | Issues: %d",
        grounding_score,
        consistency_score,
        is_acceptable,
        issues_count
    )