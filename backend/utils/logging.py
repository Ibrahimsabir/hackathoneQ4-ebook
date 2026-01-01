"""
Logging infrastructure for the retrieval pipeline.

This module provides structured logging capabilities for the retrieval system,
including different log levels, structured formats, and integration with
external services for monitoring and observability.
"""
import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime


class RetrievalLogger:
    """Structured logger for retrieval pipeline operations."""

    def __init__(self, name: str = "retrieval_pipeline", level: int = logging.INFO):
        """
        Initialize the retrieval logger.

        Args:
            name: Name of the logger
            level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid adding multiple handlers if logger already exists
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)

            # Create formatter with structured output
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)

            # Add handler to logger
            self.logger.addHandler(console_handler)

    def log_query(self, query_id: str, query_text: str, timestamp: datetime):
        """Log query processing event."""
        self.logger.info(
            f"Query processed - ID: {query_id}, Text: {query_text[:50]}..., "
            f"Timestamp: {timestamp.isoformat()}"
        )

    def log_embedding_generation(self, query_id: str, success: bool, duration_ms: float):
        """Log embedding generation event."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"Embedding generation - Query ID: {query_id}, Status: {status}, "
            f"Duration: {duration_ms}ms"
        )

    def log_similarity_search(self, query_id: str, num_results: int, duration_ms: float):
        """Log similarity search event."""
        self.logger.info(
            f"Similarity search - Query ID: {query_id}, Results: {num_results}, "
            f"Duration: {duration_ms}ms"
        )

    def log_retrieval_result(self, query_id: str, avg_similarity: float, total_chunks: int):
        """Log retrieval result summary."""
        self.logger.info(
            f"Retrieval result - Query ID: {query_id}, Avg Similarity: {avg_similarity:.3f}, "
            f"Chunks: {total_chunks}"
        )

    def log_error(self, query_id: str, error_type: str, error_message: str):
        """Log error event."""
        self.logger.error(
            f"Error - Query ID: {query_id}, Type: {error_type}, Message: {error_message}"
        )

    def log_validation_result(self, query_id: str, accuracy_score: float, validation_notes: str = ""):
        """Log validation result."""
        self.logger.info(
            f"Validation result - Query ID: {query_id}, Accuracy: {accuracy_score:.3f}, "
            f"Notes: {validation_notes}"
        )

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


# Global logger instance for the retrieval pipeline
retrieval_logger = RetrievalLogger()