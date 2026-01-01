"""
Comprehensive error handler for the ingestion pipeline
"""
import logging
import traceback
from typing import Any, Dict, Optional, Callable
from logging_config import logger
import sys
import os
from datetime import datetime


class IngestionErrorHandler:
    """
    Comprehensive error handler for the ingestion pipeline
    """

    def __init__(self, error_log_file: str = "ingestion_errors.log"):
        """
        Initialize the error handler

        Args:
            error_log_file (str): File to log errors to
        """
        self.error_log_file = error_log_file
        self.logger = logger
        self.error_count = 0
        self.errors = []

        # Set up error logging
        self._setup_error_logging()

    def _setup_error_logging(self):
        """
        Set up error logging configuration
        """
        # Create file handler for errors
        error_handler = logging.FileHandler(self.error_log_file)
        error_handler.setLevel(logging.ERROR)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(error_handler)

    def handle_error(self, error: Exception, context: str = "", should_raise: bool = False) -> Dict[str, Any]:
        """
        Handle an error with comprehensive logging

        Args:
            error (Exception): The error that occurred
            context (str): Context information about where the error occurred
            should_raise (bool): Whether to re-raise the error

        Returns:
            Dict[str, Any]: Error information
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "error_id": f"ERR_{self.error_count + 1:06d}"
        }

        # Log the error
        self.logger.error(
            f"Error ID {error_info['error_id']}: {error_info['error_type']} in {context}: {error_info['error_message']}"
        )

        # Store error info
        self.errors.append(error_info)
        self.error_count += 1

        # Write to error log file
        self._write_error_to_file(error_info)

        # Optionally re-raise the error
        if should_raise:
            raise error

        return error_info

    def _write_error_to_file(self, error_info: Dict[str, Any]):
        """
        Write error information to the error log file

        Args:
            error_info (Dict[str, Any]): Error information to write
        """
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"ERROR ID: {error_info['error_id']}\n")
                f.write(f"TIMESTAMP: {error_info['timestamp']}\n")
                f.write(f"CONTEXT: {error_info['context']}\n")
                f.write(f"ERROR TYPE: {error_info['error_type']}\n")
                f.write(f"ERROR MESSAGE: {error_info['error_message']}\n")
                f.write(f"TRACEBACK:\n{error_info['traceback']}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            self.logger.error(f"Failed to write error to log file: {e}")

    def handle_api_error(self, error: Exception, service_name: str, endpoint: str) -> Dict[str, Any]:
        """
        Handle API-specific errors

        Args:
            error (Exception): The API error
            service_name (str): Name of the service (e.g., 'Cohere', 'Qdrant')
            endpoint (str): API endpoint that failed

        Returns:
            Dict[str, Any]: Error information
        """
        context = f"{service_name} API error at {endpoint}"
        return self.handle_error(error, context)

    def handle_network_error(self, error: Exception, url: str = "") -> Dict[str, Any]:
        """
        Handle network-specific errors

        Args:
            error (Exception): The network error
            url (str): URL that caused the error

        Returns:
            Dict[str, Any]: Error information
        """
        context = f"Network error for URL: {url}" if url else "Network error"
        return self.handle_error(error, context)

    def handle_validation_error(self, error: Exception, validation_type: str) -> Dict[str, Any]:
        """
        Handle validation-specific errors

        Args:
            error (Exception): The validation error
            validation_type (str): Type of validation that failed

        Returns:
            Dict[str, Any]: Error information
        """
        context = f"Validation error ({validation_type})"
        return self.handle_error(error, context)

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors

        Returns:
            Dict[str, Any]: Error summary
        """
        error_summary = {
            "total_errors": self.error_count,
            "unique_error_types": list(set(error["error_type"] for error in self.errors)),
            "recent_errors": self.errors[-5:] if self.errors else [],
            "error_rate": self.error_count / max(1, len(self.errors)) if hasattr(self, '_total_operations') else 0
        }

        return error_summary

    def reset_error_count(self):
        """
        Reset the error counter
        """
        self.error_count = 0
        self.errors = []

    def safe_execute(self, func: Callable, *args, context: str = "", **kwargs) -> Any:
        """
        Safely execute a function with error handling

        Args:
            func (Callable): Function to execute
            *args: Arguments to pass to the function
            context (str): Context for error logging
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Any: Result of the function or None if error occurred
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context)
            return None

    def log_warning(self, message: str, context: str = ""):
        """
        Log a warning message

        Args:
            message (str): Warning message
            context (str): Context information
        """
        warning_info = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": context,
            "type": "WARNING"
        }

        self.logger.warning(f"WARNING in {context}: {message}")
        self._write_warning_to_file(warning_info)

    def _write_warning_to_file(self, warning_info: Dict[str, Any]):
        """
        Write warning information to the error log file

        Args:
            warning_info (Dict[str, Any]): Warning information to write
        """
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'-'*80}\n")
                f.write(f"WARNING at {warning_info['timestamp']}\n")
                f.write(f"CONTEXT: {warning_info['context']}\n")
                f.write(f"MESSAGE: {warning_info['message']}\n")
                f.write(f"{'-'*80}\n")
        except Exception as e:
            self.logger.error(f"Failed to write warning to log file: {e}")


def test_error_handler():
    """
    Test function to verify the error handler works
    """
    handler = IngestionErrorHandler()

    # Test basic error handling
    try:
        raise ValueError("This is a test error")
    except ValueError as e:
        error_info = handler.handle_error(e, "test context")
        print(f"Handled error: {error_info['error_id']}")

    # Test safe execution
    def failing_function():
        raise RuntimeError("This function fails")

    result = handler.safe_execute(failing_function, context="testing safe execution")
    print(f"Safe execution result: {result}")

    # Test API error
    try:
        raise ConnectionError("API connection failed")
    except ConnectionError as e:
        handler.handle_api_error(e, "Cohere", "/embed")

    # Get error summary
    summary = handler.get_error_summary()
    print(f"Error summary: {summary}")


if __name__ == "__main__":
    test_error_handler()