"""
Retry handler module for HTTP requests with exponential backoff
"""
import time
import random
import logging
from typing import Callable, Any, Optional
from functools import wraps
from requests import Response
from logging_config import logger


class RetryHandler:
    """
    Class to handle retry logic with exponential backoff for HTTP requests
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize the retry handler

        Args:
            max_retries (int): Maximum number of retry attempts
            base_delay (float): Initial delay in seconds
            max_delay (float): Maximum delay in seconds
            backoff_factor (float): Factor by which delay increases after each retry
            jitter (bool): Whether to add random jitter to delay times
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.logger = logger

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given attempt number using exponential backoff

        Args:
            attempt (int): Current attempt number (0-indexed)

        Returns:
            float: Delay in seconds
        """
        # Calculate exponential backoff
        delay = self.base_delay * (self.backoff_factor ** attempt)

        # Apply maximum delay cap
        delay = min(delay, self.max_delay)

        # Add jitter if enabled
        if self.jitter:
            # Add random jitter (±25% of the calculated delay)
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)
            # Ensure delay doesn't go below 0
            delay = max(delay, 0)

        return delay

    def should_retry(self, response: Optional[Response] = None, exception: Optional[Exception] = None) -> bool:
        """
        Determine if a request should be retried based on response or exception

        Args:
            response (Response): HTTP response object
            exception (Exception): Exception that occurred

        Returns:
            bool: True if request should be retried
        """
        # If there was an exception, retry
        if exception is not None:
            return True

        # If no response, don't retry
        if response is None:
            return False

        # Retry on server errors (5xx) and client errors (429 - Too Many Requests)
        status_code = response.status_code
        return status_code in [429, 500, 502, 503, 504]

    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic and exponential backoff

        Args:
            func (Callable): Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Any: Result of the function call
        """
        last_exception = None
        last_response = None

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)

                # If we get a response object, check if we should retry
                if isinstance(result, Response):
                    last_response = result
                    if self.should_retry(response=result):
                        if attempt < self.max_retries:
                            delay = self.calculate_delay(attempt)
                            self.logger.warning(
                                f"Request failed with status {result.status_code}, "
                                f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                            )
                            time.sleep(delay)
                            continue
                        else:
                            self.logger.error(f"Request failed after {self.max_retries + 1} attempts")
                            return result
                    else:
                        # Success case - return the result
                        return result
                else:
                    # For non-response results, assume success
                    return result

            except Exception as e:
                last_exception = e
                if self.should_retry(exception=e) and attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    self.logger.warning(
                        f"Request failed with exception: {e}, "
                        f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"Request failed after {attempt + 1} attempts: {e}")
                    raise e

        # If we've exhausted all retries, raise the last exception or return last response
        if last_exception:
            raise last_exception
        return last_response

    def retry_decorator(self, func: Callable) -> Callable:
        """
        Decorator version of retry_with_backoff

        Args:
            func (Callable): Function to decorate

        Returns:
            Callable: Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.retry_with_backoff(func, *args, **kwargs)
        return wrapper


def test_retry_handler():
    """
    Test function to verify the retry handler works
    """
    import requests
    from http_client import http_client

    retry_handler = RetryHandler(max_retries=3, base_delay=0.5)

    # Test with a function that sometimes fails
    call_count = 0

    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise requests.exceptions.ConnectionError("Simulated connection error")
        return f"Success after {call_count} calls"

    try:
        result = retry_handler.retry_with_backoff(flaky_function)
        print(f"Retry test passed: {result}")
    except Exception as e:
        print(f"Retry test failed: {e}")

    # Test with HTTP client
    def fetch_with_retry(url):
        return http_client.get(url)

    # Test with a reliable URL
    try:
        response = retry_handler.retry_with_backoff(fetch_with_retry, "https://httpbin.org/get")
        if response and response.status_code == 200:
            print("HTTP retry test passed")
        else:
            print("HTTP retry test failed")
    except Exception as e:
        print(f"HTTP retry test failed: {e}")


if __name__ == "__main__":
    test_retry_handler()