"""
HTTP client module with rate limiting for crawling book pages
"""
import requests
import time
from typing import Optional
import logging
from logging_config import logger
import random
from urllib.parse import urljoin


class RateLimitedHTTPClient:
    """
    HTTP client with built-in rate limiting to respect free-tier limits
    """

    def __init__(self, requests_per_minute: int = 30, min_delay: float = 1.0):
        """
        Initialize the rate-limited HTTP client

        Args:
            requests_per_minute (int): Maximum requests per minute (default: 30 for Cohere free tier)
            min_delay (float): Minimum delay between requests in seconds
        """
        self.max_requests_per_minute = requests_per_minute
        self.min_delay = min_delay
        self.requests_in_current_window = 0
        self.window_start_time = time.time()
        self.logger = logger

    def _enforce_rate_limit(self):
        """
        Enforce rate limiting by sleeping if necessary
        """
        current_time = time.time()
        time_elapsed = current_time - self.window_start_time

        # If we're within the same minute window
        if time_elapsed < 60:
            if self.requests_in_current_window >= self.max_requests_per_minute:
                # Wait until the window resets
                sleep_time = 60 - time_elapsed
                self.logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.requests_in_current_window = 0
                self.window_start_time = time.time()
        else:
            # Reset the window
            self.requests_in_current_window = 0
            self.window_start_time = current_time

        # Ensure minimum delay
        if self.requests_in_current_window > 0:
            time.sleep(self.min_delay)

        self.requests_in_current_window += 1

    def get(self, url: str, timeout: int = 30, **kwargs) -> Optional[requests.Response]:
        """
        Make a GET request with rate limiting

        Args:
            url (str): URL to fetch
            timeout (int): Request timeout in seconds
            **kwargs: Additional arguments to pass to requests.get

        Returns:
            requests.Response or None: Response object or None if request failed
        """
        self._enforce_rate_limit()

        try:
            self.logger.debug(f"Fetching URL: {url}")
            response = requests.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()  # Raise an exception for bad status codes
            self.logger.debug(f"Successfully fetched URL: {url}")
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching URL {url}: {e}")
            return None

    def head(self, url: str, timeout: int = 30, **kwargs) -> Optional[requests.Response]:
        """
        Make a HEAD request with rate limiting

        Args:
            url (str): URL to fetch
            timeout (int): Request timeout in seconds
            **kwargs: Additional arguments to pass to requests.head

        Returns:
            requests.Response or None: Response object or None if request failed
        """
        self._enforce_rate_limit()

        try:
            self.logger.debug(f"Making HEAD request to: {url}")
            response = requests.head(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            self.logger.debug(f"Successfully made HEAD request to: {url}")
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error making HEAD request to {url}: {e}")
            return None


# Create a global instance for use throughout the application
http_client = RateLimitedHTTPClient(requests_per_minute=30)  # Conservative rate for free tier


def test_client():
    """
    Test function to verify the HTTP client works
    """
    from config import SITEMAP_URL

    # Test with a simple URL
    response = http_client.get("https://httpbin.org/get")
    if response:
        print("HTTP client test successful")
        print(f"Status code: {response.status_code}")
    else:
        print("HTTP client test failed")


if __name__ == "__main__":
    test_client()