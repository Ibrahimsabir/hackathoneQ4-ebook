"""
Optimized components for better performance
"""
import asyncio
import aiohttp
import concurrent.futures
from typing import List, Dict, Any, Optional
import logging
from logging_config import logger
import time
from functools import wraps
from config import CHUNK_SIZE_LIMIT
import re


def async_timer(func):
    """Decorator to time async functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper


def sync_timer(func):
    """Decorator to time sync functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper


class OptimizedHTTPClient:
    """
    Optimized HTTP client using async/await for better performance
    """

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.logger = logger

    @async_timer
    async def fetch_batch(self, urls: List[str], timeout: int = 30) -> List[Optional[str]]:
        """
        Fetch multiple URLs concurrently

        Args:
            urls (List[str]): List of URLs to fetch
            timeout (int): Request timeout in seconds

        Returns:
            List[Optional[str]]: List of response contents
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_one(session, url):
            async with semaphore:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            self.logger.error(f"HTTP {response.status} for {url}")
                            return None
                except Exception as e:
                    self.logger.error(f"Error fetching {url}: {e}")
                    return None

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_one(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions in the results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Exception in request {i}: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)

            return processed_results


class OptimizedContentExtractor:
    """
    Optimized content extractor with better performance
    """

    def __init__(self):
        self.logger = logger
        # Pre-compile regex patterns for better performance
        self.nav_pattern = re.compile(
            r'<nav[^>]*>.*?</nav>|<header[^>]*>.*?</header>|<footer[^>]*>.*?</footer>|<aside[^>]*>.*?</aside>',
            re.IGNORECASE | re.DOTALL
        )
        self.script_pattern = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
        self.style_pattern = re.compile(r'<style[^>]*>.*?</style>', re.IGNORECASE | re.DOTALL)
        self.link_pattern = re.compile(r'<link[^>]*>', re.IGNORECASE)

    @sync_timer
    def extract_content_batch(self, html_contents: List[str]) -> List[Dict[str, str]]:
        """
        Extract content from multiple HTML strings in batch

        Args:
            html_contents (List[str]): List of HTML content strings

        Returns:
            List[Dict[str, str]]: List of extracted content
        """
        from bs4 import BeautifulSoup

        results = []
        for html in html_contents:
            if not html:
                results.append({"title": "", "content": "", "section_heading": ""})
                continue

            # Use regex to quickly remove common navigation elements
            clean_html = self.nav_pattern.sub('', html)
            clean_html = self.script_pattern.sub('', clean_html)
            clean_html = self.style_pattern.sub('', clean_html)
            clean_html = self.link_pattern.sub('', clean_html)

            # Parse with BeautifulSoup
            soup = BeautifulSoup(clean_html, 'html.parser')

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "No Title"

            # Extract main heading
            h1_tag = soup.find('h1')
            section_heading = h1_tag.get_text().strip() if h1_tag else "No Heading"

            # Get clean text content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            content = re.sub(r'\s+', ' ', content).strip()

            results.append({
                "title": title,
                "content": content,
                "section_heading": section_heading
            })

        return results


class OptimizedChunker:
    """
    Optimized chunker with better performance
    """

    def __init__(self, max_tokens: int = CHUNK_SIZE_LIMIT, overlap_ratio: float = 0.2):
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.logger = logger

    def chunk_batch(self, content_list: List[str], urls: List[str], titles: List[str]) -> List[Dict[str, str]]:
        """
        Chunk multiple content items in batch

        Args:
            content_list (List[str]): List of content to chunk
            urls (List[str]): Corresponding URLs
            titles (List[str]): Corresponding titles

        Returns:
            List[Dict[str, str]]: List of chunks with metadata
        """
        all_chunks = []
        for content, url, title in zip(content_list, urls, titles):
            chunks = self.chunk_by_headings(content, url, title)
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_by_headings(self, content: str, url: str, page_title: str) -> List[Dict[str, str]]:
        """
        Optimized chunking by headings with size limits
        """
        # Split content by headings
        heading_pattern = r'(\n\s*#+\s+.*?\n|\n\s*<h[1-6][^>]*>.*?</h[1-6]>\n|\n\s*<h[1-6][^>]*>.*?\n)'
        parts = re.split(heading_pattern, content, flags=re.IGNORECASE)

        chunks = []
        current_section = ""
        current_heading = page_title

        i = 0
        while i < len(parts):
            part = parts[i].strip()

            if self._is_heading(part):
                # Save previous section if it exists
                if current_section.strip():
                    section_chunks = self._chunk_section(current_section, current_heading, url, page_title)
                    chunks.extend(section_chunks)

                # Extract the heading text
                current_heading = self._extract_heading_text(part)
                current_section = ""
            else:
                # Add content to current section
                current_section += part + "\n"

            i += 1

        # Process the last section
        if current_section.strip():
            section_chunks = self._chunk_section(current_section, current_heading, url, page_title)
            chunks.extend(section_chunks)

        return chunks

    def _is_heading(self, text: str) -> bool:
        """Check if text is a heading"""
        return bool(re.match(r'^\s*#+\s+', text.strip())) or bool(re.search(r'<h[1-6][^>]*>.*?</h[1-6]>', text, re.IGNORECASE))

    def _extract_heading_text(self, heading: str) -> str:
        """Extract clean heading text"""
        clean_heading = re.sub(r'^\s*#+\s*', '', heading.strip())
        clean_heading = re.sub(r'<[^>]+>', '', clean_heading)
        return clean_heading.strip() or "Untitled Section"

    def _chunk_section(self, section: str, heading: str, url: str, page_title: str) -> List[Dict[str, str]]:
        """Chunk a single section efficiently"""
        chunks = []

        if self._count_tokens(section) <= self.max_tokens:
            chunk_data = {
                "content": section.strip(),
                "url": url,
                "page_title": page_title,
                "section_heading": heading,
                "chunk_id": f"{hash(url + heading + section[:50]) % 100000}"
            }
            chunks.append(chunk_data)
        else:
            # Split into smaller chunks
            sentences = self._split_into_sentences(section)
            current_chunk = ""
            chunk_num = 1

            for sentence in sentences:
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence

                if self._count_tokens(test_chunk) <= self.max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk.strip():
                        chunk_data = {
                            "content": current_chunk.strip(),
                            "url": url,
                            "page_title": page_title,
                            "section_heading": f"{heading} (Part {chunk_num})",
                            "chunk_id": f"{hash(url + heading + str(chunk_num)) % 100000}"
                        }
                        chunks.append(chunk_data)
                        chunk_num += 1

                    current_chunk = sentence

            if current_chunk.strip():
                chunk_data = {
                    "content": current_chunk.strip(),
                    "url": url,
                    "page_title": page_title,
                    "section_heading": f"{heading} (Part {chunk_num})",
                    "chunk_id": f"{hash(url + heading + str(chunk_num)) % 100000}"
                }
                chunks.append(chunk_data)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        """Count approximate tokens"""
        if not text:
            return 0
        words = re.findall(r'\b\w+\b', text)
        return len(words)


class OptimizedIngestionPipeline:
    """
    Optimized ingestion pipeline that uses batch processing and concurrency
    """

    def __init__(self):
        self.http_client = OptimizedHTTPClient(max_concurrent=3)  # Conservative for free tier
        self.content_extractor = OptimizedContentExtractor()
        self.chunker = OptimizedChunker()
        self.logger = logger

    async def process_urls_batch(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Process a batch of URLs efficiently

        Args:
            urls (List[str]): List of URLs to process

        Returns:
            List[Dict[str, str]]: List of processed content
        """
        self.logger.info(f"Processing {len(urls)} URLs in batch")

        # Fetch all HTML content concurrently
        html_contents = await self.http_client.fetch_batch(urls)

        # Extract content from all HTML strings
        content_items = self.content_extractor.extract_content_batch(html_contents)

        # Prepare data for chunking
        valid_contents = []
        valid_urls = []
        valid_titles = []

        for i, (content_item, url) in enumerate(zip(content_items, urls)):
            if content_item['content']:  # Only process if content was extracted
                valid_contents.append(content_item['content'])
                valid_urls.append(url)
                valid_titles.append(content_item['title'])

        # Chunk all content
        all_chunks = self.chunker.chunk_batch(valid_contents, valid_urls, valid_titles)

        self.logger.info(f"Successfully processed {len(all_chunks)} chunks from {len(urls)} URLs")
        return all_chunks


def run_performance_optimization_demo():
    """
    Demo function to show the performance optimizations
    """
    import asyncio

    async def demo():
        print("Testing optimized components...")

        # Test optimized pipeline
        pipeline = OptimizedIngestionPipeline()

        # Use some test URLs
        test_urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/json"
        ]

        print(f"Processing {len(test_urls)} test URLs...")
        chunks = await pipeline.process_urls_batch(test_urls)

        print(f"Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"Chunk {i+1}: {len(chunk['content'])} chars from {chunk['url']}")

    # Run the async demo
    asyncio.run(demo())


if __name__ == "__main__":
    run_performance_optimization_demo()