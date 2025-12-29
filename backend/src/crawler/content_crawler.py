import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from src.utils.config import Config
from src.utils.logging_config import logger
from src.models.document_chunk import DocumentChunk
import time

class ContentCrawler:
    """
    Crawler to extract content from the deployed book website using its sitemap.
    """
    def __init__(self):
        """
        Initialize the content crawler.
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; BookCrawler/1.0)'
        })

    def fetch_sitemap(self) -> Optional[List[str]]:
        """
        Fetch and parse the sitemap to get all URLs.

        Returns:
            List of URLs from the sitemap, or None if failed
        """
        try:
            response = self.session.get(Config.BOOK_SITEMAP_URL)
            response.raise_for_status()

            # Parse the sitemap XML
            root = ET.fromstring(response.content)

            # Handle both regular sitemap and sitemap index
            urls = []
            for url_element in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url/{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                urls.append(url_element.text.strip())

            # If no URLs found with namespace, try without namespace
            if not urls:
                for url_element in root.findall('.//url/loc'):
                    urls.append(url_element.text.strip())

            logger.info(f"Found {len(urls)} URLs in sitemap")
            return urls

        except Exception as e:
            logger.error(f"Failed to fetch sitemap: {str(e)}")
            return None

    def extract_content_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract clean text content from a single URL.

        Args:
            url: URL to extract content from

        Returns:
            Dictionary with content and metadata, or None if failed
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove navigation and other non-content elements
            for element in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                element.decompose()

            # Try to find main content area (Docusaurus specific selectors)
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='main-wrapper') or
                soup.find('div', class_='container') or
                soup.find('div', class_='theme-doc-markdown') or
                soup.find('div', class_='markdown') or
                soup  # fallback to entire soup if no specific content area found
            )

            # Extract text content
            content = main_content.get_text(separator=' ', strip=True)

            # Extract title
            title = soup.find('title')
            title = title.text.strip() if title else urlparse(url).path.split('/')[-1]

            # Extract headings for context
            headings = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                headings.append({
                    'level': heading.name[1],  # Extract number from h1, h2, etc.
                    'text': heading.get_text(strip=True)
                })

            # Extract first h1 as section title if available
            section_title = ""
            h1 = soup.find('h1')
            if h1:
                section_title = h1.get_text(strip=True)

            return {
                'url': url,
                'title': title,
                'content': content,
                'section_title': section_title,
                'headings': headings
            }

        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {str(e)}")
            return None

    async def crawl_all_content(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Crawl all URLs concurrently and extract content.

        Args:
            urls: List of URLs to crawl

        Returns:
            List of dictionaries with content and metadata
        """
        results = []
        failed_urls = []

        # Process URLs with some concurrency but not too many at once
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def crawl_single_url(url):
            async with semaphore:
                try:
                    # Add a small delay to be respectful to the server
                    await asyncio.sleep(0.1)

                    content_data = self.extract_content_from_url(url)
                    if content_data:
                        logger.info(f"Successfully crawled: {url}")
                        return content_data
                    else:
                        failed_urls.append(url)
                        return None
                except Exception as e:
                    logger.error(f"Error crawling {url}: {str(e)}")
                    failed_urls.append(url)
                    return None

        # Create tasks for all URLs
        tasks = [crawl_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        results = [r for r in results if r is not None]

        logger.info(f"Crawling completed: {len(results)} successful, {len(failed_urls)} failed")
        if failed_urls:
            logger.warning(f"Failed URLs: {failed_urls[:5]}...")  # Show first 5 failed URLs

        return results

    def chunk_content(self, content_data: Dict[str, Any], max_chunk_size: int = 1000, overlap: int = 200) -> List[DocumentChunk]:
        """
        Split content into semantically coherent chunks.

        Args:
            content_data: Dictionary with content and metadata
            max_chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks for context preservation

        Returns:
            List of DocumentChunk objects
        """
        content = content_data['content']
        url = content_data['url']
        title = content_data['title']
        section_title = content_data['section_title']
        headings = content_data['headings']

        chunks = []
        position = 0

        # Simple approach: split by sentences while respecting max_chunk_size
        sentences = content.split('. ')
        current_chunk = ""
        current_position = 0

        for i, sentence in enumerate(sentences):
            sentence_with_period = sentence + '. ' if i < len(sentences) - 1 else sentence

            # Check if adding this sentence would exceed the chunk size
            if len(current_chunk) + len(sentence_with_period) <= max_chunk_size:
                current_chunk += sentence_with_period
            else:
                # If the current chunk is not empty, save it
                if current_chunk.strip():
                    chunk_metadata = {
                        'heading_hierarchy': headings,
                        'content_type': 'text',
                        'language': 'en'
                    }

                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        source_url=url,
                        section_title=section_title,
                        chapter=title,
                        position=current_position,
                        metadata=chunk_metadata
                    )
                    chunks.append(chunk)
                    current_position += 1

                # Start a new chunk with potential overlap
                if len(sentence_with_period) > max_chunk_size:
                    # If the sentence itself is too long, split it
                    for j in range(0, len(sentence_with_period), max_chunk_size):
                        sub_sentence = sentence_with_period[j:j + max_chunk_size]
                        chunk_metadata = {
                            'heading_hierarchy': headings,
                            'content_type': 'text',
                            'language': 'en'
                        }

                        chunk = DocumentChunk(
                            content=sub_sentence.strip(),
                            source_url=url,
                            section_title=section_title,
                            chapter=title,
                            position=current_position,
                            metadata=chunk_metadata
                        )
                        chunks.append(chunk)
                        current_position += 1
                else:
                    # Start new chunk with overlap from previous chunk
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + sentence_with_period

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunk_metadata = {
                'heading_hierarchy': headings,
                'content_type': 'text',
                'language': 'en'
            }

            chunk = DocumentChunk(
                content=current_chunk.strip(),
                source_url=url,
                section_title=section_title,
                chapter=title,
                position=current_position,
                metadata=chunk_metadata
            )
            chunks.append(chunk)

        return chunks