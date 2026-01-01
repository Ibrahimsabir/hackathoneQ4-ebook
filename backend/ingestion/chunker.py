"""
Chunker module for splitting content using heading-aware and size-bounded strategies
"""
import re
from typing import List, Dict, Tuple
import logging
from logging_config import logger
from config import CHUNK_SIZE_LIMIT, CHUNK_OVERLAP_RATIO


class ContentChunker:
    """
    Class to chunk content using heading-aware splitting and size limits
    """

    def __init__(self, max_tokens: int = CHUNK_SIZE_LIMIT, overlap_ratio: float = CHUNK_OVERLAP_RATIO):
        """
        Initialize the content chunker

        Args:
            max_tokens (int): Maximum tokens per chunk (default from config)
            overlap_ratio (float): Ratio of overlap between chunks (default from config)
        """
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.logger = logger

    def chunk_by_headings(self, content: str, url: str, page_title: str) -> List[Dict[str, str]]:
        """
        Chunk content based on headings, respecting size limits

        Args:
            content (str): Content to chunk
            url (str): Original URL of the content
            page_title (str): Title of the page

        Returns:
            List[Dict[str, str]]: List of chunks with metadata
        """
        self.logger.info(f"Chunking content from {url} by headings")

        # Split content by headings (h1, h2, h3, etc.)
        heading_pattern = r'(\n\s*#+\s+.*?\n|\n\s*<h[1-6][^>]*>.*?</h[1-6]>\n|\n\s*<h[1-6][^>]*>.*?\n)'
        parts = re.split(heading_pattern, content, flags=re.IGNORECASE)

        chunks = []
        current_section = ""
        current_heading = page_title  # Start with page title as default heading

        i = 0
        while i < len(parts):
            part = parts[i].strip()

            # Check if this part is a heading
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

        self.logger.info(f"Created {len(chunks)} chunks from {url}")
        return chunks

    def _is_heading(self, text: str) -> bool:
        """
        Check if the text is a heading

        Args:
            text (str): Text to check

        Returns:
            bool: True if text is a heading
        """
        # Check for markdown headings
        if re.match(r'^\s*#+\s+', text.strip()):
            return True

        # Check for HTML headings
        if re.search(r'<h[1-6][^>]*>.*?</h[1-6]>', text, re.IGNORECASE):
            return True

        return False

    def _extract_heading_text(self, heading: str) -> str:
        """
        Extract the text content from a heading

        Args:
            heading (str): Raw heading text

        Returns:
            str: Clean heading text
        """
        # Remove markdown heading markers
        clean_heading = re.sub(r'^\s*#+\s*', '', heading.strip())

        # Remove HTML tags
        clean_heading = re.sub(r'<[^>]+>', '', clean_heading)

        return clean_heading.strip() or "Untitled Section"

    def _chunk_section(self, section: str, heading: str, url: str, page_title: str) -> List[Dict[str, str]]:
        """
        Chunk a single section of content, respecting size limits

        Args:
            section (str): Section content to chunk
            heading (str): Heading for this section
            url (str): Original URL
            page_title (str): Page title

        Returns:
            List[Dict[str, str]]: List of chunks from this section
        """
        chunks = []

        # If section is smaller than max size, use as single chunk
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
                # Check if adding this sentence would exceed the limit
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence

                if self._count_tokens(test_chunk) <= self.max_tokens:
                    current_chunk = test_chunk
                else:
                    # Save current chunk and start a new one
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

                    # Start new chunk with current sentence
                    current_chunk = sentence

            # Add the last chunk if it has content
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
        """
        Split text into sentences

        Args:
            text (str): Text to split

        Returns:
            List[str]: List of sentences
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _count_tokens(self, text: str) -> int:
        """
        Count approximate number of tokens in text (using word count as approximation)

        Args:
            text (str): Text to count tokens for

        Returns:
            int: Approximate token count
        """
        # Simple tokenization: split on whitespace and punctuation
        if not text:
            return 0
        # This is a simple approximation - in a real implementation you might use
        # a proper tokenizer like the one from the transformers library
        words = re.findall(r'\b\w+\b', text)
        return len(words)

    def chunk_content_list(self, content_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk a list of content items

        Args:
            content_list (List[Dict[str, str]]): List of content items with url, title, content

        Returns:
            List[Dict[str, str]]: List of chunks with metadata
        """
        all_chunks = []
        for item in content_list:
            url = item.get('url', '')
            title = item.get('title', '')
            content = item.get('content', '')

            if content:
                chunks = self.chunk_by_headings(content, url, title)
                all_chunks.extend(chunks)

        return all_chunks


def test_chunker():
    """
    Test function to verify the chunker works
    """
    chunker = ContentChunker(max_tokens=100)  # Small for testing

    # Test content with headings
    test_content = """
# Introduction
This is the introduction section with some content that we want to chunk.

## Background
Here is some background information that might be longer and need to be split into multiple chunks if it's too long.

# Main Content
This is the main content section with more information.

## Subsection 1
This is a subsection with more details.

## Subsection 2
This is another subsection with additional information.
"""

    chunks = chunker.chunk_by_headings(test_content, "https://example.com", "Test Page")

    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Heading: {chunk['section_heading']}")
        print(f"  Length: {len(chunk['content'])} characters")
        print(f"  Preview: {chunk['content'][:100]}...")


if __name__ == "__main__":
    test_chunker()