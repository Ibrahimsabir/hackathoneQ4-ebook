"""
Content extractor module for extracting clean text from Docusaurus pages
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
import re
from urllib.parse import urljoin
from logging_config import logger
from .http_client import http_client


class ContentExtractor:
    """
    Class to extract clean text content from Docusaurus pages
    Removes navigation, footer, and boilerplate text
    """

    def __init__(self):
        self.logger = logger

    def extract_content(self, url: str) -> Dict[str, str]:
        """
        Extract clean content from a single Docusaurus page

        Args:
            url (str): URL of the page to extract content from

        Returns:
            Dict[str, str]: Dictionary with 'title', 'content', and 'section_heading' keys
        """
        self.logger.info(f"Extracting content from: {url}")

        # Fetch the page
        response = http_client.get(url)
        if not response:
            self.logger.error(f"Failed to fetch content from {url}")
            return {"title": "", "content": "", "section_heading": ""}

        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = self._extract_title(soup)

        # Extract section heading (main heading)
        section_heading = self._extract_main_heading(soup)

        # Extract clean content
        content = self._extract_clean_content(soup)

        self.logger.info(f"Successfully extracted content from {url}")

        return {
            "title": title,
            "content": content,
            "section_heading": section_heading
        }

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract the page title from the soup object

        Args:
            soup (BeautifulSoup): Parsed HTML soup

        Returns:
            str: Page title
        """
        # Try to find the title in various common locations
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()

        # Look for h1 tags which might contain the title
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()

        # Look for meta tags with title
        meta_title = soup.find('meta', attrs={'property': 'og:title'})
        if meta_title:
            return meta_title.get('content', '').strip()

        return "No Title Found"

    def _extract_main_heading(self, soup: BeautifulSoup) -> str:
        """
        Extract the main heading from the page

        Args:
            soup (BeautifulSoup): Parsed HTML soup

        Returns:
            str: Main heading
        """
        # Look for the first h1 tag as the main heading
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()

        # If no h1, look for h2
        h2_tag = soup.find('h2')
        if h2_tag:
            return h2_tag.get_text().strip()

        return "No Heading Found"

    def _extract_clean_content(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text content from the soup object, removing navigation and boilerplate

        Args:
            soup (BeautifulSoup): Parsed HTML soup

        Returns:
            str: Clean text content
        """
        # Remove navigation elements
        for nav in soup.find_all(['nav', 'header', 'footer', 'aside']):
            nav.decompose()

        # Remove common Docusaurus navigation and layout elements
        for element in soup.find_all(['div', 'section', 'article'], class_=re.compile(
                r'(navbar|nav|sidebar|toc|footer|header|menu|navigation|skipToContent|theme-edit-this-page|theme-last-updated|theme-draft|theme-back-to-top-button|theme-admonition|theme-pagination|theme-previous-next|theme-doc-footer|theme-doc-sidebar|theme-doc-breadcrumbs|theme-search|theme-search-page)', re.IGNORECASE)):
            element.decompose()

        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        # Get text and clean it up
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_content_batch(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Extract content from multiple URLs

        Args:
            urls (List[str]): List of URLs to extract content from

        Returns:
            List[Dict[str, str]]: List of dictionaries with extracted content
        """
        results = []
        for i, url in enumerate(urls, 1):
            self.logger.info(f"Processing {i}/{len(urls)}: {url}")
            content_data = self.extract_content(url)
            if content_data['content']:  # Only add if content was successfully extracted
                results.append({
                    **content_data,
                    'url': url
                })
            else:
                self.logger.warning(f"Failed to extract content from {url}")

        return results


def test_extractor():
    """
    Test function to verify the content extractor works
    """
    extractor = ContentExtractor()

    # Test with a sample URL (this would need to be a real Docusaurus page)
    test_url = "https://hackathone-q4-ebook.vercel.app/"
    content_data = extractor.extract_content(test_url)

    print(f"Title: {content_data['title']}")
    print(f"Section Heading: {content_data['section_heading']}")
    print(f"Content length: {len(content_data['content'])} characters")
    print(f"Content preview: {content_data['content'][:200]}...")


if __name__ == "__main__":
    test_extractor()