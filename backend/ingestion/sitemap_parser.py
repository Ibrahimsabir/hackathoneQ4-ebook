"""
Sitemap parser module for extracting URLs from the book's sitemap.xml
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Set
from logging_config import logger


class SitemapParser:
    """
    Class to parse sitemap.xml and extract all book page URLs
    """

    def __init__(self, sitemap_url: str):
        """
        Initialize the sitemap parser

        Args:
            sitemap_url (str): URL to the sitemap.xml file
        """
        self.sitemap_url = sitemap_url
        self.logger = logger

    def parse_sitemap(self) -> List[str]:
        """
        Parse the sitemap.xml and extract all URLs

        Returns:
            List[str]: List of URLs extracted from the sitemap
        """
        self.logger.info(f"Starting to parse sitemap: {self.sitemap_url}")

        try:
            # Fetch the sitemap
            response = requests.get(self.sitemap_url)
            response.raise_for_status()

            # Parse the XML content
            soup = BeautifulSoup(response.content, 'xml')

            # Find all <url> elements and extract <loc> values
            urls = []
            url_elements = soup.find_all('url')

            for url_element in url_elements:
                loc_element = url_element.find('loc')
                if loc_element:
                    url = loc_element.text.strip()
                    urls.append(url)

            self.logger.info(f"Successfully extracted {len(urls)} URLs from sitemap")
            return urls

        except requests.RequestException as e:
            self.logger.error(f"Error fetching sitemap: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error parsing sitemap: {e}")
            raise

    def get_unique_urls(self) -> List[str]:
        """
        Get unique URLs from the sitemap, removing duplicates

        Returns:
            List[str]: List of unique URLs
        """
        urls = self.parse_sitemap()
        unique_urls = list(set(urls))  # Remove duplicates
        self.logger.info(f"Found {len(unique_urls)} unique URLs after deduplication")
        return unique_urls


def main():
    """
    Main function to test the sitemap parser
    """
    from config import SITEMAP_URL

    parser = SitemapParser(SITEMAP_URL)
    urls = parser.get_unique_urls()

    print(f"Found {len(urls)} unique URLs in the sitemap:")
    for i, url in enumerate(urls[:10], 1):  # Print first 10 URLs
        print(f"{i}. {url}")

    if len(urls) > 10:
        print(f"... and {len(urls) - 10} more URLs")


if __name__ == "__main__":
    main()