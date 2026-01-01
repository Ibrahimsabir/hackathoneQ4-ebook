"""
Content validator for validating external content before processing
"""
import re
import requests
from typing import Dict, Any, List, Optional
import logging
from logging_config import logger
from urllib.parse import urlparse
from bs4 import BeautifulSoup


class ContentValidator:
    """
    Class to validate external content before processing
    """

    def __init__(self):
        """
        Initialize the content validator
        """
        self.logger = logger
        self.content_size_limit = 10 * 1024 * 1024  # 10MB
        self.trusted_domains = set()  # Will be populated based on sitemap domain
        self.blocked_patterns = [
            r'javascript:',  # JavaScript URLs
            r'vbscript:',   # VBScript URLs
            r'data:',       # Data URLs
            r'file:',       # File URLs
        ]
        self.suspicious_patterns = [
            r'<script',     # Script tags
            r'javascript:', # JavaScript
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # eval function
            r'exec\s*\(',  # exec function
        ]

    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate a URL before processing

        Args:
            url (str): URL to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "normalized_url": url
        }

        try:
            # Check if URL is valid
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Invalid URL format")
                return validation_result

            # Normalize URL
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                normalized_url += f"?{parsed.query}"
            validation_result["normalized_url"] = normalized_url

            # Check for blocked patterns
            for pattern in self.blocked_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"URL contains blocked pattern: {pattern}")
                    return validation_result

            # Add domain to trusted domains if it's not already there
            if not self.trusted_domains:
                self.trusted_domains.add(parsed.netloc)

            # Check if domain is trusted (for this implementation, we trust the same domain as the sitemap)
            if parsed.netloc not in self.trusted_domains:
                validation_result["warnings"].append(f"URL domain '{parsed.netloc}' not in trusted domains")

        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Error validating URL: {str(e)}")

        return validation_result

    def validate_content(self, content: str, content_type: str = "text/html") -> Dict[str, Any]:
        """
        Validate content before processing

        Args:
            content (str): Content to validate
            content_type (str): Type of content (e.g., text/html, application/json)

        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "content_type": content_type,
            "size": len(content)
        }

        # Check content size
        if len(content) > self.content_size_limit:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Content size {len(content)} exceeds limit {self.content_size_limit}")
            return validation_result

        # Check for suspicious patterns based on content type
        if "html" in content_type.lower():
            validation_result.update(self._validate_html_content(content))
        elif "json" in content_type.lower():
            validation_result.update(self._validate_json_content(content))
        else:
            validation_result.update(self._validate_generic_content(content))

        return validation_result

    def _validate_html_content(self, content: str) -> Dict[str, Any]:
        """
        Validate HTML content specifically

        Args:
            content (str): HTML content to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        result = {
            "errors": [],
            "warnings": [],
            "has_suspicious_elements": False,
            "suspicious_elements": []
        }

        try:
            # Parse HTML to look for suspicious elements
            soup = BeautifulSoup(content, 'html.parser')

            # Look for suspicious patterns
            for pattern in self.suspicious_patterns:
                matches = soup.find_all(string=re.compile(pattern, re.IGNORECASE))
                if matches:
                    result["warnings"].extend([f"Suspicious pattern found: {pattern}" for _ in matches])
                    result["has_suspicious_elements"] = True
                    result["suspicious_elements"].append(pattern)

            # Look for script tags
            script_tags = soup.find_all('script')
            if script_tags:
                result["warnings"].append(f"Found {len(script_tags)} script tags")
                result["has_suspicious_elements"] = True
                result["suspicious_elements"].append("script_tags")

            # Look for iframe tags
            iframe_tags = soup.find_all('iframe')
            if iframe_tags:
                result["warnings"].append(f"Found {len(iframe_tags)} iframe tags")
                result["has_suspicious_elements"] = True
                result["suspicious_elements"].append("iframe_tags")

        except Exception as e:
            result["errors"].append(f"Error parsing HTML content: {str(e)}")

        return result

    def _validate_json_content(self, content: str) -> Dict[str, Any]:
        """
        Validate JSON content specifically

        Args:
            content (str): JSON content to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        result = {
            "errors": [],
            "warnings": []
        }

        try:
            import json
            parsed_json = json.loads(content)

            # Check for potential security issues in JSON
            self._check_json_for_suspicious_content(parsed_json, result)
        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Error validating JSON: {str(e)}")

        return result

    def _check_json_for_suspicious_content(self, data: Any, result: Dict[str, Any], path: str = ""):
        """
        Recursively check JSON data for suspicious content

        Args:
            data (Any): JSON data to check
            result (Dict[str, Any]): Result dictionary to update
            path (str): Current path in the JSON structure
        """
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(key, str) and any(suspicious in key.lower() for suspicious in ['script', 'exec', 'eval']):
                    result["warnings"].append(f"Suspicious key found at {current_path}: {key}")
                self._check_json_for_suspicious_content(value, result, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                self._check_json_for_suspicious_content(item, result, current_path)
        elif isinstance(data, str):
            for pattern in self.suspicious_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    result["warnings"].append(f"Suspicious pattern '{pattern}' found in string at {path}")

    def _validate_generic_content(self, content: str) -> Dict[str, Any]:
        """
        Validate generic content

        Args:
            content (str): Content to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        result = {
            "errors": [],
            "warnings": []
        }

        # Check for suspicious patterns in generic content
        for pattern in self.suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                result["warnings"].append(f"Suspicious pattern found: {pattern}")

        return result

    def validate_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Validate an HTTP response before processing

        Args:
            response (requests.Response): HTTP response to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type', ''),
            "content_length": int(response.headers.get('content-length', 0))
        }

        # Check status code
        if response.status_code != 200:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"HTTP status code {response.status_code} is not 200")
            return validation_result

        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(ct in content_type for ct in ['text/html', 'application/json', 'text/plain']):
            validation_result["warnings"].append(f"Unexpected content type: {content_type}")

        # Check content length
        content_length = int(response.headers.get('content-length', 0))
        if content_length > self.content_size_limit:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Content length {content_length} exceeds limit {self.content_size_limit}")

        # Validate content
        try:
            content = response.text
            content_validation = self.validate_content(content, content_type)
            validation_result.update({
                "content_validation": content_validation,
                "is_valid": validation_result["is_valid"] and content_validation["is_valid"]
            })
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Error reading response content: {str(e)}")

        return validation_result

    def set_trusted_domains(self, domains: List[str]):
        """
        Set trusted domains for URL validation

        Args:
            domains (List[str]): List of trusted domains
        """
        self.trusted_domains.update(domains)
        self.logger.info(f"Set trusted domains: {domains}")


def test_content_validator():
    """
    Test function to verify the content validator works
    """
    validator = ContentValidator()

    # Test URL validation
    print("Testing URL validation:")
    test_urls = [
        "https://example.com/page",
        "javascript:alert('test')",
        "http://trusted-domain.com/page"
    ]

    for url in test_urls:
        result = validator.validate_url(url)
        print(f"URL: {url}")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Errors: {result['errors']}")
        print(f"  Warnings: {result['warnings']}")
        print()

    # Test content validation
    print("Testing content validation:")
    test_contents = [
        "<html><body><p>Normal content</p></body></html>",
        "<html><body><script>alert('test');</script></body></html>",
        '{"normal": "json"}',
        '{"suspicious_eval": "eval(this)"}'
    ]

    for i, content in enumerate(test_contents):
        content_type = "text/html" if i < 2 else "application/json"
        result = validator.validate_content(content, content_type)
        print(f"Content {i+1}: {content[:30]}...")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Errors: {result['errors']}")
        print(f"  Warnings: {result['warnings']}")
        print()

    # Test setting trusted domains
    validator.set_trusted_domains(["example.com", "trusted-domain.com"])
    print("Trusted domains set")


if __name__ == "__main__":
    test_content_validator()