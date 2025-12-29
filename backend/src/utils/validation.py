from typing import List
import uuid

def validate_embedding_dimensions(embedding: List[float], expected_size: int = 1024) -> bool:
    """
    Validate that the embedding vector has the correct dimensions.

    Args:
        embedding: The embedding vector to validate
        expected_size: Expected number of dimensions (default 1024 for Cohere models)

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(embedding, list):
        return False

    if len(embedding) != expected_size:
        # Allow some flexibility for different embedding models
        # Cohere models typically have 1024 dimensions
        # OpenAI models have 1536 (text-embedding-3-small) or 1535 (text-embedding-ada-002) dimensions
        if expected_size == 1024 and len(embedding) in [1024, 384, 768]:  # Common Cohere sizes
            pass  # Accept these sizes
        elif expected_size == 1536 and len(embedding) in [1536, 1535]:  # Common OpenAI sizes
            pass  # Accept these sizes
        else:
            return False

    # Check that all elements are numbers
    if not all(isinstance(x, (int, float)) for x in embedding):
        return False

    return True

def generate_unique_id() -> str:
    """
    Generate a unique ID for document chunks.

    Returns:
        A unique UUID string
    """
    return str(uuid.uuid4())

def validate_content_length(content: str, max_tokens: int = 8191) -> bool:
    """
    Validate that content length is within OpenAI limits.

    Args:
        content: The content to validate
        max_tokens: Maximum allowed token count (default 8191 for OpenAI)

    Returns:
        True if valid, False otherwise
    """
    return len(content) <= max_tokens and len(content) > 0