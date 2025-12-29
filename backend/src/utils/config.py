import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage API keys and service endpoints."""

    # Cohere Configuration
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    COHERE_EMBEDDING_MODEL: str = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")

    # Qdrant Configuration
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "book_content_chunks")

    # Book Configuration
    BOOK_SITEMAP_URL: str = os.getenv("BOOK_SITEMAP_URL", "https://hackathone-q4-ebook.vercel.app/sitemap.xml")

    # Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    RATE_LIMIT_DELAY: float = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))

    # Validation
    @classmethod
    def validate(cls) -> list[str]:
        """Validate that all required configuration values are present."""
        errors = []

        if not cls.COHERE_API_KEY:
            errors.append("COHERE_API_KEY is required")

        if not cls.QDRANT_URL:
            errors.append("QDRANT_URL is required")

        if not cls.QDRANT_API_KEY:
            errors.append("QDRANT_API_KEY is required")

        return errors

# Validate configuration on import
config_errors = Config.validate()
if config_errors:
    raise ValueError(f"Configuration errors: {', '.join(config_errors)}")