"""
Configuration settings for the RAG chatbot system.
"""
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # API keys
    OPENAI_API_KEY: str
    QDRANT_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Model settings
    model_name: str = "mistralai/devstral-2512:free"  # OpenRouter free model

    # Qdrant settings
    QDRANT_URL: str

    # Application settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    cors_origins: str = "*"  # Comma-separated list of origins

    # RAG settings
    max_tokens: int = 1000
    temperature: float = 0.3
    top_p: float = 0.9
    max_context_chunks: int = 5
    confidence_threshold: float = 0.6

    # Timeouts and limits
    retrieval_timeout: int = 30
    agent_timeout: int = 60
    max_query_length: int = 1000
    max_selected_text_length: int = 2000

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


# Create settings instance
settings = Settings()


def get_cors_origins() -> List[str]:
    """
    Parse CORS origins from settings.
    """
    origins_str = settings.cors_origins
    if origins_str == "*":
        return ["*"]
    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]