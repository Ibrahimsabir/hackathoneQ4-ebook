"""
Backend package initialization.
Load environment variables before anything else.
"""
from dotenv import load_dotenv
import os

# Load environment variables as the first thing
load_dotenv()

# Ensure critical environment variables are set
_required_vars = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
_missing_vars = [var for var in _required_vars if not os.getenv(var)]

if _missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(_missing_vars)}. "
        "Please ensure your .env file is properly configured."
    )
