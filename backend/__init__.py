"""
Backend root package initialization.
Load environment variables at the earliest point.
"""
from dotenv import load_dotenv
import os
from pathlib import Path

# Find the .env file in the project root (parent of backend directory)
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
env_file = project_root / ".env"

# Load environment variables before any other imports
load_dotenv(dotenv_path=env_file)

# Debug: Print to verify loading
print(f"DEBUG: Loading .env from: {env_file}")
print(f"DEBUG: QDRANT_URL present: {bool(os.getenv('QDRANT_URL'))}")
