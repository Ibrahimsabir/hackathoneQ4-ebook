"""
Main FastAPI application for the RAG chatbot system.
"""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from project root
# backend/src/api/main.py -> go up 3 levels to project root
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / ".env"
load_dotenv(dotenv_path=env_file)

# Debug output
print(f"DEBUG [main.py]: Loading .env from: {env_file}")
print(f"DEBUG [main.py]: .env exists: {env_file.exists()}")
print(f"DEBUG [main.py]: QDRANT_URL = {os.getenv('QDRANT_URL', 'NOT SET')[:80] if os.getenv('QDRANT_URL') else 'NOT SET'}")

# Import routers
from .routers import chat, health

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for the RAG chatbot system integrated with the Docusaurus book",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(health.router, prefix="/api", tags=["health"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )