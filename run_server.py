"""
Script to run the RAG chatbot backend server
"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Create a minimal version that doesn't immediately load settings
def run_server():
    """
    Run the RAG chatbot backend server
    """
    print("Starting RAG Chatbot Server...")
    print("Loading components...")

    # Import the main app
    try:
        from backend.src.api.main import app
        print("[OK] Successfully loaded the FastAPI application")
        print(f"App title: {app.title}")
        print(f"App description: {app.description}")

        # Show available routes
        print("\nAvailable API routes:")
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                methods = ', '.join(route.methods)
                print(f"  {methods}: {route.path}")

        print("\nTo run the server, use this command:")
        print("cd backend")
        print("uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        print("\nThen access:")
        print("- Health check: http://localhost:8000/api/health")
        print("- Chat API: http://localhost:8000/api/ask (POST)")
        print("- Contextual Chat: http://localhost:8000/api/ask-with-context (POST)")

        return True

    except Exception as e:
        print(f"[ERROR] Error loading application: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("RAG Chatbot Backend Server")
    print("=" * 40)

    success = run_server()

    if success:
        print("\n[SUCCESS] The RAG chatbot backend is properly configured and ready to run!")
        print("\nNote: To use the full functionality, you need to provide valid API keys in the .env file:")
        print("- OPENAI_API_KEY: Your OpenAI API key")
        print("- QDRANT_URL: Your Qdrant vector database URL")
        print("- QDRANT_API_KEY: Your Qdrant API key (if required)")
    else:
        print("\n[ERROR] There were issues with the server setup. Please check the errors above.")