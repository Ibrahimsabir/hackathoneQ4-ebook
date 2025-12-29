# RAG Chatbot Content Ingestion Pipeline

This system ingests content from the deployed book, generates embeddings using Cohere, and stores them in Qdrant for semantic search.

## Prerequisites

- Python 3.8+
- Cohere API key
- Qdrant Cloud account and API key

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your credentials:
```env
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=book_content_chunks
COHERE_EMBEDDING_MODEL=embed-english-v3.0
BOOK_SITEMAP_URL=https://hackathone-q4-ebook.vercel.app/sitemap.xml
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BATCH_SIZE=100
RATE_LIMIT_DELAY=1.0
```

## Usage

### Command Line Interface

Run the ingestion pipeline with Cohere:
```bash
python cli_cohere.py ingest
```

Validate the storage:
```bash
python cli_cohere.py validate
```

Test search functionality:
```bash
python cli_cohere.py search --query "What is Physical AI?"
```

View metrics:
```bash
python cli_cohere.py metrics
```

Re-index all content:
```bash
python cli_cohere.py reindex
```

### Direct Usage

Run the Cohere-based ingestion pipeline directly:
```bash
python -m src.ingestion_service_cohere
```

## Components

### Crawler
- Fetches URLs from the book's sitemap
- Extracts clean text content from each page
- Handles navigation elements and boilerplate removal

### Embedding Generator
- Uses Cohere's embed-english-v3.0 model
- Generates 1024-dimensional embeddings
- Handles batching and rate limiting
- Enhanced error handling for quota management

### Storage
- Stores embeddings in Qdrant Cloud
- Maintains metadata for retrieval
- Supports similarity search

### Validation
- Verifies embedding dimensions
- Checks content quality
- Validates storage integrity

## Configuration

The system can be configured via environment variables:

- `COHERE_API_KEY`: Your Cohere API key
- `COHERE_EMBEDDING_MODEL`: Cohere embedding model to use (default: embed-english-v3.0)
- `QDRANT_URL`: Qdrant Cloud cluster URL
- `QDRANT_API_KEY`: Qdrant API key
- `QDRANT_COLLECTION_NAME`: Name of the collection (default: book_content_chunks)
- `BOOK_SITEMAP_URL`: URL to the book's sitemap (default: https://hackathone-q4-ebook.vercel.app/sitemap.xml)
- `CHUNK_SIZE`: Maximum size of content chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `BATCH_SIZE`: Number of items to process in each batch (default: 100)
- `RATE_LIMIT_DELAY`: Delay between batches in seconds (default: 1.0)

## Testing

Run the basic functionality test:
```bash
python test_ingestion.py
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sitemap       │───▶│   Crawler &      │───▶│  Content        │
│   Parser        │    │   Content        │    │  Preprocessor   │
└─────────────────┘    │   Extractor      │    └─────────────────┘
                       └──────────────────┘            │
                              │                       │
                              ▼                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Chunking       │───▶│  Embedding      │
                       │   Engine         │    │  Generator      │
                       └──────────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Validation     │───▶│  Vector Store   │
                       │   & Storage      │    │  (Qdrant)       │
                       └──────────────────┘    └─────────────────┘
```

## Error Handling

The system includes comprehensive error handling:
- Retry mechanisms for API calls
- Rate limiting with exponential backoff
- Graceful degradation when services are unavailable
- Detailed logging for debugging