# Quickstart Guide: Content Ingestion, Embedding Generation, and Vector Storage

**Feature**: RAG Chatbot for Docusaurus Book Integration
**Component**: Spec 1 - Content Ingestion, Embedding Generation, and Vector Storage
**Created**: 2025-12-29

## Overview

This guide provides instructions for setting up and running the content ingestion pipeline that will crawl your Docusaurus book, generate embeddings using Jina AI, and store them in Qdrant Cloud.

## Prerequisites

### 1. Environment Setup
- Python 3.8+ installed
- pip package manager
- Git for version control

### 2. External Services
- **Jina AI Account**: Sign up at [jina.ai](https://jina.ai) for an API key
- **Qdrant Cloud Account**: Sign up at [qdrant.tech](https://qdrant.tech) for a free cluster
- **Book Access**: Ensure the sitemap is accessible at https://hackathone-q4-ebook.vercel.app/sitemap.xml

## Configuration

### 1. Environment Variables
Create a `.env` file with the following variables:

```bash
# Jina AI Configuration
JINA_API_KEY=your_jina_api_key_here

# Qdrant Cloud Configuration
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

# Book Configuration
BOOK_SITEMAP_URL=https://hackathone-q4-ebook.vercel.app/sitemap.xml

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BATCH_SIZE=32
RATE_LIMIT_DELAY=1.0
```

### 2. Required Python Packages
The ingestion pipeline requires the following Python packages:

```bash
requests>=2.28.0
beautifulsoup4>=4.11.0
jina>=0.3.0
qdrant-client>=1.3.0
python-dotenv>=0.19.0
tqdm>=4.64.0
lxml>=4.9.0
```

## Setup Instructions

### 1. Clone and Navigate to Project
```bash
git clone <your-repo-url>
cd <project-directory>
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Or install required packages directly:
pip install requests beautifulsoup4 jina qdrant-client python-dotenv tqdm lxml
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your actual API keys and configuration
```

## Running the Ingestion Pipeline

### 1. Basic Ingestion
Run the complete ingestion pipeline with default settings:

```bash
python -m scripts.ingestion_pipeline --sitemap-url https://hackathone-q4-ebook.vercel.app/sitemap.xml
```

### 2. Advanced Options
```bash
# Specify custom chunk size and overlap
python -m scripts.ingestion_pipeline --chunk-size 800 --chunk-overlap 100

# Run only the crawling phase
python -m scripts.ingestion_pipeline --phase crawl

# Run only the embedding phase
python -m scripts.ingestion_pipeline --phase embed

# Run only the storage phase
python -m scripts.ingestion_pipeline --phase store
```

### 3. Monitoring Progress
The ingestion pipeline provides real-time progress updates:

```
🚀 Starting ingestion pipeline...
📚 Crawling 150 URLs from sitemap...
✅ Crawled 50/150 URLs (33%)
✅ Crawled 100/150 URLs (67%)
✅ Crawling completed: 150/150 URLs processed
✂️  Processing content into chunks...
✅ Chunking completed: 1,245 chunks created
🧮 Generating embeddings...
✅ Embedded 500/1245 chunks (40%)
✅ Embedded 1000/1245 chunks (80%)
✅ Embedding completed: 1,245/1,245 chunks processed
💾 Storing embeddings in Qdrant...
✅ Stored 500/1245 embeddings (40%)
✅ Stored 1000/1245 embeddings (80%)
✅ Storage completed: 1,245/1,245 embeddings stored
🎉 Ingestion pipeline completed successfully!
```

## Verification

### 1. Check Qdrant Collection
Verify that embeddings were stored correctly:

```python
from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Check collection size
collection_info = client.get_collection("book_content_chunks")
print(f"Total chunks in collection: {collection_info.points_count}")
```

### 2. Test Retrieval
Perform a test semantic search to verify the embeddings work correctly:

```bash
python -m scripts.test_retrieval --query "your test query here"
```

## Troubleshooting

### Common Issues

#### API Rate Limits
- **Issue**: Jina AI API rate limit exceeded
- **Solution**: Increase `RATE_LIMIT_DELAY` in your configuration or reduce `BATCH_SIZE`

#### Connection Issues
- **Issue**: Cannot connect to Qdrant Cloud
- **Solution**: Verify your QDRANT_URL and QDRANT_API_KEY are correct

#### Content Extraction Problems
- **Issue**: Empty content extracted from pages
- **Solution**: Check that the sitemap URLs are accessible and the page structure hasn't changed

### Logging
Enable detailed logging by setting the environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## Next Steps

Once the ingestion pipeline completes successfully:

1. **Validate Quality**: Run the validation script to ensure retrieval quality
2. **Test Integration**: Connect the retrieval service to your chatbot frontend
3. **Monitor Performance**: Track ingestion metrics and optimize as needed