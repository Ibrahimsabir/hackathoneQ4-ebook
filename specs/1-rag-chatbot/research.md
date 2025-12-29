# Research Document: Content Ingestion, Embedding Generation, and Vector Storage

**Feature**: RAG Chatbot for Docusaurus Book Integration
**Component**: Spec 1 - Content Ingestion, Embedding Generation, and Vector Storage
**Created**: 2025-12-29

## 0.1 OpenAI Embedding Model Research

### Decision: OpenAI Embedding Model Selection
- **Chosen Model**: OpenAI text-embedding-ada-002
- **Rationale**: OpenAI's text-embedding-ada-002 provides high-quality embeddings optimized for semantic search and retrieval tasks. The model is well-documented and widely used in RAG applications.

### Key Specifications
- **Model**: text-embedding-ada-002
- **Dimensions**: 1536-dimensional vectors
- **Input Length**: Up to 8191 tokens per request
- **Batch Size**: Up to 2048 texts per request
- **Rate Limits**: Based on your OpenAI account limits and usage tier

### API Usage Patterns
- **Endpoint**: `https://api.openai.com/v1/embeddings`
- **Authentication**: Bearer token via Authorization header
- **Request Format**: JSON with "model" and "input" fields
- **Response Format**: JSON with embedding vectors

### Rate Limit Handling Strategy
- **Exponential Backoff**: Implement with base delay of 1 second, max 60 seconds
- **Batch Processing**: Use optimal batch size (typically 100-200) to balance efficiency and rate limits
- **Retry Logic**: Retry failed requests up to 3 times before skipping

## 0.2 Qdrant Cloud Integration Research

### Decision: Qdrant Cloud Collection Schema
- **Chosen Schema**: Collection with 1536-dimensional vectors (matching OpenAI text-embedding-ada-002 output)
- **Rationale**: Qdrant Cloud provides efficient vector storage with semantic search capabilities, and the free tier supports the required functionality.

### Key Specifications
- **Vector Dimensions**: 1536 (to match OpenAI text-embedding-ada-002 embeddings)
- **Distance Metric**: Cosine similarity (optimal for semantic search)
- **Storage Limits**: Free tier supports up to 1GB storage
- **API Access**: REST and gRPC APIs available

### Collection Design
- **Collection Name**: `book_content_chunks`
- **Vector Field**: `content_vector` (768 dimensions)
- **Payload Fields**:
  - `content`: Original text content
  - `source_url`: URL of the source page
  - `section_title`: Title of the section
  - `chapter`: Chapter identifier
  - `position`: Position within document
  - `created_at`: Timestamp

### Metadata Storage Strategy
- Store document metadata in Qdrant payload for efficient filtering
- Use structured metadata for semantic search enhancement
- Include source information for citation purposes

## 0.3 Docusaurus Content Structure Analysis

### Decision: Content Extraction Strategy
- **Chosen Approach**: CSS selectors to extract main content while excluding navigation
- **Rationale**: Docusaurus follows predictable HTML structure with consistent class names for main content areas.

### HTML Structure Understanding
- **Main Content**: Located within `<main>` tag or `.main-wrapper` class
- **Navigation Elements**: Typically in `.navbar`, `.sidebar`, `.pagination-nav`
- **Content Sections**: Organized in `.markdown` or `.theme-doc-markdown` classes
- **Headings**: Use standard HTML heading tags (h1-h6) for structure

### Content Extraction Selectors
- **Primary Content**: `main article` or `.main-wrapper .container`
- **Exclude**: `.navbar`, `.sidebar`, `.pagination-nav`, `.theme-edit-this-page`
- **Headings**: `h1, h2, h3, h4, h5, h6` for structural context
- **Paragraphs**: `p, li, blockquote` for content extraction

### Navigation and Boilerplate Avoidance
- Use CSS selectors to exclude common Docusaurus navigation elements
- Focus on content within article or markdown containers
- Preserve heading hierarchy for chunking context