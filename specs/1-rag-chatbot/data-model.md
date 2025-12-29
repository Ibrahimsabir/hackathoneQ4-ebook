# Data Model: Content Ingestion, Embedding Generation, and Vector Storage

**Feature**: RAG Chatbot for Docusaurus Book Integration
**Component**: Spec 1 - Content Ingestion, Embedding Generation, and Vector Storage
**Created**: 2025-12-29

## Core Entities

### DocumentChunk
**Description**: A semantically coherent segment of book content with its associated embedding and metadata.

**Fields**:
- `id` (string): Unique identifier for the chunk (UUID format)
- `content` (string): The text content of the chunk
- `embedding` (array of numbers): 1536-dimensional vector representation of the content
- `source_url` (string): URL where the content originated
- `section_title` (string): Title of the section this chunk belongs to
- `chapter` (string): Chapter or document identifier
- `position` (integer): Position of the chunk within the document
- `metadata` (object): Additional metadata including:
  - `heading_hierarchy`: Array of headings leading to this chunk
  - `content_type`: Type of content (text, code, etc.)
  - `language`: Language of the content (for multilingual support)
- `created_at` (timestamp): Timestamp of ingestion
- `updated_at` (timestamp): Timestamp of last update

**Constraints**:
- `id` must be unique across all chunks
- `embedding` must be exactly 1536 elements (for OpenAI text-embedding-ada-002 compatibility)
- `content` must not exceed 8191 tokens (OpenAI limit)
- `source_url` must be a valid URL

**Indexes**:
- Primary: `id`
- Secondary: `source_url`, `section_title`, `chapter`

### CrawlSession
**Description**: Tracks a single execution of the crawling process.

**Fields**:
- `id` (string): Unique identifier for the crawl session
- `start_time` (timestamp): When the crawl started
- `end_time` (timestamp): When the crawl ended
- `status` (string): Current status (pending, running, completed, failed)
- `total_urls` (integer): Total number of URLs to crawl
- `processed_urls` (integer): Number of URLs successfully processed
- `failed_urls` (integer): Number of URLs that failed to process
- `error_details` (array of objects): Details about any errors encountered

**Constraints**:
- `id` must be unique
- `status` must be one of: pending, running, completed, failed
- `end_time` must be >= `start_time` if session is completed

### ContentExtraction
**Description**: Represents the extracted content from a single web page before chunking.

**Fields**:
- `id` (string): Unique identifier for the extraction
- `source_url` (string): URL of the source page
- `raw_content` (string): Raw HTML content from the page
- `clean_content` (string): Clean, extracted text content
- `title` (string): Title of the page
- `headings` (array of objects): Headings found in the page with their hierarchy
- `metadata` (object): Additional page metadata
- `extracted_at` (timestamp): When the content was extracted

**Constraints**:
- `source_url` must be unique per crawl session
- `clean_content` must not be empty

## Relationships

```
CrawlSession (1) → (0..n) ContentExtraction
ContentExtraction (1) → (1..n) DocumentChunk
```

### Validation Rules

#### DocumentChunk Validation
- Content must be between 50 and 2000 tokens (or within OpenAI limits, max 8191)
- Embedding vector must have exactly 1536 dimensions
- Source URL must be valid and accessible
- All required fields must be present

#### ContentExtraction Validation
- Raw content must be valid HTML
- Clean content must contain actual text (not just markup)
- Headings must maintain proper hierarchy

#### CrawlSession Validation
- Status transitions must follow: pending → running → (completed | failed)
- Processed and failed counts must not exceed total URLs
- End time must be set when status is completed or failed

## State Transitions

### CrawlSession States
```
PENDING → RUNNING → COMPLETED
              ↓
            FAILED
```

### DocumentChunk States
Document chunks are created as part of the ingestion process and are immutable once stored in the vector database.

## Storage Schema for Qdrant

### Collection: book_content_chunks
- **Vector Configuration**:
  - Name: `content_vector`
  - Size: 1536
  - Distance: Cosine

- **Payload Schema**:
```json
{
  "content": "string",
  "source_url": "string",
  "section_title": "string",
  "chapter": "string",
  "position": "integer",
  "heading_hierarchy": "keyword[]",
  "content_type": "keyword",
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

### Indexes in Qdrant
- Point ID: DocumentChunk.id
- Payload indexes: source_url, section_title, chapter for efficient filtering