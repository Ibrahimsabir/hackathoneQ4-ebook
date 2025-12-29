# Troubleshooting Guide

This guide helps resolve common issues with the RAG Chatbot Content Ingestion Pipeline.

## Common Issues

### 1. OpenAI API Quota Exceeded (Error 429)

**Symptoms:**
- Error message: "You exceeded your current quota, please check your plan and billing details"
- Process stops during embedding generation

**Solutions:**
1. Check your OpenAI account balance and billing details
2. Use the enhanced ingestion pipeline which has better quota management:
   ```bash
   python cli_enhanced.py ingest
   ```
3. Use the resume feature to continue from where you left off:
   ```bash
   python cli_enhanced.py resume
   ```
4. Reduce the batch size:
   ```bash
   python cli_enhanced.py ingest --batch-size 10
   ```

### 2. Rate Limit Errors

**Symptoms:**
- Error message: "RateLimitError" or "Too many requests"
- Process slows down or stops temporarily

**Solutions:**
1. The system automatically handles rate limits with exponential backoff
2. Increase the rate limit delay in your `.env` file:
   ```env
   RATE_LIMIT_DELAY=5.0
   ```
3. Use smaller batch sizes to reduce API pressure

### 3. Qdrant Connection Issues

**Symptoms:**
- Error connecting to Qdrant Cloud
- "Connection refused" or timeout errors

**Solutions:**
1. Verify your QDRANT_URL and QDRANT_API_KEY in the `.env` file
2. Check that your Qdrant Cloud cluster is running and accessible
3. Ensure your IP address is not blocked by Qdrant security settings

### 4. Sitemap Fetching Issues

**Symptoms:**
- Error fetching sitemap
- "404 Not Found" or "Connection error"

**Solutions:**
1. Verify the BOOK_SITEMAP_URL in your `.env` file
2. Check that the sitemap URL is accessible in a web browser
3. Ensure the sitemap is properly formatted XML

### 5. Content Extraction Problems

**Symptoms:**
- Empty content extracted
- Navigation elements included in content

**Solutions:**
1. The crawler is designed to work with Docusaurus sites
2. Check that the site structure hasn't changed significantly
3. Verify the site is publicly accessible

## Performance Tips

### Managing API Costs
1. **Use Caching**: The system caches embeddings to avoid reprocessing
2. **Optimize Batch Sizes**: Smaller batches (20) reduce quota usage per request
3. **Resume Capability**: Use the `resume` command to continue interrupted processes
4. **Monitor Usage**: Check your OpenAI usage regularly

### Optimizing Processing Speed
1. **Adjust Chunk Size**: Smaller chunks may process faster but require more API calls
2. **Optimize Batch Size**: Balance between API limits and processing efficiency
3. **Parallel Processing**: The system handles multiple URLs concurrently

## Debugging Steps

### 1. Enable Detailed Logging
Add this to your `.env` file:
```env
LOG_LEVEL=DEBUG
```

### 2. Run Test Script
```bash
python test_ingestion.py
```

### 3. Check Configuration
Verify all required environment variables:
```bash
python -c "from src.utils.config import Config; print(Config.validate())"
```

### 4. Check Cache Status
The system maintains an `embedding_cache.json` file that stores processed embeddings to avoid reprocessing.

## Recovery Procedures

### If Process Interrupts
1. Use the resume command:
   ```bash
   python cli_enhanced.py resume
   ```
2. The system tracks progress in `ingestion_state.json`

### If Collection Gets Corrupted
1. Re-index the entire collection:
   ```bash
   python cli.py reindex
   ```

## API Limits and Quotas

### OpenAI Embedding Limits
- **text-embedding-3-small**: 8,191 tokens per input
- Rate limits vary by subscription tier
- Requests are cached to reduce repeated API calls

### Qdrant Limits
- Free tier has storage limitations
- Monitor collection size with the metrics command:
  ```bash
  python cli.py metrics
  ```

## When to Seek Help

Contact support if you encounter:
- Persistent connection issues after verifying credentials
- Consistent API errors after quota checks
- Unexpected system behavior not covered in this guide
- Performance issues that don't improve with configuration changes