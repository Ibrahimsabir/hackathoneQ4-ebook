from typing import List, TypeVar, Callable, Awaitable, Any
import asyncio
from src.utils.logging_config import logger

T = TypeVar('T')
R = TypeVar('R')

async def process_in_batches(
    items: List[T],
    processor: Callable[[List[T]], Awaitable[List[R]]],
    batch_size: int = 100,
    delay: float = 0.1
) -> List[R]:
    """
    Process a list of items in batches asynchronously.

    Args:
        items: List of items to process
        processor: Async function that processes a batch of items
        batch_size: Number of items to process in each batch
        delay: Delay between batches (in seconds)

    Returns:
        List of processed results
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
            batch_results = await processor(batch)
            results.extend(batch_results)

            # Add delay between batches to respect rate limits
            if delay > 0 and i + batch_size < len(items):
                await asyncio.sleep(delay)

        except Exception as e:
            logger.error(f"Failed to process batch {i//batch_size + 1}: {str(e)}")
            raise

    return results


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    return chunks


async def process_with_semaphore(
    items: List[T],
    processor: Callable[[T], Awaitable[R]],
    max_concurrent: int = 5
) -> List[R]:
    """
    Process items with a concurrency limit using semaphore.

    Args:
        items: List of items to process
        processor: Async function that processes a single item
        max_concurrent: Maximum number of concurrent operations

    Returns:
        List of processed results
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(item):
        async with semaphore:
            return await processor(item)

    tasks = [process_with_limit(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions and raise them
    for result in results:
        if isinstance(result, Exception):
            raise result

    return results