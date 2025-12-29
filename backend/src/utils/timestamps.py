from datetime import datetime, timezone
from typing import Optional

def get_current_timestamp() -> datetime:
    """
    Get the current UTC timestamp.

    Returns:
        Current UTC datetime
    """
    return datetime.now(timezone.utc)

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format a timestamp as ISO string.

    Args:
        timestamp: Optional datetime to format (defaults to current time)

    Returns:
        ISO formatted timestamp string
    """
    if timestamp is None:
        timestamp = get_current_timestamp()
    return timestamp.isoformat()

def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse an ISO timestamp string to datetime object.

    Args:
        timestamp_str: ISO formatted timestamp string

    Returns:
        Parsed datetime object
    """
    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))