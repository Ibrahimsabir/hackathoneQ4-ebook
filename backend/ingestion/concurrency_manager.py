"""
Concurrency manager for handling concurrent indexing operations
"""
import os
import time
import threading
from typing import Dict, Optional
import logging
from logging_config import logger
import json
from datetime import datetime, timedelta


class ConcurrencyManager:
    """
    Class to manage concurrency and prevent conflicts during indexing
    """

    def __init__(self, lock_dir: str = "./locks"):
        """
        Initialize the concurrency manager

        Args:
            lock_dir (str): Directory to store lock files
        """
        self.lock_dir = lock_dir
        self._lock = threading.Lock()  # For thread-safe operations on lock tracking
        self.active_locks: Dict[str, Dict] = {}  # Track active locks in memory
        self.logger = logger

        # Create lock directory if it doesn't exist
        os.makedirs(lock_dir, exist_ok=True)

    def acquire_lock(self, resource_id: str, timeout: int = 300, owner: str = "default") -> bool:
        """
        Acquire a lock for a resource

        Args:
            resource_id (str): ID of the resource to lock
            timeout (int): Lock timeout in seconds
            owner (str): Owner identifier for the lock

        Returns:
            bool: True if lock was acquired, False otherwise
        """
        with self._lock:
            lock_key = self._get_lock_key(resource_id)

            # Check if there's an existing lock file
            lock_file = os.path.join(self.lock_dir, f"{lock_key}.lock")

            if os.path.exists(lock_file):
                # Read the existing lock file
                try:
                    with open(lock_file, 'r') as f:
                        lock_data = json.load(f)

                    # Check if the lock has expired
                    lock_time = datetime.fromisoformat(lock_data['timestamp'])
                    if datetime.now() - lock_time < timedelta(seconds=timeout):
                        # Lock is still valid
                        self.logger.debug(f"Resource {resource_id} is locked by {lock_data['owner']}")
                        return False
                    else:
                        # Lock has expired, remove it
                        self.logger.info(f"Removing expired lock for {resource_id}")
                        os.remove(lock_file)
                except Exception as e:
                    self.logger.warning(f"Error reading lock file: {e}")
                    # If there's an error reading the file, try to remove it
                    try:
                        os.remove(lock_file)
                    except:
                        pass

            # Create new lock file
            lock_data = {
                "resource_id": resource_id,
                "owner": owner,
                "timestamp": datetime.now().isoformat(),
                "timeout": timeout
            }

            try:
                with open(lock_file, 'w') as f:
                    json.dump(lock_data, f)

                # Track in memory
                self.active_locks[lock_key] = lock_data
                self.logger.debug(f"Acquired lock for resource {resource_id}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to acquire lock for {resource_id}: {e}")
                return False

    def release_lock(self, resource_id: str) -> bool:
        """
        Release a lock for a resource

        Args:
            resource_id (str): ID of the resource to unlock

        Returns:
            bool: True if lock was released, False otherwise
        """
        with self._lock:
            lock_key = self._get_lock_key(resource_id)
            lock_file = os.path.join(self.lock_dir, f"{lock_key}.lock")

            # Remove from memory tracking
            if lock_key in self.active_locks:
                del self.active_locks[lock_key]

            # Remove lock file
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                    self.logger.debug(f"Released lock for resource {resource_id}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to release lock for {resource_id}: {e}")
                    return False

            return True  # If file doesn't exist, consider it already released

    def _get_lock_key(self, resource_id: str) -> str:
        """
        Generate a lock key for a resource

        Args:
            resource_id (str): Resource ID

        Returns:
            str: Lock key
        """
        import hashlib
        return f"lock_{hashlib.md5(resource_id.encode()).hexdigest()[:16]}"

    def is_locked(self, resource_id: str) -> bool:
        """
        Check if a resource is currently locked

        Args:
            resource_id (str): ID of the resource to check

        Returns:
            bool: True if resource is locked, False otherwise
        """
        lock_key = self._get_lock_key(resource_id)
        lock_file = os.path.join(self.lock_dir, f"{lock_key}.lock")

        if os.path.exists(lock_file):
            try:
                with open(lock_file, 'r') as f:
                    lock_data = json.load(f)

                # Check if the lock has expired
                lock_time = datetime.fromisoformat(lock_data['timestamp'])
                timeout = lock_data['timeout']

                if datetime.now() - lock_time < timedelta(seconds=timeout):
                    return True
                else:
                    # Lock has expired, remove it
                    self.logger.info(f"Removing expired lock for {resource_id}")
                    os.remove(lock_file)
                    return False
            except Exception:
                # If there's an error reading the file, assume it's locked
                return True

        return False

    def cleanup_expired_locks(self):
        """
        Clean up any expired locks
        """
        with self._lock:
            for filename in os.listdir(self.lock_dir):
                if filename.endswith('.lock'):
                    lock_file = os.path.join(self.lock_dir, filename)
                    try:
                        with open(lock_file, 'r') as f:
                            lock_data = json.load(f)

                        # Check if the lock has expired
                        lock_time = datetime.fromisoformat(lock_data['timestamp'])
                        timeout = lock_data['timeout']

                        if datetime.now() - lock_time >= timedelta(seconds=timeout):
                            os.remove(lock_file)
                            lock_key = filename.replace('.lock', '')
                            if lock_key in self.active_locks:
                                del self.active_locks[lock_key]
                            self.logger.info(f"Cleaned up expired lock: {lock_data['resource_id']}")
                    except Exception as e:
                        self.logger.error(f"Error cleaning up lock file {filename}: {e}")


def test_concurrency_manager():
    """
    Test function to verify the concurrency manager works
    """
    cm = ConcurrencyManager()

    # Test lock acquisition
    resource_id = "test_resource_1"

    # Acquire lock
    acquired = cm.acquire_lock(resource_id, timeout=10, owner="test_process")
    print(f"Lock acquired: {acquired}")

    # Try to acquire same lock again (should fail)
    acquired_again = cm.acquire_lock(resource_id, timeout=10, owner="another_process")
    print(f"Second lock attempt: {acquired_again}")

    # Check if locked
    is_locked = cm.is_locked(resource_id)
    print(f"Resource is locked: {is_locked}")

    # Release lock
    released = cm.release_lock(resource_id)
    print(f"Lock released: {released}")

    # Check if still locked
    is_locked = cm.is_locked(resource_id)
    print(f"Resource is locked after release: {is_locked}")

    # Test with timeout
    acquired = cm.acquire_lock(resource_id, timeout=1, owner="test_process_2")
    print(f"New lock acquired: {acquired}")

    # Wait for timeout
    time.sleep(2)

    # Check if expired
    is_locked = cm.is_locked(resource_id)
    print(f"Resource is locked after timeout: {is_locked}")

    # Clean up expired locks
    cm.cleanup_expired_locks()


if __name__ == "__main__":
    test_concurrency_manager()