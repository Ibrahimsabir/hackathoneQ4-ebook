"""
Recovery manager for handling interrupted indexing operations
"""
import os
import json
import pickle
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from logging_config import logger


class RecoveryManager:
    """
    Class to manage recovery from interrupted indexing operations
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints", state_file: str = "ingestion_state.json"):
        """
        Initialize the recovery manager

        Args:
            checkpoint_dir (str): Directory to store checkpoints
            state_file (str): File to store overall state
        """
        self.checkpoint_dir = checkpoint_dir
        self.state_file = state_file
        self.logger = logger

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

    def create_checkpoint(self, checkpoint_id: str, data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Create a checkpoint for recovery

        Args:
            checkpoint_id (str): Unique identifier for the checkpoint
            data (Dict[str, Any]): Data to save for recovery
            metadata (Dict[str, Any]): Additional metadata about the checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "metadata": metadata or {}
        }

        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            self.logger.info(f"Checkpoint created: {checkpoint_id}")
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}")

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for recovery

        Args:
            checkpoint_id (str): Unique identifier for the checkpoint

        Returns:
            Optional[Dict[str, Any]]: Checkpoint data or None if not found
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            self.logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    def delete_checkpoint(self, checkpoint_id: str):
        """
        Delete a checkpoint

        Args:
            checkpoint_id (str): Unique identifier for the checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                self.logger.info(f"Checkpoint deleted: {checkpoint_id}")
            except Exception as e:
                self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")

    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints

        Returns:
            List[str]: List of checkpoint IDs
        """
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                checkpoints.append(filename[:-5])  # Remove .json extension
        return checkpoints

    def save_state(self, state: Dict[str, Any]):
        """
        Save the overall ingestion state

        Args:
            state (Dict[str, Any]): State to save
        """
        state_path = os.path.join(self.checkpoint_dir, self.state_file)

        state_data = {
            "timestamp": datetime.now().isoformat(),
            "state": state
        }

        try:
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
            self.logger.info("State saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load the overall ingestion state

        Returns:
            Optional[Dict[str, Any]]: State data or None if not found
        """
        state_path = os.path.join(self.checkpoint_dir, self.state_file)

        if not os.path.exists(state_path):
            self.logger.info("No state file found")
            return None

        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            self.logger.info("State loaded successfully")
            return state_data.get('state', {})
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return None

    def create_progress_checkpoint(self,
                                 task_name: str,
                                 processed_items: int,
                                 total_items: int,
                                 processed_urls: List[str] = None):
        """
        Create a progress checkpoint for a task

        Args:
            task_name (str): Name of the task
            processed_items (int): Number of items processed
            total_items (int): Total number of items
            processed_urls (List[str]): List of URLs that have been processed
        """
        checkpoint_data = {
            "task_name": task_name,
            "processed_items": processed_items,
            "total_items": total_items,
            "progress_percentage": (processed_items / total_items * 100) if total_items > 0 else 0,
            "processed_urls": processed_urls or [],
            "remaining_items": total_items - processed_items
        }

        checkpoint_id = f"progress_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.create_checkpoint(checkpoint_id, checkpoint_data)

        # Also update the overall state
        current_state = self.load_state() or {}
        current_state[task_name] = checkpoint_data
        self.save_state(current_state)

        return checkpoint_id

    def resume_from_checkpoint(self, task_name: str, all_urls: List[str]) -> Dict[str, Any]:
        """
        Resume processing from a checkpoint

        Args:
            task_name (str): Name of the task to resume
            all_urls (List[str]): Complete list of URLs to process

        Returns:
            Dict[str, Any]: Resume information including remaining URLs and progress
        """
        # Load the overall state first
        state = self.load_state()
        if not state or task_name not in state:
            self.logger.info(f"No checkpoint found for task: {task_name}")
            return {
                "resume_from": 0,
                "remaining_urls": all_urls,
                "processed_urls": [],
                "progress_percentage": 0.0
            }

        task_state = state[task_name]
        processed_urls = task_state.get("processed_urls", [])

        # Find remaining URLs
        remaining_urls = [url for url in all_urls if url not in processed_urls]

        resume_info = {
            "resume_from": len(processed_urls),
            "remaining_urls": remaining_urls,
            "processed_urls": processed_urls,
            "progress_percentage": task_state.get("progress_percentage", 0.0)
        }

        self.logger.info(f"Resuming from checkpoint: {len(processed_urls)} URLs already processed, {len(remaining_urls)} remaining")
        return resume_info

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Clean up old checkpoints, keeping only the most recent ones

        Args:
            keep_last_n (int): Number of recent checkpoints to keep
        """
        all_checkpoints = self.list_checkpoints()

        # Filter to only progress checkpoints and sort by timestamp
        progress_checkpoints = [cp for cp in all_checkpoints if cp.startswith('progress_')]
        progress_checkpoints.sort(reverse=True)  # Most recent first

        # Delete old checkpoints
        checkpoints_to_delete = progress_checkpoints[keep_last_n:]
        for cp_id in checkpoints_to_delete:
            self.delete_checkpoint(cp_id)

        self.logger.info(f"Cleaned up {len(checkpoints_to_delete)} old checkpoints, kept {keep_last_n}")

    def reset_state(self):
        """
        Reset the entire state
        """
        state_path = os.path.join(self.checkpoint_dir, self.state_file)
        if os.path.exists(state_path):
            try:
                os.remove(state_path)
                self.logger.info("State reset successfully")
            except Exception as e:
                self.logger.error(f"Failed to reset state: {e}")

        # Clean up all checkpoints
        all_checkpoints = self.list_checkpoints()
        for cp_id in all_checkpoints:
            self.delete_checkpoint(cp_id)

        self.logger.info("All checkpoints deleted")


def test_recovery_manager():
    """
    Test function to verify the recovery manager works
    """
    rm = RecoveryManager()

    # Test checkpoint creation and loading
    test_data = {
        "urls_processed": ["url1", "url2", "url3"],
        "current_position": 3,
        "total_items": 10
    }

    metadata = {
        "task": "content_extraction",
        "timestamp": datetime.now().isoformat()
    }

    # Create a checkpoint
    cp_id = "test_checkpoint_001"
    rm.create_checkpoint(cp_id, test_data, metadata)

    # Load the checkpoint
    loaded_data = rm.load_checkpoint(cp_id)
    print(f"Loaded checkpoint data: {loaded_data}")

    # Test progress checkpoint
    urls = [f"url_{i}" for i in range(20)]
    rm.create_progress_checkpoint("extraction", 5, 20, urls[:5])

    # Test resume functionality
    resume_info = rm.resume_from_checkpoint("extraction", urls)
    print(f"Resume info: {resume_info}")

    # List all checkpoints
    checkpoints = rm.list_checkpoints()
    print(f"Available checkpoints: {checkpoints}")

    # Clean up
    rm.cleanup_old_checkpoints(keep_last_n=2)


if __name__ == "__main__":
    test_recovery_manager()