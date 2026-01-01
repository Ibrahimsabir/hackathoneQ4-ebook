"""
Performance monitoring and metrics collection for the ingestion pipeline
"""
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
from logging_config import logger
import json
import os
from threading import Lock


class PerformanceMonitor:
    """
    Class to monitor performance and collect metrics for the ingestion pipeline
    """

    def __init__(self, metrics_file: str = "performance_metrics.json"):
        """
        Initialize the performance monitor

        Args:
            metrics_file (str): File to store performance metrics
        """
        self.metrics_file = metrics_file
        self.logger = logger
        self.metrics: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "operations": {},
            "timings": {},
            "throughput": {},
            "errors": {},
            "summary": {}
        }
        self.lock = Lock()  # For thread safety
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.operation_errors: Dict[str, int] = {}

    def start_operation(self, operation_name: str) -> float:
        """
        Start timing an operation

        Args:
            operation_name (str): Name of the operation

        Returns:
            float: Start time
        """
        start_time = time.time()
        self.logger.debug(f"Starting operation: {operation_name}")
        return start_time

    def end_operation(self, operation_name: str, start_time: float, success: bool = True) -> Dict[str, Any]:
        """
        End timing an operation and record metrics

        Args:
            operation_name (str): Name of the operation
            start_time (float): Start time of the operation
            success (bool): Whether the operation was successful

        Returns:
            Dict[str, Any]: Operation metrics
        """
        end_time = time.time()
        duration = end_time - start_time

        with self.lock:
            # Record timing
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            self.operation_times[operation_name].append(duration)

            # Record count
            if operation_name not in self.operation_counts:
                self.operation_counts[operation_name] = 0
            self.operation_counts[operation_name] += 1

            # Record errors
            if not success:
                if operation_name not in self.operation_errors:
                    self.operation_errors[operation_name] = 0
                self.operation_errors[operation_name] += 1

        operation_metrics = {
            "operation": operation_name,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "success": success
        }

        self.logger.debug(f"Completed operation: {operation_name}, duration: {duration:.3f}s")

        # Update summary metrics
        self._update_summary_metrics(operation_name, duration, success)

        return operation_metrics

    def _update_summary_metrics(self, operation_name: str, duration: float, success: bool):
        """
        Update summary metrics for the operation

        Args:
            operation_name (str): Name of the operation
            duration (float): Duration of the operation
            success (bool): Whether the operation was successful
        """
        # Update timing metrics
        if operation_name not in self.metrics["timings"]:
            self.metrics["timings"][operation_name] = {
                "total_calls": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "avg_time": 0,
                "success_count": 0,
                "error_count": 0
            }

        timing_data = self.metrics["timings"][operation_name]
        timing_data["total_calls"] += 1
        timing_data["total_time"] += duration
        timing_data["min_time"] = min(timing_data["min_time"], duration)
        timing_data["max_time"] = max(timing_data["max_time"], duration)
        timing_data["avg_time"] = timing_data["total_time"] / timing_data["total_calls"]
        timing_data["success_count"] += 1 if success else 0
        timing_data["error_count"] += 0 if success else 1

        # Calculate success rate
        total_ops = timing_data["success_count"] + timing_data["error_count"]
        timing_data["success_rate"] = timing_data["success_count"] / total_ops if total_ops > 0 else 0

    def measure_function(self, func_name: str = None):
        """
        Decorator to measure function performance

        Args:
            func_name (str): Name to use for the function (optional, will use function name if not provided)
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                op_name = func_name or func.__name__
                start_time = self.start_operation(op_name)
                try:
                    result = func(*args, **kwargs)
                    self.end_operation(op_name, start_time, success=True)
                    return result
                except Exception as e:
                    self.end_operation(op_name, start_time, success=False)
                    raise e
            return wrapper
        return decorator

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation

        Args:
            operation_name (str): Name of the operation

        Returns:
            Dict[str, Any]: Operation statistics
        """
        if operation_name not in self.operation_times:
            return {"error": f"No data for operation: {operation_name}"}

        times = self.operation_times[operation_name]
        count = self.operation_counts.get(operation_name, 0)
        errors = self.operation_errors.get(operation_name, 0)

        stats = {
            "operation": operation_name,
            "count": count,
            "errors": errors,
            "success_rate": (count - errors) / count if count > 0 else 0,
            "total_time": sum(times),
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times) if times else 0
        }

        if len(times) > 1:
            stats["std_dev"] = statistics.stdev(times)

        return stats

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all operations

        Returns:
            Dict[str, Any]: All operation statistics
        """
        all_stats = {}
        for operation_name in self.operation_times:
            all_stats[operation_name] = self.get_operation_stats(operation_name)
        return all_stats

    def calculate_throughput(self, operation_name: str, time_window_seconds: int = 60) -> float:
        """
        Calculate throughput for an operation (operations per second)

        Args:
            operation_name (str): Name of the operation
            time_window_seconds (int): Time window in seconds

        Returns:
            float: Throughput in operations per second
        """
        if operation_name not in self.operation_times:
            return 0

        # For simplicity, we'll calculate average throughput since start
        # In a real system, you'd want to track recent operations within the time window
        total_time = sum(self.operation_times[operation_name])
        total_ops = len(self.operation_times[operation_name])

        if total_time > 0:
            return total_ops / total_time  # ops per second
        else:
            return 0

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics

        Returns:
            Dict[str, Any]: Performance summary
        """
        total_operations = sum(self.operation_counts.values())
        total_errors = sum(self.operation_errors.values())
        total_time = sum(sum(times) for times in self.operation_times.values())

        summary = {
            "start_time": self.metrics.get("start_time"),
            "current_time": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - datetime.fromisoformat(self.metrics["start_time"].replace("Z", "+00:00").split("+")[0] if "Z" in self.metrics["start_time"] else self.metrics["start_time"])) if self.metrics.get("start_time") else 0,
            "total_operations": total_operations,
            "total_errors": total_errors,
            "total_time_seconds": total_time,
            "error_rate": total_errors / total_operations if total_operations > 0 else 0,
            "average_operation_time": total_time / total_operations if total_operations > 0 else 0,
            "operations_by_type": self.operation_counts,
            "errors_by_type": self.operation_errors
        }

        return summary

    def save_metrics(self):
        """
        Save metrics to file
        """
        try:
            # Create a serializable version of metrics
            serializable_metrics = {
                "start_time": self.metrics["start_time"],
                "operations": self.operation_counts,
                "timings": self.metrics["timings"],
                "errors": self.operation_errors,
                "summary": self.get_performance_summary()
            }

            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2, default=str)
            self.logger.info(f"Performance metrics saved to {self.metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}")

    def load_metrics(self) -> Dict[str, Any]:
        """
        Load metrics from file

        Returns:
            Dict[str, Any]: Loaded metrics
        """
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    loaded_metrics = json.load(f)
                self.logger.info(f"Performance metrics loaded from {self.metrics_file}")

                # Restore internal state
                self.metrics = loaded_metrics
                self.operation_counts = loaded_metrics.get("operations", {})
                self.operation_errors = loaded_metrics.get("errors", {})
                self.metrics["timings"] = loaded_metrics.get("timings", {})

                return loaded_metrics
            else:
                self.logger.info(f"Performance metrics file {self.metrics_file} not found")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load performance metrics: {e}")
            return {}

    def reset_metrics(self):
        """
        Reset all performance metrics
        """
        with self.lock:
            self.metrics = {
                "start_time": datetime.now().isoformat(),
                "operations": {},
                "timings": {},
                "throughput": {},
                "errors": {},
                "summary": {}
            }
            self.operation_times = {}
            self.operation_counts = {}
            self.operation_errors = {}
        self.logger.info("Performance metrics reset")

    def generate_performance_report(self) -> str:
        """
        Generate a performance report

        Returns:
            str: Performance report
        """
        summary = self.get_performance_summary()
        all_stats = self.get_all_stats()

        report = f"""
PERFORMANCE METRICS REPORT
==========================

Summary:
- Start time: {summary['start_time']}
- Current time: {summary['current_time']}
- Uptime: {summary['uptime_seconds']} seconds
- Total operations: {summary['total_operations']}
- Total errors: {summary['total_errors']}
- Error rate: {summary['error_rate']:.2%}
- Average operation time: {summary['average_operation_time']:.4f}s

Operation Details:
"""
        for op_name, stats in all_stats.items():
            report += f"""
  {op_name}:
    - Count: {stats['count']}
    - Success rate: {stats['success_rate']:.2%}
    - Avg time: {stats['avg_time']:.4f}s
    - Min time: {stats['min_time']:.4f}s
    - Max time: {stats['max_time']:.4f}s
    - Median time: {stats['median_time']:.4f}s
"""

        return report


def test_performance_monitor():
    """
    Test function to verify the performance monitor works
    """
    pm = PerformanceMonitor()

    # Test timing operations
    start = pm.start_operation("test_operation")
    time.sleep(0.1)  # Simulate work
    pm.end_operation("test_operation", start, success=True)

    start = pm.start_operation("test_operation")
    time.sleep(0.05)  # Simulate work
    pm.end_operation("test_operation", start, success=True)

    start = pm.start_operation("test_operation")
    time.sleep(0.08)  # Simulate work
    pm.end_operation("test_operation", start, success=False)  # Simulate error

    # Test decorator
    @pm.measure_function("decorated_function")
    def sample_function():
        time.sleep(0.02)
        return "result"

    result = sample_function()
    print(f"Function result: {result}")

    # Get stats
    stats = pm.get_operation_stats("test_operation")
    print(f"Test operation stats: {stats}")

    all_stats = pm.get_all_stats()
    print(f"All stats: {all_stats}")

    summary = pm.get_performance_summary()
    print(f"Summary: {summary}")

    # Generate report
    report = pm.generate_performance_report()
    print(report)

    # Save and load metrics
    pm.save_metrics()
    pm.load_metrics()


if __name__ == "__main__":
    test_performance_monitor()