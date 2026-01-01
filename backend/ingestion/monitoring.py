"""
Monitoring and alerting module for the ingestion pipeline
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable
from logging_config import logger
import json
import os


class IngestionMonitor:
    """
    Class to handle monitoring and alerting for the ingestion pipeline
    """

    def __init__(self, metrics_file: str = "ingestion_metrics.json", alert_thresholds: Dict[str, Any] = None):
        """
        Initialize the ingestion monitor

        Args:
            metrics_file (str): File to store metrics
            alert_thresholds (Dict[str, Any]): Alert thresholds configuration
        """
        self.metrics_file = metrics_file
        self.alert_thresholds = alert_thresholds or {
            "crawl_failure_rate": 0.2,  # Alert if >20% of crawls fail
            "embedding_failure_rate": 0.1,  # Alert if >10% of embeddings fail
            "storage_failure_rate": 0.05,  # Alert if >5% of storage operations fail
            "processing_time_per_page": 30,  # Alert if processing takes longer than 30 seconds per page
            "api_error_rate": 0.05,  # Alert if >5% of API calls fail
            "low_similarity_score": 0.3  # Alert if similarity scores are below 0.3
        }
        self.logger = logger
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "total_pages": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "successful_storage": 0,
            "failed_storage": 0,
            "total_api_calls": 0,
            "failed_api_calls": 0,
            "total_processing_time": 0,
            "alerts_triggered": []
        }
        self.alert_callbacks: List[Callable] = []

    def start_monitoring(self):
        """
        Start monitoring the ingestion process
        """
        self.metrics["start_time"] = datetime.now().isoformat()
        self.logger.info("Monitoring started for ingestion pipeline")

    def stop_monitoring(self):
        """
        Stop monitoring the ingestion process
        """
        self.metrics["end_time"] = datetime.now().isoformat()
        self.logger.info("Monitoring stopped for ingestion pipeline")
        self.save_metrics()

    def record_crawl_result(self, success: bool, processing_time: float = None):
        """
        Record the result of a crawl operation

        Args:
            success (bool): Whether the crawl was successful
            processing_time (float): Time taken for the crawl
        """
        if success:
            self.metrics["successful_crawls"] += 1
        else:
            self.metrics["failed_crawls"] += 1

        if processing_time:
            self.metrics["total_processing_time"] += processing_time

        self.metrics["total_pages"] += 1
        self._check_crawl_failure_rate()

    def record_embedding_result(self, success: bool):
        """
        Record the result of an embedding operation

        Args:
            success (bool): Whether the embedding was successful
        """
        if success:
            self.metrics["successful_embeddings"] += 1
        else:
            self.metrics["failed_embeddings"] += 1

        self._check_embedding_failure_rate()

    def record_storage_result(self, success: bool):
        """
        Record the result of a storage operation

        Args:
            success (bool): Whether the storage was successful
        """
        if success:
            self.metrics["successful_storage"] += 1
        else:
            self.metrics["failed_storage"] += 1

        self._check_storage_failure_rate()

    def record_api_call(self, success: bool):
        """
        Record the result of an API call

        Args:
            success (bool): Whether the API call was successful
        """
        self.metrics["total_api_calls"] += 1
        if not success:
            self.metrics["failed_api_calls"] += 1

        self._check_api_error_rate()

    def record_similarity_score(self, score: float):
        """
        Record a similarity score for monitoring

        Args:
            score (float): Similarity score
        """
        if score < self.alert_thresholds["low_similarity_score"]:
            alert_msg = f"Low similarity score detected: {score:.3f} (threshold: {self.alert_thresholds['low_similarity_score']})"
            self._trigger_alert(alert_msg, "similarity_score_low")

    def _check_crawl_failure_rate(self):
        """
        Check if crawl failure rate exceeds threshold
        """
        if self.metrics["total_pages"] > 0:
            failure_rate = self.metrics["failed_crawls"] / self.metrics["total_pages"]
            if failure_rate > self.alert_thresholds["crawl_failure_rate"]:
                alert_msg = f"Crawl failure rate too high: {failure_rate:.2%} (threshold: {self.alert_thresholds['crawl_failure_rate']:.2%})"
                self._trigger_alert(alert_msg, "crawl_failure_rate")

    def _check_embedding_failure_rate(self):
        """
        Check if embedding failure rate exceeds threshold
        """
        total_embeddings = self.metrics["successful_embeddings"] + self.metrics["failed_embeddings"]
        if total_embeddings > 0:
            failure_rate = self.metrics["failed_embeddings"] / total_embeddings
            if failure_rate > self.alert_thresholds["embedding_failure_rate"]:
                alert_msg = f"Embedding failure rate too high: {failure_rate:.2%} (threshold: {self.alert_thresholds['embedding_failure_rate']:.2%})"
                self._trigger_alert(alert_msg, "embedding_failure_rate")

    def _check_storage_failure_rate(self):
        """
        Check if storage failure rate exceeds threshold
        """
        total_storage = self.metrics["successful_storage"] + self.metrics["failed_storage"]
        if total_storage > 0:
            failure_rate = self.metrics["failed_storage"] / total_storage
            if failure_rate > self.alert_thresholds["storage_failure_rate"]:
                alert_msg = f"Storage failure rate too high: {failure_rate:.2%} (threshold: {self.alert_thresholds['storage_failure_rate']:.2%})"
                self._trigger_alert(alert_msg, "storage_failure_rate")

    def _check_api_error_rate(self):
        """
        Check if API error rate exceeds threshold
        """
        if self.metrics["total_api_calls"] > 0:
            error_rate = self.metrics["failed_api_calls"] / self.metrics["total_api_calls"]
            if error_rate > self.alert_thresholds["api_error_rate"]:
                alert_msg = f"API error rate too high: {error_rate:.2%} (threshold: {self.alert_thresholds['api_error_rate']:.2%})"
                self._trigger_alert(alert_msg, "api_error_rate")

    def _trigger_alert(self, message: str, alert_type: str):
        """
        Trigger an alert with the given message

        Args:
            message (str): Alert message
            alert_type (str): Type of alert
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message
        }

        self.metrics["alerts_triggered"].append(alert)
        self.logger.warning(f"ALERT: {message}")

        # Execute any registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable):
        """
        Add a callback function to be executed when alerts are triggered

        Args:
            callback (Callable): Callback function
        """
        self.alert_callbacks.append(callback)

    def save_metrics(self):
        """
        Save metrics to the metrics file
        """
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)
            self.logger.info(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics to {self.metrics_file}: {e}")

    def load_metrics(self) -> Dict[str, Any]:
        """
        Load metrics from the metrics file

        Returns:
            Dict[str, Any]: Loaded metrics
        """
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    self.metrics = json.load(f)
                self.logger.info(f"Metrics loaded from {self.metrics_file}")
            else:
                self.logger.info(f"Metrics file {self.metrics_file} not found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load metrics from {self.metrics_file}: {e}")

        return self.metrics

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics

        Returns:
            Dict[str, Any]: Current metrics
        """
        return self.metrics.copy()

    def generate_monitoring_report(self) -> str:
        """
        Generate a monitoring report

        Returns:
            str: Monitoring report
        """
        # Calculate derived metrics
        total_crawls = self.metrics["successful_crawls"] + self.metrics["failed_crawls"]
        total_embeddings = self.metrics["successful_embeddings"] + self.metrics["failed_embeddings"]
        total_storage = self.metrics["successful_storage"] + self.metrics["failed_storage"]

        crawl_success_rate = (self.metrics["successful_crawls"] / total_crawls * 100) if total_crawls > 0 else 0
        embedding_success_rate = (self.metrics["successful_embeddings"] / total_embeddings * 100) if total_embeddings > 0 else 0
        storage_success_rate = (self.metrics["successful_storage"] / total_storage * 100) if total_storage > 0 else 0

        avg_processing_time = (self.metrics["total_processing_time"] / self.metrics["total_pages"]) if self.metrics["total_pages"] > 0 else 0

        report = f"""
INGESTION MONITORING REPORT
==========================

Pipeline Statistics:
- Total pages processed: {self.metrics['total_pages']}
- Successful crawls: {self.metrics['successful_crawls']}
- Failed crawls: {self.metrics['failed_crawls']}
- Successful embeddings: {self.metrics['successful_embeddings']}
- Failed embeddings: {self.metrics['failed_embeddings']}
- Successful storage: {self.metrics['successful_storage']}
- Failed storage: {self.metrics['failed_storage']}

Success Rates:
- Crawl success rate: {crawl_success_rate:.2f}%
- Embedding success rate: {embedding_success_rate:.2f}%
- Storage success rate: {storage_success_rate:.2f}%

Performance Metrics:
- Total processing time: {self.metrics['total_processing_time']:.2f}s
- Average processing time per page: {avg_processing_time:.2f}s
- Total API calls: {self.metrics['total_api_calls']}
- Failed API calls: {self.metrics['failed_api_calls']}

Alerts:
- Total alerts triggered: {len(self.metrics['alerts_triggered'])}
"""
        if self.metrics['alerts_triggered']:
            report += "\nRecent Alerts:\n"
            for alert in self.metrics['alerts_triggered'][-5:]:  # Show last 5 alerts
                report += f"  - {alert['timestamp']}: {alert['message']}\n"
        else:
            report += "- No alerts triggered\n"

        return report


def test_monitoring():
    """
    Test function to verify the monitoring works
    """
    monitor = IngestionMonitor()

    # Register a simple alert callback
    def alert_callback(alert):
        print(f"ALERT CALLBACK: {alert['message']}")

    monitor.add_alert_callback(alert_callback)

    # Start monitoring
    monitor.start_monitoring()

    # Simulate some operations
    for i in range(10):
        monitor.record_crawl_result(success=True, processing_time=2.5)
        monitor.record_embedding_result(success=True)
        monitor.record_storage_result(success=True)
        monitor.record_api_call(success=True)

    # Simulate some failures to trigger alerts
    for i in range(3):
        monitor.record_crawl_result(success=False, processing_time=1.0)
        monitor.record_embedding_result(success=False)
        monitor.record_storage_result(success=False)
        monitor.record_api_call(success=False)

    # Record some similarity scores
    monitor.record_similarity_score(0.8)
    monitor.record_similarity_score(0.2)  # This should trigger an alert

    # Stop monitoring
    monitor.stop_monitoring()

    # Print report
    print(monitor.generate_monitoring_report())


if __name__ == "__main__":
    test_monitoring()