"""
Health check endpoints for monitoring the ingestion system
"""
from flask import Flask, jsonify
import logging
from datetime import datetime
from logging_config import logger
from .vector_storage import QdrantVectorStorage
from .embedder import CohereEmbedder
import requests
import time


class HealthMonitor:
    """
    Class to provide health check endpoints for monitoring
    """

    def __init__(self, app: Flask = None):
        """
        Initialize the health monitor

        Args:
            app (Flask): Flask app instance
        """
        self.app = app or Flask(__name__)
        self.logger = logger
        self.qdrant_storage = QdrantVectorStorage()
        self.embedder = CohereEmbedder()
        self.start_time = datetime.now()
        self.setup_routes()

    def setup_routes(self):
        """
        Set up health check routes
        """
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return self.get_health_status()

        @self.app.route('/health/ready', methods=['GET'])
        def readiness_check():
            return self.get_readiness_status()

        @self.app.route('/health/live', methods=['GET'])
        def liveness_check():
            return self.get_liveness_status()

        @self.app.route('/health/details', methods=['GET'])
        def detailed_health_check():
            return self.get_detailed_health_status()

    def get_health_status(self):
        """
        Basic health check endpoint

        Returns:
            Response: Health status JSON
        """
        try:
            # Check if basic services are available
            qdrant_healthy = self._check_qdrant_health()
            cohere_healthy = self._check_cohere_health()

            is_healthy = qdrant_healthy and cohere_healthy

            health_data = {
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - self.start_time),
                "checks": {
                    "qdrant": {"status": "ok" if qdrant_healthy else "error"},
                    "cohere": {"status": "ok" if cohere_healthy else "error"}
                }
            }

            status_code = 200 if is_healthy else 503
            return jsonify(health_data), status_code

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 503

    def get_readiness_status(self):
        """
        Readiness check - whether the service is ready to accept traffic

        Returns:
            Response: Readiness status JSON
        """
        try:
            # Check if all dependencies are ready
            qdrant_ready = self._check_qdrant_readiness()
            cohere_ready = self._check_cohere_readiness()

            is_ready = qdrant_ready and cohere_ready

            readiness_data = {
                "status": "ready" if is_ready else "not_ready",
                "timestamp": datetime.now().isoformat(),
                "dependencies": {
                    "qdrant": {"ready": qdrant_ready},
                    "cohere": {"ready": cohere_ready}
                }
            }

            status_code = 200 if is_ready else 503
            return jsonify(readiness_data), status_code

        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 503

    def get_liveness_status(self):
        """
        Liveness check - whether the service is alive

        Returns:
            Response: Liveness status JSON
        """
        try:
            # Check if the service is alive (basic health)
            is_alive = True  # Service is alive if it can respond to this request

            liveness_data = {
                "status": "alive",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - self.start_time),
                "process": True
            }

            return jsonify(liveness_data), 200

        except Exception as e:
            self.logger.error(f"Liveness check failed: {e}")
            return jsonify({
                "status": "dead",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 503

    def get_detailed_health_status(self):
        """
        Detailed health check with comprehensive status

        Returns:
            Response: Detailed health status JSON
        """
        try:
            # Get detailed status of all components
            qdrant_status = self._get_qdrant_status()
            cohere_status = self._get_cohere_status()
            system_status = self._get_system_status()

            all_healthy = (
                qdrant_status["healthy"] and
                cohere_status["healthy"] and
                system_status["healthy"]
            )

            detailed_health = {
                "status": "healthy" if all_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - self.start_time),
                "components": {
                    "qdrant": qdrant_status,
                    "cohere": cohere_status,
                    "system": system_status
                }
            }

            status_code = 200 if all_healthy else 503
            return jsonify(detailed_health), status_code

        except Exception as e:
            self.logger.error(f"Detailed health check failed: {e}")
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 503

    def _check_qdrant_health(self) -> bool:
        """
        Check if Qdrant is healthy

        Returns:
            bool: True if Qdrant is healthy
        """
        try:
            # Test Qdrant connection by getting collection info
            info = self.qdrant_storage.get_collection_info()
            return info.get("point_count", -1) >= 0  # If we can get info, it's accessible
        except Exception as e:
            self.logger.error(f"Qdrant health check failed: {e}")
            return False

    def _check_cohere_health(self) -> bool:
        """
        Check if Cohere is healthy

        Returns:
            bool: True if Cohere is healthy
        """
        try:
            # Test Cohere by making a simple embedding request
            test_embedding = self.embedder.generate_embeddings(["health check"])
            return len(test_embedding) > 0 if test_embedding else False
        except Exception as e:
            self.logger.error(f"Cohere health check failed: {e}")
            return False

    def _check_qdrant_readiness(self) -> bool:
        """
        Check if Qdrant is ready

        Returns:
            bool: True if Qdrant is ready
        """
        return self._check_qdrant_health()  # For our use case, health = readiness

    def _check_cohere_readiness(self) -> bool:
        """
        Check if Cohere is ready

        Returns:
            bool: True if Cohere is ready
        """
        return self._check_cohere_health()  # For our use case, health = readiness

    def _get_qdrant_status(self) -> dict:
        """
        Get detailed Qdrant status

        Returns:
            dict: Qdrant status information
        """
        try:
            info = self.qdrant_storage.get_collection_info()
            return {
                "healthy": True,
                "status": "available",
                "details": {
                    "collection_name": self.qdrant_storage.collection_name,
                    "vector_size": info.get("vector_size", "unknown"),
                    "point_count": info.get("point_count", 0),
                    "distance": info.get("distance", "unknown")
                }
            }
        except Exception as e:
            self.logger.error(f"Qdrant status check failed: {e}")
            return {
                "healthy": False,
                "status": "unavailable",
                "error": str(e)
            }

    def _get_cohere_status(self) -> dict:
        """
        Get detailed Cohere status

        Returns:
            dict: Cohere status information
        """
        try:
            # Test embedding generation
            start_time = time.time()
            test_embedding = self.embedder.generate_embeddings(["test"])
            response_time = time.time() - start_time

            return {
                "healthy": len(test_embedding) > 0 if test_embedding else False,
                "status": "available",
                "details": {
                    "model": self.embedder.model,
                    "response_time_ms": round(response_time * 1000, 2),
                    "dimension": len(test_embedding[0]) if test_embedding and test_embedding[0] else "unknown"
                }
            }
        except Exception as e:
            self.logger.error(f"Cohere status check failed: {e}")
            return {
                "healthy": False,
                "status": "unavailable",
                "error": str(e)
            }

    def _get_system_status(self) -> dict:
        """
        Get system status

        Returns:
            dict: System status information
        """
        import psutil
        import os

        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('.').percent
            process_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            return {
                "healthy": True,
                "status": "running",
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "process_memory_mb": round(process_memory, 2),
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
                }
            }
        except Exception as e:
            self.logger.error(f"System status check failed: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }

    def run_server(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """
        Run the health check server

        Args:
            host (str): Host to bind to
            port (int): Port to bind to
            debug (bool): Whether to run in debug mode
        """
        self.logger.info(f"Starting health check server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)


def create_health_app():
    """
    Create a health check app instance
    """
    health_monitor = HealthMonitor()
    return health_monitor.app


def test_health_monitor():
    """
    Test function to verify the health monitor works
    """
    health_monitor = HealthMonitor()

    # Test health check
    with health_monitor.app.test_client() as client:
        response = client.get('/health')
        print(f"Health check status: {response.status_code}")
        print(f"Health check data: {response.get_json()}")

        response = client.get('/health/ready')
        print(f"Readiness check status: {response.status_code}")

        response = client.get('/health/live')
        print(f"Liveness check status: {response.status_code}")

        response = client.get('/health/details')
        print(f"Detailed health check status: {response.status_code}")


if __name__ == "__main__":
    # Run the health check server
    health_monitor = HealthMonitor()
    health_monitor.run_server(host="0.0.0.0", port=8000, debug=True)