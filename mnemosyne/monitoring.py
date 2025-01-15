"""
Performance monitoring module with Prometheus integration.
"""
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field
from loguru import logger
import time
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server,
    REGISTRY,
    CollectorRegistry
)


class MetricType(str, Enum):
    """Types of metrics to track."""
    COUNTER = "counter"  # Monotonically increasing counter
    HISTOGRAM = "histogram"  # Distribution of values
    GAUGE = "gauge"  # Point-in-time value


class MetricLabel(str, Enum):
    """Common metric labels."""
    AGENT_TYPE = "agent_type"
    OPERATION = "operation"
    STATUS = "status"
    ERROR_TYPE = "error_type"


class PerformanceMetrics:
    """
    Manages performance metrics collection and reporting.
    """
    
    def __init__(self, port: int = 8000):
        """Initialize metrics with Prometheus registry."""
        # Create metrics
        self.request_count = Counter(
            "mnemosyne_requests_total",
            "Total number of requests",
            ["agent_type", "operation", "status"]
        )
        
        self.request_latency = Histogram(
            "mnemosyne_request_duration_seconds",
            "Request duration in seconds",
            ["agent_type", "operation"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.error_count = Counter(
            "mnemosyne_errors_total",
            "Total number of errors",
            ["agent_type", "error_type"]
        )
        
        self.active_workers = Gauge(
            "mnemosyne_active_workers",
            "Number of active workers",
            ["agent_type"]
        )
        
        self.queue_size = Gauge(
            "mnemosyne_queue_size",
            "Number of items in queue",
            ["agent_type"]
        )
        
        self.memory_usage = Gauge(
            "mnemosyne_memory_mb",
            "Memory usage in MB",
            ["component"]
        )
        
        # Start Prometheus HTTP server
        start_http_server(port)
        logger.info("Started Prometheus metrics server on port {}", port)
        
        # Initialize aggregation state
        self._reset_aggregation()
        
        # Start background tasks
        self._start_background_tasks()

    def _reset_aggregation(self):
        """Reset aggregation state."""
        self._agg_lock = threading.Lock()
        self._agg_data = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "errors": defaultdict(int)
        })

    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        async def update_memory_metrics():
            while True:
                try:
                    # TODO: Get actual memory usage
                    # For now, use dummy values
                    self.memory_usage.labels("orchestrator").set(100)
                    self.memory_usage.labels("workers").set(200)
                    self.memory_usage.labels("cache").set(50)
                except Exception as e:
                    logger.exception("Error updating memory metrics")
                await asyncio.sleep(60)  # Update every minute
        
        # Start memory metrics task
        asyncio.create_task(update_memory_metrics())

    def track_request(self,
                   agent_type: str,
                   operation: str) -> "RequestTracker":
        """
        Create a context manager to track request metrics.
        
        Args:
            agent_type: Type of agent making request
            operation: Operation being performed
            
        Returns:
            Context manager for tracking request
        """
        return RequestTracker(
            metrics=self,
            agent_type=agent_type,
            operation=operation
        )

    def track_worker(self,
                   agent_type: str) -> "WorkerTracker":
        """
        Create a context manager to track worker metrics.
        
        Args:
            agent_type: Type of worker
            
        Returns:
            Context manager for tracking worker
        """
        return WorkerTracker(
            metrics=self,
            agent_type=agent_type
        )

    def record_error(self,
                   agent_type: str,
                   error_type: str):
        """Record an error occurrence."""
        self.error_count.labels(
            agent_type=agent_type,
            error_type=error_type
        ).inc()
        
        with self._agg_lock:
            self._agg_data[agent_type]["errors"][error_type] += 1

    def update_queue_size(self,
                       agent_type: str,
                       size: int):
        """Update queue size metric."""
        self.queue_size.labels(agent_type=agent_type).set(size)

    def get_summary(self,
                  agent_type: Optional[str] = None,
                  window: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """
        Get summary metrics for the specified window.
        
        Args:
            agent_type: Optional agent type to filter by
            window: Time window to summarize
            
        Returns:
            Dictionary of summary metrics
        """
        with self._agg_lock:
            if agent_type:
                data = {
                    agent_type: self._agg_data[agent_type]
                }
            else:
                data = dict(self._agg_data)
            
            summary = {}
            for agent, metrics in data.items():
                if metrics["count"] > 0:
                    avg_time = metrics["total_time"] / metrics["count"]
                else:
                    avg_time = 0.0
                    
                summary[agent] = {
                    "requests": metrics["count"],
                    "avg_latency": avg_time,
                    "error_counts": dict(metrics["errors"])
                }
            
            return summary


class RequestTracker:
    """Context manager for tracking request metrics."""
    
    def __init__(self,
                metrics: PerformanceMetrics,
                agent_type: str,
                operation: str):
        """Initialize the tracker."""
        self.metrics = metrics
        self.agent_type = agent_type
        self.operation = operation
        self.start_time = None
        self.error = None

    async def __aenter__(self):
        """Start tracking request."""
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking request and record metrics."""
        duration = time.time() - self.start_time
        
        # Record latency
        self.metrics.request_latency.labels(
            agent_type=self.agent_type,
            operation=self.operation
        ).observe(duration)
        
        # Record request count
        status = "error" if exc_type else "success"
        self.metrics.request_count.labels(
            agent_type=self.agent_type,
            operation=self.operation,
            status=status
        ).inc()
        
        # Record error if any
        if exc_type:
            self.metrics.record_error(
                self.agent_type,
                exc_type.__name__
            )
        
        # Update aggregation
        with self.metrics._agg_lock:
            agg = self.metrics._agg_data[self.agent_type]
            agg["count"] += 1
            agg["total_time"] += duration


class WorkerTracker:
    """Context manager for tracking worker metrics."""
    
    def __init__(self,
                metrics: PerformanceMetrics,
                agent_type: str):
        """Initialize the tracker."""
        self.metrics = metrics
        self.agent_type = agent_type

    async def __aenter__(self):
        """Start tracking worker."""
        self.metrics.active_workers.labels(
            agent_type=self.agent_type
        ).inc()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking worker."""
        self.metrics.active_workers.labels(
            agent_type=self.agent_type
        ).dec()


class MonitoredComponent:
    """Base class for components with performance monitoring."""
    
    def __init__(self, metrics: PerformanceMetrics):
        """Initialize with metrics."""
        self.metrics = metrics

    async def track_operation(self,
                          operation: str,
                          agent_type: str = "general") -> "RequestTracker":
        """Track an operation with metrics."""
        return self.metrics.track_request(agent_type, operation)

    def track_worker(self, agent_type: str = "general") -> "WorkerTracker":
        """Track a worker with metrics."""
        return self.metrics.track_worker(agent_type)

    def record_error(self,
                   error_type: str,
                   agent_type: str = "general"):
        """Record an error."""
        self.metrics.record_error(agent_type, error_type)

    def update_queue_size(self,
                       size: int,
                       agent_type: str = "general"):
        """Update queue size metric."""
        self.metrics.update_queue_size(agent_type, size) 