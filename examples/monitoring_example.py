"""
Example demonstrating performance monitoring features.
"""
import asyncio
from loguru import logger
from mnemosyne.monitoring import (
    PerformanceMetrics,
    MonitoredComponent,
    MetricType
)
import random
import time


class ExampleProcessor(MonitoredComponent):
    """Example component that processes tasks."""
    
    def __init__(self, metrics: PerformanceMetrics):
        """Initialize processor."""
        super().__init__(metrics)
        self.queue = []
    
    async def process_task(self, task_id: int):
        """Process a single task with monitoring."""
        async with self.track_operation(
            operation="process_task",
            agent_type="processor"
        ):
            # Simulate processing
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Randomly fail some tasks
            if random.random() < 0.2:
                raise ValueError("Random task failure")
            
            return f"Processed task {task_id}"
    
    async def process_batch(self, batch: list):
        """Process a batch of tasks with worker tracking."""
        async with self.track_worker(agent_type="processor"):
            results = []
            for item in batch:
                try:
                    result = await self.process_task(item)
                    results.append(result)
                except Exception as e:
                    self.record_error(
                        error_type=type(e).__name__,
                        agent_type="processor"
                    )
            return results
    
    def update_queue(self, items: list):
        """Update queue with monitoring."""
        self.queue.extend(items)
        self.update_queue_size(
            size=len(self.queue),
            agent_type="processor"
        )


async def main():
    # Initialize metrics
    metrics = PerformanceMetrics(port=8000)
    logger.info("Started metrics server on :8000")
    
    # Create processor
    processor = ExampleProcessor(metrics)
    
    # Example 1: Process individual tasks
    logger.info("Running individual task processing example...")
    
    for i in range(5):
        try:
            result = await processor.process_task(i)
            logger.info("Task {} result: {}", i, result)
        except Exception as e:
            logger.error("Task {} failed: {}", i, str(e))
    
    # Example 2: Process batches with workers
    logger.info("\nRunning batch processing example...")
    
    batches = [
        list(range(5)),
        list(range(5, 10)),
        list(range(10, 15))
    ]
    
    for i, batch in enumerate(batches):
        processor.update_queue(batch)
        logger.info("Processing batch {} (queue size: {})", i + 1, len(batch))
        
        results = await processor.process_batch(batch)
        logger.info("Batch {} results: {}", i + 1, results)
        
        # Clear processed items from queue
        processor.queue = processor.queue[len(batch):]
        processor.update_queue_size(
            size=len(processor.queue),
            agent_type="processor"
        )
    
    # Example 3: Get metrics summary
    logger.info("\nGetting metrics summary...")
    
    summary = metrics.get_summary()
    logger.info("Metrics summary:\n{}", summary)
    
    # Keep server running to allow viewing metrics
    logger.info("\nMetrics server running on :8000")
    logger.info("Visit http://localhost:8000/metrics to view Prometheus metrics")
    logger.info("Press Ctrl+C to exit")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main()) 