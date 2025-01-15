"""
Example demonstrating the Orchestrator-Workers pattern.
"""
import asyncio
from loguru import logger
from mnemosyne.workers import (
    WorkerOrchestrator,
    WorkerType,
    WorkerStatus
)


async def text_processor(content: dict) -> str:
    """Example text processing function."""
    # Simulate processing time
    await asyncio.sleep(0.5)
    text = content.get("text", "")
    return f"Processed text (len={len(text)}): {text[:50]}..."


async def image_processor(content: dict) -> dict:
    """Example image processing function."""
    # Simulate processing time
    await asyncio.sleep(1.0)
    image_path = content.get("path", "")
    return {
        "path": image_path,
        "dimensions": "800x600",
        "format": "JPEG"
    }


async def main():
    # Initialize orchestrator
    orchestrator = WorkerOrchestrator()
    
    # Example 1: Parallel Text Processing
    logger.info("Running parallel text processing example...")
    
    # Create text processing tasks
    text_tasks = [
        {"text": f"Sample text {i} for processing"} 
        for i in range(5)
    ]
    
    # Process in parallel with 3 workers
    text_results = await orchestrator.process_parallel(
        tasks=text_tasks,
        worker_type=WorkerType.TEXT,
        processor=text_processor,
        num_workers=3
    )
    
    logger.info("Text processing results: {}", text_results)
    
    # Example 2: Sequential Image Processing
    logger.info("Running sequential image processing example...")
    
    # Create image processing tasks
    image_tasks = [
        {"path": f"image_{i}.jpg"}
        for i in range(3)
    ]
    
    # Process sequentially
    image_results = await orchestrator.process_sequential(
        tasks=image_tasks,
        worker_type=WorkerType.IMAGE,
        processor=image_processor
    )
    
    logger.info("Image processing results: {}", image_results)
    
    # Example 3: Mixed Workload with Dependencies
    logger.info("Running mixed workload example...")
    
    # Create a worker pool with different types
    orchestrator.create_worker_group(
        type=WorkerType.TEXT,
        count=2,
        capabilities=["summarize", "analyze"]
    )
    orchestrator.create_worker_group(
        type=WorkerType.IMAGE,
        count=2,
        capabilities=["resize", "format"]
    )
    
    # Add tasks with dependencies
    text_task_id = orchestrator.pool.add_task(
        type=WorkerType.TEXT,
        content={"text": "Text to process before image"}
    )
    
    # Image task depends on text task
    image_task_id = orchestrator.pool.add_task(
        type=WorkerType.IMAGE,
        content={"path": "dependent_image.jpg"},
        dependencies=[text_task_id]
    )
    
    # Process text task
    text_result = await orchestrator.pool.process_task(
        worker_id=next(
            w_id for w_id, w in orchestrator.pool.workers.items()
            if w.type == WorkerType.TEXT and w.status == WorkerStatus.IDLE
        ),
        task_id=text_task_id,
        processor=text_processor
    )
    logger.info("Text task result: {}", text_result)
    
    # Process image task (will wait for text task)
    image_result = await orchestrator.pool.process_task(
        worker_id=next(
            w_id for w_id, w in orchestrator.pool.workers.items()
            if w.type == WorkerType.IMAGE and w.status == WorkerStatus.IDLE
        ),
        task_id=image_task_id,
        processor=image_processor
    )
    logger.info("Image task result: {}", image_result)


if __name__ == "__main__":
    asyncio.run(main()) 