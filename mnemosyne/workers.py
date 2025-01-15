"""
Workers module for handling distributed task processing.
"""
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field
from loguru import logger
import asyncio
from datetime import datetime
import uuid
from enum import Enum


class WorkerType(str, Enum):
    """Types of worker agents."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    GENERAL = "general"


class WorkerStatus(str, Enum):
    """Status of a worker agent."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"


class Task(BaseModel):
    """A task to be processed by a worker."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: WorkerType
    content: Dict[str, Any]  # Task-specific data
    priority: int = 0  # Higher number = higher priority
    dependencies: List[str] = []  # IDs of tasks that must complete first
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class WorkerAgent(BaseModel):
    """A worker agent that processes tasks."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: WorkerType
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: Optional[str] = None  # ID of current task
    capabilities: List[str] = []  # Specific operations this worker can handle
    metadata: Dict[str, Any] = {}


class WorkerPool:
    """
    Manages a pool of worker agents and task distribution.
    """
    
    def __init__(self):
        """Initialize an empty worker pool."""
        self.workers: Dict[str, WorkerAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []  # IDs of pending tasks
        self.results: Dict[str, Any] = {}  # Task results
        logger.info("Initialized worker pool")

    def add_worker(self, 
                  type: WorkerType,
                  capabilities: Optional[List[str]] = None) -> str:
        """
        Add a new worker to the pool.
        
        Args:
            type: Type of worker
            capabilities: List of specific operations this worker can handle
            
        Returns:
            ID of the new worker
        """
        worker = WorkerAgent(
            type=type,
            capabilities=capabilities or []
        )
        self.workers[worker.id] = worker
        logger.info("Added {} worker {} with capabilities: {}", 
                   type, worker.id, capabilities)
        return worker.id

    def add_task(self,
                type: WorkerType,
                content: Dict[str, Any],
                priority: int = 0,
                dependencies: Optional[List[str]] = None) -> str:
        """
        Add a new task to the pool.
        
        Args:
            type: Type of task
            content: Task-specific data
            priority: Task priority (higher = more important)
            dependencies: IDs of tasks that must complete first
            
        Returns:
            ID of the new task
        """
        task = Task(
            type=type,
            content=content,
            priority=priority,
            dependencies=dependencies or []
        )
        self.tasks[task.id] = task
        self._update_queue()
        logger.info("Added {} task {} with priority {}", 
                   type, task.id, priority)
        return task.id

    def _update_queue(self):
        """Update the task queue based on priorities and dependencies."""
        # Get all pending tasks
        pending = [
            task for task in self.tasks.values()
            if not task.completed_at and not task.started_at
        ]
        
        # Filter out tasks with incomplete dependencies
        ready = [
            task for task in pending
            if all(dep in self.results for dep in task.dependencies)
        ]
        
        # Sort by priority (descending) and creation time
        sorted_tasks = sorted(
            ready,
            key=lambda t: (-t.priority, t.created_at)
        )
        
        # Update queue with task IDs
        self.task_queue = [task.id for task in sorted_tasks]

    def get_next_task(self, worker_id: str) -> Optional[Task]:
        """
        Get the next task for a worker to process.
        
        Args:
            worker_id: ID of the worker requesting a task
            
        Returns:
            Next task to process, or None if no suitable tasks
        """
        if worker_id not in self.workers:
            raise ValueError(f"Unknown worker: {worker_id}")
            
        worker = self.workers[worker_id]
        if worker.status != WorkerStatus.IDLE:
            return None
            
        # Find the first task matching worker type and capabilities
        for task_id in self.task_queue:
            task = self.tasks[task_id]
            if task.type == worker.type:
                return task
                
        return None

    async def process_task(self,
                        worker_id: str,
                        task_id: str,
                        processor: Callable) -> Any:
        """
        Process a task using the specified worker.
        
        Args:
            worker_id: ID of worker processing the task
            task_id: ID of task to process
            processor: Async function to process the task
            
        Returns:
            Task result
        """
        if worker_id not in self.workers:
            raise ValueError(f"Unknown worker: {worker_id}")
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")
            
        worker = self.workers[worker_id]
        task = self.tasks[task_id]
        
        try:
            # Update status
            worker.status = WorkerStatus.BUSY
            worker.current_task = task_id
            task.started_at = datetime.now()
            
            # Process task
            logger.info("Worker {} processing task {}", worker_id, task_id)
            result = await processor(task.content)
            
            # Store result
            self.results[task_id] = result
            task.completed_at = datetime.now()
            
            # Update queue
            self._update_queue()
            
            return result
            
        except Exception as e:
            logger.exception("Error processing task {}", task_id)
            task.error = str(e)
            worker.status = WorkerStatus.ERROR
            raise
            
        finally:
            worker.status = WorkerStatus.IDLE
            worker.current_task = None


class WorkerOrchestrator:
    """
    Orchestrates task distribution and parallel processing across workers.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.pool = WorkerPool()
        logger.info("Initialized worker orchestrator")

    def create_worker_group(self,
                          type: WorkerType,
                          count: int,
                          capabilities: Optional[List[str]] = None) -> List[str]:
        """
        Create a group of workers with the same type and capabilities.
        
        Args:
            type: Type of workers to create
            count: Number of workers to create
            capabilities: List of capabilities for the workers
            
        Returns:
            List of worker IDs
        """
        worker_ids = []
        for _ in range(count):
            worker_id = self.pool.add_worker(type, capabilities)
            worker_ids.append(worker_id)
        return worker_ids

    async def process_parallel(self,
                           tasks: List[Dict[str, Any]],
                           worker_type: WorkerType,
                           processor: Callable,
                           num_workers: int = 3) -> Dict[str, Any]:
        """
        Process multiple tasks in parallel using a pool of workers.
        
        Args:
            tasks: List of task content dictionaries
            worker_type: Type of workers to use
            processor: Async function to process each task
            num_workers: Number of workers to create
            
        Returns:
            Dictionary mapping task IDs to results
        """
        # Create workers if needed
        if not any(w.type == worker_type for w in self.pool.workers.values()):
            self.create_worker_group(worker_type, num_workers)
        
        # Add tasks to pool
        task_ids = []
        for task_content in tasks:
            task_id = self.pool.add_task(
                type=worker_type,
                content=task_content
            )
            task_ids.append(task_id)
        
        # Process tasks in parallel
        async def worker_loop(worker_id: str):
            while True:
                task = self.pool.get_next_task(worker_id)
                if not task:
                    # No more tasks for this worker
                    break
                    
                await self.pool.process_task(
                    worker_id,
                    task.id,
                    processor
                )
        
        # Start worker tasks
        worker_tasks = []
        for worker_id in self.pool.workers:
            if self.pool.workers[worker_id].type == worker_type:
                worker_tasks.append(
                    asyncio.create_task(worker_loop(worker_id))
                )
        
        # Wait for all tasks to complete
        await asyncio.gather(*worker_tasks)
        
        # Return results for requested tasks
        return {
            task_id: self.pool.results[task_id]
            for task_id in task_ids
            if task_id in self.pool.results
        }

    async def process_sequential(self,
                             tasks: List[Dict[str, Any]],
                             worker_type: WorkerType,
                             processor: Callable) -> List[Any]:
        """
        Process tasks sequentially using a single worker.
        
        Args:
            tasks: List of task content dictionaries
            worker_type: Type of worker to use
            processor: Async function to process each task
            
        Returns:
            List of task results in order
        """
        # Create a worker
        worker_id = self.pool.add_worker(worker_type)
        
        results = []
        for task_content in tasks:
            # Add and process task
            task_id = self.pool.add_task(
                type=worker_type,
                content=task_content
            )
            
            result = await self.pool.process_task(
                worker_id,
                task_id,
                processor
            )
            results.append(result)
        
        return results 