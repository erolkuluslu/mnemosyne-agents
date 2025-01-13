"""
Core Orchestrator module for coordinating agent interactions.
"""
from typing import Optional, Any, List, Dict, Coroutine
from collections import deque
from loguru import logger
from pydantic import BaseModel
import aisuite as ai
import os
import sys
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .vector_store import VectorStore, VectorStoreConfig


# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/mnemosyne_{time}.log",
    rotation="500 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    provider: str  # e.g., 'openai', 'anthropic', 'google'
    model: str    # e.g., 'gpt-4', 'claude-3'
    api_key: Optional[str] = None


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator."""
    debug: bool = False
    primary_provider: LLMProviderConfig = LLMProviderConfig(
        provider="openai",
        model="gpt-4"
    )
    fallback_providers: List[LLMProviderConfig] = []
    history_length: int = 10  # Number of conversation turns to remember in memory
    vector_store: Optional[VectorStoreConfig] = None  # Vector store config for persistent memory


class Message(BaseModel):
    """A single message in the conversation."""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime = datetime.now()


class Orchestrator:
    """
    Central orchestrator for managing agent workflows and interactions.
    Uses aisuite for flexible LLM provider routing.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the orchestrator with optional configuration."""
        self.config = config or OrchestratorConfig()
        
        # Set up logging based on debug mode
        if self.config.debug:
            logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
        
        # Log initialization with context
        with logger.contextualize(
            config=self.config.dict(),
            session_id=id(self)
        ):
            logger.info("Orchestrator initialized")
            logger.debug("Full configuration: {}", self.config)
        
        # Initialize aisuite client
        self.client = ai.Client()
        
        # Initialize conversation history
        self.conversation_history: deque[Message] = deque(maxlen=self.config.history_length)
        
        # Initialize thread pool for parallel execution
        self._thread_pool = ThreadPoolExecutor()
        
        # Initialize vector store if configured
        self.vector_store = VectorStore(self.config.vector_store) if self.config.vector_store else None
        if self.vector_store:
            logger.info("Vector store initialized for persistent memory")

    def _get_model_string(self, provider_config: LLMProviderConfig) -> str:
        """Convert provider config to aisuite model string format."""
        return f"{provider_config.provider}:{provider_config.model}"

    def _log_llm_request(self, messages: List[Dict[str, str]], provider_config: LLMProviderConfig):
        """Log LLM request details in debug mode."""
        if self.config.debug:
            with logger.contextualize(
                provider=provider_config.provider,
                model=provider_config.model,
                message_count=len(messages)
            ):
                logger.debug("LLM Request:")
                for idx, msg in enumerate(messages):
                    logger.debug(f"Message {idx}: {msg}")

    def _call_llm(self, messages: List[Dict[str, str]], provider_config: Optional[LLMProviderConfig] = None) -> str:
        """
        Make a call to the LLM and return its response.
        Attempts fallback providers if primary fails.
        """
        # Use primary provider if none specified
        provider_config = provider_config or self.config.primary_provider
        
        # Log request details in debug mode
        self._log_llm_request(messages, provider_config)
        
        try:
            with logger.contextualize(
                provider=provider_config.provider,
                model=provider_config.model
            ):
                # Try primary provider
                start_time = datetime.now()
                response = self.client.chat.completions.create(
                    model=self._get_model_string(provider_config),
                    messages=messages
                )
                duration = (datetime.now() - start_time).total_seconds()
                
                logger.info("LLM call successful. Duration: {:.2f}s", duration)
                if self.config.debug:
                    logger.debug("Raw response: {}", response)
                
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error("Error with provider {}: {}", provider_config.provider, str(e))
            
            # Try fallback providers if available
            for fallback in self.config.fallback_providers:
                try:
                    with logger.contextualize(
                        provider=fallback.provider,
                        model=fallback.model,
                        is_fallback=True
                    ):
                        logger.info("Attempting fallback provider")
                        start_time = datetime.now()
                        response = self.client.chat.completions.create(
                            model=self._get_model_string(fallback),
                            messages=messages
                        )
                        duration = (datetime.now() - start_time).total_seconds()
                        
                        logger.info("Fallback successful. Duration: {:.2f}s", duration)
                        return response.choices[0].message.content
                        
                except Exception as fallback_error:
                    logger.error("Fallback provider failed: {}", str(fallback_error))
            
            # If all providers fail, raise the original error
            raise

    async def run_parallel_tasks(self, 
                               tasks: List[Coroutine],
                               timeout: Optional[float] = None) -> List[Any]:
        """
        Run multiple async tasks in parallel and return their results.
        
        Args:
            tasks: List of coroutines to execute
            timeout: Optional timeout in seconds
            
        Returns:
            List of results from completed tasks
        """
        logger.debug("Running {} tasks in parallel", len(tasks))
        start_time = datetime.now()
        
        try:
            # Run tasks with optional timeout
            if timeout:
                results = await asyncio.gather(*tasks, timeout=timeout)
            else:
                results = await asyncio.gather(*tasks)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("Parallel execution completed in {:.2f}s", duration)
            
            return results
            
        except asyncio.TimeoutError:
            logger.warning("Parallel execution timed out after {}s", timeout)
            raise
        except Exception as e:
            logger.exception("Error during parallel execution")
            raise RuntimeError(f"Parallel execution failed: {str(e)}")

    async def process_input_async(self, user_input: str) -> dict[str, Any]:
        """
        Async version of process_input that supports parallel operations.
        Maintains conversation history and handles provider routing.
        """
        with logger.contextualize(
            session_id=id(self),
            history_length=len(self.conversation_history)
        ):
            logger.info("Processing new input (async)")
            logger.debug("User input: {}", user_input)
            
            try:
                # Add user message to history
                user_message = Message(role="user", content=user_input)
                self.conversation_history.append(user_message)
                
                # Store in vector store if available
                if self.vector_store:
                    await self.vector_store.add_message(
                        role="user",
                        content=user_input,
                        metadata={"session_id": id(self)}
                    )
                
                # Prepare messages for LLM including history
                messages = [{"role": msg.role, "content": msg.content} 
                          for msg in self.conversation_history]
                
                # Get response from LLM (with potential fallbacks)
                llm_response = await self._call_llm_async(messages)
                
                # Add assistant response to history
                assistant_message = Message(
                    role="assistant",
                    content=llm_response
                )
                self.conversation_history.append(assistant_message)
                
                # Store assistant response in vector store
                if self.vector_store:
                    await self.vector_store.add_message(
                        role="assistant",
                        content=llm_response,
                        metadata={
                            "session_id": id(self),
                            "provider": self.config.primary_provider.provider,
                            "model": self.config.primary_provider.model
                        }
                    )
                
                response = {
                    "status": "success",
                    "input_received": user_input,
                    "response": llm_response,
                    "provider_used": self.config.primary_provider.provider,
                    "model_used": self.config.primary_provider.model,
                    "history_length": len(self.conversation_history),
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info("Successfully processed input (async)")
                if self.config.debug:
                    logger.debug("Full response: {}", response)
                
                return response
                
            except Exception as e:
                logger.exception("Error processing input (async)")
                return {
                    "status": "error",
                    "input_received": user_input,
                    "error": str(e),
                    "provider_used": self.config.primary_provider.provider,
                    "model_used": self.config.primary_provider.model,
                    "timestamp": datetime.now().isoformat()
                }

    async def _call_llm_async(self, messages: List[Dict[str, str]], 
                             provider_config: Optional[LLMProviderConfig] = None) -> str:
        """
        Async version of _call_llm.
        Makes a call to the LLM and returns its response.
        Attempts fallback providers if primary fails.
        """
        # Use primary provider if none specified
        provider_config = provider_config or self.config.primary_provider
        
        # Log request details in debug mode
        self._log_llm_request(messages, provider_config)
        
        try:
            with logger.contextualize(
                provider=provider_config.provider,
                model=provider_config.model
            ):
                # Try primary provider
                start_time = datetime.now()
                
                # Run LLM call in thread pool to avoid blocking
                response = await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool,
                    lambda: self.client.chat.completions.create(
                        model=self._get_model_string(provider_config),
                        messages=messages
                    )
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                
                logger.info("LLM call successful (async). Duration: {:.2f}s", duration)
                if self.config.debug:
                    logger.debug("Raw response: {}", response)
                
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error("Error with provider {} (async): {}", provider_config.provider, str(e))
            
            # Try fallback providers if available
            for fallback in self.config.fallback_providers:
                try:
                    with logger.contextualize(
                        provider=fallback.provider,
                        model=fallback.model,
                        is_fallback=True
                    ):
                        logger.info("Attempting fallback provider (async)")
                        start_time = datetime.now()
                        
                        # Run fallback in thread pool
                        response = await asyncio.get_event_loop().run_in_executor(
                            self._thread_pool,
                            lambda: self.client.chat.completions.create(
                                model=self._get_model_string(fallback),
                                messages=messages
                            )
                        )
                        
                        duration = (datetime.now() - start_time).total_seconds()
                        
                        logger.info("Fallback successful (async). Duration: {:.2f}s", duration)
                        return response.choices[0].message.content
                        
                except Exception as fallback_error:
                    logger.error("Fallback provider failed (async): {}", str(fallback_error))
            
            # If all providers fail, raise the original error
            raise 

    async def get_conversation_history(self, 
                                     limit: Optional[int] = None,
                                     role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history from vector store if available, otherwise from memory.
        
        Args:
            limit: Maximum number of messages to return
            role_filter: Optional filter by role
            
        Returns:
            List of conversation messages
        """
        if self.vector_store:
            return await self.vector_store.get_recent_history(
                limit=limit or self.config.history_length,
                role_filter=role_filter
            )
        else:
            # Return from in-memory history
            history = list(self.conversation_history)
            if role_filter:
                history = [msg for msg in history if msg.role == role_filter]
            if limit:
                history = history[-limit:]
            return [msg.dict() for msg in history]

    async def search_similar_messages(self,
                                    query: str,
                                    limit: int = 5,
                                    role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar messages in conversation history.
        Only available if vector store is configured.
        
        Args:
            query: Text to search for
            limit: Maximum number of results
            role_filter: Optional filter by role
            
        Returns:
            List of similar messages with scores
        """
        if not self.vector_store:
            logger.warning("Vector store not configured, semantic search unavailable")
            return []
            
        return await self.vector_store.search_similar(
            query=query,
            limit=limit,
            role_filter=role_filter
        )

    async def clear_history(self):
        """Clear conversation history from both memory and vector store."""
        self.conversation_history.clear()
        if self.vector_store:
            await self.vector_store.clear_history()
        logger.info("Cleared conversation history") 