"""
Caching module for embeddings and LLM responses.

Key Components:
1. Configuration Classes:
   - CacheConfig: Defines caching parameters such as TTL and maximum sizes.
   - RateLimitConfig: Sets up rate limiting for user and global requests.

2. CacheEntry Class: Represents a single cache entry with data and expiration time.

3. Cache Class: Implements the caching mechanism, allowing for storing and retrieving data based on generated keys, 
   enforcing size limits, and removing the oldest entries when necessary.

4. RateLimiter Class: Manages request limits for users and globally, ensuring the system is not overwhelmed by too many requests.

5. CacheManager Class: A higher-level interface for managing caching and rate limiting, integrating both functionalities 
   into the overall workflow of the platform.

6. Caching Decorators: Provides decorators for caching embeddings and LLM responses automatically, enhancing performance 
   without modifying core function logic.

Usage:
- Create an instance of CacheManager to manage caching and rate limiting.
- Decorate functions with caching decorators to automatically handle caching.

Capabilities:
- Reduces redundant computations and API calls, improving efficiency.
- Provides control over request limits, ensuring fair usage.
- Essential for maintaining responsiveness and reliability under heavy load.

This module is integral to building efficient and responsive AI applications across different modalities.
"""
from typing import Any, Optional, Dict, List
from pydantic import BaseModel
from loguru import logger
import hashlib
import json
import time
from datetime import datetime, timedelta
import asyncio
from functools import wraps
import numpy as np


class CacheConfig(BaseModel):
    """Configuration for caching."""
    embedding_ttl: int = 3600  # Time to live for embeddings in seconds
    llm_response_ttl: int = 1800  # Time to live for LLM responses in seconds
    max_embedding_size: int = 10000  # Maximum number of embeddings to cache
    max_llm_size: int = 1000  # Maximum number of LLM responses to cache


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""
    user_limit: int = 100  # Requests per user per minute
    global_limit: int = 1000  # Total requests per minute
    embedding_limit: int = 500  # Embedding generations per minute
    llm_limit: int = 200  # LLM calls per minute


class CacheEntry(BaseModel):
    """A single cache entry."""
    data: Any
    expires_at: datetime


class Cache:
    """
    Cache implementation with TTL and size limits.
    """
    def __init__(self, max_size: int, ttl: int):
        """Initialize cache with size and TTL limits."""
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
        logger.info("Cache initialized with max_size={}, ttl={}s", max_size, ttl)

    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, (str, bytes)):
            data = json.dumps(data, sort_keys=True).encode()
            
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        entry = self._cache.get(key)
        if not entry:
            return None
            
        if datetime.now() > entry.expires_at:
            del self._cache[key]
            return None
            
        return entry.data

    def set(self, key: str, value: Any):
        """Set item in cache with TTL."""
        # Remove oldest items if cache is full
        while len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].expires_at)
            del self._cache[oldest_key]
        
        self._cache[key] = CacheEntry(
            data=value,
            expires_at=datetime.now() + timedelta(seconds=self.ttl)
        )


class RateLimiter:
    """
    Rate limiter implementation with per-user and global limits.
    """
    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with config."""
        self.config = config
        self._user_requests: Dict[str, List[datetime]] = {}
        self._global_requests: List[datetime] = []
        self._embedding_requests: List[datetime] = []
        self._llm_requests: List[datetime] = []
        logger.info("Rate limiter initialized with config: {}", config)

    def _clean_old_requests(self, requests: List[datetime]):
        """Remove requests older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        while requests and requests[0] < cutoff:
            requests.pop(0)

    async def check_rate_limit(self, 
                             user_id: Optional[str] = None, 
                             limit_type: str = "global") -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            user_id: Optional user ID for per-user limits
            limit_type: Type of limit to check (global/embedding/llm)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        now = datetime.now()
        
        # Check user limit if user_id provided
        if user_id:
            if user_id not in self._user_requests:
                self._user_requests[user_id] = []
            
            user_requests = self._user_requests[user_id]
            self._clean_old_requests(user_requests)
            
            if len(user_requests) >= self.config.user_limit:
                logger.warning("User {} rate limited", user_id)
                return False
            
            user_requests.append(now)
        
        # Check type-specific limit
        if limit_type == "embedding":
            self._clean_old_requests(self._embedding_requests)
            if len(self._embedding_requests) >= self.config.embedding_limit:
                logger.warning("Embedding rate limit reached")
                return False
            self._embedding_requests.append(now)
            
        elif limit_type == "llm":
            self._clean_old_requests(self._llm_requests)
            if len(self._llm_requests) >= self.config.llm_limit:
                logger.warning("LLM rate limit reached")
                return False
            self._llm_requests.append(now)
        
        # Check global limit
        self._clean_old_requests(self._global_requests)
        if len(self._global_requests) >= self.config.global_limit:
            logger.warning("Global rate limit reached")
            return False
            
        self._global_requests.append(now)
        return True


class CacheManager:
    """
    Manages caching and rate limiting for the platform.
    """
    def __init__(self, 
                 cache_config: Optional[CacheConfig] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """Initialize cache manager."""
        self.cache_config = cache_config or CacheConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        
        # Initialize caches
        self.embedding_cache = Cache(
            max_size=self.cache_config.max_embedding_size,
            ttl=self.cache_config.embedding_ttl
        )
        self.llm_cache = Cache(
            max_size=self.cache_config.max_llm_size,
            ttl=self.cache_config.llm_response_ttl
        )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(self.rate_limit_config)
        
        logger.info("Cache manager initialized")

    def cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding vector."""
        key = self.embedding_cache._generate_key(text)
        self.embedding_cache.set(key, embedding)

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        key = self.embedding_cache._generate_key(text)
        return self.embedding_cache.get(key)

    def cache_llm_response(self, prompt: str, response: str) -> None:
        """Cache an LLM response."""
        key = self.llm_cache._generate_key(prompt)
        self.llm_cache.set(key, response)

    def get_cached_llm_response(self, prompt: str) -> Optional[str]:
        """Get cached LLM response if available."""
        key = self.llm_cache._generate_key(prompt)
        return self.llm_cache.get(key)

    async def check_rate_limit(self, 
                             user_id: Optional[str] = None,
                             limit_type: str = "global") -> bool:
        """Check if request is within rate limits."""
        return await self.rate_limiter.check_rate_limit(user_id, limit_type)


# Decorator for caching embeddings
def cache_embedding(cache_manager: CacheManager):
    """Decorator to cache embedding results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(text: str, *args, **kwargs):
            # Check cache first
            cached = cache_manager.get_cached_embedding(text)
            if cached is not None:
                logger.debug("Using cached embedding for: {}", text[:50])
                return cached
            
            # Check rate limit
            if not await cache_manager.check_rate_limit(limit_type="embedding"):
                logger.warning("Rate limit reached for embedding generation")
                raise RuntimeError("Rate limit exceeded for embedding generation")
            
            # Generate and cache embedding
            embedding = await func(text, *args, **kwargs)
            cache_manager.cache_embedding(text, embedding)
            return embedding
        return wrapper
    return decorator


# Decorator for caching LLM responses
def cache_llm_response(cache_manager: CacheManager):
    """Decorator to cache LLM responses."""
    def decorator(func):
        @wraps(func)
        async def wrapper(prompt: str, *args, **kwargs):
            # Check cache first
            cached = cache_manager.get_cached_llm_response(prompt)
            if cached is not None:
                logger.debug("Using cached LLM response for: {}", prompt[:50])
                return cached
            
            # Check rate limit
            if not await cache_manager.check_rate_limit(limit_type="llm"):
                logger.warning("Rate limit reached for LLM calls")
                raise RuntimeError("Rate limit exceeded for LLM calls")
            
            # Generate and cache response
            response = await func(prompt, *args, **kwargs)
            cache_manager.cache_llm_response(prompt, response)
            return response
        return wrapper
    return decorator 