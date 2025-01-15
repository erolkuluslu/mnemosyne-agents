"""
Example demonstrating caching and rate limiting features.
"""
from mnemosyne.caching import (
    CacheManager,
    CacheConfig,
    RateLimitConfig,
    cache_embedding,
    cache_llm_response
)
from loguru import logger
import asyncio
import numpy as np
from datetime import datetime
import sys


class ExampleProcessor:
    """Example processor to demonstrate caching and rate limiting."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    @cache_embedding(cache_manager=lambda self: self.cache_manager)
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Simulate embedding generation."""
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Generate fake embedding
        return np.random.rand(384)  # Same dimension as our vector store
    
    @cache_llm_response(cache_manager=lambda self: self.cache_manager)
    async def generate_response(self, prompt: str) -> str:
        """Simulate LLM response generation."""
        # Simulate processing time
        await asyncio.sleep(1.0)
        
        # Generate fake response
        return f"Response to: {prompt} at {datetime.now().isoformat()}"


async def main():
    # Initialize cache manager with custom config
    cache_manager = CacheManager(
        cache_config=CacheConfig(
            embedding_ttl=60,  # 1 minute for testing
            llm_response_ttl=30,  # 30 seconds for testing
            max_embedding_size=5,
            max_llm_size=3
        ),
        rate_limit_config=RateLimitConfig(
            user_limit=5,  # 5 requests per user per minute
            global_limit=10,  # 10 total requests per minute
            embedding_limit=8,  # 8 embeddings per minute
            llm_limit=6  # 6 LLM calls per minute
        )
    )
    
    # Initialize processor
    processor = ExampleProcessor(cache_manager)
    
    try:
        # Example 1: Embedding Caching
        logger.info("\nTesting embedding caching...")
        
        # Generate embedding for same text multiple times
        text = "This is a test text for embedding."
        
        logger.info("First call - should generate new embedding")
        embedding1 = await processor.generate_embedding(text)
        
        logger.info("Second call - should use cached embedding")
        embedding2 = await processor.generate_embedding(text)
        
        # Verify embeddings are identical
        assert np.array_equal(embedding1, embedding2)
        logger.info("Embedding cache working correctly")
        
        # Example 2: LLM Response Caching
        logger.info("\nTesting LLM response caching...")
        
        # Generate response for same prompt multiple times
        prompt = "What is the meaning of life?"
        
        logger.info("First call - should generate new response")
        response1 = await processor.generate_response(prompt)
        
        logger.info("Second call - should use cached response")
        response2 = await processor.generate_response(prompt)
        
        # Verify responses are identical
        assert response1 == response2
        logger.info("LLM cache working correctly")
        
        # Example 3: Rate Limiting
        logger.info("\nTesting rate limiting...")
        
        # Test user rate limiting
        user_id = "test_user"
        logger.info("Testing user rate limiting...")
        
        for i in range(7):  # Try more than the limit
            try:
                if await cache_manager.check_rate_limit(user_id=user_id):
                    logger.info("Request {} allowed", i + 1)
                else:
                    logger.warning("Request {} rate limited", i + 1)
            except Exception as e:
                logger.error("Request {} failed: {}", i + 1, str(e))
        
        # Test embedding rate limiting
        logger.info("\nTesting embedding rate limiting...")
        texts = [f"Text {i}" for i in range(10)]
        
        for i, text in enumerate(texts):
            try:
                embedding = await processor.generate_embedding(text)
                logger.info("Embedding {} generated", i + 1)
            except RuntimeError as e:
                logger.warning("Embedding {} rate limited: {}", i + 1, str(e))
        
        logger.info("Rate limiting tests completed")
        
    except Exception as e:
        logger.exception("Error during example")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 