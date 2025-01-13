"""
Example demonstrating vector store integration for persistent memory.
Shows conversation history storage and semantic search capabilities.
"""
from mnemosyne.orchestrator import (
    Orchestrator, 
    OrchestratorConfig, 
    LLMProviderConfig
)
from mnemosyne.vector_store import VectorStoreConfig
from loguru import logger
import asyncio
import sys


async def main():
    # Initialize orchestrator with vector store
    orchestrator = Orchestrator(
        config=OrchestratorConfig(
            debug=True,
            primary_provider=LLMProviderConfig(
                provider="openai",
                model="gpt-4"
            ),
            vector_store=VectorStoreConfig(
                db_path="data/example_store",
                collection_name="example_history"
            )
        )
    )
    
    try:
        # Example conversation
        logger.info("Starting conversation...")
        
        # First message about Python
        response1 = await orchestrator.process_input_async(
            "What are the key features of Python?"
        )
        logger.info("Response 1: {}", response1["response"])
        
        # Second message about a different topic
        response2 = await orchestrator.process_input_async(
            "Tell me about machine learning."
        )
        logger.info("Response 2: {}", response2["response"])
        
        # Third message about Python again
        response3 = await orchestrator.process_input_async(
            "How does Python handle memory management?"
        )
        logger.info("Response 3: {}", response3["response"])
        
        # Demonstrate semantic search
        logger.info("\nSearching for messages about Python...")
        python_messages = await orchestrator.search_similar_messages(
            query="Python programming language features",
            limit=2
        )
        
        logger.info("Found {} relevant messages:", len(python_messages))
        for msg in python_messages:
            logger.info("- Role: {}, Content: {}", msg["role"], msg["content"])
        
        # Get recent history
        logger.info("\nGetting recent history...")
        history = await orchestrator.get_conversation_history(limit=5)
        
        logger.info("Recent conversation:")
        for msg in history:
            logger.info("- {}: {}", msg["role"], msg["content"])
        
        # Filter by role
        logger.info("\nGetting only assistant responses...")
        assistant_msgs = await orchestrator.get_conversation_history(
            role_filter="assistant"
        )
        
        logger.info("Assistant messages:")
        for msg in assistant_msgs:
            logger.info("- {}", msg["content"])
            
        # Clear history
        logger.info("\nClearing conversation history...")
        await orchestrator.clear_history()
        
        # Verify it's cleared
        empty_history = await orchestrator.get_conversation_history()
        logger.info("History after clearing: {} messages", len(empty_history))
        
    except Exception as e:
        logger.exception("Error during example")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 