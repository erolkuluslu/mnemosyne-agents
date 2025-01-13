"""
Basic example demonstrating Orchestrator usage with multiple LLM providers.
Shows enhanced logging and debugging features.
"""
from loguru import logger
from mnemosyne.orchestrator import Orchestrator, OrchestratorConfig, LLMProviderConfig
import os
from dotenv import load_dotenv
import sys

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize the orchestrator with multiple providers and debug mode
    orchestrator = Orchestrator(
        config=OrchestratorConfig(
            debug=True,  # Enable detailed logging
            history_length=5,
            # Configure primary provider
            primary_provider=LLMProviderConfig(
                provider="openai",
                model="gpt-4"
            ),
            # Configure fallback providers
            fallback_providers=[
                LLMProviderConfig(
                    provider="anthropic",
                    model="claude-3-sonnet-20240229"
                ),
                LLMProviderConfig(
                    provider="google",
                    model="gemini-pro"
                )
            ]
        )
    )
    
    logger.info("Starting conversation with enhanced logging...")
    
    try:
        # First question - will try primary provider first
        with logger.contextualize(question_number=1):
            question1 = "What are the three laws of robotics?"
            logger.info("Asking question: {}", question1)
            response1 = orchestrator.process_input(question1)
            logger.info("Received response from {}: {}", 
                       response1["provider_used"], 
                       response1["response"])
        
        # Follow-up question (demonstrates memory and potentially fallback providers)
        with logger.contextualize(question_number=2):
            question2 = "Who created these laws?"
            logger.info("Asking follow-up: {}", question2)
            response2 = orchestrator.process_input(question2)
            logger.info("Received response from {}: {}", 
                       response2["provider_used"], 
                       response2["response"])
            
        # Log conversation summary
        logger.info("Conversation completed successfully")
        logger.debug("Final history length: {}", response2["history_length"])
        
    except Exception as e:
        logger.exception("Error during conversation")
        sys.exit(1)

if __name__ == "__main__":
    main() 