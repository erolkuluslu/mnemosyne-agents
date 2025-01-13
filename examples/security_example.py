"""
Example demonstrating security features including encryption and retry mechanisms.
"""
from mnemosyne.security import Security, SecurityConfig
from mnemosyne.orchestrator import (
    Orchestrator, 
    OrchestratorConfig, 
    LLMProviderConfig
)
from loguru import logger
import asyncio
import sys
from datetime import datetime


class SimulatedFailure(Exception):
    """Simulated failure for testing retries."""
    pass


class ExampleAgent:
    """Example agent to demonstrate retry mechanism."""
    
    def __init__(self, security: Security):
        self.security = security
        self.fail_count = 0
    
    @Security.with_retries(max_tries=3, initial_wait=1.0, max_wait=5.0)
    async def flaky_operation(self) -> str:
        """Simulated operation that sometimes fails."""
        self.fail_count += 1
        
        if self.fail_count <= 2:  # Fail first two attempts
            logger.warning("Operation failed, attempt {}", self.fail_count)
            raise SimulatedFailure("Simulated failure")
            
        return "Operation succeeded on attempt 3!"


async def main():
    # Initialize security with custom config
    security = Security(
        config=SecurityConfig(
            max_retries=3,
            initial_wait=1.0,
            max_wait=30.0
        )
    )
    
    try:
        # Example 1: Data Encryption
        logger.info("\nTesting data encryption...")
        
        # Sample data to encrypt
        sensitive_data = {
            "user_id": "12345",
            "timestamp": datetime.now().isoformat(),
            "ocr_results": "Confidential document text...",
            "audio_transcript": "Sensitive conversation content..."
        }
        
        # Encrypt data
        encrypted = security.encrypt_data(sensitive_data)
        logger.info("Data encrypted: {} bytes", len(encrypted))
        
        # Decrypt data
        decrypted = security.decrypt_data(encrypted)
        logger.info("Data decrypted successfully")
        logger.debug("Decrypted content: {}", decrypted)
        
        # Verify data integrity
        assert decrypted == sensitive_data
        logger.info("Data integrity verified")
        
        # Example 2: Retry Mechanism
        logger.info("\nTesting retry mechanism...")
        
        # Create example agent
        agent = ExampleAgent(security)
        
        # Try flaky operation
        result = await agent.flaky_operation()
        logger.info("Final result: {}", result)
        
    except Exception as e:
        logger.exception("Error during example")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 