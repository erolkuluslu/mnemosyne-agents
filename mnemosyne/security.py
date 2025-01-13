"""
Security module for encryption and retry mechanisms.
"""
from typing import TypeVar, Callable, Any, Optional
from pydantic import BaseModel
from loguru import logger
import asyncio
from datetime import datetime
from cryptography.fernet import Fernet
import base64
import os
import json
from functools import wraps
import backoff


T = TypeVar('T')  # Generic type for retry decorator


class SecurityConfig(BaseModel):
    """Configuration for security features."""
    encryption_key: Optional[str] = None  # If not provided, will generate one
    max_retries: int = 3
    initial_wait: float = 1.0  # Initial wait time in seconds
    max_wait: float = 30.0  # Maximum wait time in seconds
    timeout: float = 10.0  # Default timeout in seconds


class Security:
    """
    Handles encryption and retry mechanisms.
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security with optional configuration."""
        self.config = config or SecurityConfig()
        
        # Initialize encryption key
        if not self.config.encryption_key:
            self.config.encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            logger.info("Generated new encryption key")
        
        self.fernet = Fernet(self.config.encryption_key.encode())
        logger.info("Security module initialized")

    def encrypt_data(self, data: Any) -> bytes:
        """
        Encrypt any serializable data.
        
        Args:
            data: Data to encrypt (must be JSON serializable)
            
        Returns:
            Encrypted bytes
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(data)
            
            # Encrypt
            encrypted = self.fernet.encrypt(json_data.encode())
            
            logger.debug("Data encrypted successfully")
            return encrypted
            
        except Exception as e:
            logger.exception("Encryption failed")
            raise RuntimeError(f"Failed to encrypt data: {str(e)}")

    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """
        Decrypt data back to its original form.
        
        Args:
            encrypted_data: Data to decrypt
            
        Returns:
            Decrypted data in its original form
        """
        try:
            # Decrypt
            decrypted = self.fernet.decrypt(encrypted_data)
            
            # Parse JSON
            data = json.loads(decrypted.decode())
            
            logger.debug("Data decrypted successfully")
            return data
            
        except Exception as e:
            logger.exception("Decryption failed")
            raise RuntimeError(f"Failed to decrypt data: {str(e)}")

    @staticmethod
    def with_retries(
        max_tries: Optional[int] = None,
        initial_wait: Optional[float] = None,
        max_wait: Optional[float] = None
    ):
        """
        Decorator for adding exponential backoff retry logic to functions.
        
        Args:
            max_tries: Maximum number of retry attempts
            initial_wait: Initial wait time between retries in seconds
            max_wait: Maximum wait time between retries in seconds
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            @backoff.on_exception(
                backoff.expo,
                Exception,
                max_tries=max_tries,
                max_time=max_wait,
                base=initial_wait
            )
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        "Attempt failed for {}: {}. Retrying...", 
                        func.__name__, 
                        str(e)
                    )
                    raise
            return wrapper
        return decorator


# Convenience function to get a configured security instance
def get_security(config: Optional[SecurityConfig] = None) -> Security:
    """Get a configured security instance."""
    return Security(config) 