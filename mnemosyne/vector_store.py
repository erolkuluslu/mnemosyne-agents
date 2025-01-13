"""
Vector store module for persistent memory using LanceDB.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import lancedb
import os
from pathlib import Path
from loguru import logger
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store."""
    db_path: str = "data/vector_store"
    collection_name: str = "conversation_history"
    embedding_model: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # Default for all-MiniLM-L6-v2


class ConversationEntry(BaseModel):
    """Schema for conversation entries in the vector store."""
    id: str
    role: str
    content: str
    embedding: List[float]
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class VectorStore:
    """
    Vector store for persistent memory using LanceDB.
    Handles conversation history storage and retrieval.
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize the vector store with optional configuration."""
        self.config = config or VectorStoreConfig()
        
        # Create db directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
        
        # Initialize LanceDB
        self.db = lancedb.connect(self.config.db_path)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Initialize or get collection
        self._init_collection()
        
        logger.info(
            "VectorStore initialized with path: {} and collection: {}", 
            self.config.db_path,
            self.config.collection_name
        )

    def _init_collection(self):
        """Initialize or get the conversation history collection."""
        try:
            self.collection = self.db.open_table(self.config.collection_name)
            logger.debug("Opened existing collection: {}", self.config.collection_name)
        except Exception:
            logger.info("Creating new collection: {}", self.config.collection_name)
            schema = {
                "id": "string",
                "role": "string",
                "content": "string",
                "embedding": f"vector({self.config.dimension})",
                "timestamp": "timestamp",
                "metadata": "json"
            }
            self.collection = self.db.create_table(
                self.config.collection_name,
                schema=schema
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using sentence-transformers."""
        return self.embedding_model.encode(text).tolist()

    async def add_message(self, 
                         role: str, 
                         content: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata about the message
            
        Returns:
            ID of the added message
        """
        try:
            # Generate unique ID based on timestamp
            message_id = f"{role}_{datetime.now().isoformat()}"
            
            # Get embedding
            embedding = self._get_embedding(content)
            
            # Create entry
            entry = ConversationEntry(
                id=message_id,
                role=role,
                content=content,
                embedding=embedding,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            # Add to collection
            self.collection.add([entry.dict()])
            
            logger.debug("Added message to vector store. ID: {}", message_id)
            return message_id
            
        except Exception as e:
            logger.exception("Error adding message to vector store")
            raise RuntimeError(f"Failed to add message: {str(e)}")

    async def search_similar(self, 
                           query: str, 
                           limit: int = 5,
                           role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar messages in the conversation history.
        
        Args:
            query: Text to search for
            limit: Maximum number of results
            role_filter: Optional filter by role
            
        Returns:
            List of similar messages with scores
        """
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Build search query
            search = self.collection.search(query_embedding)
            if role_filter:
                search = search.where(f"role = '{role_filter}'")
            
            # Execute search
            results = search.limit(limit).to_list()
            
            logger.debug("Found {} similar messages", len(results))
            return results
            
        except Exception as e:
            logger.exception("Error searching vector store")
            raise RuntimeError(f"Search failed: {str(e)}")

    async def get_recent_history(self, 
                               limit: int = 10,
                               role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the most recent conversation history.
        
        Args:
            limit: Maximum number of messages to return
            role_filter: Optional filter by role
            
        Returns:
            List of recent messages
        """
        try:
            # Build query
            query = self.collection
            if role_filter:
                query = query.where(f"role = '{role_filter}'")
            
            # Execute query
            results = query.order_by("timestamp", "desc").limit(limit).to_list()
            
            logger.debug("Retrieved {} recent messages", len(results))
            return results
            
        except Exception as e:
            logger.exception("Error retrieving history")
            raise RuntimeError(f"History retrieval failed: {str(e)}")

    async def clear_history(self):
        """Clear all conversation history."""
        try:
            self.db.drop_table(self.config.collection_name)
            self._init_collection()
            logger.info("Cleared conversation history")
            
        except Exception as e:
            logger.exception("Error clearing history")
            raise RuntimeError(f"Failed to clear history: {str(e)}") 