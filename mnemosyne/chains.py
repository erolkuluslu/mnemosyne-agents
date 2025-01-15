"""
Prompt chaining module for defining and executing multi-step workflows.
"""
from typing import Any, Dict, List, Optional, Union, Callable, Set
from pydantic import BaseModel, Field
from loguru import logger
import asyncio
from datetime import datetime
from enum import Enum
import uuid


class ChainNodeType(str, Enum):
    """Types of chain nodes."""
    PROMPT = "prompt"  # LLM prompt node
    BRANCH = "branch"  # Branching decision node
    MERGE = "merge"   # Merge multiple paths
    TOOL = "tool"     # External tool/function call


class ChainNode(BaseModel):
    """A single node in the prompt chain."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ChainNodeType
    name: str
    content: Union[str, Dict[str, Any]]  # Prompt text or tool config
    next_nodes: List[str] = []  # IDs of next nodes
    prev_nodes: List[str] = []  # IDs of previous nodes
    requires_all_prev: bool = False  # For merge nodes: require all prev nodes to complete
    metadata: Dict[str, Any] = {}


class ChainContext(BaseModel):
    """Context passed between chain nodes during execution."""
    variables: Dict[str, Any] = {}
    results: Dict[str, Any] = {}
    current_path: List[str] = []  # Node IDs in current execution path
    completed_nodes: Set[str] = set()  # IDs of completed nodes


class ChainValidationError(Exception):
    """Raised when chain validation fails."""
    pass


class PromptChain:
    """
    Manages a chain of prompts and tools with support for branching.
    """
    
    def __init__(self, name: str):
        """Initialize an empty chain."""
        self.name = name
        self.nodes: Dict[str, ChainNode] = {}
        self.start_nodes: List[str] = []  # Nodes with no predecessors
        self.end_nodes: List[str] = []    # Nodes with no successors
        logger.info("Created new prompt chain: {}", name)

    def add_node(self, 
                node_type: ChainNodeType,
                name: str,
                content: Union[str, Dict[str, Any]],
                requires_all_prev: bool = False) -> str:
        """
        Add a new node to the chain.
        
        Args:
            node_type: Type of node
            name: Node name
            content: Node content (prompt text or tool config)
            requires_all_prev: For merge nodes, require all prev nodes
            
        Returns:
            ID of the new node
        """
        node = ChainNode(
            type=node_type,
            name=name,
            content=content,
            requires_all_prev=requires_all_prev
        )
        
        self.nodes[node.id] = node
        logger.debug("Added {} node '{}' ({})", node_type, name, node.id)
        
        # Update start/end nodes
        self._update_topology()
        return node.id

    def connect(self, from_id: str, to_id: str):
        """Connect two nodes in the chain."""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError("Invalid node IDs")
            
        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]
        
        if to_id not in from_node.next_nodes:
            from_node.next_nodes.append(to_id)
        if from_id not in to_node.prev_nodes:
            to_node.prev_nodes.append(from_id)
            
        self._update_topology()
        logger.debug("Connected node {} to {}", from_id, to_id)

    def _update_topology(self):
        """Update start and end node lists."""
        self.start_nodes = [
            node.id for node in self.nodes.values()
            if not node.prev_nodes
        ]
        self.end_nodes = [
            node.id for node in self.nodes.values()
            if not node.next_nodes
        ]

    async def validate_node(self, 
                          node: ChainNode, 
                          context: ChainContext) -> bool:
        """
        Validate a node before execution.
        
        Args:
            node: Node to validate
            context: Current execution context
            
        Returns:
            True if validation passes
        """
        try:
            # Check if node can be executed
            if node.requires_all_prev:
                # All previous nodes must be completed
                if not all(n in context.completed_nodes for n in node.prev_nodes):
                    logger.warning(
                        "Node {} requires all prev nodes to complete", 
                        node.id
                    )
                    return False
            else:
                # At least one previous node must be completed
                if node.prev_nodes and not any(
                    n in context.completed_nodes for n in node.prev_nodes
                ):
                    logger.warning(
                        "Node {} requires at least one prev node", 
                        node.id
                    )
                    return False
            
            # Validate based on node type
            if node.type == ChainNodeType.PROMPT:
                # Check if prompt can be formatted with context
                try:
                    node.content.format(**context.variables)
                except KeyError as e:
                    logger.warning(
                        "Missing variable {} for node {}", 
                        str(e), 
                        node.id
                    )
                    return False
                    
            elif node.type == ChainNodeType.TOOL:
                # Check if required tool configs are present
                required_configs = {"name", "params"}
                if not all(k in node.content for k in required_configs):
                    logger.warning(
                        "Missing tool configs for node {}: {}", 
                        node.id,
                        required_configs - node.content.keys()
                    )
                    return False
            
            return True
            
        except Exception as e:
            logger.exception("Error validating node {}", node.id)
            return False

    async def execute_node(self,
                         node: ChainNode,
                         context: ChainContext,
                         tools: Dict[str, Callable] = None) -> Any:
        """
        Execute a single node.
        
        Args:
            node: Node to execute
            context: Current execution context
            tools: Available tool functions
            
        Returns:
            Node execution result
        """
        try:
            if node.type == ChainNodeType.PROMPT:
                # Format and execute prompt
                prompt = node.content.format(**context.variables)
                # TODO: Call LLM with prompt
                result = f"LLM response to: {prompt}"
                
            elif node.type == ChainNodeType.BRANCH:
                # Evaluate branch condition
                condition = node.content.get("condition", "true")
                result = eval(condition, {"context": context.variables})
                
            elif node.type == ChainNodeType.TOOL:
                # Execute tool function
                tool_name = node.content["name"]
                if tool_name not in tools:
                    raise ValueError(f"Tool {tool_name} not found")
                    
                tool_func = tools[tool_name]
                result = await tool_func(**node.content["params"])
                
            else:  # MERGE
                # Combine results from previous nodes
                merge_strategy = node.content.get("strategy", "last")
                prev_results = [
                    context.results[n]
                    for n in node.prev_nodes
                    if n in context.completed_nodes
                ]
                
                if merge_strategy == "last":
                    result = prev_results[-1]
                elif merge_strategy == "list":
                    result = prev_results
                else:
                    raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            
            return result
            
        except Exception as e:
            logger.exception("Error executing node {}", node.id)
            raise

    async def execute(self,
                     initial_context: Optional[Dict[str, Any]] = None,
                     tools: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """
        Execute the entire chain.
        
        Args:
            initial_context: Initial variables
            tools: Available tool functions
            
        Returns:
            Dictionary of execution results
        """
        context = ChainContext(
            variables=initial_context or {},
            results={},
            current_path=[],
            completed_nodes=set()
        )
        
        # Start with all start nodes
        next_nodes = self.start_nodes.copy()
        
        while next_nodes:
            current_id = next_nodes.pop(0)
            current_node = self.nodes[current_id]
            
            # Validate node
            if not await self.validate_node(current_node, context):
                logger.warning("Validation failed for node {}", current_id)
                continue
            
            # Execute node
            try:
                result = await self.execute_node(
                    current_node,
                    context,
                    tools
                )
                
                # Update context
                context.results[current_id] = result
                context.completed_nodes.add(current_id)
                context.current_path.append(current_id)
                
                # Update variables if node provides output
                if isinstance(result, dict):
                    context.variables.update(result)
                
                # Add next nodes to queue
                for next_id in current_node.next_nodes:
                    if next_id not in next_nodes:
                        next_nodes.append(next_id)
                
            except Exception as e:
                logger.exception("Error in node {}", current_id)
                raise
        
        return context.results


# Helper functions to create common chain patterns
def create_linear_chain(
    name: str,
    prompts: List[str],
    tools: Optional[List[Dict[str, Any]]] = None
) -> PromptChain:
    """Create a linear chain of prompts and tools."""
    chain = PromptChain(name)
    prev_id = None
    
    # Add prompts
    for i, prompt in enumerate(prompts):
        node_id = chain.add_node(
            ChainNodeType.PROMPT,
            f"prompt_{i}",
            prompt
        )
        if prev_id:
            chain.connect(prev_id, node_id)
        prev_id = node_id
    
    # Add tools
    if tools:
        for i, tool in enumerate(tools):
            node_id = chain.add_node(
                ChainNodeType.TOOL,
                f"tool_{i}",
                tool
            )
            if prev_id:
                chain.connect(prev_id, node_id)
            prev_id = node_id
    
    return chain


def create_branching_chain(
    name: str,
    condition_prompt: str,
    true_branch: List[str],
    false_branch: List[str]
) -> PromptChain:
    """Create a chain with conditional branching."""
    chain = PromptChain(name)
    
    # Add condition node
    condition_id = chain.add_node(
        ChainNodeType.PROMPT,
        "condition",
        condition_prompt
    )
    
    # Add branch node
    branch_id = chain.add_node(
        ChainNodeType.BRANCH,
        "branch",
        {"condition": "context.get('condition', False)"}
    )
    chain.connect(condition_id, branch_id)
    
    # Add true branch
    prev_id = branch_id
    for i, prompt in enumerate(true_branch):
        node_id = chain.add_node(
            ChainNodeType.PROMPT,
            f"true_{i}",
            prompt
        )
        chain.connect(prev_id, node_id)
        prev_id = node_id
    true_end_id = prev_id
    
    # Add false branch
    prev_id = branch_id
    for i, prompt in enumerate(false_branch):
        node_id = chain.add_node(
            ChainNodeType.PROMPT,
            f"false_{i}",
            prompt
        )
        chain.connect(prev_id, node_id)
        prev_id = node_id
    false_end_id = prev_id
    
    # Add merge node
    merge_id = chain.add_node(
        ChainNodeType.MERGE,
        "merge",
        {"strategy": "last"},
        requires_all_prev=False
    )
    chain.connect(true_end_id, merge_id)
    chain.connect(false_end_id, merge_id)
    
    return chain 