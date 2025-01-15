"""
Code generation agent with sandbox restrictions.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from loguru import logger
import ast
import os
from pathlib import Path
import tempfile
from mnemosyne.monitoring import MonitoredComponent
from mnemosyne.security import Security


class CodeSandbox(BaseModel):
    """Sandbox configuration for code operations."""
    allowed_paths: List[Path]
    max_file_size: int = 1024 * 1024  # 1MB
    allowed_imports: List[str] = ["os", "sys", "pathlib", "typing"]


class CodeAgent(MonitoredComponent):
    """Agent for code generation and manipulation."""
    
    def __init__(self, metrics, security: Security, sandbox: CodeSandbox):
        """Initialize code agent."""
        super().__init__(metrics)
        self.security = security
        self.sandbox = sandbox
        logger.info("Initialized code agent with sandbox")

    def _validate_path(self, path: Path) -> bool:
        """Check if path is allowed in sandbox."""
        try:
            path = Path(path).resolve()
            return any(
                str(path).startswith(str(allowed.resolve()))
                for allowed in self.sandbox.allowed_paths
            )
        except Exception:
            return False

    def _validate_code(self, code: str) -> bool:
        """Validate code for security."""
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Check imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name not in self.sandbox.allowed_imports:
                            logger.warning(
                                "Disallowed import: {}", 
                                name.name
                            )
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module not in self.sandbox.allowed_imports:
                        logger.warning(
                            "Disallowed import from: {}", 
                            node.module
                        )
                        return False
            
            return True
            
        except SyntaxError:
            logger.warning("Invalid Python syntax")
            return False
        except Exception as e:
            logger.exception("Error validating code")
            return False

    async def generate_code(self,
                        prompt: str,
                        context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate code based on prompt.
        
        Args:
            prompt: Description of code to generate
            context: Additional context (e.g. existing code)
            
        Returns:
            Generated code
        """
        async with self.track_operation(
            operation="generate_code",
            agent_type="code"
        ):
            # TODO: Call LLM to generate code
            # For now, return dummy code
            code = """
            def hello_world():
                print("Hello, World!")
                return 42
            """
            
            # Validate generated code
            if not self._validate_code(code):
                raise ValueError("Generated code failed validation")
            
            return code

    async def read_file(self, path: str) -> str:
        """Read file contents with sandbox validation."""
        async with self.track_operation(
            operation="read_file",
            agent_type="code"
        ):
            path = Path(path)
            if not self._validate_path(path):
                raise ValueError(f"Path not allowed: {path}")
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            if path.stat().st_size > self.sandbox.max_file_size:
                raise ValueError("File too large")
            
            return path.read_text()

    async def write_file(self, path: str, content: str):
        """Write file contents with sandbox validation."""
        async with self.track_operation(
            operation="write_file",
            agent_type="code"
        ):
            path = Path(path)
            if not self._validate_path(path):
                raise ValueError(f"Path not allowed: {path}")
            
            if len(content.encode()) > self.sandbox.max_file_size:
                raise ValueError("Content too large")
            
            # Validate code if it's a Python file
            if path.suffix == ".py":
                if not self._validate_code(content):
                    raise ValueError("Code failed validation")
            
            # Write to temp file first
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
                tmp.write(content)
            
            # Move temp file to target
            os.replace(tmp.name, path)

    async def analyze_code(self,
                       code: str,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze code for quality and issues.
        
        Args:
            code: Code to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        async with self.track_operation(
            operation="analyze_code",
            agent_type="code"
        ):
            # Validate code first
            if not self._validate_code(code):
                raise ValueError("Code failed validation")
            
            # TODO: Call LLM to analyze code
            # For now, return dummy analysis
            return {
                "complexity": "low",
                "maintainability": "good",
                "issues": [],
                "suggestions": [
                    "Add docstrings",
                    "Add type hints"
                ]
            } 