"""
Data analysis agent for processing and analyzing data.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from loguru import logger
import pandas as pd
import numpy as np
from mnemosyne.monitoring import MonitoredComponent
from mnemosyne.security import Security


class DataConfig(BaseModel):
    """Configuration for data operations."""
    max_rows: int = 1000000  # 1M rows
    max_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: List[str] = [".csv", ".json", ".xlsx"]


class DataAgent(MonitoredComponent):
    """Agent for data analysis and processing."""
    
    def __init__(self, metrics, security: Security, config: DataConfig):
        """Initialize data agent."""
        super().__init__(metrics)
        self.security = security
        self.config = config
        logger.info("Initialized data agent")

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data size and content."""
        if len(data) > self.config.max_rows:
            logger.warning("Data exceeds max rows")
            return False
        
        # Check memory usage
        memory_usage = data.memory_usage(deep=True).sum()
        if memory_usage > self.config.max_size:
            logger.warning("Data exceeds max size")
            return False
            
        return True

    async def load_data(self, path: str) -> pd.DataFrame:
        """Load data from file with validation."""
        async with self.track_operation(
            operation="load_data",
            agent_type="data"
        ):
            # Check file type
            if not any(path.endswith(ext) for ext in self.config.allowed_file_types):
                raise ValueError("Unsupported file type")
            
            # Load data based on type
            if path.endswith(".csv"):
                data = pd.read_csv(path)
            elif path.endswith(".json"):
                data = pd.read_json(path)
            elif path.endswith(".xlsx"):
                data = pd.read_excel(path)
            
            # Validate loaded data
            if not self._validate_data(data):
                raise ValueError("Data validation failed")
            
            return data

    async def analyze_data(self,
                       data: pd.DataFrame,
                       analysis_type: str) -> Dict[str, Any]:
        """
        Analyze data based on specified type.
        
        Args:
            data: DataFrame to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results
        """
        async with self.track_operation(
            operation="analyze_data",
            agent_type="data"
        ):
            # Validate data first
            if not self._validate_data(data):
                raise ValueError("Data validation failed")
            
            results = {}
            
            if analysis_type == "summary":
                results["summary"] = data.describe().to_dict()
                results["missing"] = data.isnull().sum().to_dict()
                results["dtypes"] = data.dtypes.astype(str).to_dict()
                
            elif analysis_type == "correlation":
                # Only numeric columns
                numeric_data = data.select_dtypes(include=[np.number])
                results["correlation"] = numeric_data.corr().to_dict()
                
            elif analysis_type == "distribution":
                results["distribution"] = {
                    col: data[col].value_counts().to_dict()
                    for col in data.columns
                }
                
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            return results

    async def process_data(self,
                       data: pd.DataFrame,
                       operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process data with specified operations.
        
        Args:
            data: DataFrame to process
            operations: List of operations to apply
            
        Returns:
            Processed DataFrame
        """
        async with self.track_operation(
            operation="process_data",
            agent_type="data"
        ):
            # Validate input data
            if not self._validate_data(data):
                raise ValueError("Input data validation failed")
            
            result = data.copy()
            
            for op in operations:
                op_type = op["type"]
                
                if op_type == "filter":
                    col = op["column"]
                    condition = op["condition"]
                    value = op["value"]
                    
                    if condition == "equals":
                        result = result[result[col] == value]
                    elif condition == "greater_than":
                        result = result[result[col] > value]
                    elif condition == "less_than":
                        result = result[result[col] < value]
                    else:
                        raise ValueError(f"Unknown condition: {condition}")
                        
                elif op_type == "transform":
                    col = op["column"]
                    transform = op["transform"]
                    
                    if transform == "normalize":
                        result[col] = (result[col] - result[col].mean()) / result[col].std()
                    elif transform == "fillna":
                        value = op.get("value", result[col].mean())
                        result[col] = result[col].fillna(value)
                    else:
                        raise ValueError(f"Unknown transform: {transform}")
                        
                else:
                    raise ValueError(f"Unknown operation type: {op_type}")
                
                # Validate intermediate result
                if not self._validate_data(result):
                    raise ValueError("Intermediate result validation failed")
            
            return result 