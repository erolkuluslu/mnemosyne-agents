"""
Sample code for testing code analysis agent.
"""
from typing import List, Dict, Optional
import os
import sys
from pathlib import Path


def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
        
    Returns:
        Dictionary with statistics
    """
    if not numbers:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0
        }
    
    total = sum(numbers)
    count = len(numbers)
    
    return {
        "mean": total / count,
        "min": min(numbers),
        "max": max(numbers),
        "sum": total
    }


class FileProcessor:
    """Process files with size validation."""
    
    def __init__(self, max_size: int = 1024 * 1024):
        """Initialize processor with max file size."""
        self.max_size = max_size
    
    def process_file(self, path: str) -> Optional[str]:
        """
        Process a file if it's within size limit.
        
        Args:
            path: Path to file
            
        Returns:
            File contents or None if too large
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if file_path.stat().st_size > self.max_size:
            return None
        
        return file_path.read_text()


def main():
    """Main function."""
    # Example usage
    numbers = [1.5, 2.7, 3.2, 4.8, 5.1]
    stats = calculate_statistics(numbers)
    print("Statistics:", stats)
    
    processor = FileProcessor(max_size=1024)  # 1KB limit
    try:
        content = processor.process_file("test.txt")
        if content:
            print("File contents:", content)
        else:
            print("File too large")
    except FileNotFoundError as e:
        print("Error:", e)


if __name__ == "__main__":
    main() 