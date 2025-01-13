"""
Example demonstrating ImageAgent's OCR capabilities.
"""
from mnemosyne.agents.image_agent import ImageAgent, ImageConfig
from mnemosyne.orchestrator import Orchestrator, OrchestratorConfig, LLMProviderConfig
from loguru import logger
from pathlib import Path
import sys


def main():
    # Initialize image agent with OCR configuration
    image_agent = ImageAgent(
        config=ImageConfig(
            languages=["en"],  # English OCR
            device="cpu",      # Use CPU for inference
            # Image processing settings
            preprocessing=True,
            contrast_enhance=True,
            denoise=True,
            min_confidence=0.5
        )
    )
    
    # Initialize orchestrator for processing extracted text
    orchestrator = Orchestrator(
        config=OrchestratorConfig(
            debug=True,
            primary_provider=LLMProviderConfig(
                provider="openai",
                model="gpt-4"
            )
        )
    )
    
    try:
        # Path to your image file
        image_path = "path/to/your/image.png"  # Replace with actual path
        
        if not Path(image_path).exists():
            logger.error("Image file not found: {}", image_path)
            sys.exit(1)
        
        # List supported languages
        supported_langs = image_agent.supported_languages()
        logger.info("Supported OCR languages: {}", supported_langs)
        
        # First, detect text regions
        logger.info("Detecting text regions...")
        regions = image_agent.detect_text_regions(image_path)
        logger.info("Found {} text regions", len(regions))
        
        # Extract text from the image
        logger.info("Extracting text...")
        result = image_agent.extract_text(image_path)
        
        logger.info("Extracted text: {}", result["text"])
        if result["blocks"]:
            logger.info("Text blocks found: {}", len(result["blocks"]))
            for i, block in enumerate(result["blocks"], 1):
                logger.info("Block {}: '{}' (confidence: {:.2f})",
                          i, block["text"], block["confidence"])
        
        # Process the extracted text with the orchestrator
        prompt = f"Please analyze this text extracted from an image and provide a summary: {result['text']}"
        response = orchestrator.process_input(prompt)
        
        logger.info("Analysis: {}", response["response"])
        
    except Exception as e:
        logger.exception("Error in OCR example")
        sys.exit(1)


if __name__ == "__main__":
    main() 