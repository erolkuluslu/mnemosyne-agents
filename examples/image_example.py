"""
Example demonstrating the usage of the ImageAgent for OCR and image generation.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from mnemosyne.agents import ImageAgent, ImageConfig, GenerationConfig
from mnemosyne.agents.image_agent import GENERATION_AVAILABLE
from loguru import logger
import torch

# Load environment variables
load_dotenv()

def main():
    # Configure the image agent
    config = ImageConfig(
        languages=["en"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_bboxes=True,  # Get bounding boxes for visualization
        return_confidence=True,  # Get confidence scores
        min_confidence=0.5,  # Minimum confidence threshold
    )

    # Add generation config if available
    if GENERATION_AVAILABLE:
        config.generation = GenerationConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_inference_steps=50,
            guidance_scale=7.5,
            width=512,
            height=512
        )
    
    # Initialize the agent
    agent = ImageAgent(config)
    
    # Example 1: OCR on an image
    image_path = "path/to/your/image.png"  # Replace with actual image path
    if os.path.exists(image_path):
        logger.info("Performing OCR on image")
        try:
            # Get detailed OCR results
            results = agent.extract_text(image_path)
            
            # Process results
            if isinstance(results, str):
                # Simple text output
                logger.info("Extracted text: {}", results)
            else:
                # Detailed results with bounding boxes and confidence
                logger.info("Found {} text regions:", len(results))
                for i, result in enumerate(results, 1):
                    logger.info(
                        "Region {}: '{}' (confidence: {:.2f})",
                        i, result["text"], result["confidence"]
                    )
                    if "bbox" in result:
                        logger.debug("Bounding box: {}", result["bbox"])
        except Exception as e:
            logger.error("OCR failed: {}", str(e))
    
    # Only run generation examples if available
    if GENERATION_AVAILABLE:
        # Example 2: Generate an image from text
        logger.info("Generating image from text prompt")
        try:
            prompt = "A serene landscape with mountains and a lake at sunset, digital art style"
            image = agent.generate_image(
                prompt,
                negative_prompt="blurry, low quality, distorted"
            )
            
            # Save the generated image
            output_path = "generated_landscape.png"
            image.save(output_path)
            logger.info("Image saved to: {}", output_path)
        except Exception as e:
            logger.error("Image generation failed: {}", str(e))
        
        # Example 3: Generate a diagram
        logger.info("Generating a flowchart diagram")
        try:
            description = "Software development lifecycle with stages: Planning, Development, Testing, Deployment, Maintenance"
            diagram = agent.generate_diagram(
                description,
                style="flowchart",
                guidance_scale=9.0  # Higher guidance for more precise diagram
            )
            
            # Save the generated diagram
            output_path = "generated_diagram.png"
            diagram.save(output_path)
            logger.info("Diagram saved to: {}", output_path)
        except Exception as e:
            logger.error("Diagram generation failed: {}", str(e))
    else:
        logger.warning("Image generation features are not available. Install required dependencies to enable them.")

if __name__ == "__main__":
    main() 