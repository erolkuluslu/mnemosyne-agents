"""
Example demonstrating parallel task execution with evaluation.
Shows how to run OCR and content moderation in parallel, then evaluate results.
"""
from mnemosyne.orchestrator import Orchestrator, OrchestratorConfig, LLMProviderConfig
from mnemosyne.agents.evaluator_agent import EvaluatorAgent, EvaluatorConfig
from mnemosyne.agents.image_agent import ImageAgent
from loguru import logger
import asyncio
from pathlib import Path
import sys


async def main():
    # Initialize agents
    orchestrator = Orchestrator(
        config=OrchestratorConfig(
            debug=True,
            primary_provider=LLMProviderConfig(
                provider="openai",
                model="gpt-4"
            )
        )
    )
    
    evaluator = EvaluatorAgent(
        config=EvaluatorConfig(
            confidence_threshold=0.7,
            check_types=["factual_coherence"]
        )
    )
    
    image_agent = ImageAgent()  # Using default config
    
    try:
        # Example: Process image and evaluate text in parallel
        image_path = "path/to/your/image.jpg"  # Replace with actual path
        
        if not Path(image_path).exists():
            logger.error("Image file not found: {}", image_path)
            sys.exit(1)
            
        logger.info("Starting parallel processing")
        
        # Define tasks to run in parallel
        tasks = [
            # Task 1: OCR the image
            image_agent.extract_text(image_path),
            
            # Task 2: Get moderation check from LLM
            orchestrator.process_input_async(
                "Please analyze this image for any concerning content."
            )
        ]
        
        # Run tasks in parallel with timeout
        results = await orchestrator.run_parallel_tasks(tasks, timeout=30.0)
        
        # Unpack results
        ocr_text, moderation_result = results
        
        logger.info("OCR Text: {}", ocr_text)
        logger.info("Moderation Result: {}", moderation_result)
        
        # Evaluate the results
        evaluation = await evaluator.evaluate_output(
            output=moderation_result["response"],
            context={"ocr_text": ocr_text}
        )
        
        logger.info("Evaluation Results:")
        logger.info("Passed: {}", evaluation["passed"])
        logger.info("Confidence: {:.2f}", evaluation["confidence"])
        
        if not evaluation["passed"]:
            logger.warning("Evaluation failed. Suggestions:")
            for suggestion in evaluation["suggestions"]:
                logger.warning("- {}", suggestion)
                
    except asyncio.TimeoutError:
        logger.error("Processing timed out")
        sys.exit(1)
    except Exception as e:
        logger.exception("Error during processing")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 