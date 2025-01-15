"""
Example demonstrating the enhanced evaluator features.
"""
import asyncio
from loguru import logger
from mnemosyne.evaluator import (
    ContentEvaluator,
    EvaluationConfig,
    EvaluationCriteria
)


async def main():
    # Create evaluator config
    config = EvaluationConfig(
        criteria=[
            EvaluationCriteria.FACTUAL_ACCURACY,
            EvaluationCriteria.STYLE_CONSISTENCY,
            EvaluationCriteria.POLICY_COMPLIANCE,
            EvaluationCriteria.RELEVANCE
        ],
        weights={
            EvaluationCriteria.FACTUAL_ACCURACY: 0.4,
            EvaluationCriteria.STYLE_CONSISTENCY: 0.2,
            EvaluationCriteria.POLICY_COMPLIANCE: 0.2,
            EvaluationCriteria.RELEVANCE: 0.2
        },
        thresholds={
            EvaluationCriteria.FACTUAL_ACCURACY: 0.8,
            EvaluationCriteria.STYLE_CONSISTENCY: 0.7,
            EvaluationCriteria.POLICY_COMPLIANCE: 0.9,
            EvaluationCriteria.RELEVANCE: 0.7
        },
        refinement_threshold=0.75
    )
    
    # Initialize evaluator
    evaluator = ContentEvaluator(config)
    
    # Example 1: Single Evaluation
    logger.info("Running single evaluation example...")
    
    content = """
    The Earth orbits the Sun at an average distance of 93 million miles.
    This journey takes approximately 365.25 days to complete.
    The Earth's atmosphere is composed primarily of nitrogen and oxygen.
    """
    
    result = await evaluator.evaluate(content)
    
    logger.info("Evaluation scores:")
    for score in result.scores:
        logger.info(
            "{}: {:.2f} - {}",
            score.criterion.value,
            score.score,
            score.feedback
        )
    logger.info("Overall score: {:.2f}", result.overall_score)
    
    if result.needs_refinement:
        logger.info("Content needs refinement")
        logger.info("Refinement prompt: {}", result.refinement_prompt)
    
    # Example 2: Evaluation with Context
    logger.info("\nRunning evaluation with context example...")
    
    content_with_context = """
    Our new product features advanced AI capabilities.
    It can process natural language and generate responses.
    The system is built on cutting-edge technology.
    """
    
    context = {
        "target_audience": "technical professionals",
        "style_guide": {
            "tone": "professional",
            "formality": "high"
        },
        "domain": "artificial intelligence"
    }
    
    result = await evaluator.evaluate(content_with_context, context)
    
    logger.info("Evaluation with context scores:")
    for score in result.scores:
        logger.info(
            "{}: {:.2f} - {}",
            score.criterion.value,
            score.score,
            score.feedback
        )
    
    # Example 3: Automatic Refinement
    logger.info("\nRunning automatic refinement example...")
    
    content_to_refine = """
    AI is really good at doing stuff.
    It helps people work better and faster.
    Everyone should use AI because it's amazing.
    """
    
    refined_content, refinement_results = await evaluator.evaluate_and_refine(
        content_to_refine,
        max_iterations=3
    )
    
    logger.info("Refinement history:")
    for i, result in enumerate(refinement_results):
        logger.info(
            "Iteration {}: Score {:.2f}",
            i + 1,
            result.overall_score
        )
    
    logger.info("Final content:\n{}", refined_content)


if __name__ == "__main__":
    asyncio.run(main()) 