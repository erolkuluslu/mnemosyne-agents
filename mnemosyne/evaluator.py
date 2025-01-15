"""
Enhanced evaluator module for multi-criteria output evaluation and refinement.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from loguru import logger
import asyncio
from datetime import datetime
from enum import Enum
import json


class EvaluationCriteria(str, Enum):
    """Types of evaluation criteria."""
    FACTUAL_ACCURACY = "factual_accuracy"
    STYLE_CONSISTENCY = "style_consistency"
    POLICY_COMPLIANCE = "policy_compliance"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    CREATIVITY = "creativity"
    SAFETY = "safety"


class EvaluationScore(BaseModel):
    """Score for a single evaluation criterion."""
    criterion: EvaluationCriteria
    score: float  # 0.0 to 1.0
    feedback: str
    suggestions: List[str] = []


class EvaluationResult(BaseModel):
    """Complete evaluation result for a piece of content."""
    scores: List[EvaluationScore]
    overall_score: float
    needs_refinement: bool
    refinement_prompt: Optional[str] = None
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator("overall_score")
    def validate_score(cls, v):
        """Validate score is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Score must be between 0 and 1")
        return v


class EvaluationConfig(BaseModel):
    """Configuration for the evaluator."""
    criteria: List[EvaluationCriteria]
    weights: Dict[EvaluationCriteria, float]
    thresholds: Dict[EvaluationCriteria, float]
    refinement_threshold: float = 0.7  # Overall score below this triggers refinement

    @validator("weights")
    def validate_weights(cls, v, values):
        """Validate weights match criteria and sum to 1."""
        if "criteria" in values:
            if set(v.keys()) != set(values["criteria"]):
                raise ValueError("Weights must match criteria exactly")
            if abs(sum(v.values()) - 1.0) > 0.001:
                raise ValueError("Weights must sum to 1")
        return v

    @validator("thresholds")
    def validate_thresholds(cls, v, values):
        """Validate thresholds match criteria and are between 0 and 1."""
        if "criteria" in values:
            if set(v.keys()) != set(values["criteria"]):
                raise ValueError("Thresholds must match criteria exactly")
            if not all(0 <= t <= 1 for t in v.values()):
                raise ValueError("Thresholds must be between 0 and 1")
        return v


class ContentEvaluator:
    """
    Evaluates content based on multiple criteria and suggests refinements.
    """
    
    def __init__(self, config: EvaluationConfig):
        """Initialize the evaluator with config."""
        self.config = config
        logger.info(
            "Initialized evaluator with {} criteria", 
            len(config.criteria)
        )

    async def evaluate_criterion(self,
                             content: str,
                             criterion: EvaluationCriteria,
                             context: Optional[Dict[str, Any]] = None) -> EvaluationScore:
        """
        Evaluate content for a single criterion.
        
        Args:
            content: Content to evaluate
            criterion: Criterion to evaluate against
            context: Additional context for evaluation
            
        Returns:
            Evaluation score with feedback
        """
        # Create prompt based on criterion
        prompt = self._create_evaluation_prompt(content, criterion, context)
        
        # TODO: Call LLM to evaluate
        # For now, simulate evaluation
        await asyncio.sleep(0.5)
        
        # Parse response into score and feedback
        if criterion == EvaluationCriteria.FACTUAL_ACCURACY:
            score = 0.85
            feedback = "Content appears mostly accurate"
            suggestions = ["Consider adding sources for key claims"]
        else:
            score = 0.75
            feedback = f"Acceptable {criterion.value}"
            suggestions = [f"Could improve {criterion.value}"]
        
        return EvaluationScore(
            criterion=criterion,
            score=score,
            feedback=feedback,
            suggestions=suggestions
        )

    def _create_evaluation_prompt(self,
                              content: str,
                              criterion: EvaluationCriteria,
                              context: Optional[Dict[str, Any]] = None) -> str:
        """Create an evaluation prompt for the LLM."""
        prompts = {
            EvaluationCriteria.FACTUAL_ACCURACY: """
                Evaluate the factual accuracy of this content:
                ---
                {content}
                ---
                Consider:
                1. Are all statements supported by evidence?
                2. Are there any contradictions?
                3. Are statistics and numbers accurate?
                
                Provide a score from 0.0 to 1.0 and explain your reasoning.
            """,
            EvaluationCriteria.STYLE_CONSISTENCY: """
                Evaluate the style consistency of this content:
                ---
                {content}
                ---
                Consider:
                1. Is the tone consistent?
                2. Is the formality level maintained?
                3. Are transitions smooth?
                
                Provide a score from 0.0 to 1.0 and explain your reasoning.
            """,
            # Add prompts for other criteria...
        }
        
        base_prompt = prompts.get(
            criterion,
            "Evaluate the {criterion} of this content: {content}"
        )
        
        return base_prompt.format(
            content=content,
            criterion=criterion.value,
            **(context or {})
        )

    def _create_refinement_prompt(self,
                               content: str,
                               scores: List[EvaluationScore],
                               context: Optional[Dict[str, Any]] = None) -> str:
        """Create a refinement prompt based on evaluation scores."""
        # Identify areas needing improvement
        improvements = []
        for score in scores:
            if score.score < self.config.thresholds[score.criterion]:
                improvements.extend(score.suggestions)
        
        prompt = f"""
            Please improve this content:
            ---
            {content}
            ---
            Focus on these aspects:
            {json.dumps(improvements, indent=2)}
            
            Maintain any correct and high-quality portions while addressing the issues.
        """
        
        return prompt

    async def evaluate(self,
                    content: str,
                    context: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate content against all configured criteria.
        
        Args:
            content: Content to evaluate
            context: Additional context for evaluation
            
        Returns:
            Complete evaluation result
        """
        # Evaluate each criterion
        scores = []
        for criterion in self.config.criteria:
            score = await self.evaluate_criterion(
                content,
                criterion,
                context
            )
            scores.append(score)
        
        # Calculate overall score
        overall_score = sum(
            score.score * self.config.weights[score.criterion]
            for score in scores
        )
        
        # Check if refinement is needed
        needs_refinement = overall_score < self.config.refinement_threshold
        refinement_prompt = None
        
        if needs_refinement:
            refinement_prompt = self._create_refinement_prompt(
                content,
                scores,
                context
            )
        
        return EvaluationResult(
            scores=scores,
            overall_score=overall_score,
            needs_refinement=needs_refinement,
            refinement_prompt=refinement_prompt
        )

    async def evaluate_and_refine(self,
                               content: str,
                               context: Optional[Dict[str, Any]] = None,
                               max_iterations: int = 3) -> tuple[str, List[EvaluationResult]]:
        """
        Evaluate content and automatically refine until it meets thresholds.
        
        Args:
            content: Initial content to evaluate and refine
            context: Additional context for evaluation
            max_iterations: Maximum number of refinement attempts
            
        Returns:
            Tuple of (final content, list of evaluation results)
        """
        current_content = content
        results = []
        
        for i in range(max_iterations):
            # Evaluate current content
            result = await self.evaluate(current_content, context)
            results.append(result)
            
            # Check if refinement is needed
            if not result.needs_refinement:
                logger.info("Content meets all criteria after {} iterations", i + 1)
                break
                
            # Refine content
            if result.refinement_prompt:
                # TODO: Call LLM with refinement prompt
                # For now, simulate refinement
                await asyncio.sleep(1.0)
                current_content = f"Refined ({i + 1}): {current_content}"
            
            logger.info(
                "Completed refinement iteration {} with score {:.2f}",
                i + 1,
                result.overall_score
            )
        
        return current_content, results 