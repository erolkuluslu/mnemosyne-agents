"""
Example demonstrating the prompt chaining library features.
"""
import asyncio
from loguru import logger
from mnemosyne.chains import (
    PromptChain,
    ChainNodeType,
    create_linear_chain,
    create_branching_chain
)


async def example_tool(text: str) -> dict:
    """Example tool that processes text."""
    return {
        "processed_text": f"Processed: {text}",
        "length": len(text)
    }


async def main():
    # Example 1: Linear Chain
    logger.info("Creating linear chain example...")
    linear_chain = create_linear_chain(
        "text_processor",
        prompts=[
            "Analyze this text: {input_text}",
            "Summarize the analysis: {processed_text}"
        ],
        tools=[{
            "name": "text_processor",
            "params": {"text": "{input_text}"}
        }]
    )
    
    # Execute linear chain
    linear_results = await linear_chain.execute(
        initial_context={"input_text": "Hello, World!"},
        tools={"text_processor": example_tool}
    )
    logger.info("Linear chain results: {}", linear_results)
    
    # Example 2: Branching Chain
    logger.info("Creating branching chain example...")
    branching_chain = create_branching_chain(
        "text_classifier",
        condition_prompt="Is this text a question: {input_text}",
        true_branch=[
            "Answer the question: {input_text}",
            "Verify the answer's accuracy"
        ],
        false_branch=[
            "Process the statement: {input_text}",
            "Generate related insights"
        ]
    )
    
    # Add dynamic modification during execution
    question_node_id = branching_chain.add_node(
        ChainNodeType.PROMPT,
        "dynamic_question",
        "Additional question about: {input_text}"
    )
    
    # We'll connect this dynamically during execution
    # based on some condition
    
    # Execute branching chain
    branching_results = await branching_chain.execute(
        initial_context={"input_text": "What is the meaning of life?"}
    )
    logger.info("Branching chain results: {}", branching_results)
    
    # Example 3: Dynamic Chain Modification
    logger.info("Demonstrating dynamic chain modification...")
    dynamic_chain = PromptChain("dynamic_example")
    
    # Add initial nodes
    start_id = dynamic_chain.add_node(
        ChainNodeType.PROMPT,
        "start",
        "Initial analysis of: {input_text}"
    )
    
    process_id = dynamic_chain.add_node(
        ChainNodeType.TOOL,
        "process",
        {
            "name": "text_processor",
            "params": {"text": "{input_text}"}
        }
    )
    
    dynamic_chain.connect(start_id, process_id)
    
    # Execute first part
    context = {
        "input_text": "This is a dynamic chain example"
    }
    
    initial_results = await dynamic_chain.execute(
        initial_context=context,
        tools={"text_processor": example_tool}
    )
    
    # Based on results, add new nodes
    if len(context["input_text"]) > 10:
        logger.info("Adding detailed analysis branch...")
        detail_id = dynamic_chain.add_node(
            ChainNodeType.PROMPT,
            "detailed_analysis",
            "Detailed analysis of: {processed_text}"
        )
        dynamic_chain.connect(process_id, detail_id)
        
        # Execute the modified chain
        final_results = await dynamic_chain.execute(
            initial_context=context,
            tools={"text_processor": example_tool}
        )
        logger.info("Final results after modification: {}", final_results)


if __name__ == "__main__":
    asyncio.run(main()) 