"""
Comprehensive demo of Mnemosyne Agents functionality.
"""
import asyncio
import gradio as gr
import pandas as pd
from pathlib import Path
from loguru import logger
from mnemosyne.orchestrator import Orchestrator, OrchestratorConfig
from mnemosyne.security import Security, SecurityConfig
from mnemosyne.caching import CacheManager, CacheConfig, RateLimitConfig
from mnemosyne.monitoring import PerformanceMetrics
from mnemosyne.agents.code_agent import CodeAgent, CodeSandbox
from mnemosyne.agents.data_agent import DataAgent, DataConfig


async def initialize_components():
    """Initialize all components with proper configuration."""
    # Set up monitoring
    metrics = PerformanceMetrics(port=8000)
    logger.info("Initialized performance metrics")

    # Set up security
    security = Security(
        config=SecurityConfig(
            encryption_key="your-secure-key-here",
            max_retries=3,
            retry_delay=1.0
        )
    )
    logger.info("Initialized security")

    # Set up caching
    cache_manager = CacheManager(
        cache_config=CacheConfig(
            ttl_seconds=3600,
            max_size_bytes=1024 * 1024 * 100  # 100MB
        ),
        rate_limit_config=RateLimitConfig(
            requests_per_minute=60,
            burst_limit=10
        )
    )
    logger.info("Initialized cache manager")

    # Set up domain agents
    code_agent = CodeAgent(
        metrics=metrics,
        security=security,
        sandbox=CodeSandbox(
            allowed_paths=[Path("./examples")],
            max_file_size=1024 * 1024,  # 1MB
            allowed_imports=["os", "sys", "pathlib", "typing"]
        )
    )
    logger.info("Initialized code agent")

    data_agent = DataAgent(
        metrics=metrics,
        security=security,
        config=DataConfig(
            max_rows=1000000,
            max_size=100 * 1024 * 1024,  # 100MB
            allowed_file_types=[".csv", ".json", ".xlsx"]
        )
    )
    logger.info("Initialized data agent")

    # Set up orchestrator
    orchestrator = Orchestrator(
        config=OrchestratorConfig(
            debug=True,
            history_length=10
        ),
        metrics=metrics,
        security=security,
        cache_manager=cache_manager,
        agents={
            "code": code_agent,
            "data": data_agent
        }
    )
    logger.info("Initialized orchestrator")

    return orchestrator, metrics


class DemoUI:
    """Gradio UI for demonstrating functionality."""

    def __init__(self):
        """Initialize demo UI."""
        self.orchestrator = None
        self.metrics = None

    async def startup(self):
        """Initialize components on startup."""
        self.orchestrator, self.metrics = await initialize_components()

    async def process_code(self, code: str) -> dict:
        """Process code with code agent."""
        try:
            agent = self.orchestrator.agents["code"]
            analysis = await agent.analyze_code(code)
            return {
                "status": "success",
                "analysis": analysis
            }
        except Exception as e:
            logger.exception("Error processing code")
            return {
                "status": "error",
                "message": str(e)
            }

    async def process_data(self, file: str) -> dict:
        """Process data with data agent."""
        try:
            agent = self.orchestrator.agents["data"]
            data = await agent.load_data(file)
            analysis = await agent.analyze_data(data, "summary")
            return {
                "status": "success",
                "analysis": analysis
            }
        except Exception as e:
            logger.exception("Error processing data")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_metrics(self) -> dict:
        """Get current performance metrics."""
        try:
            return self.metrics.get_summary()
        except Exception as e:
            logger.exception("Error getting metrics")
            return {
                "status": "error",
                "message": str(e)
            }

    def launch(self):
        """Launch Gradio interface."""
        # Initialize components
        asyncio.run(self.startup())

        # Create interface
        with gr.Blocks(title="Mnemosyne Agents Demo") as demo:
            gr.Markdown("# Mnemosyne Agents Demo")
            
            with gr.Tab("Code Analysis"):
                code_input = gr.Code(
                    label="Enter Python Code",
                    language="python"
                )
                code_button = gr.Button("Analyze Code")
                code_output = gr.JSON(label="Analysis Results")
                
                code_button.click(
                    fn=self.process_code,
                    inputs=[code_input],
                    outputs=[code_output]
                )
            
            with gr.Tab("Data Analysis"):
                file_input = gr.File(label="Upload Data File")
                data_button = gr.Button("Analyze Data")
                data_output = gr.JSON(label="Analysis Results")
                
                data_button.click(
                    fn=self.process_data,
                    inputs=[file_input],
                    outputs=[data_output]
                )
            
            with gr.Tab("Monitoring"):
                metrics_button = gr.Button("Get Metrics")
                metrics_output = gr.JSON(label="Current Metrics")
                
                metrics_button.click(
                    fn=self.get_metrics,
                    inputs=[],
                    outputs=[metrics_output]
                )

        # Launch interface
        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    # Create and launch demo
    demo = DemoUI()
    demo.launch() 