"""
Example demonstrating AudioAgent usage for speech-to-text conversion.
"""
from mnemosyne.agents.audio_agent import AudioAgent, AudioConfig
from mnemosyne.orchestrator import Orchestrator, OrchestratorConfig, LLMProviderConfig
from loguru import logger
from pathlib import Path
import sys


def main():
    # Initialize audio agent with base model
    audio_agent = AudioAgent(
        config=AudioConfig(
            model_type="base",  # Use base model for faster processing
            device="cpu",       # Use CPU for inference
            language="en"       # Default to English
        )
    )
    
    # Initialize orchestrator for processing transcribed text
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
        # Path to your audio file
        audio_path = "path/to/your/audio.wav"  # Replace with actual path
        
        if not Path(audio_path).exists():
            logger.error("Audio file not found: {}", audio_path)
            sys.exit(1)
        
        # First, detect the language
        detected_lang = audio_agent.detect_language(audio_path)
        logger.info("Detected language: {}", detected_lang)
        
        # Transcribe the audio
        transcription = audio_agent.transcribe(audio_path)
        logger.info("Transcription: {}", transcription["text"])
        
        # Process the transcribed text with the orchestrator
        prompt = f"Please analyze this transcribed text and provide a summary: {transcription['text']}"
        response = orchestrator.process_input(prompt)
        
        logger.info("Analysis: {}", response["response"])
        
    except Exception as e:
        logger.exception("Error processing audio")
        sys.exit(1)


if __name__ == "__main__":
    main() 