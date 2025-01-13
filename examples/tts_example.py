"""
Example demonstrating AudioAgent's text-to-speech capabilities.
"""
from mnemosyne.agents.audio_agent import AudioAgent, AudioConfig
from loguru import logger
import sys


def main():
    # Initialize audio agent with TTS configuration
    audio_agent = AudioAgent(
        config=AudioConfig(
            # TTS settings
            tts_model="tts_models/en/ljspeech/tacotron2-DDC",
            tts_language="en",
            # Audio processing settings
            noise_reduction=True,
            volume_normalize=True
        )
    )
    
    try:
        # List available voices and languages
        voices = audio_agent.list_available_voices()
        languages = audio_agent.list_available_languages()
        
        logger.info("Available voices: {}", voices)
        logger.info("Available languages: {}", languages)
        
        # Example text to synthesize
        text = "Hello! This is a test of the text-to-speech system."
        
        # Synthesize speech and save to file
        output_path = "test_output.wav"
        logger.info("Synthesizing speech to: {}", output_path)
        
        audio_path = audio_agent.synthesize(
            text=text,
            output_path=output_path
        )
        
        logger.info("Speech synthesized successfully to: {}", audio_path)
        
        # Now let's try transcribing the synthesized audio
        logger.info("Transcribing the synthesized audio...")
        
        transcription = audio_agent.transcribe(audio_path)
        logger.info("Transcription: {}", transcription["text"])
        
    except Exception as e:
        logger.exception("Error in TTS example")
        sys.exit(1)


if __name__ == "__main__":
    main() 