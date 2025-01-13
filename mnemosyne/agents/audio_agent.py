"""
Audio processing agent with speech-to-text and text-to-speech capabilities.
"""
from typing import Optional, Union, BinaryIO, List
from pathlib import Path
from faster_whisper import WhisperModel
from TTS.api import TTS
import soundfile as sf
import numpy as np
from loguru import logger
from pydantic import BaseModel
import tempfile
import os
import noisereduce as nr


class AudioConfig(BaseModel):
    """Configuration for audio processing."""
    # STT Configuration
    model_type: str = "base"  # Whisper model type: tiny, base, small, medium, large
    device: str = "cpu"       # Device to run inference on: cpu or cuda
    compute_type: str = "int8"  # Compute type: int8, int8_float16, float16, float32
    language: Optional[str] = None  # Language code (e.g., 'en' for English)
    
    # TTS Configuration
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"  # Default English TTS model
    tts_language: str = "en"
    tts_voice: Optional[str] = None
    
    # Audio Processing
    sample_rate: int = 16000
    noise_reduction: bool = False
    volume_normalize: bool = True


class AudioAgent:
    """
    Agent for processing audio inputs and outputs.
    Handles speech-to-text conversion and text-to-speech synthesis.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize the audio agent with optional configuration."""
        self.config = config or AudioConfig()
        
        # Initialize STT (Whisper)
        with logger.contextualize(model_type=self.config.model_type):
            logger.info("Initializing Whisper model")
            self.stt_model = WhisperModel(
                model_size_or_path=self.config.model_type,
                device=self.config.device,
                compute_type=self.config.compute_type
            )
            logger.info("Whisper model loaded successfully")
        
        # Initialize TTS
        with logger.contextualize(tts_model=self.config.tts_model):
            logger.info("Initializing TTS model")
            self.tts_model = TTS(self.config.tts_model)
            logger.info("TTS model loaded successfully")
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply preprocessing steps to the audio data.
        """
        try:
            # Ensure correct sample rate
            if sample_rate != self.config.sample_rate:
                # TODO: Add resampling logic if needed
                pass
            
            # Apply noise reduction if enabled
            if self.config.noise_reduction:
                logger.debug("Applying noise reduction")
                audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate)
            
            # Normalize volume if enabled
            if self.config.volume_normalize:
                logger.debug("Normalizing volume")
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data
            
        except Exception as e:
            logger.error("Error in audio preprocessing: {}", str(e))
            raise ValueError(f"Failed to preprocess audio: {str(e)}")
    
    def _load_audio(self, audio_input: Union[str, Path, BinaryIO, np.ndarray]) -> str:
        """
        Load audio from various input types and return a path to the audio file.
        Creates a temporary file if needed.
        """
        try:
            if isinstance(audio_input, (str, Path)):
                # Load, preprocess, and save back if preprocessing is enabled
                if self.config.noise_reduction or self.config.volume_normalize:
                    audio_data, sample_rate = sf.read(str(audio_input))
                    audio_data = self._preprocess_audio(audio_data, sample_rate)
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        sf.write(temp_file.name, audio_data, sample_rate)
                        return temp_file.name
                return str(audio_input)
            elif isinstance(audio_input, np.ndarray):
                # Preprocess and save numpy array to temporary file
                audio_data = self._preprocess_audio(audio_input, self.config.sample_rate)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, audio_data, self.config.sample_rate)
                    return temp_file.name
            else:
                # Save file-like object to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_input.read())
                    # Load, preprocess, and save back if preprocessing is enabled
                    if self.config.noise_reduction or self.config.volume_normalize:
                        audio_data, sample_rate = sf.read(temp_file.name)
                        audio_data = self._preprocess_audio(audio_data, sample_rate)
                        sf.write(temp_file.name, audio_data, sample_rate)
                    return temp_file.name
            
        except Exception as e:
            logger.error("Error loading audio: {}", str(e))
            raise ValueError(f"Failed to load audio: {str(e)}")
    
    def synthesize(self, 
                  text: str,
                  output_path: Optional[str] = None,
                  language: Optional[str] = None,
                  voice: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech using the TTS model.
        
        Args:
            text: Text to convert to speech
            output_path: Optional path to save the audio file
            language: Optional language override
            voice: Optional voice override
            
        Returns:
            Path to the generated audio file if output_path is provided,
            otherwise plays the audio and returns None
        """
        try:
            with logger.contextualize(operation="synthesize"):
                logger.info("Starting text-to-speech synthesis")
                
                # Use provided values or fall back to config
                language = language or self.config.tts_language
                voice = voice or self.config.tts_voice
                
                # Generate temporary path if none provided
                if output_path is None:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        output_path = temp_file.name
                
                # Synthesize speech
                logger.debug("Running TTS inference")
                self.tts_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language,
                    speaker=voice
                )
                
                logger.info("Speech synthesis completed successfully")
                return output_path
                
        except Exception as e:
            logger.exception("Speech synthesis failed")
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")
    
    def list_available_voices(self) -> List[str]:
        """List available voices for the current TTS model."""
        try:
            voices = self.tts_model.speakers
            return voices if voices else []
        except Exception:
            return []
    
    def list_available_languages(self) -> List[str]:
        """List available languages for the current TTS model."""
        try:
            languages = self.tts_model.languages
            return languages if languages else []
        except Exception:
            return []
    
    def transcribe(self, 
                  audio_input: Union[str, Path, BinaryIO, np.ndarray],
                  language: Optional[str] = None) -> dict:
        """
        Transcribe speech from audio input to text.
        
        Args:
            audio_input: Audio file path, binary data, or numpy array
            language: Optional language code to override config
            
        Returns:
            Dictionary containing:
                - text: The transcribed text
                - language: Detected or specified language
                - segments: List of transcribed segments with timestamps
        """
        try:
            with logger.contextualize(operation="transcribe"):
                logger.info("Starting transcription")
                
                # Load and preprocess audio
                audio_path = self._load_audio(audio_input)
                temp_file_created = not isinstance(audio_input, (str, Path))
                
                try:
                    # Run transcription
                    logger.debug("Running Whisper inference")
                    segments, info = self.stt_model.transcribe(
                        audio_path,
                        language=language or self.config.language,
                        task="transcribe"
                    )
                    
                    # Convert segments to list and join text
                    segments_list = list(segments)  # Convert generator to list
                    full_text = " ".join(seg.text for seg in segments_list)
                    
                    # Format segments for output
                    formatted_segments = [
                        {
                            "text": seg.text,
                            "start": seg.start,
                            "end": seg.end,
                            "words": [{"word": w.word, "probability": w.probability} 
                                    for w in (seg.words or [])]
                        }
                        for seg in segments_list
                    ]
                    
                    logger.info("Transcription completed successfully")
                    
                    return {
                        "text": full_text,
                        "language": info.language,
                        "segments": formatted_segments,
                        "language_probability": info.language_probability
                    }
                    
                finally:
                    # Clean up temporary file if we created one
                    if temp_file_created and os.path.exists(audio_path):
                        os.unlink(audio_path)
                
        except Exception as e:
            logger.exception("Transcription failed")
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def detect_language(self, audio_input: Union[str, Path, BinaryIO, np.ndarray]) -> str:
        """
        Detect the language of speech in the audio.
        
        Args:
            audio_input: Audio file path, binary data, or numpy array
            
        Returns:
            Detected language code
        """
        try:
            with logger.contextualize(operation="detect_language"):
                logger.info("Starting language detection")
                
                # Load and preprocess audio
                audio_path = self._load_audio(audio_input)
                temp_file_created = not isinstance(audio_input, (str, Path))
                
                try:
                    # Run language detection
                    _, info = self.stt_model.transcribe(
                        audio_path,
                        task="transcribe",
                        language=None  # Force language detection
                    )
                    
                    detected_lang = info.language
                    logger.info("Detected language: {} (probability: {:.2f})", 
                              detected_lang, info.language_probability)
                    
                    return detected_lang
                    
                finally:
                    # Clean up temporary file if we created one
                    if temp_file_created and os.path.exists(audio_path):
                        os.unlink(audio_path)
                
        except Exception as e:
            logger.exception("Language detection failed")
            raise RuntimeError(f"Language detection failed: {str(e)}") 