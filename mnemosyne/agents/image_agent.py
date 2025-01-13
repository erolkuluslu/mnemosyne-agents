"""
Image processing agent with OCR, image analysis, and generation capabilities.
"""
from typing import Optional, Union, BinaryIO, List, Dict, Any, Tuple
from pathlib import Path
import easyocr
import cv2
import numpy as np
from PIL import Image
from loguru import logger
from pydantic import BaseModel, Field
import tempfile
import os
import torch

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionControlNetPipeline,
        ControlNetModel,
    )
    from compel import Compel
    from transformers import CLIPTokenizer
    GENERATION_AVAILABLE = True
except ImportError:
    logger.warning("Image generation dependencies not available. Some features will be disabled.")
    GENERATION_AVAILABLE = False


class GenerationConfig(BaseModel):
    """Configuration for image generation."""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_images: int = 1
    seed: Optional[int] = None
    
    # ControlNet settings
    use_controlnet: bool = False
    controlnet_model: Optional[str] = None
    controlnet_conditioning_scale: float = 1.0
    
    # Advanced settings
    use_compel: bool = True  # Use compel for advanced prompt weighting
    use_safety_checker: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True


class ImageConfig(BaseModel):
    """Configuration for image processing."""
    # OCR Configuration
    languages: List[str] = ["en"]  # List of language codes for OCR
    device: str = "cpu"  # Device to run inference on: cpu or cuda
    
    # Image Processing
    min_confidence: float = 0.5  # Minimum confidence for OCR detection
    preprocessing: bool = True  # Whether to apply preprocessing
    contrast_enhance: bool = True  # Whether to enhance contrast
    denoise: bool = True  # Whether to apply denoising
    
    # Output Configuration
    return_bboxes: bool = True  # Whether to return bounding boxes
    return_confidence: bool = True  # Whether to return confidence scores
    
    # Generation Configuration
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class ImageAgent:
    """
    Agent for processing images, handling OCR, image analysis, and generation.
    Uses EasyOCR for text detection and Stable Diffusion for image generation.
    """
    
    def __init__(self, config: Optional[ImageConfig] = None):
        """Initialize the image agent with optional configuration."""
        self.config = config or ImageConfig()
        
        # Initialize OCR
        with logger.contextualize(languages=self.config.languages):
            logger.info("Initializing OCR model")
            self.reader = easyocr.Reader(
                lang_list=self.config.languages,
                gpu=self.config.device == "cuda"
            )
            logger.info("OCR model loaded successfully")
        
        # Initialize generation pipeline
        self._init_generation_pipeline()
    
    def _init_generation_pipeline(self):
        """Initialize the image generation pipeline."""
        if not GENERATION_AVAILABLE:
            logger.warning("Image generation is disabled due to missing dependencies")
            self.pipeline = None
            self.compel = None
            return

        try:
            with logger.contextualize(
                model=self.config.generation.model_id,
                device=self.config.generation.device
            ):
                logger.info("Initializing generation pipeline")
                
                # Basic pipeline
                if not self.config.generation.use_controlnet:
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        self.config.generation.model_id,
                        torch_dtype=self.config.generation.dtype,
                        safety_checker=None if not self.config.generation.use_safety_checker else "default"
                    )
                else:
                    # ControlNet pipeline
                    controlnet = ControlNetModel.from_pretrained(
                        self.config.generation.controlnet_model,
                        torch_dtype=self.config.generation.dtype
                    )
                    self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                        self.config.generation.model_id,
                        controlnet=controlnet,
                        torch_dtype=self.config.generation.dtype,
                        safety_checker=None if not self.config.generation.use_safety_checker else "default"
                    )
                
                # Move to device
                self.pipeline = self.pipeline.to(self.config.generation.device)
                
                # Optimizations
                if self.config.generation.enable_attention_slicing:
                    self.pipeline.enable_attention_slicing()
                if self.config.generation.enable_vae_slicing:
                    self.pipeline.enable_vae_slicing()
                
                # Initialize Compel for advanced prompt weighting
                if self.config.generation.use_compel:
                    self.compel = Compel(
                        tokenizer=self.pipeline.tokenizer,
                        text_encoder=self.pipeline.text_encoder,
                        truncate_long_prompts=False
                    )
                
                logger.info("Generation pipeline initialized successfully")
                
        except Exception as e:
            logger.exception("Failed to initialize generation pipeline")
            raise RuntimeError(f"Generation pipeline initialization failed: {str(e)}")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        control_image: Optional[Union[str, Path, Image.Image]] = None,
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate an image from a text prompt using Stable Diffusion.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Optional text description of what to avoid
            control_image: Optional control image for ControlNet
            **kwargs: Additional generation parameters to override config
            
        Returns:
            One or more PIL Images depending on num_images setting
        """
        if not GENERATION_AVAILABLE:
            raise RuntimeError("Image generation is not available. Please install required dependencies.")

        try:
            with logger.contextualize(operation="generate"):
                logger.info("Starting image generation")
                logger.debug("Prompt: {}", prompt)
                
                # Process prompt with Compel if enabled
                if self.config.generation.use_compel:
                    prompt_embeds = self.compel(prompt)
                    neg_prompt_embeds = self.compel(
                        negative_prompt or self.config.generation.negative_prompt
                    )
                else:
                    prompt_embeds = None
                    neg_prompt_embeds = None
                
                # Prepare generation parameters
                generator = None
                if self.config.generation.seed is not None:
                    generator = torch.Generator(device=self.config.generation.device)
                    generator.manual_seed(self.config.generation.seed)
                
                # Generate image(s)
                if self.config.generation.use_controlnet and control_image:
                    # Load and preprocess control image
                    if isinstance(control_image, (str, Path)):
                        control_image = Image.open(str(control_image))
                    control_image = control_image.resize(
                        (self.config.generation.width, self.config.generation.height)
                    )
                    
                    result = self.pipeline(
                        prompt=prompt if not prompt_embeds else None,
                        prompt_embeds=prompt_embeds,
                        negative_prompt=negative_prompt if not neg_prompt_embeds else None,
                        negative_prompt_embeds=neg_prompt_embeds,
                        image=control_image,
                        num_inference_steps=self.config.generation.num_inference_steps,
                        guidance_scale=self.config.generation.guidance_scale,
                        width=self.config.generation.width,
                        height=self.config.generation.height,
                        num_images_per_prompt=self.config.generation.num_images,
                        generator=generator,
                        controlnet_conditioning_scale=self.config.generation.controlnet_conditioning_scale,
                        **kwargs
                    ).images
                else:
                    result = self.pipeline(
                        prompt=prompt if not prompt_embeds else None,
                        prompt_embeds=prompt_embeds,
                        negative_prompt=negative_prompt if not neg_prompt_embeds else None,
                        negative_prompt_embeds=neg_prompt_embeds,
                        num_inference_steps=self.config.generation.num_inference_steps,
                        guidance_scale=self.config.generation.guidance_scale,
                        width=self.config.generation.width,
                        height=self.config.generation.height,
                        num_images_per_prompt=self.config.generation.num_images,
                        generator=generator,
                        **kwargs
                    ).images
                
                logger.info("Image generation completed successfully")
                return result[0] if len(result) == 1 else result
                
        except Exception as e:
            logger.exception("Image generation failed")
            raise RuntimeError(f"Image generation failed: {str(e)}")
    
    def generate_diagram(
        self,
        description: str,
        style: str = "flowchart",
        **kwargs
    ) -> Image.Image:
        """
        Generate a diagram from a text description.
        
        Args:
            description: Text description of the desired diagram
            style: Type of diagram (flowchart, mindmap, etc.)
            **kwargs: Additional generation parameters
            
        Returns:
            PIL Image of the generated diagram
        """
        if not GENERATION_AVAILABLE:
            raise RuntimeError("Image generation is not available. Please install required dependencies.")

        # Construct a specialized prompt for diagram generation
        diagram_prompt = f"A clean, professional {style} diagram showing {description}. " \
                        f"Minimalist design, high contrast, clear text and arrows."
        
        # Add style-specific negative prompts
        negative_prompt = "blurry, photo-realistic, complex background, " \
                        "natural scene, artistic, painterly"
        
        # Generate with specific settings for diagrams
        return self.generate_image(
            prompt=diagram_prompt,
            negative_prompt=negative_prompt,
            width=1024,  # Larger size for diagrams
            height=768,
            guidance_scale=8.5,  # Higher guidance for more precise output
            num_inference_steps=60,  # More steps for detail
            **kwargs
        )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to the image.
        """
        try:
            processed = image.copy()
            
            if self.config.preprocessing:
                # Convert to grayscale if not already
                if len(processed.shape) == 3:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                
                # Enhance contrast if enabled
                if self.config.contrast_enhance:
                    logger.debug("Enhancing contrast")
                    processed = cv2.equalizeHist(processed)
                
                # Apply denoising if enabled
                if self.config.denoise:
                    logger.debug("Applying denoising")
                    processed = cv2.fastNlMeansDenoising(processed)
            
            return processed
            
        except Exception as e:
            logger.error("Error in image preprocessing: {}", str(e))
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def _load_image(self, image_input: Union[str, Path, BinaryIO, np.ndarray]) -> np.ndarray:
        """
        Load image from various input types and convert to numpy array.
        """
        try:
            if isinstance(image_input, (str, Path)):
                # Load from file path
                image = cv2.imread(str(image_input))
                if image is None:
                    raise ValueError(f"Failed to load image from {image_input}")
                return image
            elif isinstance(image_input, np.ndarray):
                # Already a numpy array
                return image_input.copy()
            else:
                # Assume it's a file-like object, save to temp file first
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_file.write(image_input.read())
                    temp_path = temp_file.name
                
                try:
                    image = cv2.imread(temp_path)
                    if image is None:
                        raise ValueError(f"Failed to load image from file-like object")
                    return image
                finally:
                    os.unlink(temp_path)  # Clean up temp file
            
        except Exception as e:
            logger.error("Error loading image: {}", str(e))
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, BinaryIO],
        **kwargs
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Extract text from an image using OCR.
        
        Args:
            image: Input image as file path, PIL Image, numpy array, or file object
            **kwargs: Additional parameters to pass to the OCR reader
            
        Returns:
            If return_bboxes and return_confidence are False, returns the extracted text as a string.
            Otherwise, returns a list of dictionaries containing text, bounding boxes, and confidence scores.
        """
        try:
            with logger.contextualize(operation="ocr"):
                logger.info("Starting OCR text extraction")
                
                # Load and preprocess image
                if isinstance(image, (str, Path)):
                    logger.debug("Loading image from path: {}", image)
                    image = cv2.imread(str(image))
                elif isinstance(image, Image.Image):
                    logger.debug("Converting PIL Image to numpy array")
                    image = np.array(image)
                elif isinstance(image, BinaryIO):
                    logger.debug("Loading image from file object")
                    image = cv2.imdecode(
                        np.frombuffer(image.read(), np.uint8),
                        cv2.IMREAD_COLOR
                    )
                
                if image is None:
                    raise ValueError("Failed to load image")
                
                # Preprocess image
                processed_image = self._preprocess_image(image)
                
                # Perform OCR
                logger.debug("Running OCR")
                results = self.reader.readtext(
                    processed_image,
                    detail=self.config.return_bboxes or self.config.return_confidence,
                    **kwargs
                )
                
                # Format results
                if not (self.config.return_bboxes or self.config.return_confidence):
                    # Return simple text string
                    text = " ".join([result[1] for result in results])
                    logger.info("OCR completed successfully")
                    return text
                else:
                    # Return detailed results
                    formatted_results = []
                    for bbox, text, conf in results:
                        if conf >= self.config.min_confidence:
                            result = {"text": text}
                            if self.config.return_bboxes:
                                result["bbox"] = bbox
                            if self.config.return_confidence:
                                result["confidence"] = conf
                            formatted_results.append(result)
                    
                    logger.info("OCR completed successfully")
                    return formatted_results
                
        except Exception as e:
            logger.exception("OCR text extraction failed")
            raise RuntimeError(f"OCR text extraction failed: {str(e)}")
    
    def detect_text_regions(self, 
                          image_input: Union[str, Path, BinaryIO, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect regions containing text in the image.
        
        Args:
            image_input: Image file path, binary data, or numpy array
            
        Returns:
            List of detected regions with coordinates and confidence scores
        """
        try:
            with logger.contextualize(operation="detect_regions"):
                logger.info("Starting text region detection")
                
                # Load and preprocess image
                image = self._load_image(image_input)
                processed_image = self._preprocess_image(image)
                
                # Run detection
                results = self.reader.detect(
                    processed_image,
                    min_size=10,
                    text_threshold=self.config.min_confidence
                )
                
                # Format regions
                regions = []
                if results is not None and len(results) == 2:
                    boxes, scores = results
                    for box, score in zip(boxes, scores):
                        if score >= self.config.min_confidence:
                            regions.append({
                                "bbox": box.tolist(),
                                "confidence": float(score)
                            })
                
                logger.info("Detected {} text regions", len(regions))
                return regions
                
        except Exception as e:
            logger.exception("Region detection failed")
            raise RuntimeError(f"Region detection failed: {str(e)}")
    
    def supported_languages(self) -> List[str]:
        """List supported OCR languages."""
        return self.reader.lang_list 