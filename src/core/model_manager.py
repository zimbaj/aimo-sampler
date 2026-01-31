"""
Model Manager - Handles loading and caching of AI models.

Uses HuggingFace Transformers for MusicGen with efficient memory management.
"""

import gc
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from rich.console import Console

console = Console()


class ModelManager:
    """Manages AI model loading, caching, and memory optimization."""
    
    SUPPORTED_MODELS = {
        "musicgen-small": "facebook/musicgen-small",
        "musicgen-medium": "facebook/musicgen-medium", 
        "musicgen-large": "facebook/musicgen-large",
        "musicgen-melody": "facebook/musicgen-melody",
        "musicgen-stereo-small": "facebook/musicgen-stereo-small",
        "musicgen-stereo-medium": "facebook/musicgen-stereo-medium",
    }
    
    def __init__(
        self,
        model_name: str = "musicgen-medium",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the model manager.
        
        Args:
            model_name: Name of the model (musicgen-small/medium/large)
            device: Device to run on (cuda/cpu)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device if cuda requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            console.print("[yellow]⚠ CUDA not available, falling back to CPU[/yellow]")
            console.print("[dim]For GPU acceleration, install PyTorch with CUDA:[/dim]")
            console.print("[dim]pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118[/dim]")
            self.device = "cpu"
        else:
            self.device = device
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                console.print(f"[green]✓ Using GPU: {gpu_name}[/green]")
        
        self._model = None
        self._processor = None
        self._sample_rate = 32000  # MusicGen default
        
        # Generation parameters
        self._duration = 5.0
        self._temperature = 1.0
        self._top_k = 250
        self._top_p = 0.0
        self._cfg_coef = 3.0
        
    @property
    def model_id(self) -> str:
        """Get the HuggingFace model ID."""
        if self.model_name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[self.model_name]
        # Allow direct model IDs
        return self.model_name
    
    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        return self._sample_rate
    
    def _check_gpu_memory(self) -> dict:
        """Check available GPU memory."""
        if not torch.cuda.is_available():
            return {"available": False, "total": 0, "free": 0}
        
        torch.cuda.empty_cache()
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = total - allocated
        
        return {
            "available": True,
            "total": total / 1e9,  # GB
            "free": free / 1e9,
            "allocated": allocated / 1e9,
        }
    
    def load_model(self, force_reload: bool = False):
        """
        Load the MusicGen model using transformers.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded model instance
        """
        if self._model is not None and not force_reload:
            return self._model
        
        # Unload existing model first
        if self._model is not None:
            self.unload_model()
        
        console.print(f"[cyan]Loading model: {self.model_id}[/cyan]")
        
        # Check GPU memory
        gpu_info = self._check_gpu_memory()
        if gpu_info["available"]:
            console.print(
                f"[dim]GPU Memory: {gpu_info['free']:.1f}GB free / "
                f"{gpu_info['total']:.1f}GB total[/dim]"
            )
        
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            
            # Load processor
            console.print("[dim]Loading processor...[/dim]")
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
            )
            
            # Load model
            console.print("[dim]Loading model weights...[/dim]")
            
            # Determine dtype and device
            if self.device == "cuda" and torch.cuda.is_available():
                # Load directly to GPU with float16 for efficiency
                self._model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_id,
                    cache_dir=str(self.cache_dir),
                    torch_dtype=torch.float16,
                    device_map="cuda",
                )
                console.print(f"[green]✓ Model loaded on GPU (cuda)[/green]")
            else:
                # Load to CPU
                self._model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_id,
                    cache_dir=str(self.cache_dir),
                    torch_dtype=torch.float32,
                )
                console.print(f"[yellow]Model loaded on CPU[/yellow]")
            
            # Get sample rate from model config
            self._sample_rate = self._model.config.audio_encoder.sampling_rate
            
            # Verify device
            model_device = next(self._model.parameters()).device
            console.print(f"[dim]Model device: {model_device}[/dim]")
            console.print(f"[dim]Sample rate: {self._sample_rate} Hz[/dim]")
            
            # Report memory usage after loading
            if gpu_info["available"]:
                new_gpu_info = self._check_gpu_memory()
                used = gpu_info["free"] - new_gpu_info["free"]
                console.print(f"[dim]Model using {used:.1f}GB VRAM[/dim]")
            
            return self._model
            
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise
    
    def unload_model(self):
        """Unload the model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        console.print("[dim]Model unloaded, memory freed[/dim]")
    
    def get_model(self):
        """Get the loaded model, loading it if necessary."""
        if self._model is None:
            self.load_model()
        return self._model
    
    def get_processor(self):
        """Get the loaded processor, loading model if necessary."""
        if self._processor is None:
            self.load_model()
        return self._processor
    
    def set_generation_params(
        self,
        duration: float = 5.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.0,
    ):
        """
        Set generation parameters.
        
        Args:
            duration: Duration in seconds
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter (0 = disabled)
            cfg_coef: Classifier-free guidance coefficient
        """
        self._duration = duration
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._cfg_coef = cfg_coef
    
    def generate(self, prompts: List[str]) -> np.ndarray:
        """
        Generate audio from text prompts.
        
        Args:
            prompts: List of text descriptions
            
        Returns:
            Generated audio as numpy array [batch, samples]
        """
        model = self.get_model()
        processor = self.get_processor()
        
        # Process inputs
        inputs = processor(
            text=prompts,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to device
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Calculate max new tokens from duration
        # MusicGen generates at ~50 tokens per second of audio
        tokens_per_second = 50
        max_new_tokens = int(self._duration * tokens_per_second)
        
        # Generate
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p if self._top_p > 0 else None,
                guidance_scale=self._cfg_coef,
            )
        
        # Convert to numpy
        audio_np = audio_values.cpu().numpy()
        
        return audio_np
    
    @staticmethod
    def list_available_models() -> dict:
        """List all available models with their requirements."""
        return {
            "musicgen-small": {
                "id": "facebook/musicgen-small",
                "vram": "~4GB",
                "quality": "Good",
                "speed": "Fast",
            },
            "musicgen-medium": {
                "id": "facebook/musicgen-medium", 
                "vram": "~8GB",
                "quality": "Great",
                "speed": "Medium",
            },
            "musicgen-large": {
                "id": "facebook/musicgen-large",
                "vram": "~16GB", 
                "quality": "Excellent",
                "speed": "Slow",
            },
            "musicgen-melody": {
                "id": "facebook/musicgen-melody",
                "vram": "~8GB",
                "quality": "Great",
                "speed": "Medium",
                "note": "Supports melody conditioning",
            },
            "musicgen-stereo-small": {
                "id": "facebook/musicgen-stereo-small",
                "vram": "~4GB",
                "quality": "Good",
                "speed": "Fast",
                "note": "Stereo output",
            },
            "musicgen-stereo-medium": {
                "id": "facebook/musicgen-stereo-medium",
                "vram": "~8GB",
                "quality": "Great",
                "speed": "Medium",
                "note": "Stereo output",
            },
        }
