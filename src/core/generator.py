"""
Audio Generator - Main generation logic combining model and prompt processing.

This is the primary interface for generating audio samples.
"""

from pathlib import Path
from typing import Optional, Union, List
import torch
import numpy as np

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .model_manager import ModelManager
from .prompt_processor import PromptProcessor
from ..audio.processor import AudioProcessor
from ..audio.export import AudioExporter

console = Console()

# Instrument definitions for clean sample generation
INSTRUMENT_PRESETS = {
    # Drums & Percussion
    "kick": {
        "prompt": "single isolated kick drum hit, clean punchy drum sample, studio quality, no reverb",
        "duration": 1.5,
        "temperature": 0.7,
        "cfg_coef": 4.5,
    },
    "snare": {
        "prompt": "single isolated snare drum hit, clean crisp drum sample, studio quality, no reverb",
        "duration": 1.5,
        "temperature": 0.7,
        "cfg_coef": 4.5,
    },
    "hihat": {
        "prompt": "single isolated hi-hat cymbal hit, clean crisp metallic, studio quality, no reverb",
        "duration": 1.0,
        "temperature": 0.7,
        "cfg_coef": 4.0,
    },
    "clap": {
        "prompt": "single isolated hand clap, clean percussive, studio quality, no reverb",
        "duration": 1.0,
        "temperature": 0.7,
        "cfg_coef": 4.0,
    },
    "tom": {
        "prompt": "single isolated tom drum hit, clean resonant, studio quality, no reverb",
        "duration": 2.0,
        "temperature": 0.7,
        "cfg_coef": 4.5,
    },
    "cymbal": {
        "prompt": "single isolated crash cymbal hit, clean metallic, studio quality",
        "duration": 3.0,
        "temperature": 0.7,
        "cfg_coef": 4.0,
    },
    "rim": {
        "prompt": "single isolated rim shot, clean sharp percussive, studio quality, no reverb",
        "duration": 1.0,
        "temperature": 0.7,
        "cfg_coef": 4.5,
    },
    # Bass
    "bass": {
        "prompt": "single sustained bass note, clean sub bass, deep low frequency, studio quality",
        "duration": 3.0,
        "temperature": 0.8,
        "cfg_coef": 4.0,
    },
    "sub": {
        "prompt": "single pure sub bass tone, clean deep sine wave like, very low frequency",
        "duration": 3.0,
        "temperature": 0.6,
        "cfg_coef": 5.0,
    },
    # Synths
    "lead": {
        "prompt": "single sustained synthesizer lead note, clean melodic tone, studio quality",
        "duration": 3.0,
        "temperature": 0.8,
        "cfg_coef": 3.5,
    },
    "pad": {
        "prompt": "single sustained synthesizer pad chord, clean warm evolving texture, ambient",
        "duration": 5.0,
        "temperature": 0.9,
        "cfg_coef": 3.0,
    },
    "pluck": {
        "prompt": "single plucked synthesizer note, clean short staccato melodic, studio quality",
        "duration": 2.0,
        "temperature": 0.7,
        "cfg_coef": 4.0,
    },
    "keys": {
        "prompt": "single piano or keyboard note, clean acoustic, studio quality, no reverb",
        "duration": 3.0,
        "temperature": 0.7,
        "cfg_coef": 4.0,
    },
    "strings": {
        "prompt": "single sustained orchestral string note, clean legato, studio quality",
        "duration": 4.0,
        "temperature": 0.8,
        "cfg_coef": 3.5,
    },
    # FX
    "riser": {
        "prompt": "single build up riser effect, clean tension increasing energy, cinematic",
        "duration": 4.0,
        "temperature": 0.9,
        "cfg_coef": 3.0,
    },
    "impact": {
        "prompt": "single cinematic impact hit, clean powerful boom, studio quality",
        "duration": 2.0,
        "temperature": 0.7,
        "cfg_coef": 4.5,
    },
    "sweep": {
        "prompt": "single filter sweep effect, clean whoosh transition, studio quality",
        "duration": 3.0,
        "temperature": 0.8,
        "cfg_coef": 3.5,
    },
    "noise": {
        "prompt": "single white noise burst, clean textural, studio quality",
        "duration": 2.0,
        "temperature": 0.7,
        "cfg_coef": 4.0,
    },
}


class AudioGenerator:
    """Main audio generation class."""
    
    def __init__(
        self,
        model_name: str = "musicgen-medium",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        enhance_prompts: bool = True,
    ):
        """
        Initialize the audio generator.
        
        Args:
            model_name: Name of the model to use
            device: Device to run on (cuda/cpu)
            cache_dir: Directory to cache model files
            enhance_prompts: Whether to enhance prompts automatically
        """
        self.model_manager = ModelManager(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
        )
        self.prompt_processor = PromptProcessor(enhance_prompts=enhance_prompts)
        self.audio_processor = AudioProcessor()
        self.audio_exporter = AudioExporter()
        
        self._model_loaded = False
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded before generation."""
        if not self._model_loaded:
            self.model_manager.load_model()
            self._model_loaded = True
    
    def generate(
        self,
        prompt: str,
        duration: Optional[float] = None,
        bpm: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.0,
        variations: int = 1,
        seed: Optional[int] = None,
        use_recommended_params: bool = False,
    ) -> List[np.ndarray]:
        """
        Generate audio from a text prompt.
        
        Args:
            prompt: Natural language description of the desired sound
            duration: Duration in seconds (auto-detected if None)
            bpm: BPM for rhythmic content
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            cfg_coef: Classifier-free guidance coefficient
            variations: Number of variations to generate
            seed: Random seed for reproducibility
            use_recommended_params: Auto-adjust params based on prompt
            
        Returns:
            List of generated audio as numpy arrays
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Auto-detect duration if not specified
        if duration is None:
            duration = self.prompt_processor.suggest_duration(prompt)
            console.print(f"[dim]Auto-detected duration: {duration}s[/dim]")
        
        # Get recommended parameters if requested
        if use_recommended_params:
            rec_params = self.prompt_processor.get_recommended_params(prompt)
            temperature = rec_params.get("temperature", temperature)
            top_k = rec_params.get("top_k", top_k)
            cfg_coef = rec_params.get("cfg_coef", cfg_coef)
        
        # Process and enhance the prompt
        enhanced_prompt = self.prompt_processor.process(prompt, duration, bpm)
        
        console.print(f"\n[cyan]Generating:[/cyan] {prompt}")
        console.print(f"[dim]Enhanced prompt: {enhanced_prompt}[/dim]")
        console.print(f"[dim]Duration: {duration}s | Temp: {temperature} | CFG: {cfg_coef}[/dim]")
        
        # Set generation parameters
        self.model_manager.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
        )
        
        # Create prompt list for batch generation
        prompts = [enhanced_prompt] * variations
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating audio...", total=None)
            
            # Generate using model manager
            output = self.model_manager.generate(prompts)
            
            progress.update(task, completed=True)
        
        # Convert to list of audio outputs
        audio_outputs = []
        sample_rate = self.model_manager.sample_rate
        
        for i in range(output.shape[0]):
            audio_data = output[i]
            
            # MusicGen outputs [channels, samples], we might need to squeeze
            if audio_data.ndim > 1:
                # Take first channel or average if stereo
                if audio_data.shape[0] <= 2:
                    audio_data = audio_data[0]  # Take first channel
            
            audio_outputs.append({
                "audio": audio_data,
                "sample_rate": sample_rate,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "duration": duration,
            })
        
        console.print(f"[green]✓ Generated {len(audio_outputs)} sample(s)[/green]")
        
        return audio_outputs
    
    def generate_and_save(
        self,
        prompt: str,
        output_dir: Union[str, Path] = "./output",
        output_format: str = "wav",
        filename: Optional[str] = None,
        duration: Optional[float] = None,
        bpm: Optional[int] = None,
        normalize: bool = True,
        trim_silence: bool = True,
        fade_in_ms: int = 10,
        fade_out_ms: int = 50,
        variations: int = 1,
        **generation_kwargs,
    ) -> List[Path]:
        """
        Generate audio and save to files.
        
        Args:
            prompt: Natural language description
            output_dir: Directory to save files
            output_format: Output format (wav, mp3, flac)
            filename: Custom filename (auto-generated if None)
            duration: Duration in seconds
            bpm: BPM for rhythmic content
            normalize: Whether to normalize audio
            trim_silence: Whether to trim silence
            fade_in_ms: Fade in duration in milliseconds
            fade_out_ms: Fade out duration in milliseconds
            variations: Number of variations to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of paths to saved files
        """
        # Generate audio
        outputs = self.generate(
            prompt=prompt,
            duration=duration,
            bpm=bpm,
            variations=variations,
            **generation_kwargs,
        )
        
        saved_paths = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, output in enumerate(outputs):
            audio = output["audio"]
            sample_rate = output["sample_rate"]
            
            # Apply post-processing
            if normalize:
                audio = self.audio_processor.normalize(audio)
            
            if trim_silence:
                audio = self.audio_processor.trim_silence(audio, sample_rate)
            
            if fade_in_ms > 0 or fade_out_ms > 0:
                audio = self.audio_processor.apply_fade(
                    audio, sample_rate, fade_in_ms, fade_out_ms
                )
            
            # Generate filename
            if filename:
                if variations > 1:
                    base_name = f"{filename}_{i+1}"
                else:
                    base_name = filename
            else:
                base_name = self.audio_exporter.generate_filename(prompt, i if variations > 1 else None)
            
            # Save file
            file_path = self.audio_exporter.save(
                audio=audio,
                sample_rate=sample_rate,
                output_dir=output_dir,
                filename=base_name,
                format=output_format,
            )
            
            saved_paths.append(file_path)
            console.print(f"[green]✓ Saved:[/green] {file_path}")
        
        return saved_paths
    
    def generate_clean_sample(
        self,
        instrument: str,
        style: Optional[str] = None,
        duration: Optional[float] = None,
        variations: int = 1,
        seed: Optional[int] = None,
    ) -> List[dict]:
        """
        Generate clean, isolated instrument samples optimized for use in DAWs.
        
        Args:
            instrument: Type of instrument (kick, snare, bass, lead, etc.)
            style: Optional style modifier (e.g., "808", "analog", "acoustic")
            duration: Override default duration for the instrument
            variations: Number of variations to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of generated audio outputs
        """
        instrument = instrument.lower().strip()
        
        # Get instrument preset or use default
        if instrument in INSTRUMENT_PRESETS:
            preset = INSTRUMENT_PRESETS[instrument].copy()
        else:
            # Generic fallback for unknown instruments
            preset = {
                "prompt": f"single isolated {instrument} sample, clean studio quality, no reverb",
                "duration": 3.0,
                "temperature": 0.75,
                "cfg_coef": 4.0,
            }
            console.print(f"[yellow]Unknown instrument '{instrument}', using generic preset[/yellow]")
        
        # Build enhanced prompt with style
        base_prompt = preset["prompt"]
        if style:
            # Insert style into prompt
            style_additions = {
                "808": "808 style, hip hop, trap, punchy",
                "analog": "analog synthesizer, warm vintage",
                "digital": "digital synthesizer, modern clean",
                "acoustic": "acoustic natural organic",
                "electronic": "electronic synthesized",
                "dark": "dark moody low-passed",
                "bright": "bright airy high frequency presence",
                "aggressive": "aggressive distorted hard",
                "soft": "soft gentle mellow",
                "vintage": "vintage retro classic warm",
                "modern": "modern contemporary clean",
                "lo-fi": "lo-fi crunchy textured warm",
                "hi-fi": "hi-fi pristine crystal clear",
            }
            style_desc = style_additions.get(style.lower(), style)
            base_prompt = f"{base_prompt}, {style_desc}"
        
        # Use provided duration or preset default
        sample_duration = duration if duration is not None else preset["duration"]
        
        console.print(f"\n[cyan]Generating clean {instrument} sample[/cyan]")
        if style:
            console.print(f"[dim]Style: {style}[/dim]")
        console.print(f"[dim]Duration: {sample_duration}s | Variations: {variations}[/dim]")
        
        # Generate with optimized settings for clean samples
        return self.generate(
            prompt=base_prompt,
            duration=sample_duration,
            temperature=preset["temperature"],
            cfg_coef=preset["cfg_coef"],
            top_k=200,  # Slightly lower for more focused output
            variations=variations,
            seed=seed,
        )
    
    def generate_clean_sample_and_save(
        self,
        instrument: str,
        output_dir: Union[str, Path] = "./output",
        output_format: str = "wav",
        style: Optional[str] = None,
        duration: Optional[float] = None,
        variations: int = 1,
        normalize: bool = True,
        trim_silence: bool = True,
        seed: Optional[int] = None,
    ) -> List[Path]:
        """
        Generate clean instrument samples and save to files.
        
        Args:
            instrument: Type of instrument (kick, snare, bass, lead, etc.)
            output_dir: Directory to save files
            output_format: Output format (wav, mp3, flac)
            style: Optional style modifier
            duration: Override default duration
            variations: Number of variations to generate
            normalize: Whether to normalize audio
            trim_silence: Whether to trim silence
            seed: Random seed for reproducibility
            
        Returns:
            List of paths to saved files
        """
        outputs = self.generate_clean_sample(
            instrument=instrument,
            style=style,
            duration=duration,
            variations=variations,
            seed=seed,
        )
        
        saved_paths = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, output in enumerate(outputs):
            audio = output["audio"]
            sample_rate = output["sample_rate"]
            
            # Apply post-processing
            if normalize:
                audio = self.audio_processor.normalize(audio)
            
            if trim_silence:
                audio = self.audio_processor.trim_silence(audio, sample_rate)
            
            # Apply gentle fades for clean samples
            audio = self.audio_processor.apply_fade(
                audio, sample_rate, fade_in_ms=5, fade_out_ms=30
            )
            
            # Generate filename
            style_suffix = f"-{style}" if style else ""
            if variations > 1:
                base_name = f"{instrument}{style_suffix}_{i+1}"
            else:
                base_name = f"{instrument}{style_suffix}"
            
            # Save file
            file_path = self.audio_exporter.save(
                audio=audio,
                sample_rate=sample_rate,
                output_dir=output_dir,
                filename=base_name,
                format=output_format,
            )
            
            saved_paths.append(file_path)
            console.print(f"[green]✓ Saved:[/green] {file_path}")
        
        return saved_paths
    
    @staticmethod
    def list_instruments() -> dict:
        """List all available instrument presets."""
        return {
            name: {
                "duration": preset["duration"],
                "description": preset["prompt"].split(",")[0],
            }
            for name, preset in INSTRUMENT_PRESETS.items()
        }
    
    def unload(self):
        """Unload the model and free memory."""
        self.model_manager.unload_model()
        self._model_loaded = False


def calculate_bars_duration(bars: int, bpm: int) -> float:
    """
    Calculate duration in seconds for a given number of bars.
    
    Args:
        bars: Number of bars
        bpm: Tempo in beats per minute
        
    Returns:
        Duration in seconds
    """
    beats_per_bar = 4  # Assuming 4/4 time
    total_beats = bars * beats_per_bar
    seconds_per_beat = 60.0 / bpm
    return total_beats * seconds_per_beat
