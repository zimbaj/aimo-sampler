"""
Audio Exporter - File export utilities for generated audio.

Supports WAV, MP3, and FLAC formats.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np


class AudioExporter:
    """Audio file export utilities."""
    
    SUPPORTED_FORMATS = ["wav", "mp3", "flac", "ogg"]
    
    def __init__(self, default_sample_rate: int = 44100):
        """
        Initialize the audio exporter.
        
        Args:
            default_sample_rate: Default sample rate for export
        """
        self.default_sample_rate = default_sample_rate
    
    def save(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_dir: Union[str, Path],
        filename: str,
        format: str = "wav",
        mp3_bitrate: int = 320,
    ) -> Path:
        """
        Save audio to a file.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            output_dir: Directory to save the file
            filename: Filename without extension
            format: Output format (wav, mp3, flac, ogg)
            mp3_bitrate: Bitrate for MP3 encoding
            
        Returns:
            Path to the saved file
        """
        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean filename
        clean_filename = self._sanitize_filename(filename)
        file_path = output_dir / f"{clean_filename}.{format}"
        
        # Ensure audio is in correct format
        audio = self._prepare_audio(audio)
        
        if format == "wav":
            self._save_wav(audio, sample_rate, file_path)
        elif format == "mp3":
            self._save_mp3(audio, sample_rate, file_path, mp3_bitrate)
        elif format == "flac":
            self._save_flac(audio, sample_rate, file_path)
        elif format == "ogg":
            self._save_ogg(audio, sample_rate, file_path)
        
        return file_path
    
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Prepare audio for saving (convert to correct dtype)."""
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _save_wav(self, audio: np.ndarray, sample_rate: int, file_path: Path):
        """Save as WAV file."""
        try:
            import soundfile as sf
            sf.write(str(file_path), audio, sample_rate, subtype='PCM_24')
        except ImportError:
            # Fallback to scipy
            from scipy.io import wavfile
            # Convert to int16 for scipy
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(str(file_path), sample_rate, audio_int)
    
    def _save_mp3(self, audio: np.ndarray, sample_rate: int, file_path: Path, bitrate: int):
        """Save as MP3 file."""
        try:
            from pydub import AudioSegment
            
            # Convert to int16
            audio_int = (audio * 32767).astype(np.int16)
            
            # Create AudioSegment
            audio_segment = AudioSegment(
                audio_int.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit
                channels=1,
            )
            
            # Export as MP3
            audio_segment.export(
                str(file_path),
                format="mp3",
                bitrate=f"{bitrate}k",
            )
        except ImportError:
            raise ImportError(
                "pydub is required for MP3 export. Install with: pip install pydub\n"
                "Also ensure ffmpeg is installed on your system."
            )
    
    def _save_flac(self, audio: np.ndarray, sample_rate: int, file_path: Path):
        """Save as FLAC file."""
        try:
            import soundfile as sf
            sf.write(str(file_path), audio, sample_rate, format='FLAC')
        except ImportError:
            raise ImportError(
                "soundfile is required for FLAC export. Install with: pip install soundfile"
            )
    
    def _save_ogg(self, audio: np.ndarray, sample_rate: int, file_path: Path):
        """Save as OGG file."""
        try:
            import soundfile as sf
            sf.write(str(file_path), audio, sample_rate, format='OGG')
        except ImportError:
            try:
                from pydub import AudioSegment
                
                audio_int = (audio * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1,
                )
                audio_segment.export(str(file_path), format="ogg")
            except ImportError:
                raise ImportError(
                    "soundfile or pydub is required for OGG export."
                )
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Remove multiple underscores
        filename = re.sub(r'_+', '_', filename)
        # Trim to reasonable length
        filename = filename[:100]
        # Remove leading/trailing underscores
        filename = filename.strip('_')
        
        return filename
    
    def generate_filename(
        self,
        prompt: str,
        variation_index: Optional[int] = None,
        timestamp: bool = True,
    ) -> str:
        """
        Generate a filename from a prompt.
        
        Args:
            prompt: The generation prompt
            variation_index: Index for variations (None if single)
            timestamp: Whether to include timestamp
            
        Returns:
            Generated filename
        """
        # Extract key words from prompt
        words = prompt.lower().split()[:4]  # First 4 words
        base_name = "_".join(words)
        base_name = self._sanitize_filename(base_name)
        
        # Add timestamp
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{ts}"
        
        # Add variation index
        if variation_index is not None:
            base_name = f"{base_name}_v{variation_index + 1}"
        
        return base_name
    
    def get_audio_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get information about an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with audio information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            import soundfile as sf
            
            info = sf.info(str(file_path))
            return {
                "path": str(file_path),
                "format": info.format,
                "subtype": info.subtype,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration": info.duration,
                "duration_str": f"{info.duration:.2f}s",
            }
        except ImportError:
            # Basic info without soundfile
            return {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "extension": file_path.suffix,
            }
