"""
Audio Processor - Post-processing for generated audio.

Handles normalization, trimming, fading, and other audio processing.
"""

import numpy as np
from typing import Optional, Tuple


class AudioProcessor:
    """Audio post-processing utilities."""
    
    def __init__(
        self,
        target_db: float = -3.0,
        silence_threshold_db: float = -60.0,
    ):
        """
        Initialize the audio processor.
        
        Args:
            target_db: Target level for normalization in dB
            silence_threshold_db: Threshold for silence detection in dB
        """
        self.target_db = target_db
        self.silence_threshold_db = silence_threshold_db
    
    def normalize(
        self,
        audio: np.ndarray,
        target_db: Optional[float] = None,
    ) -> np.ndarray:
        """
        Normalize audio to a target level.
        
        Args:
            audio: Audio data as numpy array
            target_db: Target level in dB (uses default if None)
            
        Returns:
            Normalized audio
        """
        if target_db is None:
            target_db = self.target_db
        
        # Find peak
        peak = np.max(np.abs(audio))
        
        if peak == 0:
            return audio
        
        # Calculate target amplitude
        target_amplitude = 10 ** (target_db / 20)
        
        # Normalize
        normalized = audio * (target_amplitude / peak)
        
        return normalized
    
    def trim_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold_db: Optional[float] = None,
        min_silence_ms: int = 100,
        keep_silence_ms: int = 50,
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            threshold_db: Silence threshold in dB
            min_silence_ms: Minimum silence duration to consider
            keep_silence_ms: Amount of silence to keep at edges
            
        Returns:
            Trimmed audio
        """
        if threshold_db is None:
            threshold_db = self.silence_threshold_db
        
        # Convert threshold to amplitude
        threshold = 10 ** (threshold_db / 20)
        
        # Find non-silent regions
        amplitude = np.abs(audio)
        non_silent = amplitude > threshold
        
        if not np.any(non_silent):
            return audio
        
        # Find first and last non-silent samples
        non_silent_indices = np.where(non_silent)[0]
        start = non_silent_indices[0]
        end = non_silent_indices[-1]
        
        # Add some padding
        keep_samples = int(keep_silence_ms * sample_rate / 1000)
        start = max(0, start - keep_samples)
        end = min(len(audio), end + keep_samples)
        
        return audio[start:end]
    
    def apply_fade(
        self,
        audio: np.ndarray,
        sample_rate: int,
        fade_in_ms: int = 10,
        fade_out_ms: int = 50,
    ) -> np.ndarray:
        """
        Apply fade in and fade out to audio.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            fade_in_ms: Fade in duration in milliseconds
            fade_out_ms: Fade out duration in milliseconds
            
        Returns:
            Audio with fades applied
        """
        audio = audio.copy()
        
        # Fade in
        if fade_in_ms > 0:
            fade_in_samples = int(fade_in_ms * sample_rate / 1000)
            fade_in_samples = min(fade_in_samples, len(audio))
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_in_curve
        
        # Fade out
        if fade_out_ms > 0:
            fade_out_samples = int(fade_out_ms * sample_rate / 1000)
            fade_out_samples = min(fade_out_samples, len(audio))
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_out_curve
        
        return audio
    
    def resample(
        self,
        audio: np.ndarray,
        original_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """
        Resample audio to a different sample rate.
        
        Args:
            audio: Audio data as numpy array
            original_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if original_sr == target_sr:
            return audio
        
        try:
            import scipy.signal as signal
            
            # Calculate resampling ratio
            ratio = target_sr / original_sr
            new_length = int(len(audio) * ratio)
            
            resampled = signal.resample(audio, new_length)
            return resampled
            
        except ImportError:
            raise ImportError("scipy is required for resampling. Install with: pip install scipy")
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono.
        
        Args:
            audio: Audio data (1D or 2D array)
            
        Returns:
            Mono audio
        """
        if audio.ndim == 1:
            return audio
        
        if audio.ndim == 2:
            if audio.shape[0] == 2:
                # [2, samples] format
                return np.mean(audio, axis=0)
            elif audio.shape[1] == 2:
                # [samples, 2] format
                return np.mean(audio, axis=1)
        
        return audio
    
    def convert_to_stereo(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert mono audio to stereo.
        
        Args:
            audio: Mono audio data
            
        Returns:
            Stereo audio [2, samples]
        """
        if audio.ndim == 1:
            return np.stack([audio, audio])
        return audio
    
    def get_duration(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Get the duration of audio in seconds.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Duration in seconds
        """
        if audio.ndim == 2:
            samples = audio.shape[-1]
        else:
            samples = len(audio)
        
        return samples / sample_rate
    
    def get_peak_db(self, audio: np.ndarray) -> float:
        """
        Get the peak level in dB.
        
        Args:
            audio: Audio data
            
        Returns:
            Peak level in dB
        """
        peak = np.max(np.abs(audio))
        if peak == 0:
            return -np.inf
        return 20 * np.log10(peak)
    
    def get_rms_db(self, audio: np.ndarray) -> float:
        """
        Get the RMS level in dB.
        
        Args:
            audio: Audio data
            
        Returns:
            RMS level in dB
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms)
    
    def apply_gain(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """
        Apply gain to audio.
        
        Args:
            audio: Audio data
            gain_db: Gain in dB
            
        Returns:
            Audio with gain applied
        """
        gain = 10 ** (gain_db / 20)
        return audio * gain
    
    def clip(self, audio: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """
        Clip audio to prevent clipping distortion.
        
        Args:
            audio: Audio data
            threshold: Clipping threshold
            
        Returns:
            Clipped audio
        """
        return np.clip(audio, -threshold, threshold)
