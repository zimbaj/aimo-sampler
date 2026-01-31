"""
Prompt Processor - Enhances natural language prompts for better audio generation.

Takes user descriptions like "fat psytrance bass" and expands them with
genre-specific keywords and audio characteristics for optimal model output.
"""

import re
from typing import Optional


class PromptProcessor:
    """Process and enhance natural language prompts for audio generation."""
    
    # Genre-specific enhancements
    GENRE_KEYWORDS = {
        "psytrance": [
            "psychedelic trance", "rolling", "hypnotic", "acid", "303",
            "full-on", "progressive", "goa"
        ],
        "techno": [
            "electronic", "industrial", "minimal", "driving", "hypnotic",
            "warehouse", "berlin", "detroit"
        ],
        "dnb": [
            "drum and bass", "jungle", "breakbeat", "fast tempo", "syncopated",
            "liquid", "neurofunk"
        ],
        "house": [
            "four on the floor", "electronic dance", "disco influenced",
            "deep", "progressive", "tech house"
        ],
        "ambient": [
            "atmospheric", "ethereal", "spacious", "evolving", "textural",
            "drone", "meditation"
        ],
        "dubstep": [
            "heavy bass", "wobble", "aggressive", "half-time", "growling",
            "filthy", "brostep"
        ],
        "trance": [
            "uplifting", "euphoric", "melodic", "epic", "emotional",
            "anthemic", "progressive"
        ],
    }
    
    # Sound type enhancements
    SOUND_ENHANCEMENTS = {
        "bass": [
            "low frequency", "sub", "powerful", "rumbling", "deep"
        ],
        "kick": [
            "drum", "percussive", "punchy", "attack", "transient"
        ],
        "snare": [
            "drum", "percussive", "crack", "snap", "body"
        ],
        "hihat": [
            "hi-hat", "cymbal", "metallic", "crisp"
        ],
        "hi-hat": [
            "cymbal", "metallic", "crisp", "sizzle"
        ],
        "pad": [
            "synthesizer", "sustained", "chord", "harmony", "wash"
        ],
        "lead": [
            "synthesizer", "melody", "prominent", "cutting"
        ],
        "pluck": [
            "synthesizer", "short", "staccato", "melodic"
        ],
        "arp": [
            "arpeggiated", "sequence", "rhythmic", "pattern"
        ],
        "fx": [
            "effect", "sound design", "cinematic"
        ],
        "riser": [
            "build up", "tension", "increasing", "energy"
        ],
        "impact": [
            "hit", "cinematic", "boom", "powerful"
        ],
    }
    
    # Characteristic enhancements
    CHARACTERISTIC_MAP = {
        "fat": ["thick", "heavy", "saturated", "wide"],
        "thin": ["narrow", "light", "airy"],
        "punchy": ["tight", "transient", "attack", "impactful"],
        "soft": ["gentle", "smooth", "mellow"],
        "hard": ["aggressive", "harsh", "distorted"],
        "clean": ["pure", "clear", "undistorted"],
        "dirty": ["distorted", "gritty", "saturated", "overdriven"],
        "warm": ["analog", "saturated", "rich"],
        "cold": ["digital", "sterile", "clinical"],
        "dark": ["low-passed", "muted", "deep", "mysterious"],
        "bright": ["high frequency", "sparkly", "crisp", "airy"],
        "aggressive": ["harsh", "intense", "powerful", "driving"],
        "chill": ["relaxed", "laid-back", "smooth"],
        "epic": ["cinematic", "grand", "powerful", "dramatic"],
        "crispy": ["high frequency detail", "sharp", "defined"],
        "rolling": ["continuous", "flowing", "rhythmic", "hypnotic"],
        "growling": ["modulated", "aggressive", "morphing"],
    }
    
    def __init__(self, enhance_prompts: bool = True):
        """
        Initialize the prompt processor.
        
        Args:
            enhance_prompts: Whether to automatically enhance prompts
        """
        self.enhance_prompts = enhance_prompts
    
    def process(
        self,
        prompt: str,
        duration: Optional[float] = None,
        bpm: Optional[int] = None,
    ) -> str:
        """
        Process and enhance a user prompt.
        
        Args:
            prompt: User's natural language description
            duration: Target duration in seconds (for context)
            bpm: Target BPM (for rhythmic content)
            
        Returns:
            Enhanced prompt string
        """
        if not self.enhance_prompts:
            return prompt
        
        # Normalize the prompt
        prompt_lower = prompt.lower()
        enhanced_parts = [prompt]
        
        # Detect and add genre enhancements
        for genre, keywords in self.GENRE_KEYWORDS.items():
            if genre in prompt_lower:
                # Add 2-3 genre-specific keywords
                genre_additions = keywords[:3]
                enhanced_parts.extend(genre_additions)
                break
        
        # Detect and add sound type enhancements
        for sound_type, keywords in self.SOUND_ENHANCEMENTS.items():
            if sound_type in prompt_lower:
                # Add relevant sound characteristics
                sound_additions = keywords[:2]
                enhanced_parts.extend(sound_additions)
                break
        
        # Detect and add characteristic enhancements
        for characteristic, synonyms in self.CHARACTERISTIC_MAP.items():
            if characteristic in prompt_lower:
                # Add 1-2 synonyms for better understanding
                enhanced_parts.extend(synonyms[:2])
        
        # Add production context
        enhanced_parts.append("high quality audio")
        enhanced_parts.append("professional production")
        
        # Add duration context for very short samples
        if duration and duration < 2.0:
            enhanced_parts.append("short sample")
            enhanced_parts.append("one-shot")
        
        # Add BPM context for rhythmic content
        if bpm:
            enhanced_parts.append(f"{bpm} bpm")
        
        # Combine and deduplicate
        final_prompt = ", ".join(dict.fromkeys(enhanced_parts))
        
        return final_prompt
    
    def extract_sound_type(self, prompt: str) -> Optional[str]:
        """
        Extract the primary sound type from a prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Detected sound type or None
        """
        prompt_lower = prompt.lower()
        
        for sound_type in self.SOUND_ENHANCEMENTS.keys():
            if sound_type in prompt_lower:
                return sound_type
        
        return None
    
    def extract_genre(self, prompt: str) -> Optional[str]:
        """
        Extract the genre from a prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Detected genre or None
        """
        prompt_lower = prompt.lower()
        
        for genre in self.GENRE_KEYWORDS.keys():
            if genre in prompt_lower:
                return genre
        
        return None
    
    def suggest_duration(self, prompt: str, default: float = 5.0) -> float:
        """
        Suggest an appropriate duration based on the prompt.
        
        Args:
            prompt: User prompt
            default: Default duration if no suggestion
            
        Returns:
            Suggested duration in seconds
        """
        prompt_lower = prompt.lower()
        
        # One-shot sounds
        one_shots = ["kick", "snare", "clap", "hit", "impact", "shot"]
        if any(word in prompt_lower for word in one_shots):
            return 1.5
        
        # Short sounds
        short_sounds = ["hi-hat", "hihat", "pluck", "stab"]
        if any(word in prompt_lower for word in short_sounds):
            return 2.0
        
        # Medium sounds
        medium_sounds = ["bass", "lead", "riser", "sweep"]
        if any(word in prompt_lower for word in medium_sounds):
            return 4.0
        
        # Loops and longer content
        long_sounds = ["loop", "pad", "ambient", "atmosphere", "drone"]
        if any(word in prompt_lower for word in long_sounds):
            return 8.0
        
        return default
    
    def get_recommended_params(self, prompt: str) -> dict:
        """
        Get recommended generation parameters based on the prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Dict with recommended parameters
        """
        sound_type = self.extract_sound_type(prompt)
        genre = self.extract_genre(prompt)
        
        # Default parameters
        params = {
            "temperature": 1.0,
            "top_k": 250,
            "cfg_coef": 3.0,
        }
        
        # Adjust for sound type
        if sound_type in ["kick", "snare", "hihat", "hi-hat"]:
            # Drums need more precision
            params["temperature"] = 0.8
            params["cfg_coef"] = 4.0
        elif sound_type in ["pad", "ambient"]:
            # Pads can be more creative
            params["temperature"] = 1.1
            params["cfg_coef"] = 2.5
        elif sound_type == "bass":
            # Bass needs balance
            params["temperature"] = 0.9
            params["cfg_coef"] = 3.5
        
        return params
