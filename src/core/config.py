"""
Configuration Manager - Load and manage settings from YAML config files.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ConfigManager:
    """Manages application configuration."""
    
    DEFAULT_CONFIG = {
        "model": {
            "name": "facebook/musicgen-medium",
            "device": "cuda",
            "cache_dir": "./cache",
        },
        "generation": {
            "duration": 5.0,
            "temperature": 1.0,
            "top_k": 250,
            "top_p": 0.0,
            "cfg_coef": 3.0,
        },
        "audio": {
            "sample_rate": 44100,
            "normalize": True,
            "normalize_target_db": -3.0,
            "fade_in_ms": 10,
            "fade_out_ms": 50,
            "trim_silence": True,
            "silence_threshold_db": -60,
        },
        "output": {
            "directory": "./output",
            "format": "wav",
            "mp3_bitrate": 320,
            "naming": "timestamp",
        },
        "presets": {},
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path) if config_path else self._find_config()
        self._config = self._load_config()
    
    def _find_config(self) -> Path:
        """Find the configuration file in standard locations."""
        search_paths = [
            Path("./config/settings.yaml"),
            Path("./settings.yaml"),
            Path.home() / ".aimo" / "settings.yaml",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        # Return default path even if it doesn't exist
        return Path("./config/settings.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, merged with defaults."""
        config = self.DEFAULT_CONFIG.copy()
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                
                # Deep merge
                config = self._deep_merge(config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
        
        return config
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "model.name")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "model.name")
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """
        Save configuration to file.
        
        Args:
            path: Path to save to (uses config_path if None)
        """
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get("model", {})
    
    @property
    def generation(self) -> Dict[str, Any]:
        """Get generation configuration."""
        return self._config.get("generation", {})
    
    @property
    def audio(self) -> Dict[str, Any]:
        """Get audio configuration."""
        return self._config.get("audio", {})
    
    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._config.get("output", {})
    
    @property
    def presets(self) -> Dict[str, Any]:
        """Get presets."""
        return self._config.get("presets", {})
    
    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific preset by name.
        
        Args:
            name: Preset name
            
        Returns:
            Preset configuration or None
        """
        return self.presets.get(name)
    
    def list_presets(self) -> list:
        """List all available preset names."""
        return list(self.presets.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()
