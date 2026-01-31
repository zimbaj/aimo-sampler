# AIMO Sampler - AI-Powered Audio Sample Generator

Generate audio samples using natural language descriptions. Powered by Meta's MusicGen running locally on your GPU.

## 📚 Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Complete beginner's guide
- **[Prompt Writing Guide](docs/PROMPT_GUIDE.md)** - How to write effective prompts
- **[Configuration Reference](docs/CONFIGURATION.md)** - All settings explained

## Features

- 🎵 **Natural Language Input**: Describe what you want ("fat psytrance bass", "crispy hi-hat")
- 🖥️ **Fully Local**: All processing happens on your machine
- 🎛️ **Flexible Duration**: Generate samples from 1-30 seconds
- 🎚️ **Post-Processing**: Auto-normalization, trimming, fade in/out
- 📁 **Multiple Formats**: Export to WAV, MP3, FLAC
- ⚡ **GPU Accelerated**: Optimized for NVIDIA GPUs

## Requirements

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (8GB recommended)
- CUDA toolkit installed

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate your first sample!
python -m src.cli generate "punchy techno kick drum"

# Or use the interactive launcher (Windows)
start.bat
```

## Usage

### Basic Generation

```bash
# Generate a sample with natural language
python -m src.cli generate "fat psytrance bass with distortion"

# Specify duration (in seconds)
python -m src.cli generate "punchy techno kick" --duration 2

# Use precise mode for cleaner, more focused samples
python -m src.cli generate "808 bass" --precise

# Use bars and BPM
python -m src.cli generate "rolling drum loop" --bars 4 --bpm 140

# Multiple variations
python -m src.cli generate "ambient pad" --variations 3

# Choose model size
python -m src.cli generate "bass" --model musicgen-medium

# Fine-tune generation parameters
python -m src.cli generate "lead synth" --temperature 0.7 --cfg 5.0 --top-k 100
```

### Using Presets

```bash
# List available presets
python -m src.cli presets

# Use a preset
python -m src.cli generate --preset psytrance-bass

# Combine preset with custom description
python -m src.cli generate "extra distorted" --preset psytrance-bass
```

### More Commands

```bash
# List available models
python -m src.cli models

# Calculate duration from bars
python -m src.cli duration 4 140

# Show current config
python -m src.cli config show
```

## Sample Duration Guide

| BPM | 1 Bar | 2 Bars | 4 Bars |
|-----|-------|--------|--------|
| 90  | 2.67s | 5.33s  | 10.67s |
| 120 | 2.0s  | 4.0s   | 8.0s   |
| 140 | 1.71s | 3.43s  | 6.86s  |
| 150 | 1.6s  | 3.2s   | 6.4s   |

## Models

| Model | VRAM | Quality | Speed |
|-------|------|---------|-------|
| musicgen-small | 4GB | Good | Fast |
| musicgen-medium | 8GB | Great | Medium |
| musicgen-large | 16GB | Excellent | Slow |

## Project Structure

```
aimo-sampler/
├── src/
│   ├── cli.py              # Command-line interface
│   ├── core/
│   │   ├── generator.py    # Main generation logic
│   │   ├── model_manager.py # Model loading/caching
│   │   ├── prompt_processor.py # Prompt enhancement
│   │   └── config.py       # Configuration management
│   └── audio/
│       ├── processor.py    # Audio post-processing
│       └── export.py       # File export (WAV/MP3/FLAC)
├── config/
│   └── settings.yaml       # Configuration & presets
├── docs/                   # Documentation
├── output/                 # Generated samples
└── cache/                  # Model cache
```

## License

MIT License
