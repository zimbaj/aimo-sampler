# AIMO Sampler - Beginner's Guide

Welcome to AIMO Sampler! This guide will help you install, use, and customize your AI-powered audio sample generator.

## Table of Contents

1. [What is AIMO Sampler?](#what-is-aimo-sampler)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Your First Sample](#your-first-sample)
5. [Basic Usage](#basic-usage)
6. [Using Presets](#using-presets)
7. [Understanding Duration & BPM](#understanding-duration--bpm)
8. [Tweaking Generation Settings](#tweaking-generation-settings)
9. [Choosing the Right Model](#choosing-the-right-model)
10. [Creating Custom Presets](#creating-custom-presets)
11. [Tips for Better Results](#tips-for-better-results)
12. [Troubleshooting](#troubleshooting)

---

## What is AIMO Sampler?

AIMO Sampler is a tool that generates audio samples using artificial intelligence. Simply describe what you want in plain English (like "fat psytrance bass" or "crispy hi-hat"), and the AI will create an audio file for you.

**Key Features:**
- 🎵 Generate samples from text descriptions
- 🖥️ Runs completely on your computer (no internet needed after setup)
- 🎛️ Professional audio post-processing built-in
- 📁 Export to WAV, MP3, or FLAC

---

## Requirements

### Minimum Requirements
- **Operating System:** Windows 10/11, Linux, or macOS
- **RAM:** 8GB
- **GPU:** NVIDIA GPU with 4GB VRAM (GTX 1070 or better)
- **Storage:** 10GB free space (for models)
- **Python:** Version 3.10 or newer

### Recommended
- **RAM:** 16GB
- **GPU:** NVIDIA RTX 2070 or better (8GB VRAM)
- **Storage:** SSD for faster loading

### Check Your GPU
To check if you have a compatible NVIDIA GPU, open PowerShell and run:
```powershell
nvidia-smi
```
You should see your GPU name and memory listed.

---

## Installation

### Step 1: Install Python

1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation:
   ```powershell
   python --version
   ```

### Step 2: Download AIMO Sampler

If you have the project folder, navigate to it:
```powershell
cd D:\Projects\aimo-sampler
```

### Step 3: Create Virtual Environment

A virtual environment keeps AIMO's dependencies separate from other Python projects:

```powershell
# Create the environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# You should see (venv) at the start of your prompt
```

### Step 4: Install PyTorch with CUDA (IMPORTANT!)

This enables GPU acceleration. **Install this BEFORE other dependencies:**

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ **Important:** If you skip this step or install requirements.txt first, you may get a CPU-only version of PyTorch which will be much slower.

### Step 5: Install Other Dependencies

```powershell
pip install -r requirements.txt
```

### Step 6: Verify GPU is Working

```powershell
python -m src.cli status
```

You should see:
- ✓ CUDA Available
- Your GPU name (e.g., "NVIDIA GeForce RTX 2070")
- Recommended model based on your VRAM

If CUDA shows "Not Available", reinstall PyTorch:
```powershell
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Your First Sample

Let's generate your first audio sample!

### 1. Activate the Environment
```powershell
cd D:\Projects\aimo-sampler
.\venv\Scripts\activate
```

### 2. Generate a Sample
```powershell
python -m src.cli generate "punchy kick drum"
```

### 3. What Happens Next
1. **First run:** The AI model downloads (~1.5GB for small, ~3.5GB for medium)
2. **Model loads:** Takes 10-30 seconds
3. **Generation:** Usually 5-30 seconds depending on duration
4. **Output:** Your sample is saved to the `output` folder!

### 4. Find Your Sample
Look in the `output` folder. Your file will be named something like:
```
punchy_kick_drum_20260129_123456.wav
```

---

## Basic Usage

### Simple Generation
```powershell
python -m src.cli generate "your description here"
```

### Specify Duration
```powershell
# Generate a 2-second sample
python -m src.cli generate "snare drum" --duration 2

# Generate a 10-second pad
python -m src.cli generate "ambient pad" --duration 10
```

### Generate Multiple Variations
```powershell
# Create 3 different versions
python -m src.cli generate "bass sound" --variations 3
```

### Choose Output Format
```powershell
# Save as MP3
python -m src.cli generate "hi-hat" --format mp3

# Save as FLAC (lossless)
python -m src.cli generate "hi-hat" --format flac
```

### Custom Filename
```powershell
python -m src.cli generate "kick" --filename my_awesome_kick
```

---

## Using Presets

Presets are pre-configured prompts optimized for specific sounds.

### List All Presets
```powershell
python -m src.cli presets
```

### Use a Preset
```powershell
python -m src.cli generate --preset psytrance-bass
python -m src.cli generate --preset techno-kick
python -m src.cli generate --preset pad-ambient
```

### Available Preset Categories

| Category | Presets |
|----------|---------|
| **Bass** | `psytrance-bass`, `techno-bass`, `dnb-bass` |
| **Kicks** | `techno-kick`, `psytrance-kick` |
| **Hi-hats** | `hihat-closed`, `hihat-open`, `crash` |
| **Drums** | `snare-acoustic`, `snare-electronic`, `clap`, `tom-low` |
| **Synths** | `lead-saw`, `pad-ambient`, `pluck`, `arp` |
| **FX** | `riser`, `impact`, `sweep`, `atmosphere` |
| **Loops** | `drum-loop-techno`, `drum-loop-dnb` |

### Combine Preset with Custom Description
```powershell
# Start with preset, add your twist
python -m src.cli generate "extra distorted and aggressive" --preset psytrance-bass
```

---

## Understanding Duration & BPM

### Quick Duration Guide

For music production, samples are often measured in **bars** (measures). Here's how duration relates to bars at different tempos:

| BPM | 1 Bar | 2 Bars | 4 Bars | 8 Bars |
|-----|-------|--------|--------|--------|
| 90  | 2.67s | 5.33s  | 10.67s | 21.33s |
| 120 | 2.0s  | 4.0s   | 8.0s   | 16.0s  |
| 140 | 1.71s | 3.43s  | 6.86s  | 13.71s |
| 150 | 1.6s  | 3.2s   | 6.4s   | 12.8s  |
| 174 | 1.38s | 2.76s  | 5.52s  | 11.03s |

### Calculate Duration from Bars
```powershell
# Calculate how long 4 bars at 140 BPM is
python -m src.cli duration 4 140
# Output: 4 bars at 140 BPM = 6.86 seconds
```

### Generate Using Bars
```powershell
# Generate a 4-bar loop at 140 BPM
python -m src.cli generate "techno drum loop" --bars 4 --bpm 140
```

### Recommended Durations by Sound Type

| Sound Type | Recommended Duration |
|------------|---------------------|
| Kick, Snare, Clap | 1-2 seconds |
| Hi-hat (closed) | 0.5-1 second |
| Hi-hat (open) | 1-2 seconds |
| Bass stab | 2-4 seconds |
| Pad/Atmosphere | 8-15 seconds |
| Drum loop | 4-8 seconds (2-4 bars) |
| Riser/Sweep | 4-8 seconds |

---

## Tweaking Generation Settings

### Temperature (Creativity)

Temperature controls how "creative" or "random" the AI is:

```powershell
# Lower = more predictable, closer to training data
python -m src.cli generate "kick drum" --temperature 0.7

# Higher = more creative, more variation
python -m src.cli generate "experimental texture" --temperature 1.3
```

| Temperature | Effect | Best For |
|-------------|--------|----------|
| 0.5 - 0.8 | Conservative, predictable | Drums, kicks, standard sounds |
| 0.9 - 1.1 | Balanced (default: 1.0) | Most use cases |
| 1.2 - 1.5 | Creative, experimental | Pads, textures, unique sounds |

### CFG (Classifier-Free Guidance)

CFG controls how closely the AI follows your prompt:

```powershell
# Higher CFG = follows prompt more strictly
python -m src.cli generate "acoustic snare" --cfg 4.0

# Lower CFG = more creative interpretation
python -m src.cli generate "weird bass" --cfg 2.0
```

| CFG Value | Effect |
|-----------|--------|
| 1.0 - 2.0 | Loose interpretation, more variety |
| 2.5 - 3.5 | Balanced (default: 3.0) |
| 4.0 - 6.0 | Strict adherence to prompt |

### Combining Settings
```powershell
python -m src.cli generate "dark ambient drone" \
    --duration 15 \
    --temperature 1.2 \
    --cfg 2.5
```

### Reproducible Results (Seed)

Use a seed to get the same result every time:
```powershell
# Generate with specific seed
python -m src.cli generate "bass" --seed 12345

# Same seed = same output
python -m src.cli generate "bass" --seed 12345
```

---

## Choosing the Right Model

AIMO supports multiple model sizes. Larger = better quality but slower.

### View Available Models
```powershell
python -m src.cli models
```

### Model Comparison

| Model | VRAM Required | Quality | Speed | Best For |
|-------|--------------|---------|-------|----------|
| `musicgen-small` | 4GB | Good | Fast | Quick tests, low-end GPUs |
| `musicgen-medium` | 8GB | Great | Medium | **Recommended for RTX 2070** |
| `musicgen-large` | 16GB | Excellent | Slow | High-end GPUs, final production |

### Specify Model
```powershell
# Use small model (faster, less VRAM)
python -m src.cli generate "kick" --model musicgen-small

# Use medium model (recommended)
python -m src.cli generate "kick" --model musicgen-medium

# Use large model (best quality, needs 16GB VRAM)
python -m src.cli generate "kick" --model musicgen-large
```

### Set Default Model

Edit `config/settings.yaml`:
```yaml
model:
  name: "facebook/musicgen-medium"  # Change this
```

---

## Creating Custom Presets

### Edit the Config File

Open `config/settings.yaml` and add your presets under the `presets:` section:

```yaml
presets:
  # Your custom preset
  my-custom-bass:
    prompt: "deep wobble bass, dubstep style, aggressive, modulated"
    duration: 4.0
  
  my-ambient-pad:
    prompt: "ethereal ambient pad, soft, dreamy, reverb, slow evolving"
    duration: 12.0
  
  my-perc-loop:
    prompt: "tribal percussion loop, organic, wooden, rhythmic"
    duration: 8.0
```

### Use Your Custom Preset
```powershell
python -m src.cli generate --preset my-custom-bass
```

### Preset Tips

1. **Be specific:** "deep aggressive psytrance bass with acid 303 character" works better than "bass"
2. **Include style keywords:** "electronic", "acoustic", "cinematic", "dark", "bright"
3. **Mention characteristics:** "punchy", "soft", "distorted", "clean", "reverberant"
4. **Reference genres:** "techno", "ambient", "dubstep", "dnb", "house"

---

## Tips for Better Results

### Writing Good Prompts

**❌ Too vague:**
```
"bass"
"drum"
"sound"
```

**✅ Specific and descriptive:**
```
"deep sub bass with slight distortion, electronic, minimal techno style"
"punchy acoustic snare drum with bright crack and short decay"
"ethereal pad synthesizer, soft attack, long release, dreamy"
```

### Prompt Formula

Try this structure:
```
[adjective] [sound type], [style/genre], [characteristics], [additional details]
```

Examples:
- "punchy kick drum, techno style, tight transient, sub-bass tail"
- "bright saw lead, trance style, detuned, powerful, melodic"
- "dark ambient texture, cinematic, evolving, mysterious, reverberant"

### Keywords That Work Well

| Category | Keywords |
|----------|----------|
| **Tone** | bright, dark, warm, cold, harsh, soft |
| **Dynamics** | punchy, soft, aggressive, gentle, powerful |
| **Texture** | smooth, gritty, distorted, clean, saturated |
| **Space** | dry, reverberant, wide, tight, spacious |
| **Style** | electronic, acoustic, cinematic, vintage, modern |

### Iterate and Refine

1. Start with a basic prompt
2. Generate a few variations (`--variations 3`)
3. Listen and identify what's missing
4. Add more specific keywords
5. Repeat until satisfied

---

## Troubleshooting

### "CUDA out of memory"

Your GPU doesn't have enough memory for the model.

**Solutions:**
1. Use a smaller model: `--model musicgen-small`
2. Close other GPU-intensive applications
3. Reduce generation duration

### "Model not found" / Download Issues

The model needs to download first (~1.5-3.5GB).

**Solutions:**
1. Check your internet connection
2. Wait for the download to complete
3. Try again - downloads resume automatically

### Slow Generation

**Solutions:**
1. Use a smaller model: `--model musicgen-small`
2. Reduce duration
3. Ensure you're using GPU (not CPU)

### Check GPU is Being Used
```powershell
# Should show "CUDA available: True"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Poor Quality Output

**Solutions:**
1. Use a larger model if your GPU supports it
2. Write more descriptive prompts
3. Try different temperature/CFG settings
4. Generate multiple variations

### No Sound / Empty File

The AI may have generated silence if the prompt was unclear.

**Solutions:**
1. Be more specific in your prompt
2. Increase CFG value (`--cfg 4.0`)
3. Try a different seed

### "Module not found" Errors

Dependencies may not be installed correctly.

**Solution:**
```powershell
# Reactivate environment and reinstall
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Reference Card

```powershell
# Basic generation
python -m src.cli generate "your description"

# With duration
python -m src.cli generate "sound" --duration 5

# With preset
python -m src.cli generate --preset techno-kick

# Multiple variations
python -m src.cli generate "bass" --variations 3

# Specific model
python -m src.cli generate "pad" --model musicgen-medium

# Full customization
python -m src.cli generate "dark bass" \
    --duration 4 \
    --temperature 0.9 \
    --cfg 3.5 \
    --model musicgen-medium \
    --format wav \
    --variations 2

# View presets
python -m src.cli presets

# View models  
python -m src.cli models

# Calculate duration
python -m src.cli duration 4 140
```

---

## Getting Help

```powershell
# General help
python -m src.cli --help

# Command-specific help
python -m src.cli generate --help
```

---

**Happy sampling! 🎵**

If you create something cool, experiment with the settings, and don't be afraid to try unusual prompts - sometimes the AI surprises you!
