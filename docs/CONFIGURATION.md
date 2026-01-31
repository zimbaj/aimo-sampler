# AIMO Sampler - Configuration Reference

Complete reference for all configuration options in AIMO Sampler.

## Configuration File Location

The main configuration file is located at:
```
config/settings.yaml
```

## Full Configuration Reference

```yaml
# =============================================================================
# MODEL SETTINGS
# =============================================================================
model:
  # Model to use for generation
  # Options: 
  #   - facebook/musicgen-small   (4GB VRAM, fast)
  #   - facebook/musicgen-medium  (8GB VRAM, recommended)
  #   - facebook/musicgen-large   (16GB VRAM, best quality)
  #   - facebook/musicgen-melody  (8GB VRAM, melody conditioning)
  name: "facebook/musicgen-medium"
  
  # Device for computation
  # Options: "cuda" (GPU) or "cpu"
  device: "cuda"
  
  # Directory to cache downloaded models
  # Models are large (1.5-7GB), so choose a location with space
  cache_dir: "./cache"

# =============================================================================
# GENERATION SETTINGS
# =============================================================================
generation:
  # Default duration in seconds (can be overridden with --duration)
  duration: 5.0
  
  # Temperature: Controls randomness/creativity
  # Range: 0.1 to 2.0
  # Lower (0.5-0.8): More predictable, conservative
  # Default (1.0): Balanced
  # Higher (1.2-1.5): More creative, experimental
  temperature: 1.0
  
  # Top-K sampling: Limits vocabulary of choices
  # Range: 1 to 1000
  # Lower: More focused, less variety
  # Higher: More variety
  # Default: 250
  top_k: 250
  
  # Top-P (nucleus) sampling: Alternative to top-k
  # Range: 0.0 to 1.0
  # 0.0: Disabled (use top-k only)
  # 0.9: Consider tokens making up 90% of probability mass
  top_p: 0.0
  
  # Classifier-Free Guidance coefficient
  # Range: 1.0 to 10.0
  # Lower (1-2): Loose interpretation, more creative
  # Default (3.0): Balanced
  # Higher (4-6): Strict adherence to prompt
  cfg_coef: 3.0

# =============================================================================
# AUDIO PROCESSING SETTINGS
# =============================================================================
audio:
  # Output sample rate in Hz
  # Note: MusicGen generates at 32000 Hz internally
  # Will resample if different
  sample_rate: 44100
  
  # Normalize audio to consistent level
  normalize: true
  
  # Target level for normalization in dB
  # -3 dB leaves headroom for further processing
  # -0.1 dB for maximum loudness
  normalize_target_db: -3.0
  
  # Fade in duration in milliseconds
  # Prevents clicks at start
  fade_in_ms: 10
  
  # Fade out duration in milliseconds
  # Creates smooth ending
  fade_out_ms: 50
  
  # Remove silence from start/end
  trim_silence: true
  
  # Threshold for silence detection in dB
  # Audio below this level is considered silence
  silence_threshold_db: -60

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
output:
  # Default output directory
  directory: "./output"
  
  # Default output format
  # Options: "wav", "mp3", "flac"
  format: "wav"
  
  # Bitrate for MP3 encoding (kbps)
  # Options: 128, 192, 256, 320
  mp3_bitrate: 320
  
  # Filename generation style
  # Options:
  #   - "timestamp": prompt_20240129_123456
  #   - "prompt": prompt_only
  #   - "both": prompt_20240129_123456
  naming: "timestamp"

# =============================================================================
# PRESETS
# =============================================================================
# Presets are shortcuts for common sounds
# Each preset has:
#   - prompt: The text description
#   - duration: Default duration for this sound
#
# Use with: aimo generate --preset <name>
# =============================================================================
presets:
  # -------------------------
  # BASS SOUNDS
  # -------------------------
  psytrance-bass:
    prompt: "deep aggressive psytrance bass synthesizer, distorted, rolling, 303 acid style"
    duration: 4.0
    
  techno-bass:
    prompt: "deep techno bass, sub-heavy, clean sine wave, minimal"
    duration: 2.0
    
  dnb-bass:
    prompt: "drum and bass reese bass, detuned saw waves, growling, aggressive"
    duration: 4.0

  # -------------------------
  # KICK DRUMS
  # -------------------------
  techno-kick:
    prompt: "punchy techno kick drum, tight transient, sub-bass tail"
    duration: 1.0
    
  psytrance-kick:
    prompt: "psytrance kick drum, punchy attack, short decay, clicky"
    duration: 1.0
    
  # -------------------------
  # HI-HATS & CYMBALS
  # -------------------------
  hihat-closed:
    prompt: "crispy closed hi-hat, tight, electronic"
    duration: 0.5
    
  hihat-open:
    prompt: "open hi-hat cymbal, sizzling, electronic"
    duration: 1.5
    
  crash:
    prompt: "crash cymbal, bright, electronic production"
    duration: 3.0

  # -------------------------
  # SNARES & PERCUSSION
  # -------------------------
  snare-acoustic:
    prompt: "acoustic snare drum, punchy, bright"
    duration: 1.0
    
  snare-electronic:
    prompt: "electronic snare, punchy, layered with clap"
    duration: 1.0
    
  clap:
    prompt: "electronic clap, layered, reverb tail"
    duration: 1.5
    
  tom-low:
    prompt: "low tom drum, deep, punchy"
    duration: 2.0

  # -------------------------
  # SYNTH SOUNDS
  # -------------------------
  lead-saw:
    prompt: "bright saw wave synthesizer lead, detuned, powerful"
    duration: 4.0
    
  pad-ambient:
    prompt: "ambient pad synthesizer, soft, evolving, ethereal"
    duration: 8.0
    
  pluck:
    prompt: "pluck synthesizer, short decay, bright, melodic"
    duration: 2.0
    
  arp:
    prompt: "arpeggiated synthesizer sequence, rhythmic, electronic"
    duration: 8.0

  # -------------------------
  # FX & TEXTURES
  # -------------------------
  riser:
    prompt: "build up riser, tension, increasing pitch and intensity"
    duration: 8.0
    
  impact:
    prompt: "cinematic impact hit, deep, reverberant"
    duration: 3.0
    
  sweep:
    prompt: "filter sweep effect, white noise, rising"
    duration: 4.0
    
  atmosphere:
    prompt: "dark atmospheric texture, ambient, evolving, mysterious"
    duration: 15.0

  # -------------------------
  # LOOPS
  # -------------------------
  drum-loop-techno:
    prompt: "techno drum loop, four on the floor kick, hi-hats, minimal"
    duration: 8.0
    
  drum-loop-dnb:
    prompt: "drum and bass breakbeat, fast, syncopated, energetic"
    duration: 8.0
```

## Creating Custom Presets

### Adding a New Preset

Add to the `presets:` section in `settings.yaml`:

```yaml
presets:
  # ... existing presets ...
  
  # Your new preset
  my-custom-sound:
    prompt: "your detailed description here"
    duration: 4.0
```

### Preset Best Practices

1. **Use descriptive names:**
   - ✅ `dark-techno-bass`, `vintage-snare`
   - ❌ `bass1`, `sound2`

2. **Write detailed prompts:**
   - ✅ `"deep rolling psytrance bass, acid 303 character, distorted, aggressive"`
   - ❌ `"bass"`

3. **Set appropriate durations:**
   - One-shots (kicks, snares): 1-2s
   - Short sounds (hi-hats): 0.5-1s
   - Bass/leads: 2-4s
   - Pads/atmospheres: 8-15s
   - Loops: 4-8s

## Command Line Overrides

Any config setting can be overridden from the command line:

```powershell
# Override model
python -m src.cli generate "bass" --model musicgen-large

# Override duration
python -m src.cli generate "bass" --duration 10

# Override temperature
python -m src.cli generate "bass" --temperature 1.2

# Override CFG
python -m src.cli generate "bass" --cfg 4.0

# Override output format
python -m src.cli generate "bass" --format mp3

# Override output directory
python -m src.cli generate "bass" --output ./my-samples

# Disable normalization
python -m src.cli generate "bass" --no-normalize

# Disable silence trimming
python -m src.cli generate "bass" --no-trim
```

## Environment Variables

You can also use environment variables:

```powershell
# Set HuggingFace token for faster downloads
$env:HF_TOKEN = "your_token_here"

# Disable symlink warnings
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Set custom cache directory
$env:HF_HOME = "D:\AI\ModelCache"
```

## View Current Configuration

```powershell
python -m src.cli config show
```

## Modify Configuration via CLI

```powershell
# Set a value
python -m src.cli config set output.directory ./my-samples
python -m src.cli config set generation.temperature 0.9
python -m src.cli config set audio.normalize false
```

---

## Quick Settings Cheat Sheet

| Setting | Conservative | Balanced | Creative |
|---------|--------------|----------|----------|
| temperature | 0.7 | 1.0 | 1.3 |
| cfg_coef | 4.0 | 3.0 | 2.0 |
| top_k | 150 | 250 | 350 |

**For predictable drums:** temp=0.8, cfg=4.0
**For creative pads:** temp=1.2, cfg=2.5
**For balanced results:** temp=1.0, cfg=3.0
