"""
AIMO Sampler CLI - Command-line interface for AI audio generation.

Usage:
    aimo generate "fat psytrance bass"
    aimo generate "techno kick" --duration 2 --variations 3
    aimo presets
    aimo generate --preset psytrance-bass
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .core.config import ConfigManager
from .core.generator import AudioGenerator, calculate_bars_duration, INSTRUMENT_PRESETS
from .core.model_manager import ModelManager

console = Console()


def print_banner():
    """Print the AIMO banner."""
    banner = """
   █████╗ ██╗███╗   ███╗ ██████╗ 
  ██╔══██╗██║████╗ ████║██╔═══██╗
  ███████║██║██╔████╔██║██║   ██║
  ██╔══██║██║██║╚██╔╝██║██║   ██║
  ██║  ██║██║██║ ╚═╝ ██║╚██████╔╝
  ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝ ╚═════╝ 
    AI-Powered Audio Sampler
    """
    console.print(banner, style="cyan")


@click.group()
@click.option('--config', '-c', type=click.Path(), help='Path to config file')
@click.pass_context
def cli(ctx, config):
    """AIMO - AI-Powered Audio Sampler
    
    Generate audio samples using natural language descriptions.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = ConfigManager(config)


@cli.command()
@click.argument('prompt', required=False)
@click.option('--preset', '-p', help='Use a preset instead of/with prompt')
@click.option('--duration', '-d', type=float, help='Duration in seconds')
@click.option('--bars', '-b', type=int, help='Duration in bars (requires --bpm)')
@click.option('--bpm', type=int, help='BPM for bar calculation and rhythmic content')
@click.option('--variations', '-v', type=int, default=1, help='Number of variations to generate')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--format', '-f', type=click.Choice(['wav', 'mp3', 'flac']), help='Output format')
@click.option('--filename', help='Custom filename (without extension)')
@click.option('--model', '-m', help='Model to use (musicgen-small/medium/large)')
@click.option('--temperature', '-t', type=float, help='Generation temperature (0.5-1.5)')
@click.option('--cfg', type=float, help='Classifier-free guidance coefficient')
@click.option('--no-normalize', is_flag=True, help='Disable normalization')
@click.option('--no-trim', is_flag=True, help='Disable silence trimming')
@click.option('--no-enhance', is_flag=True, help='Disable prompt enhancement')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--precise', is_flag=True, help='Use precise mode for cleaner, more focused samples')
@click.option('--top-k', type=int, help='Top-k sampling (lower=more precise, default=250)')
@click.pass_context
def generate(
    ctx,
    prompt,
    preset,
    duration,
    bars,
    bpm,
    variations,
    output,
    format,
    filename,
    model,
    temperature,
    cfg,
    no_normalize,
    no_trim,
    no_enhance,
    seed,
    precise,
    top_k,
):
    """Generate audio samples from text descriptions.
    
    Examples:
    
        aimo generate "fat psytrance bass with distortion"
        
        aimo generate "punchy techno kick" --duration 2
        
        aimo generate "drum loop" --bars 4 --bpm 140
        
        aimo generate --preset psytrance-bass --variations 3
    """
    print_banner()
    
    config = ctx.obj['config']
    
    # Handle preset
    if preset:
        preset_config = config.get_preset(preset)
        if not preset_config:
            console.print(f"[red]Error: Preset '{preset}' not found.[/red]")
            console.print("Use 'aimo presets' to see available presets.")
            sys.exit(1)
        
        # Merge preset with prompt if both provided
        if prompt:
            prompt = f"{prompt}, {preset_config.get('prompt', '')}"
        else:
            prompt = preset_config.get('prompt', '')
        
        # Use preset duration if not specified
        if duration is None and 'duration' in preset_config:
            duration = preset_config['duration']
        
        console.print(f"[cyan]Using preset:[/cyan] {preset}")
    
    if not prompt:
        console.print("[red]Error: Please provide a prompt or use --preset[/red]")
        sys.exit(1)
    
    # Calculate duration from bars if specified
    if bars and bpm:
        duration = calculate_bars_duration(bars, bpm)
        console.print(f"[dim]{bars} bars at {bpm} BPM = {duration:.2f}s[/dim]")
    elif bars and not bpm:
        console.print("[red]Error: --bars requires --bpm to be specified[/red]")
        sys.exit(1)
    
    # Get settings from config, with CLI overrides
    model_name = model or config.get("model.name", "facebook/musicgen-medium")
    device = config.get("model.device", "cuda")
    output_dir = output or config.get("output.directory", "./output")
    output_format = format or config.get("output.format", "wav")
    
    gen_temp = temperature if temperature is not None else config.get("generation.temperature", 1.0)
    gen_cfg = cfg if cfg is not None else config.get("generation.cfg_coef", 3.0)
    gen_duration = duration if duration is not None else config.get("generation.duration", 5.0)
    gen_top_k = top_k if top_k is not None else config.get("generation.top_k", 250)
    
    # Precise mode: lower temperature, higher CFG, lower top_k for cleaner output
    if precise:
        gen_temp = temperature if temperature is not None else 0.7
        gen_cfg = cfg if cfg is not None else 5.0
        gen_top_k = top_k if top_k is not None else 100
        console.print("[cyan]🎯 Precise mode enabled:[/cyan] temp=0.7, cfg=5.0, top_k=100")
    
    normalize = not no_normalize and config.get("audio.normalize", True)
    trim_silence = not no_trim and config.get("audio.trim_silence", True)
    enhance_prompts = not no_enhance
    
    # Create generator
    try:
        generator = AudioGenerator(
            model_name=model_name,
            device=device,
            cache_dir=config.get("model.cache_dir", "./cache"),
            enhance_prompts=enhance_prompts,
        )
        
        # Generate and save
        saved_files = generator.generate_and_save(
            prompt=prompt,
            output_dir=output_dir,
            output_format=output_format,
            filename=filename,
            duration=gen_duration,
            bpm=bpm,
            normalize=normalize,
            trim_silence=trim_silence,
            fade_in_ms=config.get("audio.fade_in_ms", 10),
            fade_out_ms=config.get("audio.fade_out_ms", 50),
            variations=variations,
            temperature=gen_temp,
            cfg_coef=gen_cfg,
            top_k=gen_top_k,
            seed=seed,
        )
        
        console.print()
        console.print(Panel(
            f"[green]✓ Generated {len(saved_files)} sample(s)[/green]\n\n" +
            "\n".join([f"  • {f}" for f in saved_files]),
            title="Success",
            box=box.ROUNDED,
        ))
        
        # Cleanup
        generator.unload()
        
    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")
        raise
        sys.exit(1)


@cli.command()
@click.pass_context
def presets(ctx):
    """List all available presets."""
    print_banner()
    
    config = ctx.obj['config']
    preset_list = config.presets
    
    if not preset_list:
        console.print("[yellow]No presets found in configuration.[/yellow]")
        return
    
    # Group presets by category
    categories = {
        "Bass": [],
        "Kicks": [],
        "Hi-hats & Cymbals": [],
        "Drums & Percussion": [],
        "Synths": [],
        "FX & Textures": [],
        "Loops": [],
        "Other": [],
    }
    
    for name, preset in preset_list.items():
        prompt = preset.get('prompt', '')
        duration = preset.get('duration', '?')
        
        # Categorize
        name_lower = name.lower()
        if 'bass' in name_lower:
            categories["Bass"].append((name, prompt, duration))
        elif 'kick' in name_lower:
            categories["Kicks"].append((name, prompt, duration))
        elif 'hat' in name_lower or 'cymbal' in name_lower or 'crash' in name_lower:
            categories["Hi-hats & Cymbals"].append((name, prompt, duration))
        elif any(x in name_lower for x in ['snare', 'clap', 'tom', 'drum']):
            categories["Drums & Percussion"].append((name, prompt, duration))
        elif any(x in name_lower for x in ['lead', 'pad', 'pluck', 'arp']):
            categories["Synths"].append((name, prompt, duration))
        elif any(x in name_lower for x in ['riser', 'impact', 'sweep', 'atmosphere', 'fx']):
            categories["FX & Textures"].append((name, prompt, duration))
        elif 'loop' in name_lower:
            categories["Loops"].append((name, prompt, duration))
        else:
            categories["Other"].append((name, prompt, duration))
    
    # Display
    for category, items in categories.items():
        if not items:
            continue
        
        table = Table(title=category, box=box.ROUNDED, show_header=True)
        table.add_column("Preset", style="cyan", width=20)
        table.add_column("Description", style="white", width=50)
        table.add_column("Duration", style="green", width=10)
        
        for name, prompt, duration in items:
            # Truncate prompt
            prompt_display = prompt[:47] + "..." if len(prompt) > 50 else prompt
            table.add_row(name, prompt_display, f"{duration}s")
        
        console.print(table)
        console.print()
    
    console.print("[dim]Use: aimo generate --preset <name>[/dim]")


@cli.command()
@click.pass_context
def models(ctx):
    """List available models and their requirements."""
    print_banner()
    
    model_list = ModelManager.list_available_models()
    
    table = Table(title="Available Models", box=box.ROUNDED, show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("VRAM Required", style="yellow")
    table.add_column("Quality", style="green")
    table.add_column("Speed", style="blue")
    table.add_column("Notes", style="dim")
    
    for name, info in model_list.items():
        table.add_row(
            name,
            info.get("vram", "?"),
            info.get("quality", "?"),
            info.get("speed", "?"),
            info.get("note", ""),
        )
    
    console.print(table)
    console.print()
    console.print("[dim]Use: aimo generate --model <name>[/dim]")


@cli.group()
@click.pass_context
def config(ctx):
    """View and modify configuration."""
    pass


@config.command('show')
@click.pass_context
def config_show(ctx):
    """Show current configuration."""
    config = ctx.obj['config']
    
    import yaml
    console.print(Panel(
        yaml.dump(config.to_dict(), default_flow_style=False),
        title=f"Configuration ({config.config_path})",
        box=box.ROUNDED,
    ))


@config.command('set')
@click.argument('key')
@click.argument('value')
@click.pass_context
def config_set(ctx, key, value):
    """Set a configuration value.
    
    Example: aimo config set output.directory ./my-samples
    """
    config = ctx.obj['config']
    
    # Try to convert value to appropriate type
    if value.lower() == 'true':
        value = True
    elif value.lower() == 'false':
        value = False
    else:
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
    
    config.set(key, value)
    config.save()
    
    console.print(f"[green]Set {key} = {value}[/green]")


@cli.command()
@click.argument('bars', type=int)
@click.argument('bpm', type=int)
def duration(bars, bpm):
    """Calculate duration from bars and BPM.
    
    Example: aimo duration 4 140
    """
    dur = calculate_bars_duration(bars, bpm)
    console.print(f"[cyan]{bars} bars at {bpm} BPM = {dur:.2f} seconds[/cyan]")


@cli.command()
@click.argument('instrument')
@click.option('--style', '-s', help='Style modifier (808, analog, dark, bright, etc.)')
@click.option('--duration', '-d', type=float, help='Override default duration')
@click.option('--variations', '-v', type=int, default=1, help='Number of variations to generate')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--format', '-f', type=click.Choice(['wav', 'mp3', 'flac']), help='Output format')
@click.option('--model', '-m', help='Model to use (musicgen-small/medium/large)')
@click.option('--no-normalize', is_flag=True, help='Disable normalization')
@click.option('--no-trim', is_flag=True, help='Disable silence trimming')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.pass_context
def sample(
    ctx,
    instrument,
    style,
    duration,
    variations,
    output,
    format,
    model,
    no_normalize,
    no_trim,
    seed,
):
    """Generate clean, isolated instrument samples.
    
    Creates studio-quality single-shot samples perfect for DAW use.
    
    Available instruments:
    
    \b
    Drums:     kick, snare, hihat, clap, tom, cymbal, rim
    Bass:      bass, sub
    Synths:    lead, pad, pluck, keys, strings
    FX:        riser, impact, sweep, noise
    
    Style modifiers: 808, analog, digital, acoustic, dark, bright,
                    aggressive, soft, vintage, modern, lo-fi, hi-fi
    
    Examples:
    
        aimo sample kick
        
        aimo sample kick --style 808 --variations 3
        
        aimo sample bass --style analog --duration 4
        
        aimo sample pad --style dark
    """
    print_banner()
    
    config = ctx.obj['config']
    
    # Get settings from config, with CLI overrides
    model_name = model or config.get("model.name", "facebook/musicgen-medium")
    device = config.get("model.device", "cuda")
    output_dir = output or config.get("output.directory", "./output")
    output_format = format or config.get("output.format", "wav")
    
    normalize = not no_normalize and config.get("audio.normalize", True)
    trim_silence = not no_trim and config.get("audio.trim_silence", True)
    
    # Create generator
    try:
        generator = AudioGenerator(
            model_name=model_name,
            device=device,
            cache_dir=config.get("model.cache_dir", "./cache"),
            enhance_prompts=False,  # Use raw instrument prompts
        )
        
        # Generate and save clean samples
        saved_files = generator.generate_clean_sample_and_save(
            instrument=instrument,
            output_dir=output_dir,
            output_format=output_format,
            style=style,
            duration=duration,
            variations=variations,
            normalize=normalize,
            trim_silence=trim_silence,
            seed=seed,
        )
        
        console.print()
        console.print(Panel(
            f"[green]✓ Generated {len(saved_files)} clean {instrument} sample(s)[/green]\n\n" +
            "\n".join([f"  • {f}" for f in saved_files]),
            title="Success",
            box=box.ROUNDED,
        ))
        
        # Cleanup
        generator.unload()
        
    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")
        raise
        sys.exit(1)


@cli.command()
def instruments():
    """List all available instrument presets for clean sampling."""
    print_banner()
    
    # Group instruments by category
    categories = {
        "Drums & Percussion": ["kick", "snare", "hihat", "clap", "tom", "cymbal", "rim"],
        "Bass": ["bass", "sub"],
        "Synths & Keys": ["lead", "pad", "pluck", "keys", "strings"],
        "FX & Textures": ["riser", "impact", "sweep", "noise"],
    }
    
    for category, instruments_list in categories.items():
        table = Table(title=category, box=box.ROUNDED, show_header=True)
        table.add_column("Instrument", style="cyan", width=12)
        table.add_column("Description", style="white", width=45)
        table.add_column("Duration", style="green", width=10)
        
        for inst in instruments_list:
            if inst in INSTRUMENT_PRESETS:
                preset = INSTRUMENT_PRESETS[inst]
                desc = preset["prompt"].split(",")[0].replace("single ", "").replace("isolated ", "").strip()
                table.add_row(inst, desc.capitalize(), f"{preset['duration']}s")
        
        console.print(table)
        console.print()
    
    console.print("[cyan]Style modifiers:[/cyan]")
    console.print("[dim]  808, analog, digital, acoustic, electronic, dark, bright,[/dim]")
    console.print("[dim]  aggressive, soft, vintage, modern, lo-fi, hi-fi[/dim]")
    console.print()
    console.print("[dim]Usage: aimo sample <instrument> [--style <modifier>] [--variations N][/dim]")


@cli.command()
def status():
    """Check system status (GPU, CUDA, etc.)."""
    import torch
    
    print_banner()
    
    table = Table(title="System Status", box=box.ROUNDED, show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    # Python version
    import sys
    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", sys.executable)
    
    # PyTorch version
    table.add_row("PyTorch", torch.__version__, "")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        table.add_row("CUDA", "[green]✓ Available[/green]", f"GPU: {gpu_name}")
        table.add_row("GPU Memory", f"{gpu_memory:.1f} GB", "Total VRAM")
        
        # Recommended model based on VRAM
        if gpu_memory >= 16:
            recommended = "musicgen-large"
        elif gpu_memory >= 8:
            recommended = "musicgen-medium"
        else:
            recommended = "musicgen-small"
        table.add_row("Recommended Model", recommended, f"Based on {gpu_memory:.0f}GB VRAM")
    else:
        table.add_row("CUDA", "[red]✗ Not Available[/red]", "Will use CPU (slower)")
        table.add_row("Recommendation", "[yellow]Install CUDA PyTorch[/yellow]", "pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    console.print(table)
    console.print()
    
    if not cuda_available:
        console.print(Panel(
            "[yellow]GPU acceleration is not available.[/yellow]\n\n"
            "To enable GPU support, reinstall PyTorch with CUDA:\n\n"
            "[cyan]pip uninstall torch torchaudio -y[/cyan]\n"
            "[cyan]pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118[/cyan]",
            title="⚠ GPU Not Detected",
            box=box.ROUNDED,
        ))


def main():
    """Main entry point."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
