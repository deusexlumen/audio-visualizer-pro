#!/usr/bin/env python3
"""
Audio Visualizer Pro - CLI Entry Point

KI-Optimierter Workflow fuer Audio-Visualisierungen.
"""

import click
import json
import shutil
import subprocess
from pathlib import Path


def _check_ffmpeg():
    """Prueft ob FFmpeg installiert ist und gibt Version aus."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        click.echo("=" * 60)
        click.echo("FEHLER: FFmpeg nicht gefunden!")
        click.echo("=" * 60)
        click.echo("Audio Visualizer Pro benoetigt FFmpeg fuer Video-Encoding.")
        click.echo("")
        click.echo("Installation:")
        click.echo("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        click.echo("  macOS:         brew install ffmpeg")
        click.echo("  Windows:       https://ffmpeg.org/download.html")
        click.echo("")
        click.echo("Stelle sicher, dass ffmpeg im PATH verfuegbar ist.")
        click.echo("=" * 60)
        raise click.ClickException("FFmpeg ist erforderlich.")
    
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_line = result.stdout.splitlines()[0]
        click.echo(f"FFmpeg gefunden: {version_line}")
    except Exception:
        click.echo(f"FFmpeg gefunden: {ffmpeg_path}")
    return ffmpeg_path


@click.group()
def cli():
    """Audio Visualizer Pro - KI-Optimierter Workflow"""
    pass


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--visual', '-v', default='lumina_core',
              help='Visualizer-Typ (z.B. lumina_core, voice_flow, spectrum_genesis, spectrum_bars)')
@click.option('--output', '-o', default='output.mp4')
@click.option('--config', '-c', type=click.Path(), help='JSON Config-File')
@click.option('--resolution', '-r', default='1920x1080')
@click.option('--fps', default=60, type=int)
@click.option('--preview', is_flag=True, help='Schnelle Vorschau')
@click.option('--preview-duration', default=5.0, type=float, help='Dauer der Vorschau in Sekunden')
@click.option('--background-image', '-bg', type=click.Path(), help='Hintergrundbild')
@click.option('--background-blur', default=0.0, type=float, help='Hintergrund-Blur Radius')
@click.option('--background-vignette', default=0.0, type=float, help='Vignette Staerke (0.0-1.0)')
@click.option('--background-opacity', default=0.3, type=float, help='Hintergrund-Opazitaet (0.0-1.0)')
@click.option('--codec', default='h264', type=click.Choice(['h264', 'hevc', 'prores']), help='Video-Codec')
@click.option('--quality', default='high', type=click.Choice(['low', 'medium', 'high', 'lossless']), help='Qualitaet')
@click.option('--param', '-p', multiple=True, help='Visualizer Parameter (key=value)')
def render(audio_file, visual, output, config, resolution, fps, preview, preview_duration,
           background_image, background_blur, background_vignette, background_opacity,
           codec, quality, param):
    """Rendert Audio-Visualisierung auf der GPU."""
    
    _check_ffmpeg()
    
    try:
        width, height = map(int, resolution.split('x'))
    except ValueError:
        raise click.BadParameter(
            f"Ungueltige Aufloesung: '{resolution}'. "
            f"Format: BREITExHOEHE (z.B. 1920x1080)"
        )
    
    # Parameter parsen
    params = {}
    for p in param:
        if '=' not in p:
            raise click.BadParameter(f"Parameter muss key=value sein: {p}")
        key, value = p.split('=', 1)
        # Typ-Inferenz
        if value.lower() in ('true', 'yes', '1'):
            value = True
        elif value.lower() in ('false', 'no', '0'):
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        params[key] = value
    
    # Config-File laden (optional)
    if config:
        with open(config) as f:
            cfg_dict = json.load(f)
        if 'visual' in cfg_dict and 'params' in cfg_dict['visual']:
            params.update(cfg_dict['visual']['params'])
        if 'visual' in cfg_dict and 'type' in cfg_dict['visual']:
            visual = cfg_dict['visual']['type']
    
    from src.gpu_renderer import GPUBatchRenderer
    
    click.echo(f"[GPU] Starte Rendering: {visual} @ {width}x{height} {fps}fps")
    if preview:
        click.echo(f"[GPU] Preview-Modus: {preview_duration}s")
    
    renderer = GPUBatchRenderer(width=width, height=height, fps=fps)
    renderer.render(
        audio_path=audio_file,
        visualizer_type=visual,
        output_path=output,
        params=params if params else None,
        preview_mode=preview,
        preview_duration=preview_duration,
        background_image=background_image,
        background_blur=background_blur,
        background_vignette=background_vignette,
        background_opacity=background_opacity,
        codec=codec,
        quality=quality,
    )
    
    click.echo(f"[GPU] Fertig! Output: {output}")


@cli.command()
def list_visuals():
    """Zeigt alle verfuegbaren GPU-Visualizer an."""
    from src.gpu_visualizers import list_visualizers
    
    visuals = list_visualizers()
    click.echo("Verfuegbare GPU-Visualisierungen:")
    click.echo("")
    
    signatures = ['lumina_core', 'voice_flow', 'spectrum_genesis']
    for name in visuals:
        marker = " ⭐" if name in signatures else ""
        click.echo(f"  - {name}{marker}")
    
    click.echo("")
    click.echo("⭐ = Signature Visualizer (empfohlen)")


@cli.command()
@click.argument('name')
def create_template(name):
    """
    Erstellt ein neues GPU-Visualizer-Template.
    Generiert: src/gpu_visualizers/{name}.py mit Boilerplate.
    """
    target = Path(f"src/gpu_visualizers/{name}.py")
    if target.exists():
        click.echo(f"Fehler: {target} existiert bereits!")
        return
    
    template = f'''"""
{name}.py - Neue GPU-Visualisierung

TODO: Beschreibung hier einfuegen
"""

import numpy as np
import moderngl

from .base import BaseGPUVisualizer


class {name.title()}Visualizer(BaseGPUVisualizer):
    """
    TODO: Beschreibung hier einfuegen
    """

    PARAMS = {{
        "intensity": (0.5, 0.0, 2.0),
        "speed": (1.0, 0.0, 5.0),
    }}

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        super().__init__(ctx, width, height)
        self._build_program()

    def _build_program(self):
        vertex_shader = """
        #version 330
        in vec2 in_pos;
        void main() {{
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }}
        """
        
        fragment_shader = """
        #version 330
        uniform float u_time;
        uniform float u_rms;
        uniform float u_onset;
        uniform vec2 u_resolution;
        out vec4 f_color;
        
        void main() {{
            vec2 uv = gl_FragCoord.xy / u_resolution;
            vec3 color = vec3(0.05);
            
            // TODO: Deine Shader-Logik hier
            color += u_rms * vec3(1.0, 0.2, 0.4);
            
            f_color = vec4(color, 1.0);
        }}
        """
        
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )
        self._setup_quad()

    def render(self, features: dict, time: float):
        self.program["u_time"].value = time
        self.program["u_rms"].value = features.get("rms", 0.0)
        self.program["u_onset"].value = features.get("onset", 0.0)
        self.program["u_resolution"].value = (self.width, self.height)
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
'''
    
    target.write_text(template)
    click.echo(f"GPU-Template erstellt: {target}")
    click.echo(f"KI-Agent: Implementiere den Fragment-Shader!")


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
def analyze(audio_file):
    """Analysiert eine Audio-Datei und zeigt Features an."""
    from src.analyzer import AudioAnalyzer
    
    analyzer = AudioAnalyzer()
    features = analyzer.analyze(audio_file, fps=60)
    
    click.echo("\n=== Audio-Analyse Ergebnisse ===")
    click.echo(f"Dauer: {features.duration:.2f}s")
    click.echo(f"Sample Rate: {features.sample_rate}Hz")
    click.echo(f"Tempo: {features.tempo:.1f} BPM")
    click.echo(f"Key: {features.key or 'Unbekannt'}")
    click.echo(f"Mode: {features.mode}")
    click.echo(f"Frames: {int(features.duration * features.fps)}")
    
    click.echo("\n=== Feature-Statistiken ===")
    click.echo(f"RMS: min={features.rms.min():.3f}, max={features.rms.max():.3f}, mean={features.rms.mean():.3f}")
    click.echo(f"Onset: min={features.onset.min():.3f}, max={features.onset.max():.3f}")
    click.echo(f"Transient: max={features.transient.max():.3f}, mean={features.transient.mean():.3f}")
    click.echo(f"Voice Clarity: max={features.voice_clarity.max():.3f}, mean={features.voice_clarity.mean():.3f}")
    click.echo(f"Spectral Centroid: mean={features.spectral_centroid.mean():.3f}")


@cli.command()
@click.option('--output', '-o', default='config_template.json')
def create_config(output):
    """Erstellt eine Beispiel-Konfigurationsdatei fuer GPU-Rendering."""
    config = {
        "audio_file": "input.mp3",
        "output_file": "output.mp4",
        "visual": {
            "type": "lumina_core",
            "resolution": [1920, 1080],
            "fps": 60,
            "colors": {
                "primary": "#FF0055",
                "secondary": "#00CCFF",
                "background": "#0A0A0A"
            },
            "params": {
                "intensity": 1.0,
                "speed": 1.0
            }
        },
        "background": {
            "image": None,
            "blur": 0.0,
            "vignette": 0.3,
            "opacity": 0.3
        }
    }
    
    with open(output, 'w') as f:
        json.dump(config, f, indent=2)
    
    click.echo(f"GPU-Konfigurations-Template erstellt: {output}")


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--visual', '-v', default='lumina_core')
@click.option('--resolutions', '-r', default='1920x1080,1280x720,854x480',
              help='Komma-getrennte Aufloesungen')
@click.option('--output-prefix', '-o', default='output',
              help='Prefix fuer Output-Dateien (z.B. output -> output_1920x1080.mp4)')
@click.option('--fps', default=60, type=int)
@click.option('--preview', is_flag=True, help='Schnelle Vorschau')
@click.option('--codec', default='h264', type=click.Choice(['h264', 'hevc', 'prores']))
@click.option('--quality', default='high', type=click.Choice(['low', 'medium', 'high', 'lossless']))
def render_multi(audio_file, visual, resolutions, output_prefix, fps, preview, codec, quality):
    """Rendert in mehreren Aufloesungen gleichzeitig."""
    _check_ffmpeg()
    
    from src.gpu_renderer import GPUBatchRenderer
    from src.analyzer import AudioAnalyzer
    
    # Audio einmal analysieren
    analyzer = AudioAnalyzer()
    features = analyzer.analyze(audio_file, fps=fps)
    
    click.echo(f"[Multi] Audio analysiert: {features.duration:.1f}s @ {features.tempo:.0f} BPM")
    click.echo(f"[Multi] Rendere {visual} in mehreren Aufloesungen...")
    
    res_list = [r.strip() for r in resolutions.split(',')]
    
    for res_str in res_list:
        try:
            width, height = map(int, res_str.split('x'))
        except ValueError:
            click.echo(f"  Ueberspringe ungueltige Aufloesung: {res_str}")
            continue
        
        output_path = f"{output_prefix}_{width}x{height}.mp4"
        click.echo(f"  Rendering {width}x{height} -> {output_path}")
        
        renderer = GPUBatchRenderer(width=width, height=height, fps=fps)
        renderer.render(
            audio_path=audio_file,
            visualizer_type=visual,
            output_path=output_path,
            features=features,
            preview_mode=preview,
            preview_duration=5.0,
            codec=codec,
            quality=quality,
        )
        click.echo(f"  Fertig: {output_path}")
    
    click.echo("[Multi] Alle Aufloesungen fertig!")


@cli.command()
@click.argument('batch_file', type=click.Path(exists=True))
def batch(batch_file):
    """Fuehrt Batch-Jobs aus einer JSON-Datei aus.
    
    Beispiel batch.json:
    [
      {"audio": "song1.mp3", "visual": "lumina_core", "output": "out1.mp4"},
      {"audio": "song2.mp3", "visual": "voice_flow", "output": "out2.mp4"}
    ]
    """
    _check_ffmpeg()
    
    with open(batch_file) as f:
        jobs = json.load(f)
    
    click.echo(f"[Batch] {len(jobs)} Jobs gefunden")
    
    from src.gpu_renderer import GPUBatchRenderer
    from src.analyzer import AudioAnalyzer
    
    # Einen Renderer wiederverwenden fuer alle Jobs (schneller, vermeidet Context-Probleme)
    renderer = None
    current_resolution = None
    
    for i, job in enumerate(jobs, 1):
        click.echo(f"\n[Batch] Job {i}/{len(jobs)}: {job.get('audio', 'unknown')}")
        
        audio = job['audio']
        visual = job.get('visual', 'lumina_core')
        output = job.get('output', 'output.mp4')
        resolution = job.get('resolution', '1920x1080')
        fps = job.get('fps', 60)
        codec = job.get('codec', 'h264')
        quality = job.get('quality', 'high')
        
        width, height = map(int, resolution.split('x'))
        
        # Neuen Renderer erstellen wenn Aufloesung/FPS sich aendert
        if renderer is None or current_resolution != (width, height, fps):
            if renderer is not None:
                try:
                    renderer.__del__()
                except Exception:
                    pass
            renderer = GPUBatchRenderer(width=width, height=height, fps=fps)
            current_resolution = (width, height, fps)
        
        renderer.render(
            audio_path=audio,
            visualizer_type=visual,
            output_path=output,
            codec=codec,
            quality=quality,
        )
        click.echo(f"[Batch] Job {i} fertig: {output}")
    
    if renderer is not None:
        try:
            renderer.__del__()
        except Exception:
            pass
    
    click.echo("\n[Batch] Alle Jobs abgeschlossen!")


if __name__ == '__main__':
    cli()
