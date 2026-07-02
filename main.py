#!/usr/bin/env python3
"""
Audio Visualizer Pro - CLI Entry Point

KI-Optimierter Workflow fuer Audio-Visualisierungen.
"""

from dotenv import load_dotenv

load_dotenv()

import click
import json
import os
import shutil
import subprocess
from pathlib import Path

from config.schemas import load_and_validate_config
from src.types import Quote
from src.quote_overlay import QuoteOverlayConfig


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
@click.option('--background-image', '-bg', type=click.Path(), help='Hintergrundbild oder -video (Bild: jpg/png/webp, Video: mp4/mov/avi/mkv/webm/gif)')
@click.option('--background-blur', default=0.0, type=float, help='Hintergrund-Blur Radius')
@click.option('--background-vignette', default=0.0, type=float, help='Vignette Staerke (0.0-1.0)')
@click.option('--background-opacity', default=0.3, type=float, help='Hintergrund-Opazitaet (0.0-1.0)')
@click.option('--background-color', default='#0A0A0A', help='Hintergrund-Farbe als Hex (z.B. #0A0A0A)')
@click.option('--codec', default='h264', type=click.Choice(['h264', 'hevc', 'prores']), help='Video-Codec')
@click.option('--quality', default='high', type=click.Choice(['low', 'medium', 'high', 'lossless']), help='Qualitaet')
@click.option('--param', '-p', multiple=True, help='Visualizer Parameter (key=value)')
@click.option('--intro', type=click.Path(exists=True), help='Intro-Video, das vor dem gerenderten Video eingefuegt wird')
@click.option('--intro-fade', default=1.0, type=float, help='Crossfade-Dauer zwischen Intro und Hauptvideo in Sekunden')
def render(audio_file, visual, output, config, resolution, fps, preview, preview_duration,
           background_image, background_blur, background_vignette, background_opacity,
           background_color, codec, quality, param, intro, intro_fade):
    """Rendert Audio-Visualisierung auf der GPU."""
    
    _check_ffmpeg()
    
    try:
        width, height = map(int, resolution.split('x'))
    except ValueError:
        raise click.BadParameter(
            f"Ungueltige Aufloesung: '{resolution}'. "
            f"Format: BREITExHOEHE (z.B. 1920x1080)"
        )
    
    # Parameter parsen (CLI hat Vorrang vor Config)
    cli_params = {}
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
        cli_params[key] = value

    # Config-File laden und validieren (optional)
    cfg_visual_type = visual
    cfg_resolution = resolution
    cfg_fps = fps
    cfg_background_image = background_image
    cfg_background_blur = background_blur
    cfg_background_vignette = background_vignette
    cfg_background_opacity = background_opacity
    cfg_background_color = background_color
    cfg_codec = codec
    cfg_quality = quality
    cfg_postprocess = None
    cfg_quotes = None
    cfg_quote_overlay = None
    cfg_output = output
    cfg_intro_video = intro
    cfg_intro_fade = intro_fade

    if config:
        try:
            cfg = load_and_validate_config(config)
            cfg_visual_type = cfg.visual.type
            cfg_resolution = f"{cfg.visual.resolution[0]}x{cfg.visual.resolution[1]}"
            cfg_fps = cfg.visual.fps
            cfg_background_image = cfg.background_image or cfg_background_image
            cfg_background_blur = cfg.background_blur
            cfg_background_vignette = cfg.background_vignette
            cfg_background_opacity = cfg.background_opacity
            cfg_background_color = getattr(cfg, 'background_color', None) or cfg_background_color
            cfg_postprocess = cfg.postprocess.model_dump() if cfg.postprocess else None
            cfg_output = cfg.output_file
            if intro is None and cfg.intro_video is not None:
                cfg_intro_video = cfg.intro_video
            cfg_intro_fade = cfg.intro_fade if cfg_intro_fade == intro_fade else cfg.intro_fade_duration
            if cfg.quotes:
                cfg_quotes = [
                    Quote(
                        text=q.text,
                        start_time=q.start_time,
                        end_time=q.end_time,
                        confidence=q.confidence,
                    )
                    for q in cfg.quotes
                ]
            cfg_quote_overlay = cfg.quote_overlay
        except Exception as e:
            raise click.BadParameter(f"Ungueltige Config-Datei '{config}': {e}")

    # CLI-Optionen ueberschreiben Config-Werte
    visual = cfg_visual_type
    resolution = cfg_resolution
    fps = cfg_fps
    background_image = cfg_background_image
    background_blur = cfg_background_blur
    background_vignette = cfg_background_vignette
    background_opacity = cfg_background_opacity
    background_color = cfg_background_color
    codec = cfg_codec
    quality = cfg_quality
    output = output if output != 'output.mp4' else cfg_output
    intro = cfg_intro_video
    intro_fade = cfg_intro_fade

    try:
        width, height = map(int, resolution.split('x'))
    except ValueError:
        raise click.BadParameter(
            f"Ungueltige Aufloesung: '{resolution}'. "
            f"Format: BREITExHOEHE (z.B. 1920x1080)"
        )

    # Config-Parameter als Basis, CLI-Parameter ueberschreiben
    params = {}
    if config:
        cfg_params = cfg.visual.params.model_dump()
        params.update(cfg_params)
    params.update(cli_params)

    # Quote-Overlay-Config aus Config bauen (falls vorhanden)
    quote_config = None
    if cfg_quote_overlay is not None:
        quote_config = QuoteOverlayConfig(**cfg_quote_overlay.model_dump())

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
        background_color=background_color,
        codec=codec,
        quality=quality,
        postprocess=cfg_postprocess,
        quotes=cfg_quotes,
        quote_config=quote_config,
    )
    
    click.echo(f"[GPU] Fertig! Output: {output}")

    if intro:
        from src.intro_renderer import render_with_intro
        intro_output = str(Path(output).parent / f"{Path(output).stem}_mit_intro{Path(output).suffix}")
        click.echo(f"[Intro] Setze Intro vor: {intro_output}")
        render_with_intro(
            intro_path=intro,
            main_video_path=output,
            output_path=intro_output,
            fade_duration=intro_fade,
        )
        click.echo(f"[Intro] Fertig! Output: {intro_output}")


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


class {''.join(part.capitalize() for part in name.split('_'))}Visualizer(BaseGPUVisualizer):
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


@cli.command('create-visualizer')
@click.argument('name')
@click.option('--type', '-t', 'viz_type', default='shader',
              type=click.Choice(['shader', 'geometry', 'particles']),
              help='Template-Typ fuer den neuen Visualizer')
@click.option('--test/--no-test', default=True,
              help='Smoke-Test nach Erstellung durchfuehren (1 Frame rendern)')
@click.option('--target-dir', default='src/gpu_visualizers', type=click.Path())
def create_visualizer(name, viz_type, test, target_dir):
    """Erstellt ein neues GPU-Visualizer-Template mit reichhaltigem Startpunkt.

    Der Visualizer wird automatisch in der Registry registriert (Auto-Discovery)
    und optional direkt mit einem Smoke-Test ueberprueft.
    """
    from src.visualizer_wizard import VisualizerWizard

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    wizard = VisualizerWizard(name, viz_type=viz_type)
    file_path = wizard.write(target)
    click.echo(f"Visualizer erstellt: {file_path}")

    # Auto-Discovery aktualisieren, damit der neue Visualizer sofort verfuegbar ist.
    from src.gpu_visualizers import refresh_registry
    refresh_registry()

    # Pruefen, ob der Visualizer in der Registry auftaucht.
    from src.gpu_visualizers import list_visualizers
    available = list_visualizers()
    if wizard.module_name not in available:
        raise click.ClickException(
            f"Visualizer '{wizard.module_name}' wurde nicht in der Registry gefunden. "
            f"Verfuegbar: {available}"
        )
    click.echo(f"Auto-Registrierung erfolgreich: {wizard.module_name}")

    if test:
        click.echo("Fuehre Smoke-Test durch...")
        _smoke_test_visualizer(wizard.module_name)
        click.echo("Smoke-Test bestanden.")


def _smoke_test_visualizer(name: str):
    """Rendert einen Frame mit Dummy-Features zur Validierung eines Visualizers."""
    import numpy as np
    import moderngl

    from src.gpu_visualizers import get_visualizer, validate_visualizer_class

    cls = get_visualizer(name)
    errors = validate_visualizer_class(cls)
    if errors:
        raise click.ClickException("Validierung fehlgeschlagen:\n" + "\n".join(errors))

    ctx = moderngl.create_standalone_context()
    try:
        texture = ctx.texture((640, 480), 3)
        fbo = ctx.framebuffer(color_attachments=[texture])
        viz = cls(ctx, 640, 480)

        dummy_features = {
            "rms": np.random.rand(30).astype(np.float32),
            "onset": np.random.rand(30).astype(np.float32),
            "beat_intensity": np.random.rand(30).astype(np.float32),
            "spectral_centroid": np.random.rand(30).astype(np.float32),
            "chroma": np.random.rand(12, 30).astype(np.float32),
            "transient": np.random.rand(30).astype(np.float32),
            "voice_clarity": np.random.rand(30).astype(np.float32),
            "fps": 30,
            "frame_count": 30,
            "mode": "music",
            "tempo": 120.0,
        }

        fbo.use()
        ctx.clear(0.0, 0.0, 0.0)
        viz.render(dummy_features, 0.5)
        pixels = fbo.read(components=3)
        if len(pixels) != 640 * 480 * 3:
            raise click.ClickException("Falsche Pixel-Anzahl im gerenderten Frame")
    finally:
        ctx.release()


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--fps', default=60, type=int, help='Frames pro Sekunde fuer die Analyse')
def analyze(audio_file, fps):
    """Analysiert eine Audio-Datei und zeigt Features an."""
    from src.analyzer import AudioAnalyzer
    
    analyzer = AudioAnalyzer()
    features = analyzer.analyze(audio_file, fps=fps)
    
    click.echo("\n=== Audio-Analyse Ergebnisse ===")
    click.echo(f"Dauer: {features.duration:.2f}s")
    click.echo(f"Sample Rate: {features.sample_rate}Hz")
    click.echo(f"Tempo: {features.tempo:.1f} BPM")
    click.echo(f"Key: {features.key or 'Unbekannt'}")
    click.echo(f"Mode: {features.mode}")
    click.echo(f"Frames: {features.frame_count}")
    
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
        "postprocess": {
            "contrast": 1.0,
            "saturation": 1.0,
            "brightness": 0.0,
            "warmth": 0.0,
            "film_grain": 0.0
        },
        "background_image": None,
        "background_blur": 0.0,
        "background_vignette": 0.3,
        "background_opacity": 0.3,
        "background_color": "#0A0A0A",
        "intro_video": None,
        "intro_fade_duration": 1.0,
        "quote_overlay": {
            "enabled": False,
            "font_size": 52,
            "font_color": "#FFFFFF",
            "box_color": "#1A1A2E",
            "display_duration": 8.0,
            "position": "bottom"
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
    analyzer = AudioAnalyzer()
    
    for i, job in enumerate(jobs, 1):
        click.echo(f"\n[Batch] Job {i}/{len(jobs)}: {job.get('audio', 'unknown')}")

        audio = job.get('audio')
        if not audio:
            click.echo(f"[Batch] Fehler: Job {i} hat keinen 'audio'-Key, ueberspringe.")
            continue
        if not os.path.exists(audio):
            click.echo(f"[Batch] Fehler: Audio-Datei nicht gefunden: {audio}, ueberspringe.")
            continue

        visual = job.get('visual', 'lumina_core')
        output = job.get('output', 'output.mp4')
        resolution = job.get('resolution', '1920x1080')
        fps = job.get('fps', 60)
        codec = job.get('codec', 'h264')
        quality = job.get('quality', 'high')
        params = job.get('params', {})
        postprocess = job.get('postprocess')
        background_image = job.get('background_image')
        background_blur = job.get('background_blur', 0.0)
        background_vignette = job.get('background_vignette', 0.0)
        background_opacity = job.get('background_opacity', 0.3)

        # Quotes aus Job-JSON parsen
        quotes = None
        quote_config = None
        raw_quotes = job.get('quotes')
        raw_quote_overlay = job.get('quote_overlay')
        if raw_quotes:
            quotes = [
                Quote(
                    text=q.get('text', ''),
                    start_time=float(q.get('start_time', 0.0)),
                    end_time=float(q.get('end_time', 0.0)),
                    confidence=float(q.get('confidence', 1.0)),
                )
                for q in raw_quotes
                if q.get('text')
            ]
        if raw_quote_overlay:
            quote_config = QuoteOverlayConfig(**raw_quote_overlay)

        try:
            width, height = map(int, resolution.split('x'))
        except ValueError:
            click.echo(f"[Batch] Ueberspringe ungueltige Aufloesung: {resolution}")
            continue

        # Audio einmal pro Job analysieren und weitergeben (Caching nutzen)
        try:
            features = analyzer.analyze(audio, fps=fps)
        except Exception as e:
            click.echo(f"[Batch] Analyse fehlgeschlagen fuer {audio}: {e}")
            continue

        # Neuen Renderer erstellen wenn Aufloesung/FPS sich aendert
        if renderer is None or current_resolution != (width, height, fps):
            if renderer is not None:
                try:
                    renderer.release()
                except Exception:
                    pass
            renderer = GPUBatchRenderer(width=width, height=height, fps=fps)
            current_resolution = (width, height, fps)

        renderer.render(
            audio_path=audio,
            visualizer_type=visual,
            output_path=output,
            features=features,
            params=params if params else None,
            preview_mode=job.get('preview', False),
            preview_duration=job.get('preview_duration', 5.0),
            background_image=background_image,
            background_blur=background_blur,
            background_vignette=background_vignette,
            background_opacity=background_opacity,
            quotes=quotes,
            quote_config=quote_config,
            postprocess=postprocess,
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