"""
Audio Visualizer Pro - Grafische Benutzeroberfläche (GUI)

Eine moderne Web-basierte GUI mit Streamlit.
Ermöglicht einfache Bedienung ohne Kommandozeile.
"""

import streamlit as st
import subprocess
import sys
import json
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

# .env Datei laden
from dotenv import load_dotenv
load_dotenv()

# Füge src zum Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent))

from src.visuals.registry import VisualizerRegistry
from src.analyzer import AudioAnalyzer
from src.ai_matcher import SmartMatcher
from src.gemini_integration import GeminiIntegration
from src.types import VisualConfig, Quote
from src.quote_overlay import QuoteOverlayConfig
from src.postprocess import PostProcessor
from src.gpu_renderer import GPUBatchRenderer, GPUPreviewRenderer
from src.gpu_preview import render_gpu_preview
from src.gpu_visualizers import get_visualizer, list_visualizers

# Seiten-Config
st.set_page_config(
    page_title="Audio Visualizer Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für besseres Styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a3e 50%, #16213e 100%);
        color: #e0e0ff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #768efa 0%, #865bb2 100%);
    }
    .stButton>button:active {
        transform: translateY(0px);
    }
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox, .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stCheckbox > label {
        color: #e0e0ff !important;
    }
    .stExpander {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .preview-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .success-box {
        background: rgba(39, 174, 96, 0.15);
        border-left: 4px solid #27ae60;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    .info-box {
        background: rgba(52, 152, 219, 0.15);
        border-left: 4px solid #3498db;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    .warning-box {
        background: rgba(241, 196, 15, 0.15);
        border-left: 4px solid #f1c40f;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    .gpu-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f1e 100%);
    }
    /* Radio buttons */
    .stRadio > label {
        color: #e0e0ff !important;
    }
    /* File uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 2px dashed rgba(102, 126, 234, 0.5);
    }
</style>
""", unsafe_allow_html=True)


def check_system_requirements():
    """Prüft ob FFmpeg und essenzielle Abhängigkeiten verfügbar sind."""
    issues = []
    
    # FFmpeg prüfen
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode != 0:
            issues.append("FFmpeg ist installiert, aber funktioniert nicht korrekt.")
    except FileNotFoundError:
        issues.append("FFmpeg nicht gefunden. Bitte installieren: https://ffmpeg.org/download.html")
    except Exception as e:
        issues.append(f"FFmpeg-Check fehlgeschlagen: {e}")
    
    return issues


# Color-Grading Presets
COLOR_PRESETS = {
    "Neutral (Kein Grading)": {"contrast": 1.0, "saturation": 1.0, "brightness": 0.0, "warmth": 0.0, "film_grain": 0.0},
    "🎬 Cinematic Warm": {"contrast": 1.15, "saturation": 1.05, "brightness": -0.02, "warmth": 0.25, "film_grain": 0.15},
    "🌃 Cyberpunk Cold": {"contrast": 1.25, "saturation": 1.3, "brightness": -0.05, "warmth": -0.35, "film_grain": 0.1},
    "📼 Vintage Film": {"contrast": 0.9, "saturation": 0.75, "brightness": 0.02, "warmth": 0.4, "film_grain": 0.35},
    "🕵️ Noir": {"contrast": 1.4, "saturation": 0.0, "brightness": -0.08, "warmth": 0.0, "film_grain": 0.4},
    "🌅 Golden Hour": {"contrast": 1.1, "saturation": 1.15, "brightness": 0.05, "warmth": 0.5, "film_grain": 0.05},
    "🎵 Concert Neon": {"contrast": 1.2, "saturation": 1.4, "brightness": -0.03, "warmth": 0.15, "film_grain": 0.08},
}


def get_available_visualizers():
    """Lädt alle verfügbaren Visualizer."""
    VisualizerRegistry.autoload()
    return VisualizerRegistry.list_available()


def get_config_presets():
    """Lädt alle Config-Presets."""
    config_dir = Path("config")
    presets = {}
    if config_dir.exists():
        for json_file in config_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    presets[json_file.stem] = {
                        'file': str(json_file),
                        'visual': data.get('visual', {}).get('type', 'unknown'),
                        'description': data.get('description', '')
                    }
            except:
                pass
    return presets


def get_visualizer_info(visualizer_name: str) -> dict:
    """Gibt Informationen über einen Visualizer zurück."""
    info = {
        'pulsing_core': {
            'emoji': '🔴',
            'description': 'Pulsierender Kreis mit Chroma-Farben',
            'best_for': 'EDM, Pop',
            'color': '#FF0055'
        },
        'spectrum_bars': {
            'emoji': '📊',
            'description': '40-Balken Equalizer',
            'best_for': 'Rock, Hip-Hop',
            'color': '#00CCFF'
        },
        'chroma_field': {
            'emoji': '✨',
            'description': 'Partikel-Feld basierend auf Tonart',
            'best_for': 'Ambient, Jazz',
            'color': '#9D4EDD'
        },
        'particle_swarm': {
            'emoji': '🔥',
            'description': 'Physik-basierte Partikel-Explosionen',
            'best_for': 'Dubstep, Trap',
            'color': '#FF6B35'
        },
        'typographic': {
            'emoji': '📝',
            'description': 'Minimalistisch mit Wellenform',
            'best_for': 'Podcasts, Sprache',
            'color': '#00F5FF'
        },
        'neon_oscilloscope': {
            'emoji': '💠',
            'description': 'Retro-futuristischer Oszilloskop',
            'best_for': 'Synthwave, Cyberpunk',
            'color': '#00F5FF'
        },
        'sacred_mandala': {
            'emoji': '🕉️',
            'description': 'Heilige Geometrie mit rotierenden Mustern',
            'best_for': 'Meditation, Ambient',
            'color': '#FF9E00'
        },
        'liquid_blobs': {
            'emoji': '💧',
            'description': 'Flüssige MetaBall-ähnliche Blobs',
            'best_for': 'House, Techno',
            'color': '#00D9FF'
        },
        'neon_wave_circle': {
            'emoji': '⭕',
            'description': 'Konzentrische Neon-Ringe mit Wellen',
            'best_for': 'EDM, Techno',
            'color': '#39FF14'
        },
        'frequency_flower': {
            'emoji': '🌸',
            'description': 'Organische Blumen mit Audio-reaktiven Blütenblättern',
            'best_for': 'Indie, Folk, Pop',
            'color': '#FFB7B2'
        },
        # Signature Pro Visualizer (v2.0)
        'lumina_core': {
            'emoji': '⚡',
            'description': 'Raymarched Kern mit FBM-Noise, Beat-Explosionen und Chromatic Aberration',
            'best_for': 'EDM, Dubstep, Trap',
            'color': '#FF00AA'
        },
        'voice_flow': {
            'emoji': '🌊',
            'description': 'Organischer, atmender Flow - nie ablenkend',
            'best_for': 'Podcasts, Meditation, Ambient',
            'color': '#00E5FF'
        },
        'spectrum_genesis': {
            'emoji': '🌌',
            'description': 'Hybrid: Bars + Wellenform + SDF-Glow',
            'best_for': 'Alle Genres, Hybrid-Audio',
            'color': '#AA00FF'
        }
    }
    return info.get(visualizer_name, {
        'emoji': '🎨',
        'description': 'Visualizer',
        'best_for': 'Alle Genres',
        'color': '#ffffff'
    })


def render_preview(audio_path: str, visualizer: str, duration: float = 5.0):
    """Rendert eine Vorschau."""
    output_path = tempfile.mktemp(suffix='.mp4')
    
    cmd = [
        sys.executable, 'main.py', 'render', audio_path,
        '--visual', visualizer,
        '--output', output_path,
        '--preview',
        '--preview-duration', str(duration)
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process, output_path


def render_full(audio_path: str, visualizer: str, output_path: str, 
                resolution: str = "1920x1080", fps: int = 60):
    """Rendert das vollständige Video."""
    cmd = [
        sys.executable, 'main.py', 'render', audio_path,
        '--visual', visualizer,
        '--output', output_path,
        '--resolution', resolution,
        '--fps', str(fps)
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process


def render_with_config(audio_path: str, config_path: str, output_path: str,
                        preview: bool = False, preview_duration: float = 5.0):
    """Rendert mit Config-Datei."""
    cmd = [
        sys.executable, 'main.py', 'render', audio_path,
        '--config', config_path,
        '--output', output_path
    ]
    
    if preview:
        cmd.extend(['--preview', '--preview-duration', str(preview_duration)])
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process


def render_parameter_sliders(visualizer_type: str) -> dict:
    """Rendert Parameter-Slider fuer den ausgewaehlten Visualizer mit Session-State."""
    params = {}
    prefix = f"viz_param_{visualizer_type}_"
    
    st.markdown("#### 🔧 Parameter-Tuning")
    
    # Default-Werte pro Visualizer
    defaults = {}
    if visualizer_type == "pulsing_core":
        defaults = {
            'pulse_intensity': 0.8, 'glow_layers': 5, 'glow_radius': 25, 'ring_count': 4,
            'easing': 'expo', 'ambient_glow': 0.5, 'reflection': True, 'use_chroma_colors': True
        }
    elif visualizer_type == "spectrum_bars":
        defaults = {
            'bar_count': 40, 'smoothing': 0.3, 'bar_width': 3, 'glow_intensity': 0.5
        }
    
    # WICHTIG: Session-State vor den Widgets initialisieren, damit Slider ohne Default rendern koennen
    for param_name, default_val in defaults.items():
        key = f"{prefix}{param_name}"
        if key not in st.session_state:
            st.session_state[key] = default_val
    
    if visualizer_type == "pulsing_core":
        col1, col2 = st.columns(2)
        with col1:
            params['pulse_intensity'] = st.slider("💓 Pulse Intensität", 0.0, 2.0, step=0.1, key=f"{prefix}pulse_intensity")
            params['glow_layers'] = st.slider("✨ Glow Layer", 1, 10, step=1, key=f"{prefix}glow_layers")
            params['glow_radius'] = st.slider("🌟 Glow Radius", 5, 50, step=5, key=f"{prefix}glow_radius")
            params['ring_count'] = st.slider("⭕ Ring Anzahl", 1, 8, step=1, key=f"{prefix}ring_count")
        with col2:
            params['easing'] = st.selectbox("📈 Easing", ["expo", "cubic", "quad", "linear"], key=f"{prefix}easing")
            params['ambient_glow'] = st.slider("🌅 Ambient Glow", 0.0, 1.0, step=0.1, key=f"{prefix}ambient_glow")
            params['reflection'] = st.checkbox("🪞 Reflexion", key=f"{prefix}reflection")
            params['use_chroma_colors'] = st.checkbox("🎨 Chroma-Farben", key=f"{prefix}use_chroma_colors")
    elif visualizer_type == "spectrum_bars":
        col1, col2 = st.columns(2)
        with col1:
            params['bar_count'] = st.slider("📊 Balken Anzahl", 10, 100, step=5, key=f"{prefix}bar_count")
            params['smoothing'] = st.slider("🧈 Smoothing", 0.0, 1.0, step=0.05, key=f"{prefix}smoothing")
        with col2:
            params['bar_width'] = st.slider("📏 Balken Breite", 1, 10, step=1, key=f"{prefix}bar_width")
            params['glow_intensity'] = st.slider("✨ Glow", 0.0, 1.0, step=0.1, key=f"{prefix}glow_intensity")
    else:
        st.info(f"🔧 Parameter-Tuning für '{visualizer_type}' wird demnächst hinzugefügt!")
    
    return params


def render_gpu_parameter_sliders(gpu_viz: str) -> dict:
    """Rendert GPU-Parameter-Slider dynamisch aus dem Visualizer PARAMS."""
    params = {}
    prefix = f"gpu_param_{gpu_viz}_"
    
    try:
        viz_cls = get_visualizer(gpu_viz)
        param_spec = viz_cls.PARAMS
    except Exception:
        st.info(f"🔧 Parameter-Tuning für '{gpu_viz}' wird demnächst hinzugefügt!")
        return params
    
    # Session-State initialisieren
    for param_name, (default, min_val, max_val, step) in param_spec.items():
        key = f"{prefix}{param_name}"
        if key not in st.session_state:
            st.session_state[key] = default
    
    # Sliders erstellen
    cols = st.columns(2)
    col_idx = 0
    for param_name, (default, min_val, max_val, step) in param_spec.items():
        with cols[col_idx % 2]:
            if isinstance(step, int) and isinstance(default, int):
                params[param_name] = st.slider(
                    f"{param_name.replace('_', ' ').title()}",
                    min_val, max_val,
                    step=step,
                    key=f"{prefix}{param_name}"
                )
            else:
                params[param_name] = st.slider(
                    f"{param_name.replace('_', ' ').title()}",
                    float(min_val), float(max_val),
                    step=float(step),
                    key=f"{prefix}{param_name}"
                )
        col_idx += 1
    
    return params


def prepare_background(image_path: str, width: int, height: int, 
                         blur: float = 0.0, vignette: float = 0.0):
    """Laedt und bereitet ein Hintergrundbild vor."""
    from PIL import ImageFilter, ImageDraw
    
    bg = Image.open(image_path).convert('RGB')
    bg = bg.resize((width, height), Image.LANCZOS)
    
    if blur > 0:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=blur))
    
    if vignette > 0:
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        center_x, center_y = width // 2, height // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for r in range(int(max_dist), 0, -5):
            alpha = int(255 * (1.0 - vignette * (1.0 - r / max_dist)))
            alpha = max(0, min(255, alpha))
            draw.ellipse(
                [center_x - r, center_y - r, center_x + r, center_y + r],
                fill=alpha
            )
        
        black = Image.new('RGB', (width, height), (0, 0, 0))
        mask_inv = Image.fromarray(255 - np.array(mask))
        bg = Image.composite(black, bg, mask_inv)
    
    return bg


def composite_with_background(frame: np.ndarray, bg_image, opacity: float = 0.3):
    """Mischt einen Frame mit dem Hintergrundbild."""
    
    frame_img = Image.fromarray(frame).convert('RGB')
    opacity = max(0.0, min(1.0, opacity))
    blended = Image.blend(frame_img, bg_image, opacity)
    return np.array(blended)


def render_live_frame(audio_path: str, visualizer_type: str, params: dict,
                      resolution: str = "480x270", fps: int = 30,
                      background_image: str = None, background_blur: float = 0.0,
                      background_vignette: float = 0.0, background_opacity: float = 0.3) -> np.ndarray:
    """Rendert ein einzelnes Frame fuer die Live-Preview."""
    try:
        analyzer = AudioAnalyzer()
        features = analyzer.analyze(audio_path, fps=fps)
        
        VisualizerRegistry.autoload()
        vclass = VisualizerRegistry.get(visualizer_type)
        
        width, height = map(int, resolution.split('x'))
        config = VisualConfig(
            type=visualizer_type,
            resolution=(width, height),
            fps=fps,
            colors={"primary": "#FF0055", "secondary": "#00CCFF", "background": "#0A0A0A"},
            params=params
        )
        
        visualizer = vclass(config, features)
        visualizer.setup()
        
        # Render frame at 30% of audio where action is usually good
        frame_idx = min(int(features.frame_count * 0.3), features.frame_count - 1)
        frame = visualizer.render_frame(frame_idx)
        
        # Apply light post-processing
        post = PostProcessor({
            'contrast': 1.1,
            'saturation': 1.2,
            'bloom': 0.3,
            'vignette': 0.2
        })
        frame = post.apply(frame)
        
        # Hintergrundbild kompositieren
        if background_image and os.path.exists(background_image):
            bg = prepare_background(background_image, width, height, background_blur, background_vignette)
            frame = composite_with_background(frame, bg, background_opacity)
        
        return frame
    except Exception as e:
        st.error(f"Live-Preview Fehler: {e}")
        return None


def save_ai_config(ai_recommendation, resolution: str = "1920x1080", fps: int = 60):
    """Speichert eine AIRecommendation als temporäre JSON-Config."""
    config_obj = ai_recommendation.to_visual_config(
        resolution=tuple(int(x) for x in resolution.split("x")),
        fps=fps
    )
    config = {
        "visual": {
            "type": config_obj.type,
            "params": config_obj.params,
            "colors": config_obj.colors,
            "resolution": list(config_obj.resolution),
            "fps": config_obj.fps
        },
        "postprocess": {}
    }
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, temp_config, indent=2)
    temp_config.close()
    return temp_config.name


def save_render_config(audio_path: str, output_path: str, visualizer: str,
                       resolution: str = "1920x1080", fps: int = 60,
                       quotes: list = None, preview: bool = False,
                       preview_duration: float = 5.0,
                       colors: dict = None, params: dict = None,
                       background_image: str = None, background_blur: float = 0.0,
                       background_vignette: float = 0.0, background_opacity: float = 0.3,
                       postprocess: dict = None, turbo_mode: bool = False,
                       frame_skip: int = 1) -> str:
    """
    Erstellt eine vollstaendige Render-Config inkl. Quotes als temporäre JSON.
    
    Args:
        audio_path: Pfad zur Audio-Datei
        output_path: Pfad zur Ausgabe-Datei
        visualizer: Name des Visualizers
        resolution: Aufloesung als String "WxH"
        fps: Frames pro Sekunde
        quotes: Liste von Quote-Dictionaries
        preview: Preview-Modus
        preview_duration: Preview-Dauer in Sekunden
        colors: Farb-Config
        params: Visualizer-Parameter
        
    Returns:
        Pfad zur temporären Config-Datei
    """
    width, height = map(int, resolution.split("x"))
    
    config = {
        "audio_file": audio_path,
        "output_file": output_path,
        "visual": {
            "type": visualizer,
            "resolution": [width, height],
            "fps": fps,
            "colors": colors or {"primary": "#FF0055", "secondary": "#00CCFF", "background": "#0A0A0A"},
            "params": params or {}
        },
        "postprocess": postprocess or {
            "contrast": 1.0,
            "saturation": 1.0,
            "grain": 0.0,
            "vignette": 0.0
        },
        "background_image": background_image,
        "background_blur": background_blur,
        "background_vignette": background_vignette,
        "background_opacity": background_opacity,
        "turbo_mode": turbo_mode
    }
    
    if quotes:
        # Konvertiere Quote-Objekte zu Dictionaries fuer JSON-Serialisierung
        quote_dicts = []
        for q in quotes:
            if hasattr(q, 'text'):
                # Quote Dataclass Objekt
                quote_dicts.append({
                    "text": q.text,
                    "start_time": q.start_time,
                    "end_time": q.end_time,
                    "confidence": q.confidence
                })
            elif isinstance(q, dict):
                # Bereits ein Dictionary
                quote_dicts.append({
                    "text": q.get("text", q.get("quote", "")),
                    "start_time": q.get("start_time", 0.0),
                    "end_time": q.get("end_time", 0.0),
                    "confidence": q.get("confidence", 1.0)
                })
        config["quotes"] = quote_dicts
    
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, temp_config, indent=2)
    temp_config.close()
    return temp_config.name


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0;">🎵 Audio Visualizer Pro</h1>
        <p style="font-size: 1.3rem; color: #888; margin-top: 0.5rem;">
            KI-optimierte Audio-Visualisierungen für professionelle Musikvideos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Einstellungen")
        
        # Modus-Auswahl
        mode = st.radio(
            "Render-Modus",
            ["Schnell (Vorschau)", "Vollständig (HD)"],
            help="Vorschau = 5 Sekunden in 480p, Vollständig = Komplettes Video in HD"
        )
        
        codec = "h264"
        quality = "medium"
        
        if mode == "Vollständig (HD)":
            col1, col2, col3 = st.columns(3)
            with col1:
                resolution = st.selectbox(
                    "Auflösung",
                    ["1920x1080", "1280x720", "3840x2160", "854x480"],
                    index=0
                )
            with col2:
                fps = st.selectbox(
                    "FPS",
                    [60, 30, 24],
                    index=0
                )
            with col3:
                codec = st.selectbox(
                    "Codec",
                    ["h264", "hevc", "prores"],
                    index=0,
                    help="H.264 = Standard, HEVC = Kleine Dateien, ProRes = Professionell"
                )
                quality = st.selectbox(
                    "Qualität",
                    ["high", "medium", "low", "lossless"],
                    index=0,
                )
        else:
            resolution = "854x480"
            fps = 30
        
        st.markdown("---")
        
        # Turbo-Modus
        turbo_mode = st.checkbox(
            "⚡ Turbo-Modus",
            value=False,
            help="Schnelleres Encoding mit leicht reduzierter Qualität (ultrafast preset)."
        )
        
        # Frame-Skip fuer schnelleres Rendering
        frame_skip = st.selectbox(
            "🚀 Frame-Skip (Draft-Modus)",
            options=[("Jeder Frame (Qualität)", 1), ("Jeder 2. Frame (2x schneller)", 2), ("Jeder 3. Frame (3x schneller)", 3)],
            format_func=lambda x: x[0],
            index=0,
            help="Bei längeren Videos: nur jeden N-ten Frame rendern. FFmpeg dupliziert die Zwischenframes."
        )[1]
        
        # Post-Process (Color-Grading)
        with st.expander("🎨 Color-Grading & Post-Process", expanded=False):
            # Preset-Auswahl
            preset_names = list(COLOR_PRESETS.keys())
            selected_preset = st.selectbox("🎯 Preset", preset_names, key="pp_preset")
            
            # Preset-Werte laden (in Session-State speichern fuer konsistentes Verhalten)
            preset_values = COLOR_PRESETS[selected_preset]
            for key, val in preset_values.items():
                state_key = f"pp_{key}_state"
                if state_key not in st.session_state:
                    st.session_state[state_key] = val
            
            pp_col1, pp_col2 = st.columns(2)
            with pp_col1:
                pp_contrast = st.slider("Kontrast", 0.5, 2.0, st.session_state.get("pp_contrast_state", 1.0), 0.05, key="pp_contrast")
                pp_saturation = st.slider("Sättigung", 0.0, 2.0, st.session_state.get("pp_saturation_state", 1.0), 0.05, key="pp_saturation")
                pp_brightness = st.slider("Helligkeit", -0.5, 0.5, st.session_state.get("pp_brightness_state", 0.0), 0.05, key="pp_brightness")
            with pp_col2:
                pp_warmth = st.slider("Wärme", -1.0, 1.0, st.session_state.get("pp_warmth_state", 0.0), 0.05, key="pp_warmth", help="Positiv = warm/gelb, Negativ = kalt/blau")
                pp_grain = st.slider("Film Grain", 0.0, 1.0, st.session_state.get("pp_film_grain_state", 0.0), 0.05, key="pp_grain")
                beat_sync = st.checkbox("🥁 Quotes auf Beats synchronisieren", value=False, key="beat_sync")
            
            # Slider-Werte in Session-State speichern fuer Preset-Reset
            st.session_state["pp_contrast_state"] = pp_contrast
            st.session_state["pp_saturation_state"] = pp_saturation
            st.session_state["pp_brightness_state"] = pp_brightness
            st.session_state["pp_warmth_state"] = pp_warmth
            st.session_state["pp_film_grain_state"] = pp_grain
        
        # Info-Box
        st.markdown("""
        <div class="info-box">
            <strong>💡 Tipp:</strong><br>
            Nutze zuerst die Vorschau, um zu testen, welcher Visualizer am besten passt!
        </div>
        """, unsafe_allow_html=True)
    
    # Audio-Features global initialisieren (für KI-Modus in col2)
    features = None
    
    # Hauptbereich
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Audio-Datei")
        
        # Audio-Upload
        uploaded_file = st.file_uploader(
            "Wähle eine Audio-Datei",
            type=['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a'],
            help="Unterstützte Formate: MP3, WAV, FLAC, AAC, OGG, M4A"
        )
        
        # Audio-Player
        if uploaded_file:
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
            
            # Audio-Info anzeigen (falls analysiert)
            try:
                # Temporär speichern für Analyse
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}')
                temp_audio.write(uploaded_file.getvalue())
                temp_audio.close()
                
                analyzer = AudioAnalyzer()
                
                # Progress-Callback fuer Live-Status-Updates
                with st.status("⏳ Audio wird analysiert...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(msg, step, total):
                        pct = step / total
                        progress_bar.progress(pct, text=msg)
                        status_text.text(f"Schritt {step} von {total}: {msg}")
                    
                    features = analyzer.analyze(
                        temp_audio.name, 
                        fps=30,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(1.0, text="Fertig!")
                    status.update(label="✅ Analyse abgeschlossen", state="complete", expanded=False)
                
            except Exception as e:
                st.warning(f"Audio-Analyse fehlgeschlagen: {e}")
            
            if features:
                st.markdown(f"""
                <div class="preview-card">
                    <h4>📊 Audio-Analyse</h4>
                    <p>🎵 <strong>Dauer:</strong> {features.duration:.1f} Sekunden</p>
                    <p>⏱️ <strong>Tempo:</strong> {features.tempo:.0f} BPM</p>
                    <p>🎼 <strong>Key:</strong> {features.key or 'Unbekannt'}</p>
                    <p>🎹 <strong>Modus:</strong> {features.mode}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎨 Visualizer")
        
        # Auswahl-Modus: Auto, Manuell oder Preset
        selection_mode = st.radio(
            "Auswahl-Modus",
            ["🤖 Auto-Modus (KI empfiehlt)", "🎙️ Podcast-Genre", "🎨 Manuell auswählen", "📋 Config-Preset"],
            help="Auto-Modus analysiert dein Audio und wählt den besten Visualizer automatisch."
        )
        
        selected_visualizer = None
        config_path = None
        ai_recommendation = None
        
        if selection_mode == "📋 Config-Preset":
            presets = get_config_presets()
            if presets:
                selected_preset = st.selectbox(
                    "Config-Preset",
                    list(presets.keys()),
                    format_func=lambda x: f"{x} ({presets[x]['visual']})"
                )
                selected_visualizer = presets[selected_preset]['visual']
                config_path = presets[selected_preset]['file']
            else:
                st.warning("Keine Presets in config/ gefunden.")
        
        elif selection_mode == "🤖 Auto-Modus (KI empfiehlt)":
            if features is not None:
                matcher = SmartMatcher()
                ai_recommendation = matcher.match(features)
                selected_visualizer = ai_recommendation.visualizer
                
                info = get_visualizer_info(selected_visualizer)
                st.markdown(f"""
                <div class="preview-card" style="border-left: 4px solid {ai_recommendation.colors['primary']};">
                    <h4>🤖 KI-Empfehlung</h4>
                    <p><strong>{info['emoji']} {selected_visualizer.replace('_', ' ').title()}</strong></p>
                    <p style="color: #ccc; font-size: 0.95rem;">{ai_recommendation.reason}</p>
                    <p style="color: #888; font-size: 0.85rem;">✅ Zuversichtlichkeit: {ai_recommendation.confidence*100:.0f}%</p>
                    <p style="color: #888; font-size: 0.85rem;">🎨 Farben: Primary {ai_recommendation.colors['primary']}, Background {ai_recommendation.colors['background']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👆 Lade zuerst eine Audio-Datei hoch für die KI-Empfehlung.")
        
        elif selection_mode == "🎙️ Podcast-Genre":
            podcast_genres = {
                "📰 News / Nachrichten": "config/podcast_news.json",
                "🎤 Interview / Gespräch": "config/podcast_interview.json",
                "📖 Storytelling / Hörbuch": "config/podcast_story.json",
                "🎛️ Mixed / Variety": "config/podcast_mixed.json",
            }
            selected_genre = st.selectbox("Podcast-Genre auswählen", list(podcast_genres.keys()))
            config_path = podcast_genres[selected_genre]
            
            # Lade Preset-Info für Anzeige
            try:
                with open(config_path) as f:
                    preset_data = json.load(f)
                selected_visualizer = preset_data['visual']['type']
                info = get_visualizer_info(selected_visualizer)
                st.markdown(f"""
                <div class="preview-card">
                    <p><strong>{info['emoji']} {selected_visualizer.replace('_', ' ').title()}</strong></p>
                    <p style="color: #ccc; font-size: 0.95rem;">{preset_data.get('description', '')}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Preset konnte nicht geladen werden: {e}")
        
        else:  # Manuell
            available_visualizers = get_available_visualizers()
            
            selected_visualizer = st.selectbox(
                "Visualizer auswählen",
                available_visualizers,
                format_func=lambda x: f"{get_visualizer_info(x)['emoji']} {x.replace('_', ' ').title()}"
            )
            
            info = get_visualizer_info(selected_visualizer)
            st.markdown(f"""
            <div class="preview-card" style="border-left: 4px solid {info['color']};">
                <p style="margin: 0;"><strong>{info['description']}</strong></p>
                <p style="margin: 5px 0 0 0; color: #888; font-size: 0.9rem;">
                    Best for: {info['best_for']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # GPU-Parameter-Tuning
    gpu_viz = None
    gpu_params = {}
    if selected_visualizer:
        gpu_viz = selected_visualizer
        # Alle 13 Visualizer sind GPU-beschleunigt - kein Fallback nötig
        if gpu_viz not in list_visualizers():
            gpu_viz = "spectrum_bars"
            st.warning(f"⚠️ '{selected_visualizer}' nicht gefunden. Fallback auf 'spectrum_bars'.")
        
        prefix = f"gpu_param_{gpu_viz}_"
        
        # KI-Optimierung ausfuehren wenn triggered (MUSS vor den Slidern sein!)
        if st.session_state.get(f"{prefix}_trigger_optimize", False):
            st.session_state[f"{prefix}_trigger_optimize"] = False
            with st.spinner("🤖 KI analysiert Audio und optimiert Parameter..."):
                try:
                    temp_dir = tempfile.mkdtemp()
                    audio_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(audio_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    analyzer = AudioAnalyzer()
                    features = analyzer.analyze(audio_path, fps=30)
                    
                    feature_summary = {
                        'duration': features.duration,
                        'tempo': features.tempo,
                        'mode': features.mode,
                        'rms_mean': float(features.rms.mean()),
                        'onset_mean': float(features.onset.mean()),
                        'spectral_mean': float(features.spectral_centroid.mean()),
                    }
                    
                    viz_cls = get_visualizer(gpu_viz)
                    fallback_params = {k: v[0] for k, v in viz_cls.PARAMS.items()}
                    
                    current_params = {}
                    for k in fallback_params:
                        current_params[k] = st.session_state.get(f"{prefix}{k}", fallback_params[k])
                    
                    user_prompt = st.session_state.get(f"{prefix}user_prompt", "")
                    
                    gemini = GeminiIntegration()
                    optimized = gemini.optimize_visualizer_params(
                        gpu_viz, current_params, feature_summary, user_prompt=user_prompt
                    )
                    
                    for key, value in optimized.items():
                        full_key = f"{prefix}{key}"
                        if full_key in st.session_state:
                            del st.session_state[full_key]
                        st.session_state[full_key] = value
                    
                    st.success("✅ Parameter optimiert! Aktualisiere...")
                    st.rerun()
                except Exception as e:
                    st.error(f"KI-Optimierung fehlgeschlagen: {e}")
        
        with st.expander("🔧 Parameter-Tuning", expanded=True):
            col_ai1, col_ai2 = st.columns([3, 1])
            with col_ai2:
                if st.button("🤖 KI-Optimieren", key=f"ai_optimize_{gpu_viz}"):
                    st.session_state[f"{prefix}_trigger_optimize"] = True
                    st.rerun()
            with col_ai1:
                st.caption("🎚️ Ziehe die Regler oder lass die KI die perfekten Werte finden.")
            
            user_prompt = st.text_input(
                "💬 Dein Wunsch (optional)",
                placeholder="z.B. Mach es schlichter und ruhiger...",
                key=f"{prefix}user_prompt",
                help="Beschreibe hier, wie die Visualisierung aussehen soll. Die KI beruecksichtigt das bei der Optimierung."
            )
            
            gpu_params = render_gpu_parameter_sliders(gpu_viz)
    
    # Hintergrundbild-Bereich
    st.markdown("---")
    st.markdown("### 🖼️ Hintergrundbild")
    
    bg_image_path = None
    bg_blur = 0.0
    bg_vignette = 0.0
    bg_opacity = 0.3
    
    if uploaded_file:
        bg_col1, bg_col2 = st.columns([1, 1])
        
        with bg_col1:
            uploaded_bg = st.file_uploader(
                "Hintergrundbild hochladen (optional)",
                type=['png', 'jpg', 'jpeg', 'webp'],
                help="Ein Bild, das als Hintergrund fuer die Visualisierung dient."
            )
            
            if uploaded_bg:
                # Temporaer speichern
                bg_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_bg.name.split(".")[-1]}')
                bg_temp.write(uploaded_bg.getvalue())
                bg_temp.close()
                bg_image_path = bg_temp.name
                
                st.image(bg_image_path, caption="Hintergrundbild-Vorschau")
        
        with bg_col2:
            if bg_image_path:
                bg_blur = st.slider("🔮 Blur (Weichzeichnung)", 0.0, 20.0, 0.0, 0.5,
                                    help="Je hoeher, desto weicher das Hintergrundbild.")
                bg_vignette = st.slider("🌑 Vignette (Randabdunkelung)", 0.0, 1.0, 0.0, 0.05,
                                        help="Dunkelt die Raender ab fuer mehr Fokus in der Mitte.")
                bg_opacity = st.slider("🎨 Hintergrund-Deckkraft", 0.0, 1.0, 0.3, 0.05,
                                       help="Wie stark das Bild durchscheint (0 = nur Visualizer, 1 = nur Bild).")
            else:
                st.info("👆 Lade ein Bild hoch, um Blur und Vignette zu aktivieren.")
    else:
        st.info("👆 Lade zuerst eine Audio-Datei hoch.")
    
    # KI-Bild-Prompt Generator
    st.markdown("---")
    st.markdown("### 🎨 KI-Bild-Prompt Generator")
    
    if uploaded_file and features is not None:
        prompt_col1, prompt_col2 = st.columns([1, 1])
        
        with prompt_col1:
            if st.button("✨ Bild-Prompt generieren"):
                with st.spinner("KI erstellt Bild-Prompt basierend auf Audio..."):
                    try:
                        feature_summary = {
                            'duration': features.duration,
                            'tempo': features.tempo,
                            'mode': features.mode,
                            'rms_mean': float(features.rms.mean()),
                            'onset_mean': float(features.onset.mean()),
                            'spectral_mean': float(features.spectral_centroid.mean()),
                        }
                        
                        gemini = GeminiIntegration()
                        generated_prompt = gemini.generate_background_prompt(feature_summary)
                        st.session_state[f"bg_prompt_{uploaded_file.name}"] = generated_prompt
                        st.success("Prompt generiert!")
                    except Exception as e:
                        st.error(f"Prompt-Generierung fehlgeschlagen: {e}")
        
        with prompt_col2:
            prompt_key = f"bg_prompt_{uploaded_file.name}"
            if prompt_key in st.session_state:
                st.text_area(
                    "Generierter Prompt (fuer Midjourney / DALL-E / Stable Diffusion)",
                    value=st.session_state[prompt_key],
                    height=100,
                    key=prompt_key + "_display"
                )
                st.caption("💡 Kopiere diesen Prompt in dein bevorzugtes Bildgenerierungs-Tool.")
            else:
                st.caption("Klicke auf 'Bild-Prompt generieren', um einen passenden Prompt zu erhalten.")
    else:
        st.info("👆 Lade eine Audio-Datei hoch, um einen Bild-Prompt generieren zu lassen.")
    
    # GPU-Render-Bereich
    st.markdown("---")
    
    if uploaded_file and gpu_viz:
        col_render1, col_render2, col_render3 = st.columns([1, 2, 1])
        
        with col_render2:
            # Auto-Preview Toggle + Live-Preview
            auto_preview = st.toggle("⚡ Auto-Preview", value=False, key="auto_preview_toggle",
                                     help="Automatisch neu rendern wenn Parameter sich aendern")
            
            # Parameter-Hash fuer Auto-Preview Tracking
            preview_params_hash = hash((gpu_viz, str(sorted(gpu_params.items()) if gpu_params else []), 
                                        bg_opacity, bg_vignette, bg_blur,
                                        pp_contrast, pp_saturation, pp_brightness, pp_warmth, pp_grain))
            
            preview_needs_update = False
            if auto_preview:
                last_hash = st.session_state.get("last_preview_hash", None)
                if last_hash != preview_params_hash:
                    preview_needs_update = True
                    st.session_state["last_preview_hash"] = preview_params_hash
            
            # Preview Image Container
            preview_container = st.empty()
            
            # Manuelle Preview oder Auto-Preview Update
            if st.button("👁️ Live-Preview (GPU)") or preview_needs_update:
                with st.spinner("Rendere GPU-Frame...") if not preview_needs_update else st.empty():
                    try:
                        temp_dir = tempfile.mkdtemp()
                        audio_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(audio_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())
                        
                        pp_cfg = None
                        if pp_contrast != 1.0 or pp_saturation != 1.0 or pp_brightness != 0.0 or pp_warmth != 0.0 or pp_grain > 0.0:
                            pp_cfg = {
                                "contrast": pp_contrast,
                                "saturation": pp_saturation,
                                "brightness": pp_brightness,
                                "warmth": pp_warmth,
                                "film_grain": pp_grain,
                            }
                        
                        preview_img = render_gpu_preview(
                            audio_path=audio_path,
                            visualizer_type=gpu_viz,
                            params=gpu_params,
                            width=480,
                            height=270,
                            background_image=bg_image_path,
                            background_blur=bg_blur,
                            background_vignette=bg_vignette,
                            background_opacity=bg_opacity,
                            postprocess=pp_cfg,
                        )
                        
                        if preview_img is not None:
                            st.session_state["last_preview_img"] = preview_img
                            preview_container.image(preview_img, caption=f"👁️ Live-Preview: {gpu_viz}")
                        else:
                            preview_container.error("GPU-Preview konnte nicht gerendert werden.")
                    except Exception as e:
                        preview_container.error(f"GPU-Live-Preview Fehler: {e}")
            
            # Gespeichertes Preview anzeigen falls vorhanden
            if not preview_needs_update and "last_preview_img" in st.session_state:
                preview_container.image(st.session_state["last_preview_img"], caption=f"👁️ Live-Preview: {gpu_viz}")
            
            # Video Export Button
            button_text = "🎬 Vorschau exportieren" if mode == "Schnell (Vorschau)" else "🎬 Video exportieren"
            
            if st.button(button_text, key="gpu_main_render_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    temp_dir = tempfile.mkdtemp()
                    audio_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(audio_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    output_filename = f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    output_path = os.path.join(temp_dir, output_filename)
                    
                    w, h = map(int, resolution.split('x'))
                    
                    status_text.text("Starte GPU-Rendering...")
                    
                    renderer = GPUBatchRenderer(width=w, height=h, fps=fps)
                    
                    preview_mode = (mode == "Schnell (Vorschau)")
                    
                    # Quote-Config aus Session-State zusammenbauen
                    qck = f"quotes_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
                    quote_font_path = st.session_state.get(f"quote_font_path_{qck}", None)
                    
                    def hex_to_rgb(hex_str):
                        hex_str = hex_str.lstrip('#')
                        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
                    
                    quote_config = QuoteOverlayConfig(
                        enabled=True,
                        font_size=st.session_state.get(f"quote_font_size_{qck}", 36),
                        font_color=hex_to_rgb(st.session_state.get(f"quote_font_color_{qck}", "#FFFFFF")),
                        box_color=hex_to_rgb(st.session_state.get(f"quote_box_color_{qck}", "#000000")) + (160,),
                        position=st.session_state.get(f"quote_pos_{qck}", "bottom"),
                        text_align=st.session_state.get(f"quote_align_{qck}", "center"),
                        display_duration=st.session_state.get(f"quote_duration_{qck}", 8.0),
                        font_path=quote_font_path,
                        auto_scale_font=st.session_state.get(f"quote_autoscale_{qck}", True),
                        min_font_size=st.session_state.get(f"quote_min_font_{qck}", 16),
                        slide_animation=st.session_state.get(f"quote_slide_{qck}", "none"),
                        slide_distance=st.session_state.get(f"quote_slide_dist_{qck}", 100),
                        slide_out_animation=st.session_state.get(f"quote_slide_out_{qck}", "none"),
                        slide_out_distance=st.session_state.get(f"quote_slide_out_dist_{qck}", 100),
                        scale_in=st.session_state.get(f"quote_scale_in_{qck}", False),
                        typewriter=st.session_state.get(f"quote_typewriter_{qck}", False),
                        typewriter_speed=st.session_state.get(f"quote_tw_speed_{qck}", 15.0),
                        typewriter_mode=st.session_state.get(f"quote_tw_mode_{qck}", "char"),
                        glow_pulse=st.session_state.get(f"quote_glow_pulse_{qck}", False),
                        glow_pulse_intensity=st.session_state.get(f"quote_glow_pulse_int_{qck}", 0.5),
                    )
                    
                    render_quotes_key = f"render_quotes_{qck}"
                    edited_quotes_raw = st.session_state.get(render_quotes_key, [])
                    quotes_for_render = [Quote(**q) for q in edited_quotes_raw] if edited_quotes_raw else None
                    
                    # Post-Process Parameter zusammenbauen
                    postprocess_cfg = None
                    if pp_contrast != 1.0 or pp_saturation != 1.0 or pp_brightness != 0.0 or pp_warmth != 0.0 or pp_grain > 0.0:
                        postprocess_cfg = {
                            "contrast": pp_contrast,
                            "saturation": pp_saturation,
                            "brightness": pp_brightness,
                            "warmth": pp_warmth,
                            "film_grain": pp_grain,
                        }
                    
                    renderer.render(
                        audio_path=audio_path,
                        visualizer_type=gpu_viz,
                        output_path=output_path,
                        params=gpu_params,
                        background_image=bg_image_path,
                        background_blur=bg_blur,
                        background_vignette=bg_vignette,
                        background_opacity=bg_opacity,
                        preview_mode=preview_mode,
                        preview_duration=5.0,
                        quotes=quotes_for_render,
                        quote_config=quote_config,
                        codec=codec,
                        quality=quality,
                        postprocess=postprocess_cfg,
                        sync_quotes_to_beats=beat_sync,
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("Fertig!")
                    
                    if os.path.exists(output_path):
                        st.markdown("""
                        <div class="success-box">
                            <strong>✅ Rendering erfolgreich!</strong><br>
                            Dein Video ist bereit zum Download.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                            st.video(video_bytes)
                        
                        st.download_button(
                            label="📥 Video herunterladen",
                            data=video_bytes,
                            file_name=output_filename,
                            mime="video/mp4",
                        )
                        
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    else:
                        st.error("Rendering fehlgeschlagen: Output-Datei nicht gefunden.")
                        
                except Exception as e:
                    st.error("❌ GPU-Rendering fehlgeschlagen")
                    st.info(f"**Fehler:** {str(e)}")
                    st.info("💡 Tipp: Stelle sicher, dass FFmpeg korrekt installiert ist und eine GPU verfügbar.")
    else:
        st.info("👆 Lade zuerst eine Audio-Datei hoch und wähle einen Visualizer aus.")
    
    # --- Key-Zitate Bereich (Gemini) ---
    st.markdown("---")
    st.markdown("### 💬 Key-Zitate (Gemini KI)")
    
    if uploaded_file:
        quotes_cache_key = f"quotes_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
        
        if st.button("🔮 Key-Zitate extrahieren"):
            with st.spinner("Gemini analysiert dein Audio..."):
                try:
                    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}')
                    temp_audio.write(uploaded_file.getvalue())
                    temp_audio.close()
                    
                    gemini = GeminiIntegration()
                    quotes = gemini.extract_quotes(temp_audio.name, max_quotes=5)
                    
                    # Zeitstempel validieren: clamp auf Audio-Dauer
                    if features is not None:
                        audio_duration = features.duration
                        for q in quotes:
                            q.start_time = max(0.0, min(q.start_time, audio_duration - 1.0))
                            q.end_time = max(q.start_time + 1.0, min(q.end_time, audio_duration))
                    
                    st.session_state[quotes_cache_key] = quotes
                    
                except Exception as e:
                    st.error(f"Zitat-Extraktion fehlgeschlagen: {e}")
                    st.info("💡 Tipp: Stelle sicher, dass GEMINI_API_KEY als Umgebungsvariable gesetzt ist oder in .env hinterlegt ist.")
        
        # Zitat-Einstellungen (immer sichtbar wenn Audio geladen)
        with st.expander("⚙️ Zitat-Einstellungen", expanded=False):
            qe_col1, qe_col2 = st.columns(2)
            with qe_col1:
                quote_font_size = st.slider("Schriftgröße", 12, 72, 36, key=f"quote_font_size_{quotes_cache_key}")
                quote_display_duration = st.slider("Anzeigedauer (Sekunden)", 2.0, 20.0, 8.0, 0.5, key=f"quote_duration_{quotes_cache_key}")
                quote_position = st.selectbox("Position", ["bottom", "center", "top"], key=f"quote_pos_{quotes_cache_key}")
            with qe_col2:
                quote_text_align = st.selectbox("Text-Ausrichtung", ["center", "left", "right"], key=f"quote_align_{quotes_cache_key}")
                quote_box_color = st.color_picker("Box-Farbe", "#000000", key=f"quote_box_color_{quotes_cache_key}")
                quote_font_color = st.color_picker("Schrift-Farbe", "#FFFFFF", key=f"quote_font_color_{quotes_cache_key}")
            
            # Auto-Scale & Animationen
            st.markdown("#### ✨ Animationen & Skalierung")
            anim_col1, anim_col2, anim_col3 = st.columns(3)
            with anim_col1:
                quote_auto_scale = st.checkbox("Auto-Skalierung", value=True, key=f"quote_autoscale_{quotes_cache_key}")
                quote_min_font = st.slider("Min. Schriftgröße", 8, 36, 16, key=f"quote_min_font_{quotes_cache_key}")
                quote_scale_in = st.checkbox("Scale-In", value=False, key=f"quote_scale_in_{quotes_cache_key}")
            with anim_col2:
                quote_slide = st.selectbox("Slide-In", ["none", "up", "down", "left", "right"], key=f"quote_slide_{quotes_cache_key}")
                quote_slide_dist = st.slider("Slide-In Distanz (px)", 0, 300, 100, key=f"quote_slide_dist_{quotes_cache_key}")
                quote_slide_out = st.selectbox("Slide-Out", ["none", "up", "down", "left", "right"], key=f"quote_slide_out_{quotes_cache_key}")
                quote_slide_out_dist = st.slider("Slide-Out Distanz (px)", 0, 300, 100, key=f"quote_slide_out_dist_{quotes_cache_key}")
                quote_glow_pulse = st.checkbox("Glow-Pulse", value=False, key=f"quote_glow_pulse_{quotes_cache_key}")
                quote_glow_pulse_int = st.slider("Pulse-Stärke", 0.0, 1.0, 0.5, key=f"quote_glow_pulse_int_{quotes_cache_key}")
            with anim_col3:
                quote_typewriter = st.checkbox("Typewriter-Effekt", value=False, key=f"quote_typewriter_{quotes_cache_key}")
                quote_tw_mode = st.selectbox("Typewriter-Modus", ["char", "word"], key=f"quote_tw_mode_{quotes_cache_key}")
                quote_tw_speed = st.slider("Typewriter-Geschw.", 5.0, 50.0, 15.0, 1.0, key=f"quote_tw_speed_{quotes_cache_key}")
            
            # Schriftart-Upload
            uploaded_font = st.file_uploader("🔤 Eigene Schriftart (.ttf)", type=['ttf'], key=f"quote_font_{quotes_cache_key}")
            quote_font_path = None
            if uploaded_font:
                font_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.ttf')
                font_temp.write(uploaded_font.getvalue())
                font_temp.close()
                quote_font_path = font_temp.name
                st.session_state[f"quote_font_path_{quotes_cache_key}"] = quote_font_path
                st.success(f"✅ Schriftart geladen: {uploaded_font.name}")
            else:
                # Pruefe ob bereits eine Schriftart gespeichert ist
                quote_font_path = st.session_state.get(f"quote_font_path_{quotes_cache_key}", None)
        
        if quotes_cache_key in st.session_state and st.session_state[quotes_cache_key]:
            quotes = st.session_state[quotes_cache_key]
            st.success(f"✅ {len(quotes)} Key-Zitate gefunden!")
            
            selected_key = f"selected_{quotes_cache_key}"
            if selected_key not in st.session_state:
                st.session_state[selected_key] = [True] * len(quotes)
            
            edited_quotes = []
            
            for i, quote in enumerate(quotes):
                start_min = int(quote.start_time // 60)
                start_sec = int(quote.start_time % 60)
                
                col1, col2 = st.columns([0.08, 0.92])
                with col1:
                    enabled = st.checkbox(
                        "Aktiv",
                        value=st.session_state[selected_key][i],
                        key=f"quote_chk_{i}_{quotes_cache_key}"
                    )
                    st.session_state[selected_key][i] = enabled
                
                with col2:
                    edited_text = st.text_input(
                        f"Zitat {i+1} ({start_min}:{start_sec:02d})",
                        value=quote.text,
                        key=f"quote_txt_{i}_{quotes_cache_key}"
                    )
                    
                    if enabled:
                        edited_quotes.append({
                            'text': edited_text,
                            'start_time': quote.start_time,
                            'end_time': quote.end_time,
                            'confidence': quote.confidence
                        })
                    
                    st.caption(f"⏱️ {start_min}:{start_sec:02d} | Konfidenz: {quote.confidence*100:.0f}%")
            
            render_quotes_key = f"render_quotes_{quotes_cache_key}"
            st.session_state[render_quotes_key] = edited_quotes
            
            if edited_quotes:
                st.info(f"📌 {len(edited_quotes)} Zitate sind aktiviert und werden beim Rendering verwendet.")
            else:
                st.warning("⚠️ Keine Zitate aktiviert. Im Video werden keine Text-Overlays angezeigt.")
        else:
            st.info("👆 Lade zuerst eine Audio-Datei hoch, um Key-Zitate zu extrahieren.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Made with ❤️ | Audio Visualizer Pro</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
