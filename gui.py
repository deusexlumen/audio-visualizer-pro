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
    initial_sidebar_state="collapsed"
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


LOOKS = {
    "podcast_clean": {
        "name": "🎙️ Podcast Clean",
        "description": "Dunkel, dezent, große lesbare Quotes. Perfekt für News und Interviews.",
        "visualizer": "voice_flow",
        "params": {"flow_intensity": 0.5, "smoothness": 0.8},
        "colors": {"primary": "#667EEA", "secondary": "#764BA2", "background": "#1A1A2E"},
        "postprocess": {"contrast": 1.05, "saturation": 0.8, "brightness": 0.0, "warmth": 0.1, "film_grain": 0.05},
        "quotes": {
            "font_size": 56, "box_color": "#1a1a2e", "font_color": "#FFFFFF",
            "box_padding": 36, "box_radius": 20, "box_margin_bottom": 100,
            "max_width_ratio": 0.7, "fade_duration": 0.8, "line_spacing": 1.5,
            "max_font_size": 72, "max_chars_per_line": 40,
            "position": "bottom", "text_align": "center",
            "display_duration": 8.0, "auto_scale_font": True,
            "slide_animation": "none", "scale_in": False, "typewriter": False, "glow_pulse": False,
        },
        "background": {"opacity": 0.0, "blur": 0.0, "vignette": 0.0},
    },
    "podcast_cinematic": {
        "name": "🎬 Podcast Cinematic",
        "description": "Warm, Film-Grain, Vignette. Für Storytelling und Hörbücher.",
        "visualizer": "typographic",
        "params": {"text_size": 56, "animation_speed": 0.15, "bar_width": 4, "bar_spacing": 2},
        "colors": {"primary": "#E8A87C", "secondary": "#C38D9E", "background": "#0F0F1A"},
        "postprocess": {"contrast": 1.15, "saturation": 0.9, "brightness": -0.02, "warmth": 0.35, "film_grain": 0.2},
        "quotes": {
            "font_size": 52, "box_color": "#1a1a2e", "font_color": "#F5F5F5",
            "box_padding": 32, "box_radius": 16, "box_margin_bottom": 120,
            "max_width_ratio": 0.75, "fade_duration": 1.0, "line_spacing": 1.4,
            "max_font_size": 64, "max_chars_per_line": 38,
            "position": "bottom", "text_align": "center",
            "display_duration": 10.0, "auto_scale_font": True,
            "slide_animation": "none", "scale_in": False, "typewriter": True, "glow_pulse": False,
        },
        "background": {"opacity": 0.0, "blur": 0.0, "vignette": 0.0},
    },
    "music_energy": {
        "name": "🎵 Musik Energy",
        "description": "Dynamisch, bunt, schnelle Visuals. Für EDM, Pop und Rock.",
        "visualizer": "spectrum_bars",
        "params": {"bar_count": 64, "smoothing": 0.3, "bar_width": 3, "glow_intensity": 0.6},
        "colors": {"primary": "#FF0055", "secondary": "#00CCFF", "background": "#0A0A0A"},
        "postprocess": {"contrast": 1.1, "saturation": 1.1, "brightness": -0.02, "warmth": 0.05, "film_grain": 0.05},
        "quotes": {
            "font_size": 42, "box_color": "#0d0d1a", "font_color": "#FFFFFF",
            "box_padding": 24, "box_radius": 12, "box_margin_bottom": 80,
            "max_width_ratio": 0.8, "fade_duration": 0.5, "line_spacing": 1.25,
            "max_font_size": 56, "max_chars_per_line": 45,
            "position": "bottom", "text_align": "center",
            "display_duration": 6.0, "auto_scale_font": True,
            "slide_animation": "up", "scale_in": True, "typewriter": False, "glow_pulse": True,
        },
        "background": {"opacity": 0.0, "blur": 0.0, "vignette": 0.0},
    },
    "music_chill": {
        "name": "🌊 Musik Chill",
        "description": "Sanft, fließend, entspannt. Für Ambient, Folk und Indie.",
        "visualizer": "liquid_blobs",
        "params": {"blob_count": 5, "fluidity": 0.6},
        "colors": {"primary": "#4ECDC4", "secondary": "#96CEB4", "background": "#1A1A3E"},
        "postprocess": {"contrast": 1.05, "saturation": 0.85, "brightness": 0.02, "warmth": 0.2, "film_grain": 0.1},
        "quotes": {
            "font_size": 48, "box_color": "#1a1a2e", "font_color": "#E8E8E8",
            "box_padding": 28, "box_radius": 18, "box_margin_bottom": 100,
            "max_width_ratio": 0.75, "fade_duration": 0.7, "line_spacing": 1.35,
            "max_font_size": 64, "max_chars_per_line": 42,
            "position": "bottom", "text_align": "center",
            "display_duration": 8.0, "auto_scale_font": True,
            "slide_animation": "none", "scale_in": False, "typewriter": False, "glow_pulse": False,
        },
        "background": {"opacity": 0.0, "blur": 0.0, "vignette": 0.0},
    },
    "meditation": {
        "name": "🧘 Meditation",
        "description": "Langsam, sanft, zentriert. Für Yoga und Meditation.",
        "visualizer": "sacred_mandala",
        "params": {"rotation_speed": 0.15, "layer_count": 5},
        "colors": {"primary": "#9D4EDD", "secondary": "#C77DFF", "background": "#0F0F1E"},
        "postprocess": {"contrast": 1.0, "saturation": 0.75, "brightness": 0.0, "warmth": 0.15, "film_grain": 0.15},
        "quotes": {
            "font_size": 64, "box_color": "#0f0f1e", "font_color": "#F0F0F0",
            "box_padding": 40, "box_radius": 24, "box_margin_bottom": 150,
            "max_width_ratio": 0.65, "fade_duration": 1.2, "line_spacing": 1.6,
            "max_font_size": 80, "max_chars_per_line": 35,
            "position": "center", "text_align": "center",
            "display_duration": 12.0, "auto_scale_font": True,
            "slide_animation": "none", "scale_in": False, "typewriter": False, "glow_pulse": False,
        },
        "background": {"opacity": 0.0, "blur": 0.0, "vignette": 0.0},
    },
}


def get_available_visualizers():
    """Lädt alle verfügbaren Visualizer."""
    return list_visualizers()


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


def hex_to_rgb(hex_str):
    """Konvertiert Hex-Farbe zu RGB-Tuple. Fallback auf Weiss bei Fehler."""
    try:
        hex_str = str(hex_str).lstrip('#')
        if len(hex_str) == 3:
            hex_str = ''.join([c * 2 for c in hex_str])
        if len(hex_str) != 6:
            return (255, 255, 255)
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (255, 255, 255)


def apply_look(look_key, uploaded_file):
    """Wendet einen Look auf den Session-State an."""
    look = LOOKS[look_key]
    st.session_state["selected_look"] = look_key
    st.session_state["selected_visualizer"] = look["visualizer"]
    st.session_state["ai_color_primary"] = look["colors"]["primary"]
    st.session_state["ai_color_secondary"] = look["colors"]["secondary"]
    st.session_state["ai_color_background"] = look["colors"]["background"]

    viz = look["visualizer"]
    prefix = f"gpu_param_{viz}_"
    for param_name, value in look.get("params", {}).items():
        st.session_state[f"{prefix}{param_name}"] = value

    for k, v in look["postprocess"].items():
        st.session_state[f"pp_{k}_state"] = v
        if k == "film_grain":
            st.session_state["pp_grain"] = v
        else:
            st.session_state[f"pp_{k}"] = v

    if uploaded_file:
        qck = f"quotes_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
        quote_map = {
            "font_size": f"quote_font_size_{qck}",
            "box_color": f"quote_box_color_{qck}",
            "font_color": f"quote_font_color_{qck}",
            "box_padding": f"quote_box_padding_{qck}",
            "box_radius": f"quote_box_radius_{qck}",
            "box_margin_bottom": f"quote_box_margin_{qck}",
            "max_width_ratio": f"quote_max_width_{qck}",
            "fade_duration": f"quote_fade_duration_{qck}",
            "line_spacing": f"quote_line_spacing_{qck}",
            "max_font_size": f"quote_max_font_{qck}",
            "max_chars_per_line": f"quote_max_chars_{qck}",
            "position": f"quote_pos_{qck}",
            "text_align": f"quote_align_{qck}",
            "display_duration": f"quote_duration_{qck}",
            "auto_scale_font": f"quote_autoscale_{qck}",
            "slide_animation": f"quote_slide_{qck}",
            "scale_in": f"quote_scale_in_{qck}",
            "typewriter": f"quote_typewriter_{qck}",
            "glow_pulse": f"quote_glow_pulse_{qck}",
        }
        for k, v in look["quotes"].items():
            if k in quote_map:
                st.session_state[quote_map[k]] = v

        defaults = {
            f"quote_slide_dist_{qck}": 100,
            f"quote_slide_out_{qck}": "none",
            f"quote_slide_out_dist_{qck}": 100,
            f"quote_min_font_{qck}": 16,
            f"quote_tw_speed_{qck}": 15.0,
            f"quote_tw_mode_{qck}": "char",
            f"quote_glow_pulse_int_{qck}": 0.5,
        }
        for k, v in defaults.items():
            st.session_state[k] = v

    for k, v in look.get("background", {}).items():
        st.session_state[f"bg_{k}"] = v

    for wkey in ["viz_color_primary", "viz_color_secondary", "viz_color_background", "selected_visualizer_dropdown"]:
        if wkey in st.session_state:
            del st.session_state[wkey]


# ==================== MAIN APP ====================

def main():
    # Session-State Defaults
    if "selected_look" not in st.session_state:
        st.session_state["selected_look"] = None
    if "last_uploaded_file" not in st.session_state:
        st.session_state["last_uploaded_file"] = None

    # Header
    st.markdown("""
    <div style="text-align: left; padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size: 2rem; margin-bottom: 0;">🎵 Audio Visualizer Pro</h1>
        <p style="font-size: 1.1rem; color: #888; margin-top: 0.3rem;">
            Audio hochladen. Look wählen. Video exportieren.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # System-Check
    issues = check_system_requirements()
    if issues:
        for issue in issues:
            st.warning(issue)

    # Haupt-Layout: 2 Spalten
    left_col, right_col = st.columns([0.4, 0.6])

    # Variablen mit Defaults
    features = None
    uploaded_file = None
    gpu_viz = None
    gpu_params = {}
    pp_contrast = 1.0
    pp_saturation = 1.0
    pp_brightness = 0.0
    pp_warmth = 0.0
    pp_grain = 0.0
    beat_sync = False
    bg_image_path = None
    bg_blur = 0.0
    bg_vignette = 0.0
    bg_opacity = 0.3
    preview_clicked = False
    render_clicked = False
    resolution = "1920x1080"
    fps = 60
    codec = "h264"
    quality = "high"
    turbo_mode = False
    frame_skip = 1

    with left_col:
        st.markdown("### 📁 Audio-Upload")
        uploaded_file = st.file_uploader(
            "Wähle eine Audio-Datei",
            type=['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a'],
            help="Unterstützte Formate: MP3, WAV, FLAC, AAC, OGG, M4A"
        )

        if uploaded_file:
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')

            # Eindeutige ID fuer diese Datei
            file_id = f"{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
            temp_audio_key = f"temp_audio_{file_id}"
            features_key = f"features_{file_id}"

            # Temporaere Datei nur einmal erstellen und im Session-State cachen
            if temp_audio_key not in st.session_state:
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}')
                temp_audio.write(uploaded_file.getvalue())
                temp_audio.close()
                st.session_state[temp_audio_key] = temp_audio.name

            temp_audio_path = st.session_state[temp_audio_key]

            # Features nur einmal analysieren und cachen
            if features_key not in st.session_state:
                analyzer = AudioAnalyzer()

                with st.status("⏳ Audio wird analysiert...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(msg, step, total):
                        pct = min(1.0, step / total)
                        progress_bar.progress(pct, text=msg)
                        status_text.text(f"Schritt {step} von {total}: {msg}")

                    features = analyzer.analyze(
                        temp_audio_path,
                        fps=30,
                        progress_callback=update_progress
                    )

                    progress_bar.progress(1.0, text="Fertig!")
                    status.update(label="✅ Analyse abgeschlossen", state="complete", expanded=False)

                st.session_state[features_key] = features
            else:
                features = st.session_state[features_key]

            if features:
                st.markdown(f"""
                <div class="preview-card" style="padding: 12px;">
                    <p style="margin: 2px 0;">🎵 <strong>Dauer:</strong> {features.duration:.1f}s</p>
                    <p style="margin: 2px 0;">⏱️ <strong>Tempo:</strong> {features.tempo:.0f} BPM</p>
                    <p style="margin: 2px 0;">🎼 <strong>Key:</strong> {features.key or 'Unbekannt'}</p>
                    <p style="margin: 2px 0;">🎹 <strong>Modus:</strong> {features.mode}</p>
                </div>
                """, unsafe_allow_html=True)

                auto_look_key = f"auto_look_{file_id}"

                if auto_look_key not in st.session_state:
                    st.session_state[auto_look_key] = True

                    if features.mode == 'speech':
                        recommended_look = "podcast_clean"
                    elif features.mode == 'music' and features.tempo > 110:
                        recommended_look = "music_energy"
                    elif features.mode == 'music' and features.tempo <= 110:
                        recommended_look = "music_chill"
                    else:
                        recommended_look = "podcast_clean"

                    apply_look(recommended_look, uploaded_file)
                    st.toast(f"🤖 Auto-Look: '{LOOKS[recommended_look]['name']}' ausgewählt", icon="✨")
                    st.rerun()

        st.markdown("### ✨ Look wählen")

        if st.button("🤖 Auto", key="look_auto", width='stretch', help="Wählt den passenden Look basierend auf deinem Audio"):
            if features:
                if features.mode == 'speech':
                    recommended_look = "podcast_clean"
                elif features.mode == 'music' and features.tempo > 110:
                    recommended_look = "music_energy"
                elif features.mode == 'music' and features.tempo <= 110:
                    recommended_look = "music_chill"
                else:
                    recommended_look = "podcast_clean"
                apply_look(recommended_look, uploaded_file)
                st.rerun()
            else:
                st.warning("Lade zuerst eine Audio-Datei hoch.")

        look_cols = st.columns(2)
        look_keys = list(LOOKS.keys())
        for idx, look_key in enumerate(look_keys):
            with look_cols[idx % 2]:
                look = LOOKS[look_key]
                is_active = st.session_state.get("selected_look") == look_key
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    f"{look['name']}\n{look['description']}",
                    key=f"look_btn_{look_key}",
                    type=btn_type,
                    width='stretch'
                ):
                    apply_look(look_key, uploaded_file)
                    st.rerun()

        selected_visualizer = st.session_state.get("selected_visualizer")
        gpu_viz = selected_visualizer
        if gpu_viz and gpu_viz not in list_visualizers():
            gpu_viz = "spectrum_bars"

        with st.expander("⚙️ Feintunen", expanded=False):
            if gpu_viz:
                available_visualizers = get_available_visualizers()
                selected_visualizer = st.selectbox(
                    "Visualizer",
                    available_visualizers,
                    index=available_visualizers.index(gpu_viz) if gpu_viz in available_visualizers else 0,
                    format_func=lambda x: f"{get_visualizer_info(x)['emoji']} {x.replace('_', ' ').title()}",
                    key="selected_visualizer_dropdown"
                )
                st.session_state["selected_visualizer"] = selected_visualizer
                gpu_viz = selected_visualizer

                st.markdown("#### 🔧 Parameter")
                gpu_params = render_gpu_parameter_sliders(gpu_viz)

                st.markdown("#### 🎨 Farben")
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    viz_primary = st.color_picker("Primary", st.session_state.get("ai_color_primary", "#FF0055"), key="viz_color_primary")
                    st.session_state["ai_color_primary"] = viz_primary
                with col_f2:
                    viz_secondary = st.color_picker("Secondary", st.session_state.get("ai_color_secondary", "#00CCFF"), key="viz_color_secondary")
                    st.session_state["ai_color_secondary"] = viz_secondary
                with col_f3:
                    viz_background = st.color_picker("Background", st.session_state.get("ai_color_background", "#0A0A0A"), key="viz_color_background")
                    st.session_state["ai_color_background"] = viz_background

                st.markdown("#### 🎨 Color-Grading")
                pp_col1, pp_col2 = st.columns(2)
                with pp_col1:
                    pp_contrast = st.slider("Kontrast", 0.5, 2.0, st.session_state.get("pp_contrast_state", 1.0), 0.05, key="pp_contrast")
                    pp_saturation = st.slider("Sättigung", 0.0, 2.0, st.session_state.get("pp_saturation_state", 1.0), 0.05, key="pp_saturation")
                    pp_brightness = st.slider("Helligkeit", -0.5, 0.5, st.session_state.get("pp_brightness_state", 0.0), 0.05, key="pp_brightness")
                with pp_col2:
                    pp_warmth = st.slider("Wärme", -1.0, 1.0, st.session_state.get("pp_warmth_state", 0.0), 0.05, key="pp_warmth", help="Positiv = warm/gelb, Negativ = kalt/blau")
                    pp_grain = st.slider("Film Grain", 0.0, 1.0, st.session_state.get("pp_film_grain_state", 0.0), 0.05, key="pp_grain")
                    beat_sync = st.checkbox("🥁 Quotes auf Beats synchronisieren", value=False, key="beat_sync")

                st.session_state["pp_contrast_state"] = pp_contrast
                st.session_state["pp_saturation_state"] = pp_saturation
                st.session_state["pp_brightness_state"] = pp_brightness
                st.session_state["pp_warmth_state"] = pp_warmth
                st.session_state["pp_film_grain_state"] = pp_grain

                st.markdown("---")
                st.markdown("##### 🤖 KI-Optimierung")
                ai_col1, ai_col2 = st.columns([3, 1])
                with ai_col2:
                    prefix = f"gpu_param_{gpu_viz}_"
                    if st.button("🤖 Optimieren", key=f"ai_optimize_{gpu_viz}"):
                        st.session_state[f"{prefix}_trigger_optimize"] = True
                        st.rerun()
                with ai_col1:
                    st.caption("Die KI passt alle Einstellungen basierend auf deinem Audio an.")

                st.text_input(
                    "💬 Dein Wunsch (optional)",
                    placeholder="z.B. Mach es schlichter und ruhiger...",
                    key="ai_user_prompt",
                    help="Beschreibe hier, wie die Visualisierung aussehen soll."
                )

                if st.session_state.get(f"{prefix}_trigger_optimize", False):
                    st.session_state[f"{prefix}_trigger_optimize"] = False
                    with st.spinner("🤖 KI analysiert Audio und optimiert ALLE Einstellungen..."):
                        try:
                            # Gecachten Audio-Pfad verwenden
                            audio_path = st.session_state.get(temp_audio_key)
                            if not audio_path:
                                st.error("Audio-Pfad nicht gefunden. Bitte Datei neu hochladen.")
                                return

                            analyzer = AudioAnalyzer()
                            features_opt = analyzer.analyze(audio_path, fps=30)

                            feature_summary = {
                                'duration': features_opt.duration,
                                'tempo': features_opt.tempo,
                                'mode': features_opt.mode,
                                'rms_mean': float(features_opt.rms.mean()),
                                'rms_std': float(features_opt.rms.std()),
                                'onset_mean': float(features_opt.onset.mean()),
                                'onset_std': float(features_opt.onset.std()),
                                'spectral_mean': float(features_opt.spectral_centroid.mean()),
                                'transient_mean': float(features_opt.transient.mean()) if hasattr(features_opt, 'transient') else 0.0,
                                'voice_clarity_mean': float(features_opt.voice_clarity.mean()) if hasattr(features_opt, 'voice_clarity') else 0.0,
                            }

                            viz_cls = get_visualizer(gpu_viz)
                            fallback_params = {k: v[0] for k, v in viz_cls.PARAMS.items()}
                            param_specs = viz_cls.PARAMS

                            current_params = {}
                            for k in fallback_params:
                                current_params[k] = st.session_state.get(f"{prefix}{k}", fallback_params[k])

                            current_colors = {
                                "primary": st.session_state.get("ai_color_primary", "#FF0055"),
                                "secondary": st.session_state.get("ai_color_secondary", "#00CCFF"),
                                "background": st.session_state.get("ai_color_background", "#0A0A0A"),
                            }

                            user_prompt = st.session_state.get("ai_user_prompt", "")

                            gemini = GeminiIntegration()
                            optimized = gemini.optimize_all_settings(
                                gpu_viz, current_params, feature_summary, current_colors,
                                param_specs=param_specs, user_prompt=user_prompt
                            )

                            for key, value in optimized.get("params", {}).items():
                                full_key = f"{prefix}{key}"
                                if full_key in st.session_state:
                                    del st.session_state[full_key]
                                st.session_state[full_key] = value

                            colors = optimized.get("colors", current_colors)
                            st.session_state["ai_color_primary"] = colors.get("primary", "#FF0055")
                            st.session_state["ai_color_secondary"] = colors.get("secondary", "#00CCFF")
                            st.session_state["ai_color_background"] = colors.get("background", "#0A0A0A")

                            pp = optimized.get("postprocess", {})
                            for k, v in pp.items():
                                state_key = f"pp_{k}_state"
                                widget_key = f"pp_{k}"
                                if k == "film_grain":
                                    widget_key = "pp_grain"
                                st.session_state[state_key] = v
                                if widget_key in st.session_state:
                                    del st.session_state[widget_key]
                                st.session_state[widget_key] = v

                            bg = optimized.get("background", {})
                            for k, v in bg.items():
                                widget_key = f"bg_{k}"
                                if widget_key in st.session_state:
                                    del st.session_state[widget_key]
                                st.session_state[widget_key] = v

                            quotes_opt = optimized.get("quotes", {})
                            if quotes_opt and uploaded_file:
                                quotes_cache_key = f"quotes_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
                                quote_mappings = {
                                    "font_size": f"quote_font_size_{quotes_cache_key}",
                                    "box_color": f"quote_box_color_{quotes_cache_key}",
                                    "font_color": f"quote_font_color_{quotes_cache_key}",
                                    "position": f"quote_pos_{quotes_cache_key}",
                                    "display_duration": f"quote_duration_{quotes_cache_key}",
                                    "auto_scale_font": f"quote_autoscale_{quotes_cache_key}",
                                    "slide_animation": f"quote_slide_{quotes_cache_key}",
                                    "slide_out_animation": f"quote_slide_out_{quotes_cache_key}",
                                    "scale_in": f"quote_scale_in_{quotes_cache_key}",
                                    "typewriter": f"quote_typewriter_{quotes_cache_key}",
                                    "glow_pulse": f"quote_glow_pulse_{quotes_cache_key}",
                                    "box_padding": f"quote_box_padding_{quotes_cache_key}",
                                    "box_radius": f"quote_box_radius_{quotes_cache_key}",
                                    "box_margin_bottom": f"quote_box_margin_{quotes_cache_key}",
                                    "max_width_ratio": f"quote_max_width_{quotes_cache_key}",
                                    "fade_duration": f"quote_fade_duration_{quotes_cache_key}",
                                    "line_spacing": f"quote_line_spacing_{quotes_cache_key}",
                                    "max_font_size": f"quote_max_font_{quotes_cache_key}",
                                    "max_chars_per_line": f"quote_max_chars_{quotes_cache_key}",
                                }
                                for src_key, dst_key in quote_mappings.items():
                                    if src_key in quotes_opt:
                                        val = quotes_opt[src_key]
                                        if dst_key in st.session_state:
                                            del st.session_state[dst_key]
                                        st.session_state[dst_key] = val

                            st.success("✅ Alle Einstellungen optimiert! Aktualisiere...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"KI-Optimierung fehlgeschlagen: {e}")
            else:
                st.info("Lade zuerst eine Audio-Datei hoch und wähle einen Look.")

        with st.expander("💬 Zitate", expanded=False):
            if uploaded_file:
                quotes_cache_key = f"quotes_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"

                if st.button("🔮 Key-Zitate extrahieren", key="extract_quotes_btn"):
                    with st.spinner("Gemini analysiert dein Audio..."):
                        try:
                            # Gecachten Audio-Pfad verwenden
                            audio_path = st.session_state.get(temp_audio_key)
                            if not audio_path:
                                st.error("Audio-Pfad nicht gefunden. Bitte Datei neu hochladen.")
                                return

                            gemini = GeminiIntegration()
                            quotes = gemini.extract_quotes(audio_path, audio_duration=features.duration)

                            if features is not None:
                                audio_duration = features.duration
                                for q in quotes:
                                    q.start_time = max(0.0, min(q.start_time, audio_duration - 1.0))
                                    q.end_time = max(q.start_time + 1.0, min(q.end_time, audio_duration))

                            st.session_state[quotes_cache_key] = quotes
                        except Exception as e:
                            st.error(f"Zitat-Extraktion fehlgeschlagen: {e}")
                            st.info("💡 Tipp: Stelle sicher, dass GEMINI_API_KEY gesetzt ist.")

                st.markdown("##### ⚙️ Zitat-Einstellungen")
                qe_col1, qe_col2 = st.columns(2)
                with qe_col1:
                    quote_font_size = st.slider("Schriftgröße", 12, 96, st.session_state.get(f"quote_font_size_{quotes_cache_key}", 52), key=f"quote_font_size_{quotes_cache_key}")
                    quote_display_duration = st.slider("Anzeigedauer (Sekunden)", 2.0, 20.0, st.session_state.get(f"quote_duration_{quotes_cache_key}", 8.0), 0.5, key=f"quote_duration_{quotes_cache_key}")
                    quote_position = st.selectbox("Position", ["bottom", "center", "top"], index=["bottom", "center", "top"].index(st.session_state.get(f"quote_pos_{quotes_cache_key}", "bottom")), key=f"quote_pos_{quotes_cache_key}")
                with qe_col2:
                    quote_text_align = st.selectbox("Text-Ausrichtung", ["center", "left", "right"], index=["center", "left", "right"].index(st.session_state.get(f"quote_align_{quotes_cache_key}", "center")), key=f"quote_align_{quotes_cache_key}")
                    quote_box_color = st.color_picker("Box-Farbe", st.session_state.get(f"quote_box_color_{quotes_cache_key}", "#1a1a2e"), key=f"quote_box_color_{quotes_cache_key}")
                    quote_font_color = st.color_picker("Schrift-Farbe", st.session_state.get(f"quote_font_color_{quotes_cache_key}", "#FFFFFF"), key=f"quote_font_color_{quotes_cache_key}")

                st.markdown("##### 📐 Box & Layout")
                box_col1, box_col2, box_col3 = st.columns(3)
                with box_col1:
                    quote_box_padding = st.slider("Box-Padding", 0, 80, st.session_state.get(f"quote_box_padding_{quotes_cache_key}", 32), key=f"quote_box_padding_{quotes_cache_key}")
                    quote_box_radius = st.slider("Box-Radius (px)", 0, 50, st.session_state.get(f"quote_box_radius_{quotes_cache_key}", 16), key=f"quote_box_radius_{quotes_cache_key}")
                    quote_box_margin = st.slider("Box-Abstand Rand", 20, 300, st.session_state.get(f"quote_box_margin_{quotes_cache_key}", 100), key=f"quote_box_margin_{quotes_cache_key}")
                with box_col2:
                    quote_max_width = st.slider("Max. Breite", 0.3, 1.0, st.session_state.get(f"quote_max_width_{quotes_cache_key}", 0.75), 0.05, key=f"quote_max_width_{quotes_cache_key}")
                    quote_fade_duration = st.slider("Fade-Dauer (Sek)", 0.1, 2.0, st.session_state.get(f"quote_fade_duration_{quotes_cache_key}", 0.6), 0.1, key=f"quote_fade_duration_{quotes_cache_key}")
                    quote_line_spacing = st.slider("Zeilenabstand", 0.8, 2.0, st.session_state.get(f"quote_line_spacing_{quotes_cache_key}", 1.35), 0.05, key=f"quote_line_spacing_{quotes_cache_key}")
                with box_col3:
                    quote_min_font = st.slider("Min. Schriftgröße", 8, 36, st.session_state.get(f"quote_min_font_{quotes_cache_key}", 16), key=f"quote_min_font_{quotes_cache_key}")
                    quote_max_font = st.slider("Max. Schriftgröße", 20, 96, st.session_state.get(f"quote_max_font_{quotes_cache_key}", 72), key=f"quote_max_font_{quotes_cache_key}")
                    quote_max_chars = st.slider("Max. Zeichen/Zeile", 20, 80, st.session_state.get(f"quote_max_chars_{quotes_cache_key}", 40), key=f"quote_max_chars_{quotes_cache_key}")

                st.markdown("##### ✨ Animationen")
                anim_col1, anim_col2, anim_col3 = st.columns(3)
                with anim_col1:
                    quote_auto_scale = st.checkbox("Auto-Skalierung", value=st.session_state.get(f"quote_autoscale_{quotes_cache_key}", True), key=f"quote_autoscale_{quotes_cache_key}")
                    quote_scale_in = st.checkbox("Scale-In", value=st.session_state.get(f"quote_scale_in_{quotes_cache_key}", False), key=f"quote_scale_in_{quotes_cache_key}")
                with anim_col2:
                    slide_options = ["none", "up", "down", "left", "right"]
                    slide_val = st.session_state.get(f"quote_slide_{quotes_cache_key}", "none")
                    if slide_val not in slide_options:
                        slide_val = "none"
                    quote_slide = st.selectbox("Slide-In", slide_options, index=slide_options.index(slide_val), key=f"quote_slide_{quotes_cache_key}")
                    quote_slide_dist = st.slider("Slide-In Distanz (px)", 0, 300, st.session_state.get(f"quote_slide_dist_{quotes_cache_key}", 100), key=f"quote_slide_dist_{quotes_cache_key}")
                    slide_out_options = ["none", "up", "down", "left", "right"]
                    slide_out_val = st.session_state.get(f"quote_slide_out_{quotes_cache_key}", "none")
                    if slide_out_val not in slide_out_options:
                        slide_out_val = "none"
                    quote_slide_out = st.selectbox("Slide-Out", slide_out_options, index=slide_out_options.index(slide_out_val), key=f"quote_slide_out_{quotes_cache_key}")
                    quote_slide_out_dist = st.slider("Slide-Out Distanz (px)", 0, 300, st.session_state.get(f"quote_slide_out_dist_{quotes_cache_key}", 100), key=f"quote_slide_out_dist_{quotes_cache_key}")
                    quote_glow_pulse = st.checkbox("Glow-Pulse", value=st.session_state.get(f"quote_glow_pulse_{quotes_cache_key}", False), key=f"quote_glow_pulse_{quotes_cache_key}")
                    quote_glow_pulse_int = st.slider("Pulse-Stärke", 0.0, 1.0, st.session_state.get(f"quote_glow_pulse_int_{quotes_cache_key}", 0.5), key=f"quote_glow_pulse_int_{quotes_cache_key}")
                with anim_col3:
                    quote_typewriter = st.checkbox("Typewriter-Effekt", value=st.session_state.get(f"quote_typewriter_{quotes_cache_key}", False), key=f"quote_typewriter_{quotes_cache_key}")
                    quote_tw_mode = st.selectbox("Typewriter-Modus", ["char", "word"], index=["char", "word"].index(st.session_state.get(f"quote_tw_mode_{quotes_cache_key}", "char")), key=f"quote_tw_mode_{quotes_cache_key}")
                    quote_tw_speed = st.slider("Typewriter-Geschw.", 5.0, 50.0, st.session_state.get(f"quote_tw_speed_{quotes_cache_key}", 15.0), 1.0, key=f"quote_tw_speed_{quotes_cache_key}")

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

                        qc_col1, qc_col2 = st.columns([0.08, 0.92])
                        with qc_col1:
                            enabled = st.checkbox(
                                "Aktiv",
                                value=st.session_state[selected_key][i],
                                key=f"quote_chk_{i}_{quotes_cache_key}"
                            )
                            st.session_state[selected_key][i] = enabled

                        with qc_col2:
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
                        st.info(f"📌 {len(edited_quotes)} Zitate sind aktiviert.")
                    else:
                        st.warning("⚠️ Keine Zitate aktiviert.")
                else:
                    st.caption("Noch keine Zitate extrahiert.")
            else:
                st.info("Lade zuerst eine Audio-Datei hoch.")

        with st.expander("🖼️ Hintergrund", expanded=False):
            if uploaded_file:
                uploaded_bg = st.file_uploader(
                    "Hintergrundbild hochladen (optional)",
                    type=['png', 'jpg', 'jpeg', 'webp'],
                    key="bg_uploader"
                )

                if uploaded_bg:
                    bg_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_bg.name.split(".")[-1]}')
                    bg_temp.write(uploaded_bg.getvalue())
                    bg_temp.close()
                    bg_image_path = bg_temp.name
                    st.image(bg_image_path, caption="Hintergrundbild-Vorschau", width='stretch')

                bg_blur = st.slider("🔮 Blur (Weichzeichnung)", 0.0, 20.0,
                                    st.session_state.get("bg_blur", 0.0), 0.5,
                                    key="bg_blur")
                bg_vignette = st.slider("🌑 Vignette (Randabdunkelung)", 0.0, 1.0,
                                        st.session_state.get("bg_vignette", 0.0), 0.05,
                                        key="bg_vignette")
                bg_opacity = st.slider("🎨 Hintergrund-Deckkraft", 0.0, 1.0,
                                       st.session_state.get("bg_opacity", 0.3), 0.05,
                                       key="bg_opacity")
            else:
                st.info("Lade zuerst eine Audio-Datei hoch.")

        st.markdown("---")
        st.markdown("### 🎬 Export")

        if uploaded_file and gpu_viz:
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                preview_clicked = st.button("📹 Schnell-Vorschau (5s)", key="btn_preview", width='stretch')
            with export_col2:
                render_clicked = st.button("🎬 Video exportieren", key="btn_render", type="primary", width='stretch')

            with st.expander("🔧 Technische Einstellungen", expanded=False):
                tech_col1, tech_col2, tech_col3 = st.columns(3)
                with tech_col1:
                    resolution = st.selectbox("Auflösung", ["1920x1080", "1280x720", "3840x2160", "854x480"], index=0, key="export_resolution")
                with tech_col2:
                    fps = st.selectbox("FPS", [60, 30, 24], index=0, key="export_fps")
                with tech_col3:
                    codec = st.selectbox("Codec", ["h264", "hevc", "prores"], index=0, key="export_codec")
                    quality = st.selectbox("Qualität", ["high", "medium", "low", "lossless"], index=0, key="export_quality")

                turbo_mode = st.checkbox("⚡ Turbo-Modus", value=False, key="export_turbo")
                frame_skip = st.selectbox(
                    "🚀 Frame-Skip (Draft-Modus)",
                    options=[("Jeder Frame (Qualität)", 1), ("Jeder 2. Frame (2x schneller)", 2), ("Jeder 3. Frame (3x schneller)", 3)],
                    format_func=lambda x: x[0],
                    index=0,
                    key="export_frame_skip"
                )[1]
        else:
            st.info("Lade Audio und wähle einen Look, um zu exportieren.")

    with right_col:
        st.markdown("### 👁️ Live Preview")

        if uploaded_file and gpu_viz:
            auto_preview = st.toggle("⚡ Auto-Preview", value=False, key="auto_preview_toggle",
                                     help="Automatisch neu rendern wenn Parameter sich ändern")

            preview_params_hash = hash((gpu_viz, str(sorted(gpu_params.items()) if gpu_params else []),
                                        bg_opacity, bg_vignette, bg_blur,
                                        pp_contrast, pp_saturation, pp_brightness, pp_warmth, pp_grain))

            preview_needs_update = False
            if auto_preview:
                last_hash = st.session_state.get("last_preview_hash", None)
                if last_hash != preview_params_hash:
                    preview_needs_update = True

            preview_container = st.empty()

            manual_preview = st.button("🔄 Preview aktualisieren", key="manual_preview_btn")
            render_error = False
            if manual_preview or preview_needs_update:
                with st.spinner("Rendere GPU-Frame...") if not preview_needs_update else st.empty():
                    try:
                        # Gecachten Audio-Pfad verwenden
                        audio_path = st.session_state.get(temp_audio_key)
                        if not audio_path:
                            st.error("Audio-Pfad nicht gefunden. Bitte Datei neu hochladen.")
                            render_error = True
                        else:
                            pp_cfg = None
                            if pp_contrast != 1.0 or pp_saturation != 1.0 or pp_brightness != 0.0 or pp_warmth != 0.0 or pp_grain > 0.0:
                                pp_cfg = {
                                    "contrast": pp_contrast,
                                    "saturation": pp_saturation,
                                    "brightness": pp_brightness,
                                    "warmth": pp_warmth,
                                    "film_grain": pp_grain,
                                }

                            preview_quote_cfg = None
                            preview_quotes = None
                            qck_preview = f"quotes_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
                            edited_raw = st.session_state.get(f"render_quotes_{qck_preview}", [])
                            if edited_raw:
                                preview_quotes = [Quote(**q) for q in edited_raw]
                                preview_quote_cfg = QuoteOverlayConfig(
                                    enabled=True,
                                    font_size=st.session_state.get(f"quote_font_size_{qck_preview}", 52),
                                    font_color=hex_to_rgb(st.session_state.get(f"quote_font_color_{qck_preview}", "#FFFFFF")),
                                    box_color=hex_to_rgb(st.session_state.get(f"quote_box_color_{qck_preview}", "#1a1a2e")) + (200,),
                                    box_padding=st.session_state.get(f"quote_box_padding_{qck_preview}", 32),
                                    box_radius=st.session_state.get(f"quote_box_radius_{qck_preview}", 16),
                                    box_margin_bottom=st.session_state.get(f"quote_box_margin_{qck_preview}", 100),
                                    max_width_ratio=st.session_state.get(f"quote_max_width_{qck_preview}", 0.75),
                                    fade_duration=st.session_state.get(f"quote_fade_duration_{qck_preview}", 0.6),
                                    line_spacing=st.session_state.get(f"quote_line_spacing_{qck_preview}", 1.35),
                                    max_chars_per_line=st.session_state.get(f"quote_max_chars_{qck_preview}", 40),
                                    position=st.session_state.get(f"quote_pos_{qck_preview}", "bottom"),
                                    text_align=st.session_state.get(f"quote_align_{qck_preview}", "center"),
                                    display_duration=st.session_state.get(f"quote_duration_{qck_preview}", 8.0),
                                    auto_scale_font=st.session_state.get(f"quote_autoscale_{qck_preview}", True),
                                    min_font_size=st.session_state.get(f"quote_min_font_{qck_preview}", 16),
                                    max_font_size=st.session_state.get(f"quote_max_font_{qck_preview}", 72),
                                    slide_animation=st.session_state.get(f"quote_slide_{qck_preview}", "none"),
                                    slide_distance=st.session_state.get(f"quote_slide_dist_{qck_preview}", 100),
                                    slide_out_animation=st.session_state.get(f"quote_slide_out_{qck_preview}", "none"),
                                    slide_out_distance=st.session_state.get(f"quote_slide_out_dist_{qck_preview}", 100),
                                    scale_in=st.session_state.get(f"quote_scale_in_{qck_preview}", False),
                                    typewriter=st.session_state.get(f"quote_typewriter_{qck_preview}", False),
                                    typewriter_speed=st.session_state.get(f"quote_tw_speed_{qck_preview}", 15.0),
                                    typewriter_mode=st.session_state.get(f"quote_tw_mode_{qck_preview}", "char"),
                                    glow_pulse=st.session_state.get(f"quote_glow_pulse_{qck_preview}", False),
                                    glow_pulse_intensity=st.session_state.get(f"quote_glow_pulse_int_{qck_preview}", 0.5),
                                )

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
                                quotes=preview_quotes,
                                quote_config=preview_quote_cfg,
                            )

                            if preview_img is not None:
                                # Hash erst nach erfolgreichem Render speichern
                                st.session_state["last_preview_hash"] = preview_params_hash
                                st.session_state["last_preview_img"] = preview_img
                                preview_container.image(preview_img, caption=f"👁️ Live-Preview: {gpu_viz}", width='stretch')
                            else:
                                render_error = True
                                preview_container.error("GPU-Preview konnte nicht gerendert werden.")
                    except Exception as e:
                        render_error = True
                        preview_container.error(f"GPU-Live-Preview Fehler: {e}")

            if not render_error and not preview_needs_update and "last_preview_img" in st.session_state:
                preview_container.image(st.session_state["last_preview_img"], caption=f"👁️ Live-Preview: {gpu_viz}", width='stretch')

            if preview_clicked or render_clicked:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Gecachten Audio-Pfad verwenden
                    audio_path = st.session_state.get(temp_audio_key)
                    if not audio_path:
                        st.error("Audio-Pfad nicht gefunden. Bitte Datei neu hochladen.")
                        return

                    temp_dir = tempfile.mkdtemp()
                    output_filename = f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    output_path = os.path.join(temp_dir, output_filename)

                    w, h = map(int, resolution.split('x'))

                    status_text.text("Starte GPU-Rendering...")

                    renderer = GPUBatchRenderer(width=w, height=h, fps=fps)

                    preview_mode = preview_clicked

                    qck = f"quotes_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
                    quote_font_path = st.session_state.get(f"quote_font_path_{qck}", None)

                    quote_config = QuoteOverlayConfig(
                        enabled=True,
                        font_size=st.session_state.get(f"quote_font_size_{qck}", 52),
                        font_color=hex_to_rgb(st.session_state.get(f"quote_font_color_{qck}", "#FFFFFF")),
                        box_color=hex_to_rgb(st.session_state.get(f"quote_box_color_{qck}", "#1a1a2e")) + (200,),
                        box_padding=st.session_state.get(f"quote_box_padding_{qck}", 32),
                        box_radius=st.session_state.get(f"quote_box_radius_{qck}", 16),
                        box_margin_bottom=st.session_state.get(f"quote_box_margin_{qck}", 100),
                        max_width_ratio=st.session_state.get(f"quote_max_width_{qck}", 0.75),
                        fade_duration=st.session_state.get(f"quote_fade_duration_{qck}", 0.6),
                        line_spacing=st.session_state.get(f"quote_line_spacing_{qck}", 1.35),
                        max_chars_per_line=st.session_state.get(f"quote_max_chars_{qck}", 40),
                        position=st.session_state.get(f"quote_pos_{qck}", "bottom"),
                        text_align=st.session_state.get(f"quote_align_{qck}", "center"),
                        display_duration=st.session_state.get(f"quote_duration_{qck}", 8.0),
                        font_path=quote_font_path,
                        auto_scale_font=st.session_state.get(f"quote_autoscale_{qck}", True),
                        min_font_size=st.session_state.get(f"quote_min_font_{qck}", 16),
                        max_font_size=st.session_state.get(f"quote_max_font_{qck}", 72),
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

                        # Temporaere Dateien nicht sofort loeschen — Streamlit
                        # serviert Videos asynchron und braucht die Datei
                        # noch fuer den Browser. Windows raeumt sowieso auf.
                        # shutil.rmtree(temp_dir, ignore_errors=True)
                    else:
                        st.error("Rendering fehlgeschlagen: Output-Datei nicht gefunden.")

                except Exception as e:
                    st.error("❌ GPU-Rendering fehlgeschlagen")
                    st.info(f"**Fehler:** {str(e)}")
                    st.info("💡 Tipp: Stelle sicher, dass FFmpeg korrekt installiert ist und eine GPU verfügbar.")
        else:
            st.info("👆 Lade eine Audio-Datei hoch und wähle einen Look, um die Preview zu sehen.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Made with ❤️ | Audio Visualizer Pro</p>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
