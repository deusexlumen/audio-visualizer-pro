"""
PostProcess - Layer 3: Color Grading, LUTs, Export

Nachbearbeitung der gerenderten Frames mit verschiedenen Effekten.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Optional, Tuple
import colorsys


class PostProcessor:
    """
    Post-Processing für gerenderte Frames.
    
    Unterstützt:
    - LUTs (Color Lookup Tables)
    - Film Grain
    - Chromatic Aberration
    - Vignette
    - Kontrast/Sättigung
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lut = self._load_lut(config.get('lut'))
        self.grain_intensity = config.get('grain', 0.0)
        self.chromatic = config.get('chromatic_aberration', 0.0)
        self.vignette = config.get('vignette', 0.0)
        self.contrast = config.get('contrast', 1.0)
        self.saturation = config.get('saturation', 1.0)
        self.brightness = config.get('brightness', 1.0)
        self.bloom = config.get('bloom', 0.3)
        self.bloom_threshold = config.get('bloom_threshold', 180)
        
    def _load_lut(self, lut_path: Optional[str]) -> Optional[np.ndarray]:
        """Lädt eine .cube LUT-Datei."""
        if not lut_path:
            return None
        
        try:
            # Einfacher 3D LUT Loader (simplified)
            # Für vollständige Unterstützung wäre eine externe Bibliothek nötig
            with open(lut_path, 'r') as f:
                lines = f.readlines()
            
            # Parse LUT-Daten (vereinfacht)
            lut_data = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('LUT'):
                    values = [float(v) for v in line.split()]
                    if len(values) == 3:
                        lut_data.append(values)
            
            return np.array(lut_data) if lut_data else None
            
        except Exception as e:
            print(f"[PostProcess] Warnung: Konnte LUT nicht laden: {e}")
            return None
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Wendet alle konfigurierten Post-Processing-Effekte an.
        
        Args:
            frame: RGB-Frame als numpy array (H, W, 3)
        
        Returns:
            Bearbeiteter Frame
        """
        # Konvertiere zu PIL Image für einfachere Bearbeitung
        img = Image.fromarray(frame)
        
        # Kontrast
        if self.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)
        
        # Sättigung
        if self.saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(self.saturation)
        
        # Helligkeit
        if self.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness)
        
        # Konvertiere zurück zu numpy für weitere Bearbeitung
        result = np.array(img)
        
        # Bloom
        if self.bloom > 0:
            result = self._apply_bloom(result)
        
        # Film Grain
        if self.grain_intensity > 0:
            result = self._apply_grain(result)
        
        # Vignette
        if self.vignette > 0:
            result = self._apply_vignette(result)
        
        # Chromatic Aberration
        if self.chromatic > 0:
            result = self._apply_chromatic_aberration(result)
        
        # LUT (Color Lookup Table) anwenden
        result = self.apply_lut(result)
        
        return result
    
    def _apply_bloom(self, frame: np.ndarray) -> np.ndarray:
        """Fuegt Bloom-Effekt hinzu (Glühen bei hellen Bereichen)."""
        img = Image.fromarray(frame)
        
        # Extrahiere helle Bereiche
        bright = img.point(lambda p: p if p > self.bloom_threshold else 0)
        
        # Blurre die hellen Bereiche
        bloom = bright.filter(ImageFilter.GaussianBlur(radius=15))
        
        # Additive Komposition
        result = Image.blend(img, bloom, self.bloom)
        return np.array(result)
    
    def _apply_grain(self, frame: np.ndarray) -> np.ndarray:
        """Fügt Film Grain hinzu."""
        noise = np.random.normal(0, self.grain_intensity * 50, frame.shape)
        grainy = frame.astype(np.float32) + noise
        return np.clip(grainy, 0, 255).astype(np.uint8)
    
    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Fügt Vignette-Effekt hinzu."""
        h, w = frame.shape[:2]
        
        # Erstelle Vignette-Maske
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Distanz vom Zentrum (normalisiert)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Vignette-Kurve
        vignette = 1 - (dist / max_dist) * self.vignette
        vignette = np.clip(vignette, 0.3, 1.0)
        
        # Anwenden
        return (frame * vignette[:, :, np.newaxis]).astype(np.uint8)
    
    def _apply_chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Fügt Chromatic Aberration hinzu."""
        h, w = frame.shape[:2]
        shift = int(self.chromatic * 3)
        
        if shift == 0:
            return frame
        
        result = frame.copy()
        
        # Rot-Kanal nach links verschieben
        result[:, :-shift, 0] = frame[:, shift:, 0]
        
        # Blau-Kanal nach rechts verschieben
        result[:, shift:, 2] = frame[:, :-shift, 2]
        
        return result
    
    def apply_lut(self, frame: np.ndarray) -> np.ndarray:
        """Wendet die geladene LUT an (vereinfacht)."""
        if self.lut is None:
            return frame
        
        # Vereinfachte LUT-Anwendung
        # Für vollständige 3D LUT-Unterstützung wäre eine externe Bibliothek besser
        result = frame.copy()
        
        # Skaliere zu LUT-Indizes
        lut_size = int(round(len(self.lut) ** (1/3)))
        
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                r, g, b = frame[i, j] / 255.0
                
                # LUT-Indizes
                lr = int(r * (lut_size - 1))
                lg = int(g * (lut_size - 1))
                lb = int(b * (lut_size - 1))
                
                # Index in flachem LUT-Array
                idx = lr + lg * lut_size + lb * lut_size * lut_size
                
                if idx < len(self.lut):
                    result[i, j] = (self.lut[idx] * 255).astype(np.uint8)
        
        return result


class PostProcessPipeline:
    """
    Pipeline für Post-Processing von Videos.
    Kann auf existierende Videos angewendet werden.
    """
    
    def __init__(self, config: Dict):
        self.processor = PostProcessor(config)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Verarbeitet einen einzelnen Frame."""
        return self.processor.apply(frame)
    
    def process_video(self, input_path: str, output_path: str):
        """
        Verarbeitet ein komplettes Video.
        
        Args:
            input_path: Pfad zum Eingabe-Video
            output_path: Pfad zum Ausgabe-Video
        """
        import subprocess
        import tempfile
        
        # Temporäre Datei
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video.close()
        
        try:
            # FFmpeg für Frame-Extraktion und -Rekodierung
            # Dies ist ein vereinfachtes Beispiel
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg Fehler: {result.stderr}")
                
        finally:
            if Path(temp_video.name).exists():
                Path(temp_video.name).unlink()
