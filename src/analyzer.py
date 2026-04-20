"""
AudioAnalyzer 2.0 - Layer 1: Audio -> Features

Neu in v2.0:
- Exponential Moving Average (EMA) fuer alle Features
- Transient-Detection (Kick/Snare) fuer Musik-Modus
- Voice-Clarity-Index (80Hz-3kHz) fuer Podcast-Modus
- Verbesserte Modus-Erkennung (Musik vs Speech vs Hybrid)
- BPM-Stabilisierung ueber Zeit
"""

import librosa
import numpy as np
import hashlib
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Callable
from .types import AudioFeatures

_SLOW_FORMATS = {'.mp3', '.m4a', '.aac', '.ogg', '.wma', '.opus'}


class EMAFilter:
    """Exponential Moving Average fuer geglaettete Steuerungswerte."""
    
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.state = None
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Wendet EMA auf ein ganzes Array an (frame-basiert)."""
        if self.state is None:
            self.state = data[0]
        result = np.zeros_like(data)
        for i, val in enumerate(data):
            self.state = self.alpha * val + (1 - self.alpha) * self.state
            result[i] = self.state
        return result
    
    def reset(self):
        self.state = None


class AudioAnalyzer:
    """Audio-Analyse 2.0 mit EMA-Smoothing und erweiterten Features."""
    
    def __init__(self, cache_dir: str = ".cache/audio_features"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, audio_path: str, fps: int) -> Path:
        file_stat = Path(audio_path).stat()
        hasher = hashlib.md5()
        hasher.update(f"{file_stat.st_size}_{file_stat.st_mtime}_{fps}_v2".encode())
        return self.cache_dir / f"{hasher.hexdigest()}.npz"
    
    def _progress(self, msg: str, step: int, total: int, callback: Optional[Callable] = None):
        pct = int((step / total) * 100)
        print(f"[Analyzer] {msg} ({step}/{total})")
        if callback:
            callback(msg, step, total)
    
    def analyze(self, audio_path: str, fps: int = 60, force_reanalyze: bool = False,
                progress_callback: Optional[Callable] = None,
                ema_alpha: float = 0.15) -> AudioFeatures:
        """
        Extrahiert alle Features mit EMA-Smoothing.
        
        Args:
            ema_alpha: Glättungsfaktor (0.0 = keine Glättung, 1.0 = maximale Glättung)
        """
        cache_path = self._get_cache_path(audio_path, fps)
        
        if not force_reanalyze and cache_path.exists():
            self._progress("Lade aus Cache...", 1, 1, progress_callback)
            data = np.load(cache_path, allow_pickle=True)
            loaded_data = {}
            # Skalare Felder die aus 0-dim numpy Arrays zurueckkonvertiert werden muessen
            scalar_fields = {'duration', 'sample_rate', 'fps', 'frame_count', 'tempo', 'key', 'mode'}
            for k in data.files:
                val = data[k]
                if isinstance(val, np.ndarray):
                    if val.dtype.kind == 'U':
                        # Unicode-String
                        loaded_data[k] = str(val.item())
                    elif k in scalar_fields or val.size == 1:
                        # Skalare Werte (None, float, int, bool) aus 0-dim Array extrahieren
                        loaded_data[k] = val.item() if val.size > 0 else None
                    else:
                        loaded_data[k] = val
                else:
                    loaded_data[k] = val
            return AudioFeatures(**loaded_data)
        
        total_steps = 9
        step = 0
        
        # Audio laden
        self._progress("Lade Audio...", step := step + 1, total_steps, progress_callback)
        audio_path_obj = Path(audio_path)
        temp_wav = None
        if audio_path_obj.suffix.lower() in _SLOW_FORMATS:
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', str(audio_path), '-ar', '11025', '-ac', '1', temp_wav.name],
                    capture_output=True, check=True
                )
                audio_path = temp_wav.name
            except (subprocess.CalledProcessError, FileNotFoundError):
                if temp_wav:
                    os.unlink(temp_wav.name)
                    temp_wav = None
        
        try:
            y, sr = librosa.load(audio_path, sr=11025, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
        finally:
            if temp_wav and os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
        
        hop_length = int(sr / fps)
        expected_frames = int(duration * fps)
        
        # RMS
        self._progress("Berechne Lautstaerke...", step := step + 1, total_steps, progress_callback)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms = self._normalize(rms)
        
        # Onset
        self._progress("Erkenne Beats...", step := step + 1, total_steps, progress_callback)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset = self._normalize(onset_env)
        
        # Spektrale Features
        self._progress("Analysiere Spektrum...", step := step + 1, total_steps, progress_callback)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
        
        # Chroma
        self._progress("Erkenne Tonart...", step := step + 1, total_steps, progress_callback)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # NEU: Transient-Detection (fuer Kick/Snare)
        self._progress("Erkenne Transienten...", step := step + 1, total_steps, progress_callback)
        transient = self._detect_transients(y, sr, hop_length, expected_frames)
        
        # NEU: Voice Clarity (80Hz - 3kHz Band)
        self._progress("Analysiere Sprach-Präsenz...", step := step + 1, total_steps, progress_callback)
        voice_clarity = self._detect_voice_clarity(y, sr, hop_length, expected_frames)
        
        # Tempo & Mode
        self._progress("Klassifiziere Audio-Typ...", step := step + 1, total_steps, progress_callback)
        tempo = self._estimate_tempo_simple(onset_env, sr, hop_length)
        mode = self._detect_mode_advanced(tempo, onset_env, spec_cent, voice_clarity, rms)
        key = self._estimate_key(chroma) if duration < 600 else None
        
        # Beat-Frames fuer Audio-Sync extrahieren
        beat_frames = self._extract_beat_frames(onset_env, sr, hop_length, fps, expected_frames)
        
        # Finalisieren
        self._progress("Finalisiere...", step := step + 1, total_steps, progress_callback)
        
        # EMA-Smoothing anwenden
        ema = EMAFilter(alpha=ema_alpha)
        rms_smooth = ema.process(self._interpolate_to_length(rms, expected_frames))
        ema.reset()
        onset_smooth = ema.process(self._interpolate_to_length(onset, expected_frames))
        ema.reset()
        transient_smooth = ema.process(transient)
        ema.reset()
        voice_clarity_smooth = ema.process(voice_clarity)
        
        chroma_frames = chroma.shape[1]
        mfcc = np.zeros((13, chroma_frames), dtype=np.float32)
        tempogram = np.zeros((384, chroma_frames), dtype=np.float32)
        
        features = AudioFeatures(
            duration=duration,
            sample_rate=sr,
            fps=fps,
            rms=rms_smooth,
            onset=onset_smooth,
            spectral_centroid=self._normalize(self._interpolate_to_length(spec_cent, expected_frames)),
            spectral_rolloff=self._normalize(self._interpolate_to_length(spec_roll, expected_frames)),
            zero_crossing_rate=self._normalize(self._interpolate_to_length(zcr, expected_frames)),
            transient=transient_smooth,
            voice_clarity=voice_clarity_smooth,
            chroma=chroma,
            mfcc=mfcc,
            tempogram=tempogram,
            tempo=float(tempo),
            key=key,
            mode=mode,
            beat_frames=beat_frames
        )
        
        self._save_cache(cache_path, features)
        return features
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    def _interpolate_to_length(self, data: np.ndarray, target_length: int) -> np.ndarray:
        if len(data) == target_length:
            return data
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, data)
    
    def _detect_transients(self, y: np.ndarray, sr: int, hop_length: int, target_frames: int) -> np.ndarray:
        """
        Erkennt Transienten (Kick/Snare) via Differenz der RMS-Energie.
        Gibt ein Array zurueck mit transient-Staerke pro Frame (0-1).
        """
        # Kurze RMS-Fenster fuer schnelle Transienten
        rms_short = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=256)[0]
        # Differenz = Anstieg der Energie
        diff = np.diff(rms_short, prepend=rms_short[0])
        diff = np.maximum(diff, 0)  # nur positive Anstiege
        # Normalisieren
        diff = self._normalize(diff)
        return self._interpolate_to_length(diff, target_frames)
    
    def _detect_voice_clarity(self, y: np.ndarray, sr: int, hop_length: int, target_frames: int) -> np.ndarray:
        """
        Misst die Energie im Sprach-Band (80Hz - 3kHz) relativ zur Gesamtenergie.
        Hoeher = mehr Sprache, niedriger = mehr Musik/Noise.
        """
        # STFT fuer Frequenz-Analyse
        stft = np.abs(librosa.stft(y, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Maske fuer 80Hz - 3kHz
        voice_mask = (freqs >= 80) & (freqs <= 3000)
        
        # Energie im Sprach-Band / Gesamtenergie pro Frame
        voice_energy = np.sum(stft[voice_mask, :], axis=0)
        total_energy = np.sum(stft, axis=0) + 1e-8
        clarity = voice_energy / total_energy
        
        return self._interpolate_to_length(clarity, target_frames)
    
    def _extract_beat_frames(self, onset_env: np.ndarray, sr: int, hop_length: int, 
                             fps: int, expected_frames: int) -> np.ndarray:
        """Extrahiert Beat-Frames und konvertiert sie auf FPS-Basis.
        
        Returns:
            np.ndarray: Frame-Indizes (0-based) bei denen Beats auftreten.
        """
        try:
            # Beat-Tracking mit librosa
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env, 
                sr=sr, 
                hop_length=hop_length,
                units='frames'
            )
            
            # Konvertiere von hop_length-Frames auf Video-Frames
            # Ein hop_length-Frame entspricht hop_length/sr Sekunden
            # Video-Frame-Rate ist fps
            hop_duration = hop_length / sr
            video_frame_duration = 1.0 / fps
            
            beat_times = beats * hop_duration
            beat_video_frames = (beat_times / video_frame_duration).astype(np.int32)
            
            # Entferne Duplikate und clamp auf gueltigen Bereich
            beat_video_frames = np.unique(beat_video_frames)
            beat_video_frames = beat_video_frames[beat_video_frames < expected_frames]
            
            return beat_video_frames.astype(np.int32)
        except Exception:
            return np.array([], dtype=np.int32)
    
    def _estimate_tempo_simple(self, onset_env: np.ndarray, sr: int, hop_length: int) -> float:
        try:
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=96
            )
            tg_mean = np.mean(tempogram, axis=1)
            bpms = librosa.tempo_frequencies(len(tg_mean), hop_length=hop_length, sr=sr)
            best_idx = np.argmax(tg_mean)
            tempo = float(bpms[best_idx])
            if tempo < 40 or tempo > 250:
                return 120.0
            return tempo
        except Exception:
            return 120.0
    
    def _detect_mode_advanced(self, tempo: float, onset_env: np.ndarray, 
                               spec_cent: np.ndarray, voice_clarity: np.ndarray,
                               rms: np.ndarray) -> str:
        """
        Erweiterte Modus-Erkennung mit mehr Features.
        Returns: 'music', 'speech', 'hybrid'
        """
        onset_std = float(np.std(onset_env))
        cent_mean = float(np.mean(spec_cent))
        voice_mean = float(np.mean(voice_clarity))
        rms_var = float(np.std(rms))
        
        # Speech: hohe Voice-Clarity, niedrige spektrale Variabilitaet
        is_speech = (voice_mean > 0.45) and (rms_var < 0.15) and (cent_mean < 2000)
        
        # Music: erkennbares Tempo, rhythmische Variabilitaet, hoher Spectral-Centroid
        is_music = (tempo > 60) and (onset_std > 0.08) and (cent_mean > 1200) and (voice_mean < 0.5)
        
        if is_music and is_speech:
            return "hybrid"
        elif is_music:
            return "music"
        elif is_speech:
            return "speech"
        else:
            # Default: Hybrid wenn unklar
            return "hybrid"
    
    def _estimate_key(self, chroma: np.ndarray) -> str:
        chroma_avg = np.mean(chroma, axis=1)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return keys[np.argmax(chroma_avg)] + " major"
    
    def _save_cache(self, path: Path, features: AudioFeatures):
        data = {}
        for k, v in features.model_dump().items():
            if isinstance(v, str):
                data[k] = np.array(v, dtype='<U100')
            else:
                data[k] = v
        np.savez_compressed(path, **data)
