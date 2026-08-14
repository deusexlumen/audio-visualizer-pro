"""
AudioAnalyzer 2.0 - Layer 1: Audio -> Features

Neu in v2.0:
- Exponential Moving Average (EMA) fuer alle Features
- Transient-Detection (Kick/Snare) fuer Musik-Modus
- Voice-Clarity-Index (80Hz-3kHz) fuer Podcast-Modus
- Verbesserte Modus-Erkennung (Musik vs Speech vs Hybrid)
- BPM-Stabilisierung ueber Zeit
"""

import gc
import librosa
import numpy as np
import scipy.signal.windows
import hashlib
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Callable
from .app_logging import get_logger
from .ffmpeg_locator import get_ffmpeg_path
from .paths import user_data_dir
from .types import AudioFeatures

logger = get_logger(__name__)

_SLOW_FORMATS = {'.mp3', '.m4a', '.aac', '.ogg', '.wma', '.opus'}

# librosa hat tempo() in 0.10 nach feature.rhythm verschoben; der alte
# Alias in librosa.beat warnt und verschwindet in 1.0.
try:
    from librosa.feature.rhythm import tempo as _librosa_tempo
except ImportError:  # pragma: no cover - nur auf aelteren librosa-Versionen
    from librosa.beat import tempo as _librosa_tempo


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
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else user_data_dir("cache", "audio_features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    # Bei Format-Aenderungen an den gecachten Features hochzaehlen —
    # invalidiert alle bestehenden Caches.
    CACHE_VERSION = 9

    def _get_cache_path(self, audio_path: str, fps: int, ema_alpha: float = 0.15) -> Path:
        """Cache-Key basiert auf Datei-Inhalt (MD5) + fps + ema_alpha — nicht auf mtime."""
        path = Path(audio_path)
        file_stat = path.stat()
        hasher = hashlib.md5()
        hasher.update(str(file_stat.st_size).encode())
        try:
            with open(path, 'rb') as f:
                hasher.update(f.read(1024 * 1024))  # Erste 1MB
                if file_stat.st_size > 4 * 1024 * 1024:
                    # Mitte der Datei (reduziert Kollisionswahrscheinlichkeit)
                    mid = file_stat.st_size // 2
                    f.seek(mid - 512 * 1024)
                    hasher.update(f.read(1024 * 1024))
                    f.seek(-1024 * 1024, 2)
                    hasher.update(f.read())  # Letzte 1MB
                elif file_stat.st_size > 2 * 1024 * 1024:
                    f.seek(-1024 * 1024, 2)
                    hasher.update(f.read())  # Letzte 1MB
        except Exception:
            pass
        hasher.update(f"_{fps}_{ema_alpha}_v{self.CACHE_VERSION}".encode())
        return self.cache_dir / f"{hasher.hexdigest()}.npz"
    
    def _progress(self, msg: str, step: int, total: int, callback: Optional[Callable] = None):
        pct = int((step / total) * 100)
        logger.info(f"[Analyzer] {msg} ({step}/{total})")
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
        cache_path = self._get_cache_path(audio_path, fps, ema_alpha)
        
        if not force_reanalyze and cache_path.exists():
            self._progress("Lade aus Cache...", 1, 1, progress_callback)
            data = np.load(cache_path, allow_pickle=True)
            loaded_data = {}
            # Skalare Felder die aus 0-dim numpy Arrays zurueckkonvertiert werden muessen
            scalar_fields = {'duration', 'sample_rate', 'fps', 'frame_count', 'tempo', 'key', 'mode'}
            # Array-Felder muessen IMMER als ndarray bleiben, auch bei size 0 oder 1
            array_fields = {'rms', 'onset', 'spectral_centroid', 'spectral_rolloff',
                            'zero_crossing_rate', 'transient', 'voice_clarity', 'voice_band',
                            'chroma', 'mfcc', 'tempogram', 'beat_frames'}
            for k in data.files:
                val = data[k]
                if isinstance(val, np.ndarray):
                    if val.dtype.kind == 'U':
                        # Unicode-String
                        s = str(val.item())
                        if s == self._NONE_SENTINEL:
                            loaded_data[k] = None
                        else:
                            loaded_data[k] = s
                    elif k in array_fields:
                        # Array-Felder: immer als ndarray behalten
                        if val.size == 0:
                            loaded_data[k] = np.array([], dtype=val.dtype if val.dtype != object else np.float32)
                        else:
                            loaded_data[k] = val
                    elif k in scalar_fields or val.size == 1:
                        # Skalare Werte (None, float, int, bool) aus 0-dim Array extrahieren
                        if val.size == 0:
                            loaded_data[k] = None
                        else:
                            item = val.item()
                            # Edge-Case: Object-Arrays koennen verschachtelt sein
                            if isinstance(item, np.ndarray):
                                loaded_data[k] = None
                            else:
                                loaded_data[k] = item
                    else:
                        loaded_data[k] = val
                else:
                    loaded_data[k] = val
            return AudioFeatures(**loaded_data)
        
        total_steps = 10
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
                    [get_ffmpeg_path(), '-y', '-i', str(audio_path), '-ar', '44100', '-ac', '1', temp_wav.name],
                    capture_output=True, check=True
                )
                audio_path = temp_wav.name
            except (subprocess.CalledProcessError, FileNotFoundError):
                if temp_wav:
                    os.unlink(temp_wav.name)
                    temp_wav = None
        
        try:
            y, sr = librosa.load(audio_path, sr=44100, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            raise RuntimeError(f"Audio-Datei konnte nicht geladen werden: {audio_path}") from e
        finally:
            if temp_wav and os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
        
        # RAM-Optimierung: Fuer lange Dateien groesseres hop_length verwenden.
        # Das Ergebnis wird sowieso auf die Video-FPS interpoliert, daher ist
        # die hoehere Zeitaufloesung bei langen Podcasts/Songs nicht noetig.
        hop_length = 512 if duration > 600 else 256
        n_fft = 2048
        expected_frames = int(np.ceil(duration * fps))
        
        # === Pre-Emphasis Filter (0.97) ===
        # Hebt hochfrequente Anteile (Konsonanten, S-Laute) fuer Sprach-Analyse an
        # In-Place (spart eine Kopie des Audio-Buffers, ca. 200 MB bei 20min)
        y[1:] -= 0.97 * y[:-1]
        
        # === Windowing: Blackman-Harris zur Minimierung von Spectral Leakage ===
        window = scipy.signal.windows.blackmanharris(n_fft)
        
        # Gemeinsamer STFT fuer Frequenz-basierte Features (mit explizitem Window)
        self._progress("STFT Berechnung...", step := step + 1, total_steps, progress_callback)
        stft_complex = librosa.stft(y, hop_length=hop_length, n_fft=n_fft, window=window)
        stft = np.abs(stft_complex)
        # RAM-Optimierung: complex64-STFT wird nicht mehr gebraucht, aber belegt
        # fuer lange Dateien mehrere GB. Sofort freigeben.
        del stft_complex
        gc.collect()
        
        # RMS (Zeitbereich, profitiert trotzdem vom Pre-Emphasis)
        self._progress("Berechne Lautstaerke...", step := step + 1, total_steps, progress_callback)
        rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=n_fft)[0]
        rms = self._normalize(rms)
        
        # Onset (aus bereits windowed Power-Spektrogramm fuer Konsistenz)
        self._progress("Erkenne Beats...", step := step + 1, total_steps, progress_callback)
        onset_env = librosa.onset.onset_strength(S=stft**2, sr=sr)
        onset = self._normalize(onset_env)
        
        # Spektrale Features (aus bereits windowed STFT)
        self._progress("Analysiere Spektrum...", step := step + 1, total_steps, progress_callback)
        spec_cent = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
        spec_roll = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
        
        # Chroma (aus bereits windowed STFT)
        self._progress("Erkenne Tonart...", step := step + 1, total_steps, progress_callback)
        chroma_raw = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma = np.zeros((12, expected_frames), dtype=np.float32)
        for i in range(12):
            chroma[i, :] = self._interpolate_to_length(chroma_raw[i, :], expected_frames)
        
        # NEU: Transient-Detection (fuer Kick/Snare)
        self._progress("Erkenne Transienten...", step := step + 1, total_steps, progress_callback)
        transient = self._detect_transients(y, sr, hop_length, expected_frames)
        
        # NEU: Voice Clarity (80Hz - 3kHz Band)
        self._progress("Analysiere Sprach-Präsenz...", step := step + 1, total_steps, progress_callback)
        voice_clarity = self._detect_voice_clarity(y, sr, hop_length, expected_frames, stft=stft)
        
        # NEU: Voice Band (80Hz - 3kHz fuer gezielte Sprach-Triggerung)
        self._progress("Analysiere Sprach-Band...", step := step + 1, total_steps, progress_callback)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        idx_80 = np.searchsorted(freqs, 80.0)
        idx_3k = np.searchsorted(freqs, 3000.0)
        idx_80 = max(1, min(idx_80, len(freqs) - 1))
        idx_3k = max(idx_80 + 1, min(idx_3k, len(freqs)))
        voice_band_raw = np.mean(stft[idx_80:idx_3k, :], axis=0)
        voice_band = self._normalize(voice_band_raw)
        voice_band = self._interpolate_to_length(voice_band, expected_frames)
        
        # Tempo & Mode
        self._progress("Klassifiziere Audio-Typ...", step := step + 1, total_steps, progress_callback)
        tempo = self._estimate_tempo_simple(onset_env, sr, hop_length)
        mode = self._detect_mode_advanced(spec_cent, voice_clarity, rms)
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
        
        mfcc_raw = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc = np.zeros((13, expected_frames), dtype=np.float32)
        for i in range(13):
            mfcc[i, :] = self._interpolate_to_length(mfcc_raw[i, :], expected_frames)
        
        tempogram_raw = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        tempogram = np.zeros((tempogram_raw.shape[0], expected_frames), dtype=np.float32)
        for i in range(tempogram_raw.shape[0]):
            tempogram[i, :] = self._interpolate_to_length(tempogram_raw[i, :], expected_frames)
        
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
            voice_band=voice_band,
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
    
    def _detect_voice_clarity(self, y: np.ndarray, sr: int, hop_length: int, target_frames: int, stft: np.ndarray = None) -> np.ndarray:
        """
        Misst die Energie im Sprach-Band (80Hz - 3kHz) relativ zur Gesamtenergie.
        Hoeher = mehr Sprache, niedriger = mehr Musik/Noise.
        """
        # STFT fuer Frequenz-Analyse
        if stft is None:
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
            beat_video_frames = np.round(beat_times / video_frame_duration).astype(np.int32)
            
            # Entferne Duplikate und clamp auf gueltigen Bereich
            beat_video_frames = np.unique(beat_video_frames)
            beat_video_frames = beat_video_frames[beat_video_frames < expected_frames]
            
            return beat_video_frames.astype(np.int32)
        except Exception:
            return np.array([], dtype=np.int32)
    
    # Plausibler BPM-Bereich; alles ausserhalb gilt als Fehlschaetzung
    TEMPO_MIN_BPM = 40.0
    TEMPO_MAX_BPM = 250.0
    TEMPO_FALLBACK_BPM = 120.0

    def _estimate_tempo_simple(self, onset_env: np.ndarray, sr: int, hop_length: int) -> float:
        """Schaetzt das Tempo in BPM.

        Bevorzugt librosas eigenen Schaetzer (log-normaler Prior um 120 BPM,
        das faengt Oktav-Fehler ab). Faellt auf das Tempogram-Maximum zurueck,
        wobei ungueltige Lags ausmaskiert werden muessen: der Lag 0 hat immer
        die groesste Energie und entspricht inf BPM — ein blankes argmax
        liefert deshalb ausnahmslos den Fallback-Wert.
        """
        try:
            tempo = float(np.atleast_1d(
                _librosa_tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
            )[0])
            if self._tempo_plausible(tempo):
                return tempo
        except Exception:
            logger.warning("Tempo-Schaetzung ueber librosa fehlgeschlagen, nutze Tempogram",
                           exc_info=True)

        try:
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=96
            )
            tg_mean = np.mean(tempogram, axis=1)
            bpms = librosa.tempo_frequencies(len(tg_mean), hop_length=hop_length, sr=sr)
            valid = np.isfinite(bpms) & (bpms >= self.TEMPO_MIN_BPM) & (bpms <= self.TEMPO_MAX_BPM)
            if not np.any(valid):
                return self.TEMPO_FALLBACK_BPM
            masked = np.where(valid, tg_mean, -np.inf)
            tempo = float(bpms[int(np.argmax(masked))])
            if self._tempo_plausible(tempo):
                return tempo
        except Exception:
            pass

        return self.TEMPO_FALLBACK_BPM

    def _tempo_plausible(self, tempo: float) -> bool:
        """True, wenn der BPM-Wert endlich und im plausiblen Bereich liegt."""
        return bool(np.isfinite(tempo)) and self.TEMPO_MIN_BPM <= tempo <= self.TEMPO_MAX_BPM
    
    # Stuetzpunkte der Modus-Erkennung, gemessen am Golden-Korpus
    # (3 Musik-, 3 Podcast-Dateien, siehe docs/internal/mode-detection.md).
    # Jeweils (Sprache-Kante, Musik-Kante) — dazwischen wird linear ueberblendet.
    # ACHTUNG: cent_mean gilt fuer das Spektrum NACH dem Pre-Emphasis-Filter
    # (analyze() hebt Hoehen um ~6 dB/Oktave an). Sprache liegt roh bei
    # 500-1500 Hz, hier aber bei ~5200 Hz.
    MODE_RMS_VAR_EDGES = (0.10, 0.17)        # wenig Dynamik -> Sprache
    MODE_VOICE_MEAN_EDGES = (0.35, 0.28)     # viel Sprachband-Anteil -> Sprache
    MODE_CENT_MEAN_EDGES = (5400.0, 6300.0)  # tieferer Schwerpunkt -> Sprache
    # Ab welchem Gesamtscore die Entscheidung eindeutig ist. 0.30 liegt knapp
    # unter 1/3: zwei voll ausgeschlagene Merkmale setzen sich damit gegen ein
    # gegenlaeufiges drittes durch, echte Mischfaelle bleiben 'hybrid'.
    MODE_DECISION_MARGIN = 0.30

    @staticmethod
    def _mode_feature_score(value: float, speech_edge: float, music_edge: float) -> float:
        """Bewertet ein Merkmal: +1 = klar Sprache, -1 = klar Musik.

        Zwischen den beiden Kanten wird linear ueberblendet, damit
        Grenzfaelle nicht an einer harten Schwelle kippen.
        """
        span = speech_edge - music_edge
        if abs(span) < 1e-9:
            return 0.0
        t = float(np.clip((value - music_edge) / span, 0.0, 1.0))
        return t * 2.0 - 1.0

    def _detect_mode_advanced(self, spec_cent: np.ndarray, voice_clarity: np.ndarray,
                              rms: np.ndarray) -> str:
        """Klassifiziert das Material als 'music', 'speech' oder 'hybrid'.

        Bewertet drei Merkmale einzeln und mittelt sie, statt harte
        UND-Ketten zu verlangen: ein einzelner Ausreisser kippt die
        Entscheidung dann nicht mehr, und unklare Faelle landen bewusst
        auf 'hybrid'.

        Tempo und Onset-Streuung gehen bewusst NICHT ein: Sprache bekommt
        vom Tempo-Schaetzer ebenso plausible BPM-Werte wie Musik, und die
        Onset-Streuung trennt am Korpus nicht (Podcasts streuen dort sogar
        staerker als Musik).
        """
        cent_mean = float(np.mean(spec_cent))
        voice_mean = float(np.mean(voice_clarity))
        rms_var = float(np.std(rms))

        score = float(np.mean([
            self._mode_feature_score(rms_var, *self.MODE_RMS_VAR_EDGES),
            self._mode_feature_score(voice_mean, *self.MODE_VOICE_MEAN_EDGES),
            self._mode_feature_score(cent_mean, *self.MODE_CENT_MEAN_EDGES),
        ]))

        if score > self.MODE_DECISION_MARGIN:
            return "speech"
        if score < -self.MODE_DECISION_MARGIN:
            return "music"
        return "hybrid"
    
    def _estimate_key(self, chroma: np.ndarray) -> str:
        """
        Schaetzt Tonart mit Krumhansl-Schmuckler Key-Profilen.
        Beruecksichtigt Major und Minor, waehlt den besten Korrelations-Score.
        """
        chroma_avg = np.mean(chroma, axis=1)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Krumhansl-Schmuckler Key-Profile (normalisiert)
        major_profile = np.array([
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
        ])
        minor_profile = np.array([
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
        ])
        
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        chroma_norm = chroma_avg / (np.sum(chroma_avg) + 1e-10)
        
        best_score = -np.inf
        best_key = "C major"
        
        for shift in range(12):
            # Major-Korrelation
            major_shifted = np.roll(major_profile, shift)
            with np.errstate(invalid='ignore', divide='ignore'):
                major_score = np.corrcoef(chroma_norm, major_shifted)[0, 1]
            if not np.isnan(major_score) and major_score > best_score:
                best_score = major_score
                best_key = f"{keys[shift]} major"

            # Minor-Korrelation
            minor_shifted = np.roll(minor_profile, shift)
            with np.errstate(invalid='ignore', divide='ignore'):
                minor_score = np.corrcoef(chroma_norm, minor_shifted)[0, 1]
            if not np.isnan(minor_score) and minor_score > best_score:
                best_score = minor_score
                best_key = f"{keys[shift]} minor"

        # Wenn keine sinnvolle Korrelation gefunden (z.B. Stille), None zurueckgeben
        if np.isinf(best_score) or np.isnan(best_score):
            return None
        return best_key
    
    _NONE_SENTINEL = "__NONE__"

    def _save_cache(self, path: Path, features: AudioFeatures):
        data = {}
        for k, v in features.model_dump().items():
            if v is None:
                data[k] = np.array(self._NONE_SENTINEL, dtype='<U100')
            elif isinstance(v, str):
                data[k] = np.array(v, dtype='<U100')
            else:
                data[k] = v
        np.savez_compressed(path, **data)
