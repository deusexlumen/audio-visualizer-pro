# GUI-Render Memory-Hang & Gemini-Zitate – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the full GUI video export stable and memory-bounded for long audio files, and make the Gemini key-quote extraction robust against invalid/unavailable models and malformed JSON.

**Architecture:**
1. Replace the unbounded `queue.Queue()` between GPU frame producer and FFmpeg encoder in `GPUBatchRenderer` with a small bounded queue (`maxsize=3`) and a cancellation-aware `put` loop. This forces the producer to wait for the consumer, capping RAM to a few frames.
2. Make `GeminiIntegration` model-agnostic: read `GEMINI_MODEL` from the environment, default to a stable audio-capable model (`gemini-1.5-flash`), and iterate over a fallback list on "model not found" errors. Also normalize the `config` argument to `google.genai.types.GenerateContentConfig` for SDK 2.x compatibility.
3. Fix the GUI's `_features_to_dict` helper so KI optimization receives real audio-feature statistics instead of hard-coded defaults.
4. Add focused automated tests for the bounded queue and the model fallback.

**Tech Stack:** Python 3.11, ModernGL, FFmpeg, `google-genai` 2.x, pytest, DearPyGui

---

## Task 1: Cap the producer-consumer frame queue in `src/gpu_renderer.py`

**Files:**
- Modify: `src/gpu_renderer.py:242-265` (queue creation + encode worker) and `src/gpu_renderer.py:362` (frame put)
- Test: `tests/test_gpu_renderer.py` (add new test class)

### Background
`GPUBatchRenderer.render()` currently uses `frame_queue = queue.Queue()` (unbounded). For a 17-minute 1920×1080 video this can queue thousands of raw frames (≈6 MiB each) while FFmpeg lags behind, causing the RAM spike and GUI hang observed by the user.

### Step 1: Replace the queue creation

Change:

```python
            # === PRODUCER-CONSUMER: Render und Encode parallel ===
            # Der Render-Thread rendert Frames in eine Queue.
            # Ein separater Thread schreibt sie zu FFmpeg stdin.
            # Queue OHNE maxsize: put() blockiert nie, FFmpeg ist der einzige Engpass.
            frame_queue = queue.Queue()
```

to:

```python
            # === PRODUCER-CONSUMER: Render und Encode parallel ===
            # Der Render-Thread rendert Frames in eine Queue.
            # Ein separater Thread schreibt sie zu FFmpeg stdin.
            # Queue MIT maxsize: Producer wartet auf Encoder, RAM bleibt konstant.
            frame_queue = queue.Queue(maxsize=3)
```

### Step 2: Make `frame_queue.put` cancellation-aware

Replace the single line:

```python
                    frame_queue.put(pixels)
```

(located near the end of the main render loop) with:

```python
                    # Bei voller Queue blockieren, aber Abbruch regelmaessig pruefen
                    while True:
                        try:
                            frame_queue.put(pixels, timeout=0.1)
                            break
                        except queue.Full:
                            if cancel_event is not None and cancel_event.is_set():
                                print("[GPU] Render abgebrochen durch User (Queue voll).")
                                break
                            # Sonst kurz warten und erneut versuchen
                    if cancel_event is not None and cancel_event.is_set():
                        break
```

### Step 3: Run existing renderer tests

```bash
pytest tests/test_gpu_renderer.py -v
```

Expected: all pass.

### Step 4: Commit

```bash
git add src/gpu_renderer.py
git commit -m "fix(renderer): begrenze Producer-Consumer Queue auf 3 Frames"
```

---

## Task 2: Add automated test for bounded queue behavior

**Files:**
- Create: `tests/test_render_queue_bound.py`

### Step 1: Write the test

```python
"""
Test that the render producer blocks on a full frame queue and never
grows unbounded, even when the encoder is artificially slowed down.
"""
import queue
import subprocess
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.gpu_renderer import GPUBatchRenderer


class TestBoundedFrameQueue:
    """Regression test for unbounded frame queue memory spike."""

    def test_render_queue_never_exceeds_maxsize(self, tmp_path):
        """A slow encoder must not let the frame queue grow beyond maxsize."""
        observed_max = [0]
        original_put = queue.Queue.put

        def tracking_put(self, item, block=True, timeout=None):
            # Track actual max size seen during the render
            observed_max[0] = max(observed_max[0], self.qsize())
            return original_put(self, item, block=block, timeout=timeout)

        # Fake FFmpeg process that writes very slowly
        fake_proc = Mock()
        fake_proc.poll.return_value = None
        fake_proc.returncode = 0
        fake_proc.stdin = Mock()

        write_delay = [0.05]

        def slow_write(data):
            time.sleep(write_delay[0])

        fake_proc.stdin.write.side_effect = slow_write

        # Tiny audio features to keep the test fast
        from src.types import AudioFeatures
        frame_count = 120
        features = AudioFeatures(
            duration=frame_count / 30.0,
            sample_rate=44100,
            fps=30,
            frame_count=frame_count,
            rms=np.random.rand(frame_count).astype(np.float32),
            onset=np.random.rand(frame_count).astype(np.float32),
            spectral_centroid=np.random.rand(frame_count).astype(np.float32),
            spectral_rolloff=np.random.rand(frame_count).astype(np.float32),
            zero_crossing_rate=np.random.rand(frame_count).astype(np.float32),
            transient=np.random.rand(frame_count).astype(np.float32),
            voice_clarity=np.random.rand(frame_count).astype(np.float32),
            voice_band=np.random.rand(frame_count).astype(np.float32),
            chroma=np.random.rand(12, frame_count).astype(np.float32),
            mfcc=np.random.rand(13, frame_count).astype(np.float32),
            tempogram=np.random.rand(384, frame_count).astype(np.float32),
            tempo=120.0,
            key="C major",
            mode="music",
            beat_frames=np.array([], dtype=np.int32),
        )

        output_path = str(tmp_path / "bounded_queue.mp4")

        observed_max = [0]

        def make_tracking_put(original_put):
            def tracking_put(self, item, block=True, timeout=None):
                observed_max[0] = max(observed_max[0], self.qsize())
                return original_put(self, item, block=block, timeout=timeout)
            return tracking_put

        with patch.object(queue.Queue, "put", make_tracking_put(queue.Queue.put)):
            renderer = GPUBatchRenderer(width=320, height=180, fps=30)
            with patch("subprocess.Popen", return_value=fake_proc):
                renderer.render(
                    audio_path="dummy.mp3",
                    visualizer_type="voice_flow",
                    output_path=output_path,
                    features=features,
                    preview_mode=True,
                    preview_duration=frame_count / 30.0,
                    quality="low",
                )

        assert observed_max[0] <= 3, (
            f"Queue grew to {observed_max[0]} items, expected <= 3"
        )
```

### Step 2: Run the new test

```bash
pytest tests/test_render_queue_bound.py -v
```

Expected: PASS.

### Step 3: Commit

```bash
git add tests/test_render_queue_bound.py
git commit -m "test(renderer): verify frame queue stays bounded with slow encoder"
```

---

## Task 3: Make Gemini model configurable and add fallback

**Files:**
- Modify: `src/gemini_integration.py:121-145` (`__init__`), `src/gemini_integration.py:486-494` (`extract_quotes` generate_content call), `src/gemini_integration.py:393-398` (`transcribe_audio` generate_content call), `src/gemini_integration.py:603-610` and `src/gemini_integration.py:907-914` (`optimize_*` generate_content calls)
- Test: `tests/test_gemini_integration.py`

### Step 1: Add model constants and constructor parameter

At the top of the class, after imports, add:

```python
DEFAULT_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash",
    "gemini-3.1-flash-preview",
]
```

Change the constructor signature from:

```python
    def __init__(self, api_key: Optional[str] = None):
```

to:

```python
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
```

and replace:

```python
        self.model = "gemini-3.1-flash-lite-preview"
```

with:

```python
        self.model = model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
```

### Step 2: Add helper to normalize `config`

Add a static method:

```python
    @staticmethod
    def _make_config(response_mime_type: Optional[str] = None, **kwargs):
        """Baut ein SDK-kompatibles Config-Objekt (dict oder types)."""
        config = dict(kwargs)
        if response_mime_type:
            config["response_mime_type"] = response_mime_type
        try:
            from google.genai import types
            return types.GenerateContentConfig(**config)
        except Exception:
            # Fallback fuer aeltere oder abgespeckte SDK-Versionen
            return config
```

### Step 3: Add model-fallback caller

Add a new method:

```python
    def _generate_with_model_fallback(self, contents, response_mime_type: Optional[str] = None):
        """Fuehrt generate_content aus und probiert Fallback-Modelle bei 'not found'."""
        models_to_try = [self.model] + [
            m for m in self.FALLBACK_MODELS if m != self.model
        ]
        last_error = None
        for model_name in models_to_try:
            try:
                return self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=self._make_config(response_mime_type=response_mime_type),
                )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_model_error = (
                    "model" in err_str
                    and ("not found" in err_str or "invalid" in err_str or "not supported" in err_str)
                )
                if is_model_error:
                    print(f"[Gemini] Modell {model_name} nicht verfuegbar, probiere Fallback...")
                    continue
                # Nicht-modell-bezogene Fehler sofort weiterwerfen (Retry-Logik greift woanders)
                raise
        raise RuntimeError(
            f"Kein Gemini-Modell verfuegbar. Letzter Fehler: {last_error}"
        )
```

### Step 4: Replace all `client.models.generate_content` calls

Replace each occurrence of:

```python
            response = self._call_gemini_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, uploaded_file],
                    config={
                        "response_mime_type": "application/json",
                    }
                )
            )
```

with:

```python
            response = self._call_gemini_with_retry(
                lambda: self._generate_with_model_fallback(
                    contents=[prompt, uploaded_file],
                    response_mime_type="application/json",
                )
            )
```

There are three such blocks (transcribe, extract_quotes, optimize_all_settings). The `optimize_visualizer_params` block does **not** set `response_mime_type`; still wrap it with `_generate_with_model_fallback()` without that parameter.

### Step 5: Update tests to expect new default model and fallback

In `tests/test_gemini_integration.py`, change:

```python
            assert gemini.model == "gemini-3.1-flash-lite-preview"
```

to:

```python
            assert gemini.model == "gemini-1.5-flash"
```

Add a new test:

```python
    def test_model_fallback_on_not_found(self):
        """Wenn das konfigurierte Modell fehlt, soll ein Fallback verwendet werden."""
        with patch('src.gemini_integration.genai') as mock_genai:
            from src.gemini_integration import GeminiIntegration

            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            def side_effect(*, model, contents, config=None):
                if model == "gemini-1.5-flash":
                    raise RuntimeError("Model not found")
                resp = Mock()
                resp.text = '[{"text": "Fallback quote", "start_time": 1.0, "end_time": 3.0, "confidence": 0.9}]'
                return resp

            mock_client.models.generate_content.side_effect = side_effect

            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(b'dummy')
                tmp_path = tmp.name
            try:
                gemini = GeminiIntegration(api_key="test-key", model="gemini-1.5-flash")
                quotes = gemini.extract_quotes(tmp_path, audio_duration=60.0, max_quotes=1)
                assert len(quotes) == 1
                assert quotes[0].text == "Fallback quote"
                # Erster Versuch mit 1.5-flash, danach Fallback
                calls = [c.kwargs.get("model") or c.args[0] for c in mock_client.models.generate_content.call_args_list]
                assert "gemini-1.5-flash-8b" in calls or "gemini-2.0-flash" in calls
            finally:
                os.unlink(tmp_path)
```

### Step 6: Run Gemini tests

```bash
pytest tests/test_gemini_integration.py -v
```

Expected: all pass.

### Step 7: Commit

```bash
git add src/gemini_integration.py tests/test_gemini_integration.py
git commit -m "fix(gemini): stabiles Default-Modell, env-override und Fallback-Logik"
```

---

## Task 4: Fix `_features_to_dict` in `gui.py`

**Files:**
- Modify: `gui.py:3197-3209`

### Background
`_features_to_dict` reads `rms_mean`, `rms_std`, etc. from the `AudioFeatures` object, but those attributes do not exist, so KI optimization always receives hard-coded defaults.

### Step 1: Compute real statistics

Replace the method with:

```python
    def _features_to_dict(self, features) -> dict:
        def _mean(arr):
            arr = np.asarray(arr)
            return float(arr.mean()) if arr.size else 0.0

        def _std(arr):
            arr = np.asarray(arr)
            return float(arr.std()) if arr.size else 0.0

        return {
            'duration': float(getattr(features, 'duration', 0)),
            'tempo': float(getattr(features, 'tempo', 120)),
            'mode': str(getattr(features, 'mode', 'music')),
            'rms_mean': _mean(getattr(features, 'rms', [])),
            'rms_std': _std(getattr(features, 'rms', [])),
            'onset_mean': _mean(getattr(features, 'onset', [])),
            'onset_std': _std(getattr(features, 'onset', [])),
            'spectral_mean': _mean(getattr(features, 'spectral_centroid', [])),
            'transient_mean': _mean(getattr(features, 'transient', [])),
            'voice_clarity_mean': _mean(getattr(features, 'voice_clarity', [])),
        }
```

### Step 2: Add a focused test

Create `tests/test_gui_features_to_dict.py`:

```python
import numpy as np
import pytest

from gui import AudioVisualizerGUI
from src.types import AudioFeatures


def test_features_to_dict_uses_real_statistics():
    gui = AudioVisualizerGUI()
    features = AudioFeatures(
        duration=10.0,
        sample_rate=44100,
        fps=30,
        rms=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        onset=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        spectral_centroid=np.array([0.7, 0.8, 0.9], dtype=np.float32),
        spectral_rolloff=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        zero_crossing_rate=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        transient=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        voice_clarity=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        voice_band=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        chroma=np.zeros((12, 3), dtype=np.float32),
        mfcc=np.zeros((13, 3), dtype=np.float32),
        tempogram=np.zeros((384, 3), dtype=np.float32),
        tempo=128.0,
        key="C major",
        mode="music",
        beat_frames=np.array([], dtype=np.int32),
    )
    d = gui._features_to_dict(features)
    assert d['rms_mean'] == pytest.approx(0.2)
    assert d['onset_mean'] == pytest.approx(0.5)
    assert d['tempo'] == 128.0
    assert d['mode'] == 'music'
```

### Step 3: Run the test

```bash
pytest tests/test_gui_features_to_dict.py -v
```

Expected: PASS.

### Step 4: Commit

```bash
git add gui.py tests/test_gui_features_to_dict.py
git commit -m "fix(gui): KI-Optimierung nutzt echte Audio-Feature-Statistiken"
```

---

## Task 5: Full regression test run

**Files:** all touched + existing tests

### Step 1: Run the full test suite

```bash
pytest tests/ -v --timeout=300
```

Expected: all pass.

### Step 2: Run a CLI full-render smoke test (optional but recommended)

```bash
python main.py render \
  "assets/user_uploads/Die_Macy-Stiftung_als_Fundament_der_Kontrolle.m4a_33410687_Die_Macy-Stiftung_als_Fundament_der_Kontrolle.m4a" \
  --visual voice_flow \
  --resolution 854x480 \
  --fps 30 \
  --quality medium \
  -o output/smoke_test_cli.mp4
```

Expected: completes without memory spike; output file exists.

### Step 3: Commit any final cleanup

```bash
git add -A
git commit -m "test: volle Regression nach Render-Memory und Gemini-Fixes"
```

---

## Spec Coverage Checklist

| User requirement | Covered by |
|------------------|------------|
| GUI does not hang and RAM does not spike during long renders | Task 1, Task 2 |
| KI quote extraction is no longer buggy / unstable model | Task 3 |
| KI parameter optimization uses real audio data | Task 4 |
| Tool is 100% functional (tests + smoke test) | Task 5 |

## Placeholder Scan

- No `TBD`, `TODO`, or placeholder code in tasks.
- All code snippets are concrete and map to existing file locations.
- Test commands and expected outputs are explicit.

## Type Consistency Notes

- `_generate_with_model_fallback` signature matches existing `_call_gemini_with_retry` usage.
- `_make_config` returns either `GenerateContentConfig` or `dict`; `google-genai` 2.x accepts both.
- `_features_to_dict` returns the same keys as before, but with real values.
