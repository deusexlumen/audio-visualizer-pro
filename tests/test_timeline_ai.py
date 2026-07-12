"""Tests fuer die KI-Timeline: offline (SmartMatcher) und Gemini-Verfeinerung."""

import numpy as np
from unittest.mock import Mock, patch

from src.ai_matcher import SmartMatcher
from src.segmentation import Segment
from src.types import AudioFeatures, Timeline


def _features(seconds=30, fps=30):
    n = seconds * fps
    return AudioFeatures(
        duration=seconds, sample_rate=22050, fps=fps, frame_count=n,
        rms=np.random.rand(n).astype(np.float32),
        onset=np.random.rand(n).astype(np.float32),
        spectral_centroid=np.random.rand(n).astype(np.float32),
        spectral_rolloff=np.random.rand(n).astype(np.float32),
        zero_crossing_rate=np.random.rand(n).astype(np.float32),
        transient=np.random.rand(n).astype(np.float32),
        voice_clarity=np.zeros(n, dtype=np.float32),
        voice_band=np.zeros(n, dtype=np.float32),
        chroma=np.random.rand(12, n).astype(np.float32),
        mfcc=np.random.rand(13, n).astype(np.float32),
        tempogram=np.random.rand(384, n).astype(np.float32),
        tempo=128.0, key="C", mode="music",
        beat_frames=np.arange(0, n, fps).astype(np.int64),
    )


def _segments():
    return [
        Segment(0.0, 10.0, {"rms": 0.2, "onset_density": 0.1, "dominant_chroma": "C"}),
        Segment(10.0, 20.0, {"rms": 0.8, "onset_density": 0.5, "dominant_chroma": "G"}),
        Segment(20.0, 30.0, {"rms": 0.3, "onset_density": 0.2, "dominant_chroma": "A"}),
    ]


def test_suggest_timeline_offline_deckt_alle_segmente():
    matcher = SmartMatcher()
    tl = matcher.suggest_timeline(_features(), _segments())
    assert isinstance(tl, Timeline)
    assert len(tl.scenes) == 3
    assert tl.scenes[0].start == 0.0
    assert tl.scenes[-1].end == 30.0
    # Erste Szene ohne Crossfade
    assert tl.scenes[0].transition == "cut"
    assert tl.scenes[1].transition == "crossfade"


def test_suggest_timeline_lueckenlos_und_gueltige_visualizer():
    from src.gpu_visualizers import list_visualizers
    known = set(list_visualizers())
    tl = SmartMatcher().suggest_timeline(_features(), _segments())
    for a, b in zip(tl.scenes, tl.scenes[1:]):
        assert abs(a.end - b.start) < 1e-6
    for s in tl.scenes:
        assert s.visualizer in known


def test_suggest_timeline_ohne_segmente_eine_szene():
    tl = SmartMatcher().suggest_timeline(_features(), [])
    assert len(tl.scenes) == 1


def test_gemini_timeline_parst_und_filtert():
    with patch("src.gemini_integration.genai") as mock_genai:
        from src.gemini_integration import GeminiIntegration
        mock_client = Mock()
        resp = Mock()
        resp.text = (
            '[{"index":0,"visualizer":"lumina_core","label":"Intro"},'
            '{"index":1,"visualizer":"UNBEKANNT","label":"X"},'
            '{"index":2,"visualizer":"nebula_drift","label":"Drop"}]'
        )
        resp.usage_metadata = None
        mock_client.models.generate_content.return_value = resp
        mock_genai.Client.return_value = mock_client

        g = GeminiIntegration(api_key="test")
        result = g.generate_scene_timeline(
            segments_stats=[{"start": 0, "end": 10}, {"start": 10, "end": 20}, {"start": 20, "end": 30}],
            available_visualizers=["lumina_core", "nebula_drift"],
            audio_path=None,
            use_cache=False,
        )
        # Unbekannter Visualizer wird herausgefiltert
        assert len(result) == 2
        assert result[0]["visualizer"] == "lumina_core"
        assert result[1]["visualizer"] == "nebula_drift"
