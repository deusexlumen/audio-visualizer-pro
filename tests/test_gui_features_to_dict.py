import numpy as np
import pytest

from src.gui.helpers import _features_to_dict
from src.types import AudioFeatures


def test_features_to_dict_uses_real_statistics():
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
    d = _features_to_dict(features)
    assert d['rms_mean'] == pytest.approx(0.2)
    assert d['onset_mean'] == pytest.approx(0.5)
    assert d['tempo'] == 128.0
    assert d['mode'] == 'music'
