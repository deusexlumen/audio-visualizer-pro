"""Video-Hintergrund: Degradation statt Abbruch (Spec §14, §15, C17)."""

from unittest.mock import MagicMock, patch

import pytest

from src.studio.engine import is_video_background

pytestmark = pytest.mark.gpu


def test_is_video_background():
    assert is_video_background("clip.mp4") is True
    assert is_video_background("clip.MKV") is True
    assert is_video_background("bild.png") is False


def test_video_hintergrund_degradiert_statt_abbruch(tmp_path, dummy_audio_features):
    from src.gpu_renderer import GPUBatchRenderer
    from src.render_common import build_features_dict
    from src.studio.constraints import ConstraintSet
    from src.studio.engine import run_studio

    features_dict = build_features_dict(
        dummy_audio_features, dummy_audio_features.frame_count, 30
    )
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    with patch.object(GPUBatchRenderer, "render",
                      MagicMock(side_effect=RuntimeError("mock"))), \
         patch("src.gpu_renderer.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        sidecar = run_studio(
            str(audio), "spectrum_bars", dummy_audio_features, features_dict,
            str(out), constraints=ConstraintSet(max_overlay_alpha=1.0),
            background_image=str(tmp_path / "hintergrund.mp4"),
        )

    # Explizit KEIN Abbruch: Lauf erfolgreich, M3 deaktiviert (Spec §15)
    assert sidecar["mask"]["provider"] == "none:video_background"
    assert sidecar["verify"]["metrics"]["M3"] is None
    assert any("video" in w.lower()
               for w in sidecar["mask"].get("warnings", []))


def test_strict_ohne_provider_wirft(tmp_path, monkeypatch):
    from PIL import Image
    from src.studio import mask_service

    monkeypatch.setattr(mask_service, "_try_rembg", lambda img: None)
    monkeypatch.setattr(mask_service, "_try_opencv", lambda img: None)
    img = tmp_path / "bg.png"
    Image.new("RGB", (32, 32)).save(img)
    with pytest.raises(RuntimeError, match="strict"):
        mask_service.get_subject_mask(str(img),
                                      cache_dir=str(tmp_path / "c"),
                                      strict=True)
