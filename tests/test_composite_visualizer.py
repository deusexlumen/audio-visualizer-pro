"""Tests fuer den rezeptbasierten CompositeVisualizer und die Rezept-Registry."""

import json

import numpy as np
import pytest

from config.schemas import RecipeSchema
from src.gpu_visualizers.composite import make_recipe_visualizer_class, build_params_for_recipe
from src.gpu_visualizers.blocks import BLOCK_LIBRARY


@pytest.fixture(scope="module")
def gl_context():
    """Ein ModernGL-Kontext fuer das ganze Modul."""
    import moderngl
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"Keine GPU/OpenGL verfuegbar: {e}")
    try:
        ctx.__enter__()
    except Exception:
        pass
    yield ctx
    ctx.release()


@pytest.fixture
def dummy_features():
    return {
        "rms": np.random.rand(30).astype(np.float32),
        "onset": np.random.rand(30).astype(np.float32),
        "beat_intensity": np.random.rand(30).astype(np.float32),
        "spectral_centroid": np.random.rand(30).astype(np.float32),
        "chroma": np.random.rand(12, 30).astype(np.float32),
        "transient": np.random.rand(30).astype(np.float32),
        "voice_clarity": np.random.rand(30).astype(np.float32),
        "fps": 30, "frame_count": 30, "mode": "music", "tempo": 120.0,
    }


def _recipe(layers):
    return RecipeSchema(name="test_recipe", layers=layers).model_dump()


def test_build_params_namespaced():
    recipe = _recipe([
        {"block": "ring", "params": {"radius": 0.3}},
        {"block": "core_glow"},
    ])
    params, groups = build_params_for_recipe(recipe)
    assert "l0_radius" in params
    assert "l1_size" in params
    assert any("Ring" in g for g in groups)
    # Rezept-Default wird uebernommen
    assert params["l0_radius"][0] == 0.3


@pytest.mark.gpu
def test_composite_rendert_nicht_schwarz(gl_context, dummy_features):
    ctx = gl_context
    tex = ctx.texture((256, 144), 3, dtype="f2")
    fbo = ctx.framebuffer(color_attachments=[tex])
    try:
        recipe = _recipe([
            {"block": "core_glow", "blend": "add"},
            {"block": "ring", "blend": "screen"},
            {"block": "particles", "blend": "add"},
        ])
        cls = make_recipe_visualizer_class(recipe)
        viz = cls(ctx, 256, 144)
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0)
        viz.render(dummy_features, 0.5)
        raw = np.frombuffer(fbo.read(components=3, dtype="f2"), dtype=np.float16).astype(np.float32)
        assert not np.isnan(raw).any()
        assert float(np.ptp(raw)) > 0.01
    finally:
        fbo.release(); tex.release()


@pytest.mark.gpu
def test_jeder_baustein_rendert_solo(gl_context, dummy_features):
    ctx = gl_context
    tex = ctx.texture((256, 144), 3, dtype="f2")
    fbo = ctx.framebuffer(color_attachments=[tex])
    try:
        for block_name in BLOCK_LIBRARY:
            recipe = _recipe([{"block": block_name, "blend": "add"}])
            cls = make_recipe_visualizer_class(recipe)
            viz = cls(ctx, 256, 144)
            fbo.use()
            ctx.clear(0.0, 0.0, 0.0)
            viz.render(dummy_features, 0.5)
            raw = np.frombuffer(fbo.read(components=3, dtype="f2"), dtype=np.float16).astype(np.float32)
            assert not np.isnan(raw).any(), f"{block_name}: NaN"
    finally:
        fbo.release(); tex.release()


def test_registry_entdeckt_beispielrezept():
    from src.gpu_visualizers import refresh_registry, list_visualizers
    refresh_registry()
    assert "beispiel_puls" in list_visualizers()


def test_gemini_suggest_recipe_filtert(monkeypatch):
    from unittest.mock import Mock, patch
    with patch("src.gemini_integration.genai") as mock_genai:
        from src.gemini_integration import GeminiIntegration
        client = Mock()
        resp = Mock()
        resp.text = (
            '[{"block":"ring","blend":"add","mappings":[{"target":"radius","source":"u_energy","gain":0.2}]},'
            '{"block":"UNBEKANNT","blend":"add"},'
            '{"block":"core_glow","blend":"quatsch"}]'
        )
        resp.usage_metadata = None
        client.models.generate_content.return_value = resp
        mock_genai.Client.return_value = client
        g = GeminiIntegration(api_key="test")
        layers = g.suggest_recipe("heller kern mit ring", ["ring", "core_glow"], use_cache=False)
        assert len(layers) == 2
        assert layers[0]["block"] == "ring"
        assert layers[0]["mappings"][0]["target"] == "radius"
        # Ungueltiger blend wird auf 'add' korrigiert
        assert layers[1]["blend"] == "add"


def test_kaputtes_rezept_wird_uebersprungen(tmp_path, monkeypatch):
    from src.gpu_visualizers import _discover_recipe_visualizers
    import src.gpu_visualizers as reg

    bad = tmp_path / "kaputt.json"
    bad.write_text("{ das ist kein json")
    good = tmp_path / "gut.json"
    good.write_text(json.dumps({
        "name": "gut_test", "layers": [{"block": "ring"}]
    }))
    monkeypatch.setattr(reg, "recipe_dirs", lambda: [tmp_path])
    result = _discover_recipe_visualizers()
    assert "gut_test" in result
    assert "kaputt" not in result
