"""Tests fuer das Studio-Rezept-Schema und die Baustein-Bibliothek."""

import pytest
from pydantic import ValidationError

from config.schemas import RecipeSchema, RecipeLayerSchema
from src.gpu_visualizers.blocks import BLOCK_LIBRARY, BLEND_MODES, block_names


def test_block_library_hat_pflichtfelder():
    for name, block in BLOCK_LIBRARY.items():
        assert "glsl" in block and f"block_{name}" in block["glsl"]
        assert "params" in block and block["params"]
        assert "arg_order" in block
        # Jede arg_order-Position hat einen Param
        for arg in block["arg_order"]:
            assert arg in block["params"]


def test_gueltiges_rezept():
    r = RecipeSchema(
        name="mein_test",
        layers=[
            RecipeLayerSchema(block="ring", blend="add", params={"radius": 0.3}),
            RecipeLayerSchema(block="core_glow", blend="screen"),
        ],
    )
    assert r.display_name  # automatisch gesetzt
    assert len(r.layers) == 2


def test_rezept_ungueltiger_name():
    with pytest.raises(ValidationError):
        RecipeSchema(name="Mein Rezept")  # Grossbuchstaben/Leerzeichen verboten


def test_layer_unbekannter_baustein():
    with pytest.raises(ValidationError):
        RecipeLayerSchema(block="gibtsnicht")


def test_blend_modi_vorhanden():
    assert set(BLEND_MODES.keys()) == {"add", "screen", "max"}
    assert "ring" in block_names()
