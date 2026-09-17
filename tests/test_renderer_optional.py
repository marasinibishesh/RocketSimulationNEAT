"""Renderer smoke tests run automatically when Panda3D is installed."""
import pytest

pytest.importorskip("panda3d")

from falcon9_neat.render.rocket_model import build_booster, build_full_vehicle


def test_procedural_rocket_geometry_builds_without_external_assets():
    booster, exhaust = build_booster()
    full = build_full_vehicle()
    assert booster.getNumChildren() > 0
    assert exhaust.getParent() == booster
    assert full.getNumChildren() > 0
