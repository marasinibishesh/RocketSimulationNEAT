import numpy as np

from falcon9_neat.config import load_config
from falcon9_neat.environment import FalconLandingEnv


def test_environment_shapes_and_finite_step():
    cfg = load_config(None)
    env = FalconLandingEnv(cfg, seed=123)
    obs = env.reset(123)
    assert obs.shape == (13,)
    result = env.step(np.array([0.5, 0.0, 0.0]))
    assert result.observation.shape == (13,)
    assert np.all(np.isfinite(result.observation))
    assert np.isfinite(result.reward)
    assert "altitude_m" in result.info


def test_seeded_resets_are_reproducible():
    cfg = load_config(None)
    a = FalconLandingEnv(cfg, seed=9)
    b = FalconLandingEnv(cfg, seed=9)
    assert np.allclose(a.reset(99), b.reset(99))


def test_baseline_controller_lands_deterministic_case():
    from falcon9_neat.controllers import HeuristicLandingController

    cfg = load_config(None)
    env = FalconLandingEnv(cfg, deterministic=True)
    controller = HeuristicLandingController()
    obs = env.reset()
    result = None
    for _ in range(1500):
        result = env.step(controller.act(obs))
        obs = result.observation
        if result.terminated:
            break
    assert result is not None and result.terminated
    assert result.info["success"] is True
    assert result.info["lateral_distance_m"] < cfg.environment.pad_radius_m
