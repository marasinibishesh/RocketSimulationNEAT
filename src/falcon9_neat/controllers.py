"""Baseline controllers useful for smoke tests and visual demos."""
from __future__ import annotations

import numpy as np


class HeuristicLandingController:
    """A compact PD guidance/autopilot baseline for pre-training demos.

    It is intentionally separate from NEAT.  The baseline proves that the
    environment is controllable and makes the 3-D demo useful before an evolved
    checkpoint exists.  NEAT does not receive or imitate these control laws.
    """

    def act(self, obs: np.ndarray) -> np.ndarray:
        x, y, alt, vx, vy, vz = obs[:6]
        upx, upy = obs[6], obs[7]
        wx, wy = obs[9] * 0.5, obs[10] * 0.5
        fuel_fraction = float(obs[12])

        # Undo observation normalization.
        xm, ym = x * 500.0, y * 500.0
        altm = max(0.0, alt * 1000.0)
        vxm, vym, vzm = vx * 80.0, vy * 80.0, vz * 100.0

        # Vertical guidance follows a descent-speed envelope that approaches a
        # ~1 m/s touchdown.  Hover throttle is estimated from remaining mass.
        desired_vz = -max(1.0, min(50.0, 0.070 * altm + 0.8))
        hover_throttle = 0.297 + 0.104 * fuel_fraction
        throttle = hover_throttle + 0.018 * (desired_vz - vzm)

        # Horizontal guidance asks for a bounded tilt vector.
        ax_cmd = np.clip(-0.0060 * xm - 0.115 * vxm, -2.8, 2.8)
        ay_cmd = np.clip(-0.0060 * ym - 0.115 * vym, -2.8, 2.8)
        desired_up_x = float(np.clip(ax_cmd / 10.5, -0.23, 0.23))
        desired_up_y = float(np.clip(ay_cmd / 10.5, -0.23, 0.23))

        # Attitude PD.  Signs follow the simulator's gimbal/torque convention.
        pitch = -4.2 * (upy - desired_up_y) + 1.25 * wx
        yaw = 4.2 * (upx - desired_up_x) + 1.25 * wy
        return np.array(
            [np.clip(throttle, 0.0, 1.0), np.clip(pitch, -1.0, 1.0), np.clip(yaw, -1.0, 1.0)],
            dtype=float,
        )
