"""Training environment for autonomous first-stage landing."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from .config import AppConfig
from .math3d import angle_between, quat_from_euler
from .physics import Control, DynamicsInfo, RocketState, body_up, bottom_altitude_m, step_dynamics


@dataclass(slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    info: dict[str, Any]


class FalconLandingEnv:
    """Gym-like environment with no dependency on Gym/Gymnasium.

    World axes: X east, Y north, Z up.  The target pad is at world origin.
    Observation vector has 13 normalized values:
      [x,y,altitude, vx,vy,vz, body_up_x,y,z, wx,wy,wz, fuel_fraction]
    Action vector has 3 values:
      [throttle 0..1, gimbal_pitch -1..1, gimbal_yaw -1..1]
    """

    observation_size = 13
    action_size = 3

    def __init__(self, config: AppConfig, seed: int | None = None, deterministic: bool = False):
        self.config = config
        self.rng = np.random.default_rng(config.neat.seed if seed is None else seed)
        self.deterministic = deterministic
        self.state: RocketState | None = None
        self.last_dynamics: DynamicsInfo | None = None
        self._steady_wind = np.zeros(3)
        self._phase = np.zeros(2)
        self._terminated = False
        self._last_distance = 0.0
        self._last_altitude = 0.0
        self.outcome = "not_started"

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rcfg = self.config.rocket
        ecfg = self.config.environment
        if self.deterministic:
            x, y = 65.0, -45.0
            vx, vy = -5.0, 3.0
            tilt_x, tilt_y = math.radians(2.5), math.radians(-1.5)
            wind_dir = 0.55
        else:
            x, y = self.rng.normal(0.0, ecfg.initial_lateral_spread_m, size=2)
            speed_xy = self.rng.uniform(0.0, ecfg.initial_horizontal_speed_mps)
            heading = self.rng.uniform(-math.pi, math.pi)
            vx, vy = speed_xy * math.cos(heading), speed_xy * math.sin(heading)
            tilt_x, tilt_y = np.radians(
                self.rng.uniform(-ecfg.initial_tilt_deg, ecfg.initial_tilt_deg, size=2)
            )
            wind_dir = self.rng.uniform(-math.pi, math.pi)
        altitude = ecfg.initial_altitude_m * (1.0 if self.deterministic else self.rng.uniform(0.85, 1.12))
        vz = ecfg.initial_vertical_speed_mps * (1.0 if self.deterministic else self.rng.uniform(0.88, 1.12))
        center_z = altitude + rcfg.length_m * 0.5
        self.state = RocketState(
            position_m=np.array([x, y, center_z], dtype=float),
            velocity_mps=np.array([vx, vy, vz], dtype=float),
            quaternion=quat_from_euler(tilt_x, tilt_y, 0.0),
            omega_body_rps=np.zeros(3, dtype=float),
            propellant_kg=rcfg.landing_propellant_kg,
            time_s=0.0,
        )
        self._steady_wind = np.array(
            [ecfg.wind_speed_mps * math.cos(wind_dir), ecfg.wind_speed_mps * math.sin(wind_dir), 0.0],
            dtype=float,
        )
        self._phase = self.rng.uniform(0.0, 2.0 * math.pi, size=2)
        self._terminated = False
        self.outcome = "flying"
        self._last_distance = float(np.linalg.norm(self.state.position_m[:2]))
        self._last_altitude = max(0.0, bottom_altitude_m(self.state, rcfg))
        self.last_dynamics = None
        return self._observation()

    def _wind(self) -> np.ndarray:
        assert self.state is not None
        e = self.config.environment
        t = self.state.time_s
        gust = e.wind_gust_mps * np.array(
            [math.sin(0.63 * t + self._phase[0]), math.sin(0.41 * t + self._phase[1]), 0.0]
        )
        return self._steady_wind + gust

    def _observation(self) -> np.ndarray:
        assert self.state is not None
        s = self.state
        r = self.config.rocket
        alt = max(0.0, bottom_altitude_m(s, r))
        up = body_up(s)
        fuel_frac = s.propellant_kg / max(1.0, r.landing_propellant_kg)
        obs = np.array(
            [
                s.position_m[0] / 500.0,
                s.position_m[1] / 500.0,
                alt / 1000.0,
                s.velocity_mps[0] / 80.0,
                s.velocity_mps[1] / 80.0,
                s.velocity_mps[2] / 100.0,
                up[0],
                up[1],
                up[2],
                s.omega_body_rps[0] / 0.5,
                s.omega_body_rps[1] / 0.5,
                s.omega_body_rps[2] / 0.5,
                fuel_frac,
            ],
            dtype=float,
        )
        return np.clip(obs, -2.5, 2.5)

    def step(self, action: np.ndarray | list[float] | Control) -> StepResult:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self._terminated:
            raise RuntimeError("Episode has terminated; call reset().")

        if isinstance(action, Control):
            control = action
        else:
            a = np.asarray(action, dtype=float).reshape(3)
            control = Control(a[0], a[1], a[2])

        before_prop = self.state.propellant_kg
        self.state, self.last_dynamics = step_dynamics(
            self.state, control, self.config.rocket, self.config.environment, self._wind()
        )
        s = self.state
        r = self.config.rocket
        e = self.config.environment
        alt = bottom_altitude_m(s, r)
        lateral_dist = float(np.linalg.norm(s.position_m[:2]))
        horizontal_speed = float(np.linalg.norm(s.velocity_mps[:2]))
        vertical_speed = float(s.velocity_mps[2])
        tilt = angle_between(body_up(s), np.array([0.0, 0.0, 1.0]))
        omega = float(np.linalg.norm(s.omega_body_rps))

        # Dense shaping: reward progress toward the pad and ground while
        # penalizing excessive speed, tilt, spin and propellant use.
        progress_xy = (self._last_distance - lateral_dist) / 25.0
        progress_z = max(-1.0, min(1.0, (self._last_altitude - max(0.0, alt)) / 15.0))
        reward = 0.35 * progress_xy + 0.16 * progress_z
        reward -= 0.0025 * lateral_dist
        reward -= 0.0020 * horizontal_speed**2
        reward -= 0.0012 * max(0.0, abs(vertical_speed) - 18.0) ** 2
        reward -= 0.75 * tilt**2
        reward -= 0.08 * omega**2
        reward -= 0.00030 * max(0.0, before_prop - s.propellant_kg)
        reward += 0.025  # surviving while controlled

        terminated = False
        outcome = "flying"
        success = False

        if alt <= 0.0:
            tilt_deg = math.degrees(tilt)
            success = (
                lateral_dist <= e.pad_radius_m
                and abs(vertical_speed) <= e.touchdown_vertical_speed_mps
                and horizontal_speed <= e.touchdown_horizontal_speed_mps
                and tilt_deg <= e.touchdown_tilt_deg
                and omega <= 0.22
            )
            if success:
                # Strong terminal reward makes genuine soft landings dominate.
                precision = max(0.0, 1.0 - lateral_dist / e.pad_radius_m)
                softness = max(0.0, 1.0 - abs(vertical_speed) / e.touchdown_vertical_speed_mps)
                reward += 3500.0 + 650.0 * precision + 500.0 * softness
                outcome = "landed"
            else:
                reward -= 1900.0 + min(900.0, 8.0 * abs(vertical_speed) + 3.0 * horizontal_speed)
                outcome = "crashed"
            terminated = True
        elif lateral_dist > e.max_range_m or s.position_m[2] > 3_500.0:
            reward -= 1300.0
            terminated = True
            outcome = "out_of_bounds"
        elif s.time_s >= e.max_time_s:
            reward -= 800.0 + 0.5 * lateral_dist + 0.5 * max(0.0, alt)
            terminated = True
            outcome = "timeout"

        self._last_distance = lateral_dist
        self._last_altitude = max(0.0, alt)
        self._terminated = terminated
        self.outcome = outcome

        info = {
            "outcome": outcome,
            "success": success,
            "time_s": s.time_s,
            "altitude_m": max(0.0, alt),
            "lateral_distance_m": lateral_dist,
            "vertical_speed_mps": vertical_speed,
            "horizontal_speed_mps": horizontal_speed,
            "tilt_deg": math.degrees(tilt),
            "fuel_kg": s.propellant_kg,
            "mass_kg": self.last_dynamics.mass_kg,
            "thrust_n": self.last_dynamics.thrust_n,
            "dynamic_pressure_pa": self.last_dynamics.dynamic_pressure_pa,
            "wind_mps": self.last_dynamics.wind_mps.copy(),
        }
        return StepResult(self._observation(), float(reward), terminated, info)
