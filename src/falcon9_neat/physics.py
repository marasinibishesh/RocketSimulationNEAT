"""6-DOF rigid-body booster dynamics for the landing simulation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import EnvironmentConfig, RocketConfig
from .math3d import quat_integrate_body_rates, quat_rotate, unit

G0 = 9.80665


@dataclass(slots=True)
class Control:
    """Normalized control request produced by a controller."""

    throttle: float
    gimbal_pitch: float
    gimbal_yaw: float

    def clipped(self) -> "Control":
        return Control(
            float(np.clip(self.throttle, 0.0, 1.0)),
            float(np.clip(self.gimbal_pitch, -1.0, 1.0)),
            float(np.clip(self.gimbal_yaw, -1.0, 1.0)),
        )


@dataclass(slots=True)
class RocketState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion: np.ndarray
    omega_body_rps: np.ndarray
    propellant_kg: float
    time_s: float = 0.0

    def copy(self) -> "RocketState":
        return RocketState(
            self.position_m.copy(),
            self.velocity_mps.copy(),
            self.quaternion.copy(),
            self.omega_body_rps.copy(),
            float(self.propellant_kg),
            float(self.time_s),
        )


@dataclass(slots=True)
class DynamicsInfo:
    mass_kg: float
    thrust_n: float
    air_density_kgpm3: float
    dynamic_pressure_pa: float
    acceleration_mps2: np.ndarray
    angular_accel_rps2: np.ndarray
    wind_mps: np.ndarray


def _gravity(env: EnvironmentConfig, altitude_m: float) -> float:
    r = env.earth_radius_m
    return env.earth_gravity_mps2 * (r / (r + max(0.0, altitude_m))) ** 2


def _air_density(env: EnvironmentConfig, altitude_m: float) -> float:
    return env.sea_level_density_kgpm3 * math.exp(-max(0.0, altitude_m) / env.atmosphere_scale_height_m)


def _inertia_diagonal(mass_kg: float, rocket: RocketConfig) -> np.ndarray:
    r = rocket.diameter_m * 0.5
    l = rocket.length_m
    transverse = mass_kg * (3.0 * r * r + l * l) / 12.0
    axial = 0.5 * mass_kg * r * r
    return np.array([transverse, transverse, axial], dtype=float)


def body_up(state: RocketState) -> np.ndarray:
    return unit(quat_rotate(state.quaternion, np.array([0.0, 0.0, 1.0])))


def bottom_altitude_m(state: RocketState, rocket: RocketConfig) -> float:
    # Approximate engine-plane altitude. Sufficient for landing-contact logic.
    up = body_up(state)
    return float(state.position_m[2] - 0.5 * rocket.length_m * max(up[2], 0.15))


def step_dynamics(
    state: RocketState,
    control: Control,
    rocket: RocketConfig,
    env: EnvironmentConfig,
    wind_world_mps: np.ndarray,
) -> tuple[RocketState, DynamicsInfo]:
    """Advance one semi-implicit Euler step.

    This model intentionally favors clarity and trainability over proprietary
    vehicle fidelity. It includes variable mass, thrust vectoring, quadratic
    drag, aerodynamic weathercock stability, grid-fin-like damping/control,
    3-D gravity, rotational inertia, and fuel consumption.
    """
    dt = env.dt_s
    ctl = control.clipped()
    mass = rocket.dry_mass_kg + max(0.0, state.propellant_kg)
    altitude = max(0.0, bottom_altitude_m(state, rocket))
    rho = _air_density(env, altitude)
    gravity = _gravity(env, altitude)

    # One center Merlin-class engine is used during the terminal landing phase.
    throttle = ctl.throttle if state.propellant_kg > 0.0 else 0.0
    thrust_n = throttle * rocket.engine_max_thrust_n
    gimbal = math.radians(rocket.max_gimbal_deg)
    gp = ctl.gimbal_pitch * gimbal
    gy = ctl.gimbal_yaw * gimbal
    thrust_body = unit(np.array([math.tan(gy), -math.tan(gp), 1.0])) * thrust_n
    thrust_world = quat_rotate(state.quaternion, thrust_body)

    air_velocity = state.velocity_mps - np.asarray(wind_world_mps, dtype=float)
    speed = float(np.linalg.norm(air_velocity))
    q_dyn = 0.5 * rho * speed * speed
    if speed > 1e-7:
        drag_world = -0.5 * rho * rocket.drag_coefficient * rocket.reference_area_m2 * speed * air_velocity
    else:
        drag_world = np.zeros(3)

    gravity_force = np.array([0.0, 0.0, -mass * gravity])
    total_force = thrust_world + drag_world + gravity_force
    acceleration = total_force / mass

    # Torque from thrust vector acting below the center of mass.
    engine_arm_body = np.array([0.0, 0.0, -0.48 * rocket.length_m])
    thrust_torque_body = np.cross(engine_arm_body, thrust_body)

    # Aerodynamic weathercock torque: attempts to align the long axis with the
    # apparent wind. Grid-fin authority increases with dynamic pressure.
    body_up_world = body_up(state)
    if speed > 0.5:
        target_axis_world = -unit(air_velocity)
        alignment_world = np.cross(body_up_world, target_axis_world)
        # Transform torque direction to body frame using inverse quaternion.
        q = state.quaternion
        inv_q = np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
        alignment_body = quat_rotate(inv_q, alignment_world)
    else:
        alignment_body = np.zeros(3)

    aero_scale = q_dyn * rocket.reference_area_m2 * rocket.length_m * rocket.aero_torque_coefficient
    aero_torque_body = alignment_body * aero_scale

    # Grid-fin-like torque uses the same two controller channels to add
    # atmosphere-dependent control authority. It fades naturally near hover.
    # Match the engine-gimbal torque sign convention: +pitch produces -X
    # body torque and +yaw produces -Y body torque.
    grid_cmd = np.array([-ctl.gimbal_pitch, -ctl.gimbal_yaw, 0.0], dtype=float)
    grid_torque_body = grid_cmd * (
        q_dyn * rocket.reference_area_m2 * rocket.length_m * rocket.grid_fin_torque_coefficient
    )

    inertia = _inertia_diagonal(mass, rocket)
    omega = state.omega_body_rps
    rotational_coupling = np.cross(omega, inertia * omega)
    damping_torque = -rocket.angular_damping * inertia * omega
    torque_body = thrust_torque_body + aero_torque_body + grid_torque_body + damping_torque
    angular_accel = (torque_body - rotational_coupling) / inertia

    new_omega = omega + angular_accel * dt
    new_q = quat_integrate_body_rates(state.quaternion, new_omega, dt)
    new_velocity = state.velocity_mps + acceleration * dt
    new_position = state.position_m + new_velocity * dt

    mdot = thrust_n / max(1e-9, rocket.engine_isp_s * G0)
    new_propellant = max(0.0, state.propellant_kg - mdot * dt)

    new_state = RocketState(
        position_m=new_position,
        velocity_mps=new_velocity,
        quaternion=new_q,
        omega_body_rps=new_omega,
        propellant_kg=new_propellant,
        time_s=state.time_s + dt,
    )
    info = DynamicsInfo(
        mass_kg=mass,
        thrust_n=thrust_n,
        air_density_kgpm3=rho,
        dynamic_pressure_pa=q_dyn,
        acceleration_mps2=acceleration,
        angular_accel_rps2=angular_accel,
        wind_mps=np.asarray(wind_world_mps, dtype=float).copy(),
    )
    return new_state, info
