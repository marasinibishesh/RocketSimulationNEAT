"""Typed configuration objects and YAML loader."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(slots=True)
class RocketConfig:
    # Landing-phase model. Values are deliberately configurable and are not a
    # claim that SpaceX publishes every internal first-stage parameter.
    length_m: float = 47.7
    diameter_m: float = 3.7
    dry_mass_kg: float = 25_600.0
    landing_propellant_kg: float = 9_000.0
    engine_max_thrust_n: float = 845_000.0
    engine_isp_s: float = 282.0
    max_gimbal_deg: float = 6.0
    drag_coefficient: float = 0.82
    reference_area_m2: float = 10.75
    aero_torque_coefficient: float = 0.055
    grid_fin_torque_coefficient: float = 0.12
    angular_damping: float = 0.06


@dataclass(slots=True)
class EnvironmentConfig:
    dt_s: float = 0.05
    max_time_s: float = 55.0
    earth_gravity_mps2: float = 9.80665
    earth_radius_m: float = 6_371_000.0
    sea_level_density_kgpm3: float = 1.225
    atmosphere_scale_height_m: float = 8_500.0
    pad_radius_m: float = 14.0
    max_range_m: float = 2_500.0
    touchdown_vertical_speed_mps: float = 3.0
    touchdown_horizontal_speed_mps: float = 2.5
    touchdown_tilt_deg: float = 7.0
    wind_speed_mps: float = 7.0
    wind_gust_mps: float = 2.5
    initial_altitude_m: float = 850.0
    initial_lateral_spread_m: float = 130.0
    initial_vertical_speed_mps: float = -62.0
    initial_horizontal_speed_mps: float = 14.0
    initial_tilt_deg: float = 5.0


@dataclass(slots=True)
class NeatConfig:
    population_size: int = 96
    generations: int = 80
    episodes_per_genome: int = 3
    input_count: int = 13
    output_count: int = 3
    survival_fraction: float = 0.25
    elitism: int = 2
    compatibility_threshold: float = 3.0
    compatibility_excess: float = 1.0
    compatibility_disjoint: float = 1.0
    compatibility_weight: float = 0.4
    stagnation_generations: int = 14
    weight_mutation_rate: float = 0.80
    weight_perturb_rate: float = 0.90
    weight_perturb_sigma: float = 0.35
    add_connection_rate: float = 0.12
    add_node_rate: float = 0.045
    toggle_connection_rate: float = 0.015
    crossover_rate: float = 0.75
    interspecies_mating_rate: float = 0.01
    seed: int = 7


@dataclass(slots=True)
class RenderConfig:
    window_width: int = 1440
    window_height: int = 900
    multisamples: int = 4
    shadow_map_size: int = 2048
    camera_distance_m: float = 125.0
    camera_height_m: float = 45.0
    hud: bool = True


@dataclass(slots=True)
class AppConfig:
    rocket: RocketConfig
    environment: EnvironmentConfig
    neat: NeatConfig
    render: RenderConfig


def _build(cls: type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)  # type: ignore[arg-type]


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig(RocketConfig(), EnvironmentConfig(), NeatConfig(), RenderConfig())
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig(
        rocket=_build(RocketConfig, raw.get("rocket")),
        environment=_build(EnvironmentConfig, raw.get("environment")),
        neat=_build(NeatConfig, raw.get("neat")),
        render=_build(RenderConfig, raw.get("render")),
    )
