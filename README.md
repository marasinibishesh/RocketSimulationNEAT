# Falcon 9 NEAT Landing Simulator

A self-contained Python project that evolves a neural controller with **NEAT implemented from scratch** and applies it to a **3-D, six-degree-of-freedom reusable-booster landing simulation**. The real-time visualizer uses Panda3D, while physics and training remain independent of the renderer so training can run headlessly.

> **Scope:** this is an educational/research simulator inspired by the Falcon 9 first stage, not SpaceX flight software, a mission-planning tool, or an engineering-validated digital twin. Public vehicle dimensions and engine thrust inform the presentation, while landing-phase mass, aerodynamic coefficients, control authority, and other unpublished details are explicitly tunable approximations.

## Highlights

- NEAT from scratch: genes, innovation numbers, structural mutation, crossover, compatibility distance, speciation, fitness sharing, elitism, stagnation and checkpoint serialization.
- 6-DOF booster physics: translation, quaternion attitude, body angular rates, variable mass, propellant consumption, altitude-dependent gravity and atmosphere, quadratic drag, thrust-vector control, aerodynamic restoring torque, wind and gusts.
- Landing task: randomized initial position, velocity, attitude and wind; soft-touchdown criteria; dense progress shaping plus a large terminal landing reward.
- Procedural 3-D model: Falcon 9 inspired stage, nine-engine cluster, interstage, grid fins, landing legs and live exhaust plume. A separate showroom mode displays a procedural full-stack ~70 m vehicle.
- Procedural world: landing pad, terrain, distant low-poly hills, atmospheric fog, lighting, shadows, cinematic cameras and a telemetry HUD.
- GitHub ready: `pyproject.toml`, CLI entry points, tests, GitHub Actions workflow, MIT license, architecture/physics/NEAT docs and editable YAML configuration.

## Quick start

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Run the visual simulator immediately with the built-in baseline controller:

```bash
falcon9-demo --controller heuristic
```

View the full-stack procedural rocket model:

```bash
falcon9-demo --showroom
```

Train NEAT:

```bash
falcon9-train --config configs/default.yaml --generations 80
```

A smaller smoke-training run is useful first:

```bash
falcon9-train --generations 5 --population 24 --episodes 1
```

Evaluate a checkpoint headlessly:

```bash
falcon9-evaluate --checkpoint checkpoints/best_genome.json --episodes 20
```

Visualize the evolved network:

```bash
falcon9-demo --controller neat --checkpoint checkpoints/best_genome.json
```

You can also use the unified command:

```bash
falcon9-neat train --generations 10 --population 32
falcon9-neat evaluate --checkpoint checkpoints/best_genome.json
falcon9-neat demo --controller neat --checkpoint checkpoints/best_genome.json
```

## Controls

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `R` | Reset the current episode |
| `C` | Cycle camera modes |
| `Esc` | Exit |

## Controller interface

The NEAT network receives 13 normalized inputs:

```text
x, y, altitude,
vx, vy, vz,
body_up_x, body_up_y, body_up_z,
omega_x, omega_y, omega_z,
fuel_fraction
```

It outputs three channels:

```text
throttle, gimbal_pitch, gimbal_yaw
```

Network output 0 is remapped from `[-1, 1]` to throttle `[0, 1]`; the remaining outputs stay signed.

## Repository layout

```text
RocketSimulationNEAT/
├── configs/default.yaml          # Physics, environment, NEAT, renderer settings
├── docs/
│   ├── ARCHITECTURE.md
│   ├── NEAT.md
│   └── PHYSICS.md
├── src/falcon9_neat/
│   ├── environment.py            # Landing task and reward
│   ├── physics.py                # 6-DOF rigid-body integration
│   ├── controllers.py            # Baseline visual-demo controller
│   ├── math3d.py                 # Quaternion/vector math
│   ├── neat/                     # From-scratch NEAT implementation
│   └── render/                   # Panda3D procedural model/world/app
├── tests/
├── .github/workflows/tests.yml
├── pyproject.toml
└── LICENSE
```

## Public reference values

The default visuals use public Falcon 9 information from SpaceX: approximately **70 m overall height**, **3.7 m diameter**, **549,054 kg launch mass**, and a sea-level Merlin thrust figure of **845 kN**. SpaceX also describes the first stage as using nine Merlin engines and LOX/RP-1 propellant. The landing simulation uses only a landing-phase subset of the vehicle and therefore does **not** initialize with full launch mass or full propellant load.

Official references:

- SpaceX Falcon 9 vehicle page: https://www.spacex.com/vehicles/falcon-9/
- SpaceX Falcon Payload User's Guide (2025): https://www.spacex.com/assets/media/falcon-users-guide-2025-05-09.pdf

## Training notes

NEAT is stochastic. A strong landing controller may take many generations depending on CPU speed, population size and scenario randomization. The default configuration values favor robustness rather than the shortest training run. For faster iteration, reduce `population_size`, `episodes_per_genome` and `generations`; for a more robust final controller, increase them again.

The fitness function combines pad-centering progress, descent progress, horizontal/vertical speed penalties, attitude and angular-rate penalties, fuel use and a large terminal bonus for a valid soft landing. See `docs/NEAT.md` for details.

## Accuracy and extension points

This repository intentionally separates public facts from modeled assumptions. If you have higher-fidelity mass properties, engine throttle limits, aerodynamic tables, grid-fin models, wind profiles or telemetry, edit `configs/default.yaml` and/or replace the corresponding model in `physics.py` without changing the NEAT code.

Natural next steps include multi-engine landing burns, boostback/reentry phases, Earth curvature in position dynamics, tabulated atmosphere, landing-leg contact dynamics, sensor noise/delay, recurrent NEAT, distributed evaluation, domain randomization, replay files, and telemetry plots.

## License and trademarks

Code in this repository is MIT licensed. Falcon 9, SpaceX and related marks belong to their respective owners. This project is independent, unofficial and intended for education/research.
