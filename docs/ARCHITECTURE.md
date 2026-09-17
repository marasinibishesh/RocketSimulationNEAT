# Architecture

The project is split into three intentionally independent layers.

## 1. Simulation core

`physics.py` contains the rigid-body state and integrator. `environment.py` wraps that model into an episodic landing problem. The simulation core imports NumPy only and can run on a headless machine.

A state consists of:

- center-of-mass position `[x, y, z]`
- inertial velocity `[vx, vy, vz]`
- orientation quaternion `[w, x, y, z]`
- body angular velocity `[wx, wy, wz]`
- remaining landing propellant
- simulation time

A control consists of throttle, pitch gimbal and yaw gimbal.

## 2. Neuroevolution

`falcon9_neat/neat/` is independent from the physics details. A `Genome` is a collection of node and connection genes. An `InnovationTracker` assigns historical markings. `FeedForwardNetwork` compiles a genome into an executable phenotype. `Population` performs speciation and reproduction. `Trainer` supplies the landing environment as the fitness task.

The separation is deliberate: another task can reuse the NEAT implementation, and another controller can reuse the rocket environment.

## 3. Visualization

`falcon9_neat/render/` imports Panda3D only when the user runs the visualizer. Geometry is generated procedurally, so there are no proprietary CAD files or large binary assets in the repository. The renderer reads the exact state evolved by the headless simulator and never owns the physics.

## Data flow

```text
random scenario
     |
     v
FalconLandingEnv ---- observation (13) ----> Controller / NEAT network
     ^                                              |
     |                                              |
     +------- 6-DOF physics <---- action (3) -------+
                    |
                    +---- state ----> Panda3D renderer (optional)
```

## Why fixed-step physics?

Training needs deterministic integration for a given random seed, independent of monitor refresh rate. `environment.py` therefore advances a fixed simulation `dt`. The Panda3D app samples this model for display.

## Why procedural graphics?

A GitHub repository that depends on an external 3-D asset is easy to break and can create licensing ambiguity. The rocket, pad and terrain are generated from cylinders, frustums and boxes at runtime. This keeps the project clone-and-run and makes the scale editable in code.
