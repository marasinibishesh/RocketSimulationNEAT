# Physics model

This project models the terminal landing phase of a Falcon 9 inspired reusable booster. It is a training simulator, not an engineering-validated vehicle model.

## Coordinate frames

- World frame: +X east, +Y north, +Z up.
- Body frame: +Z points from engines toward the vehicle nose.
- Attitude: unit quaternion `[w, x, y, z]` rotates body vectors into world coordinates.

## Translational dynamics

The integrator applies:

```text
F_total = F_thrust + F_drag + F_gravity
acceleration = F_total / mass
```

Gravity varies with altitude using an inverse-square approximation. Atmospheric density follows a simple exponential atmosphere. Drag is quadratic in relative airspeed and is evaluated against steady wind plus smooth gust components.

## Propulsion

Terminal descent uses one center Merlin-class engine in the default model. Maximum thrust defaults to 845 kN, matching SpaceX's public sea-level Merlin figure. Propellant flow is derived from thrust and configured specific impulse:

```text
m_dot = thrust / (Isp * g0)
```

Throttle is continuous from 0 to 1 in the training model. Real hardware can have additional throttle, transient, ignition and operating constraints; add those constraints if your research needs them.

## Rotation

The project integrates Euler's rigid-body equation in body coordinates with a diagonal cylinder-like inertia approximation. Torques include:

- engine-gimbal torque about the center of mass
- aerodynamic alignment torque
- grid-fin-like atmosphere-dependent control torque
- angular damping

Quaternion attitude is integrated from body angular rates and renormalized every step.

## Ground contact

The current contact model is terminal rather than deformable: when the approximate engine-plane altitude reaches zero, the environment checks landing criteria. A touchdown succeeds only when the booster is inside the pad radius and below configured limits for vertical speed, horizontal speed, tilt and angular rate. Otherwise the episode is marked as a crash.

For research requiring bounce, leg compression, friction or tipping dynamics, replace terminal contact with a multi-point constraint/contact solver.

## Modeled versus public values

Public SpaceX values are used only where explicitly documented, such as overall Falcon 9 diameter and Merlin sea-level thrust. Landing propellant, first-stage dry mass in this simplified environment, aerodynamic coefficients, grid-fin torque and several control limits are configurable approximations. They should be calibrated before drawing engineering conclusions.
