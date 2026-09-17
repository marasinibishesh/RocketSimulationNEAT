"""Small 3-D math helpers used by the simulator.

Quaternion convention throughout the project is [w, x, y, z].  The local
rocket +Z axis points from the engines toward the nose.
"""
from __future__ import annotations

import math
import numpy as np

EPS = 1e-12


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v, dtype=float)
    return np.asarray(v, dtype=float) / n


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit(np.asarray(axis, dtype=float))
    half = 0.5 * angle_rad
    s = math.sin(half)
    return quat_normalize(np.array([math.cos(half), *(axis * s)], dtype=float))


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Create quaternion from intrinsic XYZ roll/pitch/yaw radians."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from body/local coordinates into world coordinates."""
    q = quat_normalize(q)
    qv = np.array([0.0, *np.asarray(v, dtype=float)], dtype=float)
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


def quat_integrate_body_rates(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Integrate body-frame angular velocity with a first-order quaternion step."""
    omega_q = np.array([0.0, *np.asarray(omega_body, dtype=float)], dtype=float)
    q_dot = 0.5 * quat_multiply(q, omega_q)
    return quat_normalize(q + q_dot * dt)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = unit(a), unit(b)
    return math.acos(clamp(float(np.dot(ua, ub)), -1.0, 1.0))


def rotation_matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
