import math
import numpy as np

from falcon9_neat.math3d import quat_from_axis_angle, quat_rotate, quat_integrate_body_rates


def test_quaternion_rotation_z_90():
    q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    out = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-7)


def test_quaternion_integration_stays_normalized():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(1000):
        q = quat_integrate_body_rates(q, np.array([0.1, -0.2, 0.3]), 0.01)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-10
