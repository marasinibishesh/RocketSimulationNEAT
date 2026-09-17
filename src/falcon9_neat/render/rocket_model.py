"""Procedural Falcon 9 inspired 3-D vehicle model."""
from __future__ import annotations

import math
from panda3d.core import NodePath, TransparencyAttrib

from .geometry import box, cylinder, frustum

WHITE = (0.92, 0.93, 0.94, 1.0)
OFFWHITE = (0.82, 0.83, 0.84, 1.0)
BLACK = (0.055, 0.06, 0.065, 1.0)
DARK = (0.12, 0.13, 0.14, 1.0)
METAL = (0.30, 0.31, 0.32, 1.0)


def _engine_cluster(parent: NodePath, z: float) -> None:
    positions = [(0,0)]
    ring_r = 0.95
    for i in range(8):
        a = 2 * math.pi * i / 8
        positions.append((ring_r * math.cos(a), ring_r * math.sin(a)))
    for i, (x, y) in enumerate(positions):
        bell = frustum(f"engine_bell_{i}", 0.38, 0.24, 1.25, METAL, segments=18)
        bell.reparentTo(parent)
        bell.setPos(x, y, z)
        # bell widens toward the engine exit, which is below the stage
        bell.setHpr(0, 180, 0)


def _grid_fins(parent: NodePath, z: float) -> None:
    # Two plates per fin suggest the characteristic lattice silhouette without
    # requiring a heavy CAD mesh.
    for i in range(4):
        angle = 90 * i
        holder = parent.attachNewNode(f"grid_fin_{i}")
        holder.setH(angle)
        holder.setPos(0, 0, z)
        plate = box("fin_plate", 0.16, 3.2, 2.15, DARK)
        plate.reparentTo(holder)
        plate.setPos(0, 2.25, 0)
        for k in (-0.7, 0.0, 0.7):
            rib = box("fin_rib", 0.22, 3.45, 0.08, METAL)
            rib.reparentTo(holder)
            rib.setPos(0, 2.25, k)


def _landing_legs(parent: NodePath, z: float) -> None:
    # Four swept structural legs. Their geometry is intentionally low-poly for
    # real-time training visualization while preserving a recognizable profile.
    for i in range(4):
        angle = math.radians(45 + 90 * i)
        holder = parent.attachNewNode(f"landing_leg_{i}")
        holder.setPos(0, 0, z)
        holder.setH(math.degrees(angle))
        strut = box("leg_strut", 0.32, 0.34, 12.5, OFFWHITE)
        strut.reparentTo(holder)
        strut.setPos(0, 2.7, -4.9)
        strut.setP(-22)
        foot = box("leg_foot", 0.65, 2.6, 0.22, DARK)
        foot.reparentTo(holder)
        foot.setPos(0, 5.0, -10.4)
        foot.setP(-4)


def build_booster(length_m: float = 47.7, diameter_m: float = 3.7) -> tuple[NodePath, NodePath]:
    """Return (rocket_root, exhaust_node). Root origin is near the CM."""
    root = NodePath("falcon9_booster")
    r = diameter_m / 2

    body = cylinder("first_stage", r, length_m * 0.88, WHITE, segments=40)
    body.reparentTo(root)
    body.setZ(0.4)

    # Black interstage and lower engine section.
    interstage = cylinder("interstage", r * 1.005, 4.0, BLACK, segments=40)
    interstage.reparentTo(root)
    interstage.setZ(length_m * 0.44 - 2.0)
    skirt = cylinder("engine_skirt", r * 1.01, 2.0, DARK, segments=32)
    skirt.reparentTo(root)
    skirt.setZ(-length_m * 0.44 + 0.5)
    _engine_cluster(root, -length_m * 0.44 - 1.1)
    _grid_fins(root, length_m * 0.36)
    _landing_legs(root, -length_m * 0.22)

    # Simple vertical black marking improves scale/readability at distance.
    mark = box("stage_mark", 0.045, 0.85, 6.5, BLACK)
    mark.reparentTo(root)
    mark.setPos(r + 0.02, 0, 5.5)

    # Exhaust is a translucent tapered plume; app updates its scale and alpha.
    exhaust = frustum("exhaust", 1.15, 0.18, 16.0, (1.0, 0.55, 0.12, 0.38), segments=24, capped=False)
    exhaust.reparentTo(root)
    exhaust.setZ(-length_m * 0.5 - 8.0)
    exhaust.setTransparency(TransparencyAttrib.MAlpha)
    exhaust.setTwoSided(True)
    exhaust.hide()
    return root, exhaust


def build_full_vehicle() -> NodePath:
    """Static full-stack Falcon 9 inspired display model (~70 m tall)."""
    root = NodePath("falcon9_full_vehicle")
    booster, _ = build_booster(47.7, 3.7)
    booster.reparentTo(root)
    booster.setZ(-11.15)

    second = cylinder("second_stage", 1.85, 10.0, WHITE, segments=40)
    second.reparentTo(root)
    second.setZ(18.0)
    fairing_cyl = cylinder("fairing_base", 2.6, 8.0, WHITE, segments=40)
    fairing_cyl.reparentTo(root)
    fairing_cyl.setZ(27.0)
    fairing_nose = frustum("fairing_nose", 2.6, 0.18, 5.1, WHITE, segments=40)
    fairing_nose.reparentTo(root)
    fairing_nose.setZ(33.55)
    return root
