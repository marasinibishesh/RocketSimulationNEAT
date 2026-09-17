"""Procedural landing site, pad, horizon markers and lighting."""
from __future__ import annotations

import math
from panda3d.core import AmbientLight, DirectionalLight, Fog, NodePath, Vec4

from .geometry import box, cylinder, frustum


def build_world(render: NodePath, shadow_map_size: int = 2048) -> NodePath:
    world = render.attachNewNode("world")
    # Ground and concrete landing pad.
    ground = box("ground", 5000.0, 5000.0, 1.0, (0.18, 0.22, 0.18, 1.0))
    ground.reparentTo(world); ground.setZ(-0.55)
    pad = cylinder("landing_pad", 28.0, 0.55, (0.32, 0.33, 0.34, 1.0), segments=64)
    pad.reparentTo(world); pad.setZ(-0.2)
    inner = cylinder("pad_inner", 17.0, 0.59, (0.72, 0.72, 0.70, 1.0), segments=64)
    inner.reparentTo(world); inner.setZ(-0.16)
    target = cylinder("target", 5.0, 0.62, (0.18, 0.18, 0.18, 1.0), segments=48)
    target.reparentTo(world); target.setZ(-0.14)
    # Crosshair stripes.
    for angle in (0, 90):
        stripe = box("pad_cross", 1.1, 24.0, 0.08, (0.88, 0.88, 0.86, 1.0))
        stripe.reparentTo(world); stripe.setH(angle); stripe.setZ(0.2)

    # Low-poly distant terrain gives depth cues and a more complete environment.
    for i in range(18):
        a = 2 * math.pi * i / 18
        dist = 900 + 120 * math.sin(i * 1.7)
        radius = 85 + 35 * ((i * 37) % 5)
        height = 100 + 35 * ((i * 19) % 7)
        hill = frustum("distant_hill", radius, 2.0, height, (0.13, 0.17, 0.13, 1.0), segments=12)
        hill.reparentTo(world)
        hill.setPos(dist * math.cos(a), dist * math.sin(a), height * 0.5 - 0.3)

    ambient = AmbientLight("ambient")
    ambient.setColor(Vec4(0.33, 0.35, 0.40, 1))
    ambient_np = world.attachNewNode(ambient)
    render.setLight(ambient_np)
    sun = DirectionalLight("sun")
    sun.setColor(Vec4(1.0, 0.92, 0.78, 1))
    sun.setShadowCaster(True, shadow_map_size, shadow_map_size)
    sun_np = world.attachNewNode(sun)
    sun_np.setPos(120, -140, 260)
    sun_np.setHpr(-38, -48, 0)
    sun.getLens().setFilmSize(420, 420)
    sun.getLens().setNearFar(20, 900)
    render.setLight(sun_np)
    render.setShaderAuto()

    fog = Fog("atmospheric_haze")
    fog.setColor(0.46, 0.60, 0.73)
    fog.setLinearRange(900.0, 2900.0)
    render.setFog(fog)
    return world
