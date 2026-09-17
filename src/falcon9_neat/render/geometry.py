"""Procedural Panda3D geometry helpers.

No external 3-D model files are required; the vehicle and world are generated
at runtime from geometric primitives so the repository stays self-contained.
"""
from __future__ import annotations

import math

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)


def _geom_node(name: str, vertices: list[tuple], triangles: list[tuple[int, int, int]]) -> NodePath:
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(len(vertices))
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")
    for pos, normal, color in vertices:
        vw.addData3f(*pos)
        nw.addData3f(*normal)
        cw.addData4f(*color)
    prim = GeomTriangles(Geom.UHStatic)
    for a, b, c in triangles:
        prim.addVertices(a, b, c)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def cylinder(
    name: str,
    radius: float,
    height: float,
    color=(0.9, 0.9, 0.9, 1.0),
    segments: int = 32,
    capped: bool = True,
) -> NodePath:
    """Cylinder centered at z=0 with its long axis along +Z."""
    vertices: list[tuple] = []
    triangles: list[tuple[int, int, int]] = []
    half = height * 0.5
    # Side wall: duplicate vertices per seam so normals stay radial.
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        x, y = radius * math.cos(a), radius * math.sin(a)
        n = (math.cos(a), math.sin(a), 0.0)
        vertices.append(((x, y, -half), n, color))
        vertices.append(((x, y, half), n, color))
    for i in range(segments):
        k = 2 * i
        triangles.extend([(k, k + 2, k + 1), (k + 1, k + 2, k + 3)])
    if capped:
        bottom_center = len(vertices)
        vertices.append(((0, 0, -half), (0, 0, -1), color))
        top_center = len(vertices)
        vertices.append(((0, 0, half), (0, 0, 1), color))
        bottom_ring = []
        top_ring = []
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            x, y = radius * math.cos(a), radius * math.sin(a)
            bottom_ring.append(len(vertices))
            vertices.append(((x, y, -half), (0, 0, -1), color))
            top_ring.append(len(vertices))
            vertices.append(((x, y, half), (0, 0, 1), color))
        for i in range(segments):
            j = (i + 1) % segments
            triangles.append((bottom_center, bottom_ring[j], bottom_ring[i]))
            triangles.append((top_center, top_ring[i], top_ring[j]))
    return _geom_node(name, vertices, triangles)


def frustum(
    name: str,
    bottom_radius: float,
    top_radius: float,
    height: float,
    color=(0.8, 0.8, 0.8, 1.0),
    segments: int = 32,
    capped: bool = True,
) -> NodePath:
    vertices: list[tuple] = []
    triangles: list[tuple[int, int, int]] = []
    half = height * 0.5
    slope = (bottom_radius - top_radius) / max(height, 1e-9)
    nz = slope
    nlen = math.sqrt(1.0 + nz * nz)
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        ca, sa = math.cos(a), math.sin(a)
        normal = (ca / nlen, sa / nlen, nz / nlen)
        vertices.append(((bottom_radius * ca, bottom_radius * sa, -half), normal, color))
        vertices.append(((top_radius * ca, top_radius * sa, half), normal, color))
    for i in range(segments):
        k = 2 * i
        triangles.extend([(k, k + 2, k + 1), (k + 1, k + 2, k + 3)])
    if capped:
        # Reuse simple flat caps with separate normals.
        bc = len(vertices)
        vertices.append(((0, 0, -half), (0, 0, -1), color))
        tc = len(vertices)
        vertices.append(((0, 0, half), (0, 0, 1), color))
        br, tr = [], []
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            ca, sa = math.cos(a), math.sin(a)
            br.append(len(vertices)); vertices.append(((bottom_radius * ca, bottom_radius * sa, -half), (0,0,-1), color))
            tr.append(len(vertices)); vertices.append(((top_radius * ca, top_radius * sa, half), (0,0,1), color))
        for i in range(segments):
            j = (i + 1) % segments
            triangles.append((bc, br[j], br[i]))
            triangles.append((tc, tr[i], tr[j]))
    return _geom_node(name, vertices, triangles)


def box(name: str, size_x: float, size_y: float, size_z: float, color=(1,1,1,1)) -> NodePath:
    hx, hy, hz = size_x / 2, size_y / 2, size_z / 2
    faces = [
        ((1,0,0), [(hx,-hy,-hz),(hx,hy,-hz),(hx,hy,hz),(hx,-hy,hz)]),
        ((-1,0,0), [(-hx,hy,-hz),(-hx,-hy,-hz),(-hx,-hy,hz),(-hx,hy,hz)]),
        ((0,1,0), [(-hx,hy,-hz),(hx,hy,-hz),(hx,hy,hz),(-hx,hy,hz)]),
        ((0,-1,0), [(hx,-hy,-hz),(-hx,-hy,-hz),(-hx,-hy,hz),(hx,-hy,hz)]),
        ((0,0,1), [(-hx,-hy,hz),(-hx,hy,hz),(hx,hy,hz),(hx,-hy,hz)]),
        ((0,0,-1), [(-hx,hy,-hz),(-hx,-hy,-hz),(hx,-hy,-hz),(hx,hy,-hz)]),
    ]
    vertices=[]; triangles=[]
    for normal, pts in faces:
        base=len(vertices)
        for p in pts: vertices.append((p,normal,color))
        triangles.extend([(base,base+1,base+2),(base,base+2,base+3)])
    return _geom_node(name, vertices, triangles)
