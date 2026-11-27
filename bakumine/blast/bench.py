"""The site of a blast: a demo bench, a free face cut into a topography, a default polygon.

plan_blast.md task B5. Nothing here is a blast model; these are the geometric helpers the
orchestration (:mod:`bakumine.blast.simulate`) needs to put a design on a ground surface.

Plan axes are those of :func:`bakumine.blast.design.face_axes`: ``f`` is the unit vector of
the face azimuth (the direction the free face looks and the rock is thrown), ``u`` is ``f``
turned 90 degrees clockwise; ``s = p . u`` runs along the face, ``t = p . f`` toward the
front. A ground surface is a level (float) or any object with ``z_at(x, y)`` (a
:class:`~bakumine.core.geometry.dtm.DTM`, a :class:`~bakumine.blast.throw.Heightmap`,
a :class:`StepBench`); NaN means "unknown there" and stays NaN.

* :class:`StepBench` / :func:`demo_site` -- a synthetic bench with a vertical face: the bench
  top behind the face line, the floor in front of it.
* :func:`cut_bench` -- **a visualisation assumption**: a blast designed on virgin topography
  (as the block modeller knows it) has no free face and no floor to throw onto, so the area
  in front of the block is taken as already excavated to the bench floor ("free face
  excavated in front of the block"). The surface is only ever lowered, never raised.
* :func:`default_polygon` -- a rectangle at the centre of a topography, for a request that
  gives no polygon.
* :func:`project_site` -- the topography (and, when asked, the blocks of a block model) of a
  project folder. Read only: nothing is written to the project.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bakumine.blast.design import face_axes
from bakumine.blast.throw import Heightmap

__all__ = [
    "DemoSite",
    "StepBench",
    "bench_for_polygon",
    "cut_bench",
    "default_polygon",
    "demo_site",
    "extent",
    "front_of_polygon",
    "ground_z",
    "project_site",
    "rectangle",
    "sample_ground",
]

Bounds = tuple[tuple[float, float], tuple[float, float]]


def _polygon(polygon: Any) -> np.ndarray:
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        raise ValueError("polygon needs at least three [x, y] vertices")
    poly = poly[:, :2]
    if not np.isfinite(poly).all():
        raise ValueError("polygon has non-finite coordinates")
    return poly


def ground_z(ground: float | Any, x: Any, y: Any) -> np.ndarray:
    """Height of a ground surface (a level or an object with ``z_at``) at plan positions."""
    x = np.asarray(x, dtype=np.float64)
    if isinstance(ground, int | float):
        return np.full(x.shape, float(ground))
    return np.asarray(ground.z_at(x, np.asarray(y, dtype=np.float64)), dtype=np.float64).reshape(
        x.shape
    )


def sample_ground(
    ground: float | Any, origin: Sequence[float], spacing: float, shape: Sequence[int]
) -> Heightmap:
    """``ground`` sampled on a regular grid: node ``[ix, iy]`` at ``origin + spacing * (ix, iy)``.

    NaN where the ground is unknown.
    """
    nx, ny = int(shape[0]), int(shape[1])
    if nx < 2 or ny < 2 or not spacing > 0:
        raise ValueError("a heightmap needs a spacing > 0 and at least 2 x 2 nodes")
    gx = float(origin[0]) + float(spacing) * np.arange(nx)
    gy = float(origin[1]) + float(spacing) * np.arange(ny)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    return Heightmap((float(origin[0]), float(origin[1])), float(spacing), ground_z(ground, xx, yy))


def rectangle(
    centre: Sequence[float], width: float, depth: float, face_azimuth: float
) -> list[list[float]]:
    """Rectangle ``width`` along the face and ``depth`` behind it, centred on ``centre``.

    Vertices: front left, front right, back right, back left (left and right for somebody
    who stands behind the blast and looks out over the face).
    """
    if not (width > 0 and depth > 0):
        raise ValueError("width and depth must be > 0")
    u, f = face_axes(face_azimuth)
    c = np.asarray(centre, dtype=np.float64)[:2]
    corners = [(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    return [[float(v) for v in c + a * width * u + b * depth * f] for a, b in corners]


def extent(polygon: Any, face_azimuth: float, apron: float, margin: float) -> Bounds:
    """``((xmin, ymin), (xmax, ymax))`` of the polygon plus ``apron`` metres in front of its
    face, widened by ``margin`` on every side."""
    poly = _polygon(polygon)
    u, f = face_axes(face_azimuth)
    s, t = poly @ u, poly @ f
    front = [
        (s.min(), t.max()),
        (s.max(), t.max()),
        (s.max(), t.max() + apron),
        (s.min(), t.max() + apron),
    ]
    pts = np.vstack([poly, [a * u + b * f for a, b in front]])
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    return (float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))


def front_of_polygon(x: Any, y: Any, polygon: Any, face_azimuth: float, apron: float) -> np.ndarray:
    """True for plan points in front of the block: within the block's width along the face,
    ahead of the polygon's front boundary at that position, and no more than ``apron``
    metres ahead of the polygon's foremost point. Points on the front boundary are not in
    front (the crest stays)."""
    poly = _polygon(polygon)
    u, f = face_axes(face_azimuth)
    ps, pt = poly @ u, poly @ f
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s = x * u[0] + y * u[1]
    t = x * f[0] + y * f[1]
    boundary = np.full(s.shape, -np.inf)  # the polygon's largest t at each s
    for (s1, t1), (s2, t2) in zip(
        zip(ps, pt, strict=True), zip(np.roll(ps, -1), np.roll(pt, -1), strict=True), strict=True
    ):
        if s1 == s2:
            hit = s == s1
            boundary = np.where(hit, np.maximum(boundary, max(t1, t2)), boundary)
            continue
        lo, hi = min(s1, s2), max(s1, s2)
        hit = (s >= lo) & (s <= hi)
        with np.errstate(invalid="ignore"):
            cross = t1 + (s - s1) * (t2 - t1) / (s2 - s1)
        boundary = np.where(hit, np.maximum(boundary, cross), boundary)
    tol = 1e-9
    return np.isfinite(boundary) & (t > boundary + tol) & (t <= pt.max() + apron)


# ------------------------------------------------------------------------------ demo bench


@dataclass(frozen=True)
class StepBench:
    """A bench with a vertical face: ``top_z`` on and behind the face line, ``floor_z`` in
    front of it. ``point`` is any plan point of the face line; defined everywhere."""

    point: tuple[float, float]
    face_azimuth: float
    top_z: float
    floor_z: float

    def z_at(self, x: Any, y: Any) -> np.ndarray:
        _, f = face_axes(self.face_azimuth)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t = (x - self.point[0]) * f[0] + (y - self.point[1]) * f[1]
        return np.where(t > 1e-9, self.floor_z, self.top_z)


@dataclass(frozen=True)
class DemoSite:
    """A synthetic bench: the block to blast, the ground and the area the ground is meant
    for (``bounds`` = ``((xmin, ymin), (xmax, ymax))``)."""

    polygon: list[list[float]]
    ground: StepBench
    top_z: float
    floor_z: float
    face_azimuth: float
    bounds: Bounds


def bench_for_polygon(
    polygon: Any,
    top_z: float = 1000.0,
    bench_height: float = 10.0,
    face_azimuth: float = 180.0,
    apron: float = 150.0,
    margin: float = 60.0,
) -> DemoSite:
    """A :class:`StepBench` whose face line passes through the foremost point of ``polygon``."""
    poly = _polygon(polygon)
    if not bench_height > 0:
        raise ValueError("bench_height must be > 0")
    _, f = face_axes(face_azimuth)
    front = poly[int(np.argmax(poly @ f))]
    ground = StepBench(
        (float(front[0]), float(front[1])),
        float(face_azimuth),
        float(top_z),
        float(top_z) - float(bench_height),
    )
    return DemoSite(
        polygon=[[float(px), float(py)] for px, py in poly],
        ground=ground,
        top_z=ground.top_z,
        floor_z=ground.floor_z,
        face_azimuth=float(face_azimuth),
        bounds=extent(poly, face_azimuth, apron, margin),
    )


def demo_site(
    top_z: float = 1000.0,
    bench_height: float = 10.0,
    width: float = 60.0,
    depth: float = 35.0,
    face_azimuth: float = 180.0,
    centre: Sequence[float] = (500.0, 500.0),
) -> DemoSite:
    """A synthetic bench for a blast without a project.

    The block to blast is a rectangle ``width`` along the face and ``depth`` behind it,
    centred on ``centre``. The ground is ``top_z`` on and behind the face line and
    ``top_z - bench_height`` in front of it (vertical face). ``bounds`` reaches 150 m in
    front of the face and 60 m to the sides and behind the block.
    """
    polygon = rectangle(centre, width, depth, face_azimuth)
    return bench_for_polygon(polygon, top_z, bench_height, face_azimuth, apron=150.0, margin=60.0)


# ------------------------------------------------------------------------------ free face


def cut_bench(
    ground: float | Any,
    polygon: Any,
    floor_z: float,
    face_azimuth: float,
    apron: float = 150.0,
    margin: float = 60.0,
    spacing: float | None = None,
) -> Heightmap:
    """The ground with a free face and a floor excavated in front of the block.

    The area in front of the block (:func:`front_of_polygon`: the width of the block,
    ``apron`` metres in the throw direction) is lowered to ``floor_z``. It is never raised:
    ground already below the floor stays as it is, and unknown ground (NaN) stays unknown.
    This is an assumption made so that a blast laid out on virgin topography has a face to
    break to and a floor to throw onto; it is not a pit design.

    The result is a heightmap over :func:`extent` (polygon and apron plus ``margin``) with
    a node spacing of 1 m, or coarser so that no side has more than about 300 nodes. On the
    grid the vertical face becomes a ramp one cell wide.
    """
    poly = _polygon(polygon)
    if not (apron > 0 and margin >= 0):
        raise ValueError("apron must be > 0 and margin >= 0")
    lo, hi = extent(poly, face_azimuth, apron, margin)
    side = max(hi[0] - lo[0], hi[1] - lo[1])
    step = max(1.0, side / 300.0) if spacing is None else float(spacing)
    if not step > 0:
        raise ValueError("spacing must be > 0")
    shape = (
        int(math.ceil((hi[0] - lo[0]) / step)) + 1,
        int(math.ceil((hi[1] - lo[1]) / step)) + 1,
    )
    grid = sample_ground(ground, lo, step, shape)
    gx = lo[0] + step * np.arange(shape[0])
    gy = lo[1] + step * np.arange(shape[1])
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    front = front_of_polygon(xx, yy, poly, face_azimuth, apron)
    z = grid.z.copy()
    with np.errstate(invalid="ignore"):
        lower = front & (z > floor_z)
    z[lower] = float(floor_z)
    return Heightmap(grid.origin, grid.spacing, z)


def default_polygon(
    surface: Any,
    centre: Sequence[float] | None = None,
    width: float = 60.0,
    depth: float = 35.0,
    face_azimuth: float = 180.0,
) -> list[list[float]]:
    """A rectangle (:func:`rectangle`) at the centre of a topography.

    ``surface`` is an object with ``bounds`` (``[[xmin, ymin, ...], [xmax, ymax, ...]]``, a
    DTM) or such a pair itself. ``centre`` defaults to the middle of the bounds.
    """
    bounds = np.asarray(getattr(surface, "bounds", surface), dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 2:
        raise ValueError("need a surface with bounds or ((xmin, ymin), (xmax, ymax))")
    if centre is None:
        centre = 0.5 * (bounds[0, :2] + bounds[1, :2])
    return rectangle(centre, width, depth, face_azimuth)


# ------------------------------------------------------------------------------ project


#: Id of a project surface made for the blast view; :func:`project_site` prefers it.
BLAST_TERRAIN_ID = "blast-terrain"


def project_site(
    folder: str | Path, topo: str | None = None, model: str | None = None
) -> tuple[Any, tuple[np.ndarray, np.ndarray] | None, dict[str, str | None]]:
    """Topography and blocks of a project folder.

    Returns ``(dtm, blocks, ids)``: the DTM object ``topo`` (default: the ``dtm`` with the id
    :data:`BLAST_TERRAIN_ID` when the project has one, else the first ``dtm`` of the project),
    ``blocks = (centres, sizes)`` of the block model ``model`` (``None`` when no model is
    asked for) and the ids used. Block sizes are along the model axes; a rotated model's
    blocks are passed on with their centres only rotated.

    Raises ``ValueError`` when the project has no DTM or an id is unknown.
    """
    from bakumine.core.geometry.dtm import DTM
    from bakumine.core.io.obj import read_obj
    from bakumine.core.project import Project

    project = Project.open(folder)

    def pick(object_id: str | None, kind: str) -> Any:
        objects = project.list(kind)
        if object_id is None:
            return objects[0] if objects else None
        for obj in objects:
            if obj.id == object_id:
                return obj
        raise ValueError(f"the project has no {kind} object {object_id!r}")

    # A surface made for the blast view (wider and finer than the modeller's topography) is
    # preferred when no id is asked for: its id is "blast-terrain".
    preferred = (
        None
        if topo is not None
        else next((obj for obj in project.list("dtm") if obj.id == BLAST_TERRAIN_ID), None)
    )
    topo_obj = preferred if preferred is not None else pick(topo, "dtm")
    if topo_obj is None:
        raise ValueError("the project has no topography (dtm object) to put the blast on")
    dtm = DTM(read_obj(project.path_of(topo_obj)))
    blocks = None
    model_obj = pick(model, "model") if model is not None else None
    if model_obj is not None:
        from bakumine.core.blockmodel.storage import load_model

        loaded = load_model(project.path_of(model_obj))
        blocks = (np.asarray(loaded.centroids()), np.asarray(loaded.sizes()))
    ids = {"topo": topo_obj.id, "model": None if model_obj is None else model_obj.id}
    return dtm, blocks, ids
