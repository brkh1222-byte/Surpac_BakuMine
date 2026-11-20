"""Section digitising endpoints (P5.T7): strings drawn on sections, and a solid from them.

    GET  /api/v1/digitise/strings?set=<name>       JSON: the saved strings of a set
    PUT  /api/v1/digitise/strings?set=<name>       replace the set with the body's strings
    GET  /api/v1/digitise/strings.str?set=<name>   the set as a Surpac ``.str`` text file
    POST /api/v1/digitise/solid                    build a solid from a set's closed strings
    DELETE /api/v1/digitise/solid/{id}             remove a solid from the scene

A *set* is one string file. Each digitised string is one segment; its points are world
coordinates (``x = east, y = north, z = up``). A closed string repeats its first point
at the end, as Surpac does. Per point, description ``d1`` is the section id the string
was drawn on and ``d2`` the string's id, so a set reads back exactly as it was saved
(coordinates rounded to millimetres by the ``.str`` writer).

Where a set lives: with a project served (``BAKUMINE_PROJECT``), in the project's
``strings/`` folder as ``<set>.str``, registered as a ``string`` object; otherwise in
memory for the life of the process (demo). A built solid is added to the scene: to the
project as a ``solid`` object (OBJ in ``solids/``), or to the demo source in memory.

The solid is :func:`bakumine.core.geometry.sections_to_solid.solid_from_strings` on the
set's closed strings, checked with :func:`bakumine.core.geometry.validation.validate`.
"""

from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from bakumine.api.scene import SceneSource, get_scene_source
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io.strings import StringFile, StringHeader, StringSegment
from bakumine.core.io.surpac_str import SurpacFormatError, format_surpac_str, parse_surpac_str

router = APIRouter(prefix="/api/v1/digitise", tags=["digitise"])
Source = Annotated[SceneSource, Depends(get_scene_source)]

DEFAULT_SET = "digitised"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
#: description fields cannot hold commas or line breaks (``.str`` layout)
_TEXT = re.compile(r"^[^,\r\n]{0,128}$")
SetName = Annotated[str, Query(alias="set", description="string set name")]


class DigitisedString(BaseModel):
    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[^,\r\n]+$")
    #: section the string was drawn on (any text without commas)
    section: str = Field("", max_length=128, pattern=r"^[^,\r\n]*$")
    #: string number in the ``.str`` file
    number: int = Field(1, ge=1, le=32000)
    closed: bool = False
    #: world points, the closing point NOT repeated
    points: list[tuple[float, float, float]]


class StringsBody(BaseModel):
    strings: list[DigitisedString]


class SolidRequest(BaseModel):
    set: str = DEFAULT_SET
    name: str = Field("Digitised solid", min_length=1, max_length=64)
    ends: Literal["flat", "point", "taper"] = "flat"
    method: Literal["min_area", "min_span", "greedy"] = "min_area"


# ---------------------------------------------------------------------- conversion


def to_string_file(strings: list[DigitisedString]) -> StringFile:
    segments = []
    for s in strings:
        points = np.asarray(s.points, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise HTTPException(422, f"string {s.id!r} has non-finite coordinates")
        if len(points) == 0:
            continue
        if s.closed:
            if len(points) < 3:
                raise HTTPException(422, f"closed string {s.id!r} needs at least 3 points")
            points = np.vstack([points, points[:1]])
        descriptions = [(s.section, s.id)] * len(points)
        segments.append(StringSegment(s.number, points, descriptions))
    return StringFile(segments, header=StringHeader("bakumine", "", "section digitising", ""))


def from_string_file(strings: StringFile) -> list[dict[str, Any]]:
    out = []
    for k, seg in enumerate(strings.segments):
        closed = seg.closed(1e-6)
        points = seg.points[:-1] if closed else seg.points
        first = seg.descriptions[0] if seg.descriptions else ()
        out.append(
            {
                "id": first[1] if len(first) > 1 and first[1] else f"s{k + 1}",
                "section": first[0] if first else "",
                "number": seg.number,
                "closed": closed,
                "points": points.tolist(),
            }
        )
    return out


# ---------------------------------------------------------------------- storage


class _Memory:
    """String sets of sources that have no folder (demo, empty), by source identity."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.texts: dict[tuple[int, str], str] = {}


_memory = _Memory()


def _project(source: SceneSource):
    from bakumine.api.project_source import ProjectSceneSource

    if isinstance(source, ProjectSceneSource):
        from bakumine.core.project import Project

        return Project.open(source.folder)
    return None


def _check_set(name: str) -> str:
    if not _NAME.match(name):
        raise HTTPException(400, "set name: letters, digits, '-' and '_', at most 64")
    return name


def _set_object_id(name: str) -> str:
    return f"strings-{name.lower()}"


def load_text(source: SceneSource, name: str) -> str | None:
    project = _project(source)
    if project is None:
        with _memory.lock:
            return _memory.texts.get((id(source), name))
    try:
        path = project.path_of(_set_object_id(name))
    except KeyError:
        return None
    return path.read_text(encoding="utf-8") if path.is_file() else None


def save_text(source: SceneSource, name: str, text: str) -> str | None:
    """Stores the set; returns the project object id (None in memory)."""
    project = _project(source)
    if project is None:
        with _memory.lock:
            _memory.texts[(id(source), name)] = text
        return None
    object_id = _set_object_id(name)
    try:
        path = project.path_of(object_id)
    except KeyError:
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / f"{name}.str"
            file.write_text(text, encoding="utf-8", newline="\n")
            project.add("string", file, name=name, object_id=object_id)
        return object_id
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return object_id


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "solid"


def add_solid(source: SceneSource, name: str, mesh: Mesh) -> str:
    """Adds (or replaces) a solid in the scene; returns its object id."""
    object_id = _slug(name)
    project = _project(source)
    if project is not None:
        from bakumine.core.io.obj import write_obj

        if any(o.id == object_id for o in project.list()):
            if project.get(object_id).kind != "solid":
                raise HTTPException(409, f"object {object_id!r} exists and is not a solid")
            project.remove(object_id)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / f"{object_id}.obj"
            write_obj(file, mesh, decimals=None, comment="solid from digitised sections")
            project.add("solid", file, name=name, object_id=object_id)
        return object_id
    add_mesh = getattr(source, "add_mesh", None)
    if add_mesh is None:
        raise HTTPException(409, "this scene cannot take new solids")
    try:
        add_mesh(object_id, name, mesh)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return object_id


# ---------------------------------------------------------------------- endpoints


def _read(source: SceneSource, name: str) -> StringFile:
    text = load_text(source, name)
    if text is None:
        return StringFile()
    return parse_surpac_str(text, source=f"{name}.str")


@router.get("/strings")
def get_strings(source: Source, name: SetName = DEFAULT_SET) -> dict[str, Any]:
    _check_set(name)
    return {"set": name, "strings": from_string_file(_read(source, name))}


@router.put("/strings")
def put_strings(body: StringsBody, source: Source, name: SetName = DEFAULT_SET) -> dict[str, Any]:
    _check_set(name)
    ids = [s.id for s in body.strings]
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "string ids must be unique")
    strings = to_string_file(body.strings)
    try:
        text = format_surpac_str(strings)
    except SurpacFormatError as exc:
        raise HTTPException(422, str(exc)) from None
    object_id = save_text(source, name, text)
    saved = from_string_file(parse_surpac_str(text))
    return {
        "set": name,
        "object_id": object_id,
        "n_strings": len(saved),
        "n_points": sum(len(s["points"]) for s in saved),
        "strings": saved,
    }


@router.get("/strings.str")
def get_strings_str(source: Source, name: SetName = DEFAULT_SET) -> Response:
    _check_set(name)
    text = load_text(source, name)
    if text is None:
        raise HTTPException(404, f"no string set {name!r}")
    return Response(
        content=text,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{name}.str"'},
    )


@router.post("/solid")
def post_solid(body: SolidRequest, source: Source) -> dict[str, Any]:
    from bakumine.core.geometry.sections_to_solid import SectionSolidError, solid_from_strings
    from bakumine.core.geometry.validation import validate

    _check_set(body.set)
    strings = _read(source, body.set)
    closed = [s for s in strings.segments if s.closed(1e-6)]
    skipped = len(strings.segments) - len(closed)
    try:
        mesh = solid_from_strings(StringFile(closed), ends=body.ends, method=body.method)
    except NotImplementedError as exc:
        raise HTTPException(422, {"message": str(exc), "problems": []}) from None
    except SectionSolidError as exc:
        problems = [
            {"kind": p.kind, "sections": list(p.sections), "message": p.message}
            for p in exc.problems
        ]
        raise HTTPException(422, {"message": str(exc), "problems": problems}) from None
    report = validate(mesh)
    object_id = add_solid(source, body.name, mesh)
    sections = _count_sections(closed)
    return {
        "id": object_id,
        "name": body.name,
        "valid": report.is_valid,
        "watertight": report.is_closed and report.is_manifold,
        "outward": report.is_outward,
        "volume": float(mesh.volume),
        "n_vertices": int(mesh.n_vertices),
        "n_faces": int(mesh.n_faces),
        "n_strings": len(closed),
        "n_sections": sections,
        "skipped_open": skipped,
        "issues": report.issues(),
    }


def _count_sections(segments: list[StringSegment]) -> int:
    ids = {seg.descriptions[0][0] for seg in segments if seg.descriptions and seg.descriptions[0]}
    ids.discard("")
    return len(ids) if ids else len(segments)


@router.delete("/solid/{object_id}")
def delete_solid(object_id: str, source: Source) -> dict[str, str]:
    """Removes a solid: a project's solid object, or a demo solid added in memory."""
    project = _project(source)
    try:
        if project is not None:
            if project.get(object_id).kind != "solid":
                raise KeyError(object_id)
            project.remove(object_id)
        else:
            remove = getattr(source, "remove_mesh", None)
            if remove is None:
                raise KeyError(object_id)
            remove(object_id)
    except KeyError:
        raise HTTPException(404, f"no removable solid {object_id!r}") from None
    return {"removed": object_id}
