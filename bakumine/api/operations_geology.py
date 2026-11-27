"""Geology operations behind the UI panel: ``bakumine geology build`` and ``geology code``
on project objects, taking both off again (``geology uncode``: the coded attributes of a
block model; ``geology remove``: the geology model itself), and what the viewer reads of
a geology model.

Like :mod:`bakumine.api.operations_implicit` they call the functions the CLI calls
(:func:`bakumine.core.geology.spec.build_geology`, :func:`bakumine.core.geology.coding.
code_geology`), write a job-log entry and add the result to the project.

**Build spec in the API** (geology_contracts.md section 10). The files of a spec are project
objects, each written ``{"object": "<id>"}`` (or the bare id):

``database``                              a database
``topography``                            a dtm; ``{"object": id, "source": "survey"}`` to
                                          state where the surface comes from
``extent.model``                          a block model
``faults.surfaces[].replace_with.file``   a dtm or a solid (the fault surface)
``units``                                 the dictionary itself (an object with ``units``),
                                          or ``{"object": id}``: the dictionary of a
                                          geology model of the project

Other files cannot be named: ``extra_intervals`` takes ``rows`` and ``replace_with.points``
a list of ``[x, y, z]``. The spec is rewritten with the objects' paths inside the project
and run with the project folder as its base; that rewritten spec is what the job log keeps
(``replay.spec``) and what ``bakumine geology build --base PROJECT`` runs again.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from bakumine.api.operations import (
    OperationError,
    _claim,
    _fail,
    _model_path,
    _record,
    _release,
    _slug,
    _will_modify,
)
from bakumine.api.operations_geometry import _ERRORS, _name, _Sources
from bakumine.core import joblog
from bakumine.core.project import Project

GEOLOGY_MANIFEST = "geology.json"


def _object_id(ref: Any, what: str) -> str:
    if isinstance(ref, dict) and isinstance(ref.get("object"), str):
        return ref["object"]
    if isinstance(ref, str) and ref:
        return ref
    raise OperationError(f'{what}: name a project object, {{"object": "<id>"}}; got {ref!r}')


def project_spec(spec: Any, sources: _Sources) -> dict[str, Any]:
    """The request's spec with every project object replaced by its path in the project.

    The objects are noted as inputs of the job. Raises :class:`OperationError` for a
    reference that is not a project object of the right kind.
    """
    if not isinstance(spec, dict):
        raise OperationError("spec must be an object (the geology build spec)")
    out = json.loads(json.dumps(spec))
    project = sources.project

    def path_of(ref: Any, what: str, *kinds: str) -> str:
        return project.get(sources.obj(_object_id(ref, what), *kinds).id).path

    if out.get("database") is None:
        raise OperationError('spec.database: name the drillhole database, {"object": "<id>"}')
    out["database"] = path_of(out["database"], "spec.database", "database")
    if out.get("topography") is not None:
        ref = out["topography"]
        path = path_of(ref, "spec.topography", "dtm")
        stated = ref.get("source") if isinstance(ref, dict) else None
        out["topography"] = {"file": path, "source": stated} if stated else path
    extent = out.get("extent")
    if isinstance(extent, dict) and "model" in extent:
        extent["model"] = path_of(extent["model"], "spec.extent.model", "model")
    units = out.get("units")
    if isinstance(units, dict) and "object" in units and "units" not in units:
        out["units"] = path_of(units, "spec.units", "geology") + "/units.json"
    elif not isinstance(units, dict):
        raise OperationError(
            "spec.units: give the unit dictionary itself (an object) or the geology model "
            'whose dictionary to use, {"object": "<id>"}'
        )
    faults = out.get("faults")
    for i, surface in enumerate((faults or {}).get("surfaces") or [] if isinstance(faults, dict)
                                else []):  # fmt: skip
        replace = surface.get("replace_with") if isinstance(surface, dict) else None
        if not isinstance(replace, dict):
            continue
        where = f"spec.faults.surfaces[{i}].replace_with"
        if "file" in replace:
            replace["file"] = path_of(replace["file"], f"{where}.file", "dtm", "solid")
        if isinstance(replace.get("points"), str):
            raise OperationError(f"{where}.points: give the points as a list of [x, y, z]")
    for i, extra in enumerate(out.get("extra_intervals") or []):
        if isinstance(extra, dict) and "file" in extra:
            raise OperationError(
                f"spec.extra_intervals[{i}]: a file cannot be named here; give the intervals "
                "as 'rows' (objects with hole_id, depth_from, depth_to and the code)"
            )
    return out


def geology_build(source: Any, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine geology build``: a geology model of the project's drillholes."""
    return _record(
        source.folder, "geology_build", request, lambda rec: _build(source, request, rec)
    )


def _build(source: Any, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.geology.spec import build_geology, build_summary, load_spec, save_geology

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    folder = Path(source.folder)
    given = request.get("spec")
    default = given.get("name") if isinstance(given, dict) else None
    name = _name(request, default if isinstance(default, str) and default.strip() else "geology")
    sources = _Sources(folder, rec)
    try:
        document = project_spec(given, sources)
        document["name"] = name
        spec = load_spec(document, base_dir=folder)
        model = build_geology(spec)
        summary = build_summary(model)
        with tempfile.TemporaryDirectory() as tmp:
            target = save_geology(model, Path(tmp) / _slug(name))
            obj = Project.open(folder).add("geology", target, name)
    except OperationError:
        raise
    except _ERRORS as exc:
        raise _fail(exc) from None
    rec.entry["outputs"] = [obj.to_dict()]
    rec.entry["replay"] = {"spec": spec.to_dict()}
    return {"id": obj.id, "kind": "geology", **summary}


# ------------------------------------------------------------------ geology code


def geology_code(source: Any, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine geology code``: code a block model of the project from a geology model."""
    return _record(source.folder, "geology_code", request, lambda rec: _code(source, request, rec))


def _text(request: dict[str, Any], key: str) -> str | None:
    value = request.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise OperationError(f"{key} must be text (an attribute name)")
    return value


def _code(source: Any, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.blockmodel.storage import load_model, save_model
    from bakumine.core.geology.coding import (
        DEFAULT_LAYERS,
        code_geology,
        coding_provenance_path,
        coding_record_path,
        read_coding_record,
        unit_tonnage,
        write_coding_provenance,
        write_coding_record,
    )
    from bakumine.core.geology.model import GeologyModel
    from bakumine.core.geology.spec import json_safe

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    folder = Path(source.folder)
    model_id = request.get("model")
    if not isinstance(model_id, str):
        raise OperationError("model: the id of the block model to code")
    path = _model_path(folder, model_id)
    geology_folder = geology_path(folder, request.get("geology"))
    layers = request.get("layers", list(DEFAULT_LAYERS))
    if not isinstance(layers, list) or not layers or not all(isinstance(v, str) for v in layers):
        raise OperationError(f"layers must be a list of {', '.join(DEFAULT_LAYERS)}")
    for key in ("subblock", "merge"):
        if not isinstance(request.get(key, True), bool):
            raise OperationError(f"{key} must be true or false")
    density = "density" if "density" not in request else _text(request, "density")
    display, ore = _text(request, "display"), _text(request, "ore_attribute")
    max_level = request.get("max_level")
    if max_level is not None and (isinstance(max_level, bool) or not isinstance(max_level, int)):
        raise OperationError("max_level must be a whole number")
    model_name = path.name
    _claim(model_id)
    try:
        # an unusable name fails now
        provenance_file = coding_provenance_path(geology_folder, model_name)
        _will_modify(rec, folder, model_id, path, None)
        if rec.entry["snapshot"]:  # undo also takes back the two records of the coding
            joblog.snapshot_files(folder, rec.id, [coding_record_path(path), provenance_file])
        rec.entry["inputs"].append(Project.open(folder).get(request["geology"]).to_dict())
        geology = GeologyModel.load(geology_folder)
        model = load_model(path)
        result = code_geology(
            model, geology, geology_name=geology_folder.name, layers=layers,
            subblock=request.get("subblock", True), max_level=max_level, density=density,
            distance=_text(request, "distance"), display=display, ore_attribute=ore,
            merge=request.get("merge", False),
        )  # fmt: skip
        header = save_model(model, path)
        coded_before = read_coding_record(path) is None and provenance_file.is_file()
        write_coding_record(
            path, model, result, geology=request["geology"], coded_before=coded_before
        )
        write_coding_provenance(geology_folder, model_name, result.provenance)
        tonnage = None
        if "lith" in result.attributes:
            tonnage = unit_tonnage(model, lith=result.attributes["lith"], density=density or "")
    except OperationError:
        raise
    except _ERRORS as exc:
        raise _fail(exc) from None
    finally:
        _release(model_id)
    kinds = {"measured": 0, "assumed": 0, "missing": 0}
    for entry in result.provenance:
        kinds[entry.kind] += 1
    return json_safe(
        {
            "model": model_id,
            "geology": request["geology"],
            "n_blocks": int(header.n_blocks),
            "attributes": dict(result.attributes),
            "skipped": dict(result.skipped),
            "n_split": int(sum(r.n_split for r in result.results)),
            "n_merged": int(result.n_merged),
            "volumes": result.volumes,
            "tonnage": tonnage,
            "provenance": kinds,
        }
    )


# ------------------------------------------------------------------ geology uncode


def geology_uncode(source: Any, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine geology uncode``: drop the attributes a geology coding wrote to a model."""
    return _record(
        source.folder, "geology_uncode", request, lambda rec: _uncode(source, request, rec)
    )


def coding_provenance_files(folder: Path, model_name: str) -> list[tuple[Any, Path]]:
    """(geology object, its record of coding the model ``model_name``) where one exists."""
    project = Project.open(folder)
    out = []
    for obj in project.list("geology"):
        file = project.path_of(obj) / "coding" / f"{model_name}.provenance.json"
        if file.is_file():
            out.append((obj, file))
    return out


def _uncode(source: Any, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.blockmodel.storage import load_model, save_model
    from bakumine.core.geology.coding import (
        coding_record_path,
        read_coding_record,
        remove_coding,
        save_coding_record,
    )

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    folder = Path(source.folder)
    model_id = request.get("model")
    if not isinstance(model_id, str):
        raise OperationError("model: the id of the block model to take the coding off")
    path = _model_path(folder, model_id)
    names = request.get("attributes")
    if names is not None and (
        not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names)
    ):
        raise OperationError("attributes must be a list of attribute names")
    merge = request.get("merge")
    if merge is not None and not isinstance(merge, bool):
        raise OperationError("merge must be true or false")
    try:
        record = read_coding_record(path)
    except ValueError as exc:
        raise OperationError(str(exc)) from None
    if record is None and names is None:
        raise OperationError(
            f"block model {model_id!r} has no geology coding to remove (no coding record "
            f"{coding_record_path(path).name}); to drop attributes of an older coding, name "
            "them in 'attributes'"
        )
    provenance = coding_provenance_files(folder, path.name)
    _claim(model_id)
    try:
        _will_modify(rec, folder, model_id, path, None)
        rec.entry["inputs"] += [obj.to_dict() for obj, _ in provenance]
        if rec.entry["snapshot"]:
            files = [coding_record_path(path), *(file for _, file in provenance)]
            joblog.snapshot_files(folder, rec.id, files)
        model = load_model(path)
        try:
            removal = remove_coding(model, record, attributes=names, merge=merge)
        except ValueError as exc:
            raise OperationError(f"block model {model_id!r}: {exc}") from None
        header = save_model(model, path)
        save_coding_record(path, removal.record)
        if removal.record is None:  # nothing of the coding is left: its records go too
            for _, file in provenance:
                file.unlink(missing_ok=True)
    except OperationError:
        raise
    except _ERRORS as exc:
        raise _fail(exc) from None
    finally:
        _release(model_id)
    # replay drops exactly these attributes, whatever record the new project has
    rec.entry["replay"] = {"attributes": removal.removed, "merge": removal.n_merged > 0}
    return {
        "model": model_id,
        "removed": removal.removed,
        "kept": removal.kept,
        "n_merged": removal.n_merged,
        "n_blocks": int(header.n_blocks),
        "geology": [obj.id for obj, _ in provenance] if removal.record is None else [],
    }


def model_coding(folder: Path, model_path: Path) -> dict[str, Any] | None:
    """What the viewer shows of a model's geology coding: ``{"geology", "layers": {layer:
    attribute}, "attributes": [{"name", "role", "created"}]}``, or None without a record."""
    from bakumine.core.geology.coding import LAYERS, read_coding_record

    try:
        record = read_coding_record(model_path)
    except ValueError:
        return None
    if record is None:
        return None
    return {
        "geology": record.get("geology"),
        "layers": {a["role"]: a["name"] for a in record["attributes"] if a.get("role") in LAYERS},
        "attributes": record["attributes"],
    }


# ------------------------------------------------------------------ geology remove


def geology_remove(source: Any, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine geology remove``: take a geology model and its files out of the project."""
    return _record(
        source.folder, "geology_remove", request, lambda rec: _remove(source, request, rec)
    )


def _remove(source: Any, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    import shutil

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    folder = Path(source.folder)
    object_id = request.get("geology")
    project = Project.open(folder)
    try:
        if not isinstance(object_id, str) or project.get(object_id).kind != "geology":
            raise KeyError(object_id)
    except KeyError:
        raise OperationError(f"no geology model {object_id!r} in the project") from None
    obj = project.get(object_id)
    path = project.path_of(obj)
    coded = sorted(
        p.name.removesuffix(".provenance.json") for p in (path / "coding").glob("*.provenance.json")
    )
    _claim(object_id)
    try:
        rec.entry["inputs"] = [obj.to_dict()]
        rec.entry["removes"] = [obj.to_dict()]
        undoable = joblog.keep_snapshots() > 0 and path.exists()
        if undoable:
            # the snapshot IS the object: moved aside, moved back by undo
            snapshot = joblog.take_object_snapshot(folder, rec.id, obj.to_dict(), path)
            try:
                project.remove(object_id, delete_files=False)
            except Exception:
                shutil.move(str(snapshot / "object"), str(path))
                joblog.drop_snapshot(folder, rec.id)
                raise
            rec.entry.update(modifies=object_id, snapshot=True)
        else:
            project.remove(object_id)
    except OperationError:
        raise
    except _ERRORS as exc:
        raise _fail(exc) from None
    finally:
        _release(object_id)
    return {
        "removed": object_id,
        "name": obj.name,
        "path": obj.path,
        "undoable": undoable,
        # block models coded from it keep their attributes (``geology uncode`` drops them)
        "coded_models": coded,
    }


# ------------------------------------------------------------------ reading a model


def geology_path(folder: Path, object_id: Any) -> Path:
    """Folder of the geology object ``object_id``; :class:`OperationError` when there is none."""
    project = Project.open(folder)
    try:
        if not isinstance(object_id, str) or project.get(object_id).kind != "geology":
            raise KeyError(object_id)
        path = project.path_of(object_id)
    except KeyError:
        raise OperationError(f"no geology model {object_id!r}") from None
    if not (path / GEOLOGY_MANIFEST).is_file():
        raise OperationError(f"geology model {object_id!r} has no {GEOLOGY_MANIFEST}")
    return path


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def geology_children(object_id: str, folder: Path) -> list[dict[str, Any]]:
    """The meshes of a geology folder the viewer can draw, in manifest order.

    One dictionary per mesh: ``id`` (``<object id>.unit.<CODE>``, ``.fault.<NAME>`` or
    ``.surface.<stem>``), ``name``, ``group`` (``Units``, ``Faults``, ``Surfaces``),
    ``colour`` (``#rrggbb`` or None), ``file`` (path inside the folder), ``code``.
    """
    from bakumine.core.geology.units import UnitDictionary

    manifest = _read(folder / GEOLOGY_MANIFEST) or {}
    try:
        units = UnitDictionary.load(folder / "units.json")
        colours = units.colours()
        names = {u.code: u.name for u in (*units.units, *units.weathering_zones)}
    except (OSError, ValueError):
        colours, names = {}, {}
    out: list[dict[str, Any]] = []

    def add(part: str, key: str, name: str, group: str, colour: str | None, file: str) -> None:
        if (folder / file).is_file():
            out.append({"id": f"{object_id}.{part}.{key}", "name": name, "group": group,
                        "colour": colour, "file": file, "code": key})  # fmt: skip

    for code, file in ((manifest.get("meshes") or {}).get("lith") or {}).items():
        add("unit", code, names.get(code) or code, "Units", colours.get(code), file)
    for fault in manifest.get("faults") or []:
        if fault.get("mesh"):
            add("fault", fault["name"], f"Fault {fault['name']}", "Faults", None, fault["mesh"])
    surfaces = []
    if manifest.get("cover"):
        surfaces.append((manifest["cover"]["unit"], manifest["cover"]["surface"]))
    surfaces += [(w["zone"], w["base"]) for w in manifest.get("weathering") or [] if w.get("base")]
    for code, file in surfaces:
        stem = Path(file).stem
        label = f"Base of {names.get(code) or code}"
        add("surface", stem, label, "Surfaces", colours.get(code), file)
    return out


def geology_info(folder: Path, object_id: str, name: str | None = None) -> dict[str, Any]:
    """``GET /api/v1/geology/{id}``: what the panel shows of a geology model."""
    path = geology_path(folder, object_id)
    manifest = _read(path / GEOLOGY_MANIFEST)
    children = geology_children(object_id, path)
    by_file = {c["file"]: c for c in children}
    try:
        from bakumine.core.geology.units import UnitDictionary

        dictionary = UnitDictionary.load(path / "units.json")
        colours = dictionary.colours()
        described = {u.code: (u.name, u.kind) for u in dictionary.units}
        described.update({z.code: (z.name, "weathering") for z in dictionary.weathering_zones})
    except (OSError, ValueError):
        colours, described = {}, {}
    meshes = (manifest.get("meshes") or {}).get("lith") or {}
    units = []
    for code in manifest["layers"]["lith"]:
        if code == "AIR":
            continue
        mesh = by_file.get(meshes.get(code, ""))
        units.append(
            {"code": code, "name": described.get(code, (None, None))[0],
             "kind": described.get(code, (None, None))[1], "colour": colours.get(code),
             "mesh": mesh["id"] if mesh else None}
        )  # fmt: skip
    faults = []
    for item in manifest.get("faults") or []:
        record = _read(path / item["file"]) or {}
        plane = record.get("plane") or {}
        mesh = by_file.get(item.get("mesh") or "")
        faults.append(
            {"name": item["name"], "source": record.get("source"),
             "dip": plane.get("dip"), "dip_azimuth": plane.get("dip_azimuth"),
             "blocks": record.get("blocks"), "misfit": record.get("misfit"),
             "mesh": mesh["id"] if mesh else None}
        )  # fmt: skip
    surfaces = []
    listed = []
    if manifest.get("cover"):
        listed.append(("cover", manifest["cover"]["unit"], manifest["cover"]["surface"]))
    listed += [("weathering", w["zone"], w["base"]) for w in manifest.get("weathering") or []]
    for role, code, file in listed:
        if file is None:
            continue
        side = _read((path / file).with_suffix(".json")) or {}
        mesh = by_file.get(file)
        surfaces.append(
            {"name": f"base:{code}", "role": role, "code": code, "colour": colours.get(code),
             "mesh": mesh["id"] if mesh else None, "diagnostics": side.get("diagnostics")}
        )  # fmt: skip
    grid = manifest["grid"]
    lo = [float(v) for v in grid["origin"]]
    hi = [o + s * (n - 1) for o, s, n in zip(lo, grid["spacing"], grid["shape"], strict=True)]
    coding = sorted(p.name.removesuffix(".provenance.json")
                    for p in (path / "coding").glob("*.provenance.json"))  # fmt: skip
    return {
        "id": object_id,
        "name": name or manifest.get("name"),
        "model_name": manifest.get("name"),
        "created": manifest.get("created"),
        "grid": grid,
        "bounds": {"min": lo, "max": hi},
        "layers": manifest["layers"],
        "blocks": manifest.get("blocks", []),
        "pooled_blocks": manifest.get("pooled_blocks", []),
        "units": units,
        "faults": faults,
        "surfaces": surfaces,
        "max_extrapolation": manifest.get("max_extrapolation"),
        "diagnostics": _read(path / "diagnostics.json") or {},
        "spec": _read(path / "spec.json"),
        "coded_models": coding,
    }


def geology_provenance(folder: Path, object_id: str) -> dict[str, Any]:
    """``GET /api/v1/geology/{id}/provenance``: the build record and every coding record."""
    path = geology_path(folder, object_id)
    coding = {
        file.name.removesuffix(".provenance.json"): _read(file)
        for file in sorted((path / "coding").glob("*.provenance.json"))
    }
    return {"build": _read(path / "provenance.json"), "coding": coding}
