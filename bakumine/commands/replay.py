"""Replay a project job log with the CLI (P6.T7).

``bakumine project replay LOG --into NEW`` turns each finished, not undone job of a
``jobs.jsonl`` (see :mod:`bakumine.core.joblog`) into the CLI commands that do the same
thing and runs them into a new project:

    import       bakumine import (archived upload files, --map) + validate + project add
    validate     bakumine validate --allow-errors
    composite    bakumine composite --out WORK/... + project add
    create_model bakumine blockmodel create WORK/... + project add
    code         bakumine blockmodel code --steps WORK/....json [--subblock] [--merge]
    estimate     bakumine estimate --plan WORK/....json
    report       bakumine blockmodel report --out WORK/....csv
    surface_from_points / import_mesh
                 bakumine surface from-points | surface import --out WORK/....obj
                 + project add (:data:`bakumine.commands.geometry.REPLAY`)

Objects a job reads that no earlier job made (solids, DTMs added to the project by hand)
are copied from the source project with ``bakumine project add``; solids and DTMs are
handed to ``--geometry-dir`` by ``bakumine project export-geometry``.
``bakumine project export-log LOG --out FILE`` writes the same steps as a shell (``.sh``)
or Python (``.py``) script instead of running them.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

from bakumine.core import joblog

PROJECT, WORK, SOURCE = "@PROJECT@", "@WORK@", "@SOURCE@"
_TOKENS = re.compile(r"(@PROJECT@|@WORK@|@SOURCE@)")

#: ImportOptions field -> CLI flag (store_true flags have no value)
_IMPORT_FLAGS = {
    "length_unit": "--length-unit",
    "convert_to_metres": "--to-metres",
    "dip_convention": "--dip-convention",
    "normalise_dip": "--normalise-dip",
    "coordinate_order": "--coordinate-order",
    "below_detection": "--below-detection",
    "delimiter": "--delimiter",
}


class ReplayError(ValueError):
    """A log entry cannot be expressed with the CLI."""


def _num(value: Any) -> str:
    return repr(float(value))


def _triple(values: Any) -> str:
    return ",".join(_num(v) for v in values)


def _run(*argv: str, note: str = "") -> dict[str, Any]:
    return {"kind": "run", "argv": list(argv), "note": note}


def _write(path: str, text: str) -> dict[str, Any]:
    return {"kind": "write", "path": path, "text": text}


def _add(obj: dict[str, str], source: str) -> dict[str, Any]:
    step = _run(
        "project", "add", PROJECT, obj["kind"], source, "--name", obj["name"], "--id", obj["id"]
    )
    step["expect"] = {"id": obj["id"], "path": obj["path"]}
    return step


def _work(job_id: str, name: str) -> str:
    return f"{WORK}/{job_id}/{name}"


def _in_project(obj: dict[str, str]) -> str:
    return f"{PROJECT}/{obj['path']}"


def _geometry(entry: dict[str, Any]) -> list[str]:
    """Steps' ``--geometry-dir`` arguments (and the export before them), when needed."""
    if not any(o["kind"] in ("solid", "dtm") for o in entry.get("inputs", [])):
        return []
    return ["--geometry-dir", f"{WORK}/geometry"]


def _export_geometry() -> dict[str, Any]:
    return _run("project", "export-geometry", PROJECT, f"{WORK}/geometry")


# ------------------------------------------------------------------ one job each


def _import(e: dict[str, Any]) -> list[dict[str, Any]]:
    from bakumine.core.drillholes.importer import ImportOptions

    replay = e.get("replay") or {}
    files: dict[str, str] = replay.get("files") or {}
    out = e["outputs"][0]
    folder = _work(e["id"], Path(out["path"]).name)
    argv = ["import"]
    for table, rel in files.items():
        path = f"{SOURCE}/{rel}"
        if table in ("collar", "survey", "assay", "geology"):
            argv += [f"--{table}", path]
        else:
            argv += ["--interval", f"{table}={path}"]
    map_file = _work(e["id"], "map.json")
    argv += ["--map", map_file, "--out", folder]
    options = {k: v for k, v in ((e["params"].get("options") or {}).items()) if v not in (None, "")}
    defaults = ImportOptions()
    for key, value in options.items():
        if key in ("text_columns", "numeric_columns"):
            flag = "--text-column" if key == "text_columns" else "--numeric-column"
            for column in [value] if isinstance(value, str) else value:
                argv += [flag, str(column)]
        elif key in ("convert_to_metres", "normalise_dip"):
            if value:
                argv.append(_IMPORT_FLAGS[key])
        elif key in _IMPORT_FLAGS:
            argv += [_IMPORT_FLAGS[key], str(value)]
        elif getattr(defaults, key, object()) != value:
            raise ReplayError(f"job {e['id']}: import option {key!r} has no CLI flag")
    return [
        _write(map_file, json.dumps(replay.get("maps") or {}, indent=2)),
        _run(*argv),
        _run("validate", folder, "--allow-errors"),
        _add(out, folder),
    ]


def _validate(e: dict[str, Any]) -> list[dict[str, Any]]:
    return [_run("validate", _in_project(e["inputs"][0]), "--allow-errors")]


def _composite(e: dict[str, Any]) -> list[dict[str, Any]]:
    p, out = e["params"], e["outputs"][0]
    base = _work(e["id"], Path(out["path"]).stem)
    argv = ["composite", _in_project(e["inputs"][0])]
    argv += ["--table", p.get("table") or "assay", "--mode", p.get("mode") or "downhole"]
    if p.get("length") is not None:
        argv += ["--length", _num(p["length"])]
    if p.get("bench_height") is not None:
        argv += ["--bench-height", _num(p["bench_height"])]
    argv += ["--bench-reference", _num(p.get("bench_reference") or 0.0)]
    if p.get("domain_column"):
        argv += ["--domain-column", p["domain_column"]]
        if p.get("domain_table"):
            argv += ["--domain-table", p["domain_table"]]
    if p.get("attributes"):
        names = p["attributes"]
        argv += ["--attributes", names if isinstance(names, str) else ",".join(names)]
    argv += ["--out", base]
    return [_run(*argv), _add(out, f"{base}{Path(out['path']).suffix}")]


def _create(e: dict[str, Any]) -> list[dict[str, Any]]:
    p, out = e["params"], e["outputs"][0]
    folder = _work(e["id"], Path(out["path"]).name)
    argv = ["blockmodel", "create", folder, f"--size={_triple(p['size'])}"]
    steps: list[dict[str, Any]] = []
    if p.get("extent_from"):
        by_id = {o["id"]: o for o in e["inputs"]}
        for ref in p["extent_from"]:
            obj = by_id[ref]
            if obj["kind"] == "database":
                argv += ["--extent-from", f"drillholes:{_in_project(obj)}"]
            else:
                argv += ["--extent-from", ref]
        if _geometry(e):
            steps.append(_export_geometry())
            argv += _geometry(e)
        padding = p.get("padding", 0.0)
        argv.append(
            f"--padding={_num(padding) if isinstance(padding, int | float) else _triple(padding)}"
        )
    else:
        argv.append(f"--origin={_triple(p['origin'])}")
        argv += ["--shape", ",".join(str(int(float(v))) for v in p["shape"])]
    argv += ["--max-depth", str(int(p.get("max_depth") or 0))]
    if p.get("angles"):
        argv.append(f"--angles={_triple(p['angles'])}")
    return [*steps, _run(*argv), _add(out, folder)]


def _code(e: dict[str, Any]) -> list[dict[str, Any]]:
    p = e["params"]
    spec = {"attributes": p.get("attributes") or [], "steps": p.get("steps") or []}
    steps_file = _work(e["id"], "coding.json")
    argv = ["blockmodel", "code", _in_project(e["outputs"][0]), "--steps", steps_file]
    argv += _geometry(e)
    if p.get("subblock"):
        argv.append("--subblock")
    if p.get("merge"):
        argv.append("--merge")
    out = [_write(steps_file, json.dumps(spec, indent=2)), _run(*argv)]
    return [_export_geometry(), *out] if _geometry(e) else out


def _estimate(e: dict[str, Any]) -> list[dict[str, Any]]:
    plan = (e.get("replay") or {}).get("plan")
    if plan is None:
        raise ReplayError(f"job {e['id']}: the estimation plan was not logged")
    comps = next((o for o in e["inputs"] if o["kind"] == "composite"), None)
    if comps is None:
        raise ReplayError(f"job {e['id']}: the composites were not logged")
    plan_file = _work(e["id"], "plan.json")
    argv = ["estimate", _in_project(e["outputs"][0]), "--composites", _in_project(comps)]
    argv += ["--plan", plan_file, *_geometry(e)]
    out = [_write(plan_file, json.dumps(plan, indent=2)), _run(*argv)]
    return [_export_geometry(), *out] if _geometry(e) else out


def _report(e: dict[str, Any]) -> list[dict[str, Any]]:
    p = e["params"]
    where = (p.get("where") or "").strip()
    if where.startswith("{"):
        return [{"kind": "comment", "text": f"job {e['id']}: report with a constraint JSON "
                 "filter is not available on the CLI; skipped"}]  # fmt: skip
    model = next(o for o in e["inputs"] if o["kind"] == "model")
    argv = ["blockmodel", "report", _in_project(model)]
    if where:
        argv.append(f"--where={where}")
    for name in p.get("group_by") or []:
        argv.append(f"--group-by={name}")
    if p.get("cutoffs"):
        argv.append("--cutoffs=" + ",".join(_num(c) for c in p["cutoffs"]))
    if p.get("cutoff_attribute"):
        argv.append(f"--cutoff-attribute={p['cutoff_attribute']}")
    for grade in p.get("grade_attributes") or []:
        argv.append(f"--grade={grade}")
    if p.get("density") is not None:
        density = p["density"]
        argv.append(f"--density={_num(density) if isinstance(density, int | float) else density}")
    if p.get("partial"):
        argv.append(f"--partial={p['partial']}")
    if p.get("bench_height") is not None:
        argv.append(f"--bench-height={_num(p['bench_height'])}")
    argv.append(f"--bench-reference={_num(p.get('bench_reference') or 0.0)}")
    argv.append(f"--out={_work(e['id'], 'report.csv')}")
    return [_run(*argv)]


_MAPPERS = {
    "import": _import,
    "validate": _validate,
    "composite": _composite,
    "create_model": _create,
    "code": _code,
    "estimate": _estimate,
    "report": _report,
}


def _intercepts(e: dict[str, Any]) -> list[dict[str, Any]]:
    from bakumine.commands.intercepts import replay

    return replay(e)


_MAPPERS["intercepts"] = _intercepts  # bakumine intercepts --params WORK/....json


def _classify(e: dict[str, Any]) -> list[dict[str, Any]]:
    from bakumine.commands.classify import replay

    return replay(e)


_MAPPERS["classify"] = _classify  # bakumine classify --params WORK/....json


def _more_mappers() -> None:
    from bakumine.commands import geometry

    _MAPPERS.update(geometry.REPLAY)  # surface_from_points, import_mesh


_more_mappers()


def replay_steps(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CLI steps that redo the finished, not undone jobs of a log, in order."""
    undone = joblog.undone_ids(entries)
    present: set[str] = set()
    steps: list[dict[str, Any]] = [_run("project", "init", PROJECT, note="new project")]
    for e in entries:
        op = e.get("operation")
        if op == "undo" or e.get("status") != "done" or e.get("id") in undone:
            continue
        if op not in _MAPPERS:
            raise ReplayError(f"job {e.get('id')}: unknown operation {op!r}")
        for obj in e.get("inputs", []):
            if obj["id"] not in present:
                add = _add(obj, f"{SOURCE}/{obj['path']}")
                add["note"] = f"{obj['kind']} {obj['id']} from the source project"
                steps.append(add)
                present.add(obj["id"])
        mapped = _MAPPERS[op](e)
        mapped.insert(0, {"kind": "comment", "text": f"job {e['id']}: {op} ({e.get('time')})"})
        steps += mapped
        present.update(o["id"] for o in e.get("outputs", []))
    return steps


# ------------------------------------------------------------------ run / export


def _subst(text: str, values: dict[str, str]) -> str:
    return _TOKENS.sub(lambda m: values[m.group(0)], text)


def run_steps(steps: list[dict[str, Any]], project: Path, source: Path, work: Path) -> None:
    from bakumine.cli import main
    from bakumine.core.project import Project

    values = {PROJECT: str(project), WORK: str(work), SOURCE: str(source)}
    for step in steps:
        if step["kind"] == "comment":
            print(f"# {step['text']}")
        elif step["kind"] == "write":
            path = Path(_subst(step["path"], values))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(step["text"], encoding="utf-8")
        else:
            argv = [_subst(a, values) for a in step["argv"]]
            if argv[:2] == ["project", "init"] and (project / "project.json").is_file():
                if Project.open(project).list():
                    raise ReplayError(f"{project} already has objects; replay into a new one")
                continue
            print(f"$ bakumine {shlex.join(argv)}", flush=True)
            code = main(argv)
            if code != 0:
                raise ReplayError(f"step failed with exit code {code}: bakumine {argv[0]} ...")
            expect = step.get("expect")
            if expect and Project.open(project).get(expect["id"]).path != expect["path"]:
                raise ReplayError(f"object {expect['id']!r} landed on a different path")


def _sh_arg(arg: str) -> str:
    if not _TOKENS.search(arg):
        return shlex.quote(arg)
    parts = _TOKENS.split(arg)
    body = "".join(
        "${" + p.strip("@") + "}" if _TOKENS.fullmatch(p) else re.sub(r'([\\"$`])', r"\\\1", p)
        for p in parts
    )
    return f'"{body}"'


def to_shell(steps: list[dict[str, Any]], source: Path, log: Path) -> str:
    lines = [
        "#!/bin/sh",
        f"# Replay of {log} written by `bakumine project export-log`.",
        "# usage: sh THIS_SCRIPT NEW_PROJECT_FOLDER",
        "# BAKUMINE (default 'uv run bakumine') and SOURCE (the logged project) can be set.",
        "set -eu",
        'PROJECT="${1:?usage: sh THIS_SCRIPT NEW_PROJECT_FOLDER}"',
        f"SOURCE=\"${{SOURCE:-{source}}}\"",
        'BAKUMINE="${BAKUMINE:-uv run bakumine}"',
        'WORK="$(mktemp -d)"',
        "trap 'rm -rf \"$WORK\"' EXIT",
        "",
    ]
    for step in steps:
        if step["kind"] == "comment":
            lines.append(f"# {step['text']}")
        elif step["kind"] == "write":
            target = _sh_arg(step["path"])
            lines += [
                f'mkdir -p "$(dirname {target})"',
                f"cat > {target} <<'BAKUMINE_EOF'",
                step["text"],
                "BAKUMINE_EOF",
            ]
        else:
            if step.get("note"):
                lines.append(f"# {step['note']}")
            lines.append("$BAKUMINE " + " ".join(_sh_arg(a) for a in step["argv"]))
    return "\n".join(lines) + "\n"


def to_python(steps: list[dict[str, Any]], source: Path, log: Path) -> str:
    return f'''"""Replay of {log} written by `bakumine project export-log`.

usage: uv run python THIS_SCRIPT NEW_PROJECT_FOLDER [SOURCE_PROJECT]
"""

import sys
import tempfile
from pathlib import Path

from bakumine.commands.replay import run_steps

STEPS = {json.dumps(steps, indent=1)}

if __name__ == "__main__":
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else Path({str(source)!r})
    with tempfile.TemporaryDirectory() as work:
        run_steps(STEPS, Path(sys.argv[1]), source, Path(work))
'''


# ------------------------------------------------------------------ subcommands


def _log_and_source(args: argparse.Namespace) -> tuple[Path, Path, list[dict[str, Any]]]:
    log = Path(args.log)
    if log.is_dir():
        log = log / joblog.LOG_FILE
    if not log.is_file():
        raise FileNotFoundError(f"{log} not found")
    source = Path(args.source) if args.source else log.parent
    return log, source.resolve(), joblog.read(log)


def run_replay(args: argparse.Namespace) -> int:
    log, source, entries = _log_and_source(args)
    steps = replay_steps(entries)
    with tempfile.TemporaryDirectory(prefix="bakumine-replay-") as work:
        run_steps(steps, Path(args.into), source, Path(work))
    print(f"replayed {log} into {args.into}")
    return 0


def run_export(args: argparse.Namespace) -> int:
    log, source, entries = _log_and_source(args)
    steps = replay_steps(entries)
    out = Path(args.out)
    style = args.format or ("py" if out.suffix == ".py" else "sh")
    text = to_python(steps, source, log) if style == "py" else to_shell(steps, source, log)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({sum(s['kind'] == 'run' for s in steps)} commands)")
    return 0


def run_export_geometry(args: argparse.Namespace) -> int:
    import numpy as np

    from bakumine.core.io.obj import read_obj
    from bakumine.core.project import Project

    project = Project.open(args.project)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for obj in project.list():
        if obj.kind in ("solid", "dtm"):
            mesh = read_obj(project.path_of(obj))
            np.save(out / f"{obj.id}_vertices.npy", np.asarray(mesh.vertices))
            np.save(out / f"{obj.id}_faces.npy", np.asarray(mesh.faces))
            n += 1
    print(f"wrote {n} solids / DTMs to {out}")
    return 0


def configure(sub: Any, guarded: Any) -> None:
    """Add ``replay``, ``export-log`` and ``export-geometry`` to ``bakumine project``."""
    replay = sub.add_parser("replay", help="redo a job log (jobs.jsonl) into a new project")
    replay.add_argument("log", help="jobs.jsonl, or the project folder holding it")
    replay.add_argument("--into", required=True, help="new project folder")
    replay.add_argument("--source", help="project the log belongs to (default: its folder)")
    replay.set_defaults(func=guarded(run_replay))
    export = sub.add_parser("export-log", help="write a job log as a shell or Python script")
    export.add_argument("log", help="jobs.jsonl, or the project folder holding it")
    export.add_argument("--out", required=True, help="script to write (.sh or .py)")
    export.add_argument("--format", choices=("sh", "py"), help="default: from --out's suffix")
    export.add_argument("--source", help="project the log belongs to (default: its folder)")
    export.set_defaults(func=guarded(run_export))
    geometry = sub.add_parser(
        "export-geometry", help="write the project's solids and DTMs for --geometry-dir"
    )
    geometry.add_argument("project", type=Path, help="project folder")
    geometry.add_argument("out", help="folder for <id>_vertices.npy / <id>_faces.npy")
    geometry.set_defaults(func=guarded(run_export_geometry))


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use `bakumine project replay`")
