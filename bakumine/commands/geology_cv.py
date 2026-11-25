"""``bakumine geology cv``: hole-out cross-validation of a geology build spec.

    bakumine geology cv DB --spec SPEC [--folds 10] [--seed 0] [--step 1.0] --out cv.csv
        [--summary cv.json]

Holes are left out whole (``--folds 0`` = one at a time), the rock-unit model of the spec is
fitted again from the remaining holes and compared with the logs of the left-out holes. When
the spec lists faults that are inferred from the logs, the fault blocks are tested the same
way. ``--out`` gets one row per scored piece of hole (``layer, hole_id, fold, depth, length,
x, y, z, logged, predicted, distance``); ``--summary`` the figures and the provenance
entries they give.

The command reads these keys of the spec (geology_contracts.md section 10) itself: ``units``,
``lithology_source``, ``lithology`` (events, cap, kernel) and ``faults`` (blocks,
fault_zone_units, min_picks, surfaces). The rock units are tested as one model for all
holes: the per-fault-block fit of the full build is not repeated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote
from bakumine.commands.geology_units import _interval_table
from bakumine.commands.geometry import _database

HELP = "Hole-out cross-validation: how well the model predicts holes it has not seen"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Leave holes out (whole holes, never single intervals), fit the geology model of the "
        "spec from the others and compare it with the logs of the left-out holes: agreement "
        "by length, confusion matrix in metres, recall and precision per unit, contact "
        "position error, and all of these by distance to the nearest remaining hole. It "
        "measures the prediction between the holes at the drill spacing, not beyond the "
        "drilled volume."
    )
    parser.add_argument("database", help="drillhole database folder")
    parser.add_argument("--spec", required=True, metavar="JSON", help="geology build spec")
    parser.add_argument("--folds", type=int, default=10,
                        help="number of folds by hole (default 10; 0 = leave-one-out)")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0, help="seed of the fold split (default 0)")
    parser.add_argument("--step", type=float, default=1.0, metavar="M",
                        help="longest piece of hole scored at one point (default 1 m)")  # fmt: skip
    parser.add_argument("--out", required=True, metavar="CSV",
                        help="per-piece table: logged and predicted label, distance")  # fmt: skip
    parser.add_argument("--summary", metavar="JSON", help="the figures and provenance entries")
    parser.set_defaults(func=_guarded(run_cv))


def _events(spec: dict[str, Any]) -> list[Any]:
    from bakumine.core.geology.lithology import EventSpec, FieldSettings
    from bakumine.core.geometry.implicit_request import anisotropy_from_spec

    lithology = spec.get("lithology")
    if not isinstance(lithology, dict) or not lithology.get("events"):
        raise CommandError("the spec has no lithology events: nothing to cross-validate")
    events = []
    for entry in lithology["events"]:
        anisotropy = entry.get("anisotropy")
        settings = FieldSettings(
            anisotropy=None if anisotropy is None else anisotropy_from_spec(anisotropy),
            cap=lithology.get("cap"),
            kernel=lithology.get("kernel") or "linear",
        )
        events.append(
            EventSpec(
                str(entry.get("mode", "argmax")),
                tuple(str(u) for u in entry["units"]),
                settings,
                entry.get("thickness"),
            )
        )
    return events


def run_cv(args: argparse.Namespace) -> int:
    import pandas as pd

    from bakumine.core.geology.units import UnitDictionary
    from bakumine.core.geology.validation import (
        cross_validate,
        cross_validate_fault_blocks,
        fault_network_builder,
        stack_builder,
    )

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        raise CommandError(f"spec file {spec_path} does not exist")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not spec.get("units"):
        raise CommandError("the spec must be an object that names the unit dictionary ('units')")
    units = UnitDictionary.load(spec_path.parent / spec["units"])
    source = spec.get("lithology_source") or {}
    table_name = source.get("table") or units.source_table or "geology"
    column = source.get("column") or units.source_column
    if not column:
        raise CommandError("the spec names no lithology column (lithology_source.column)")

    database = _database(args.database)
    table = _interval_table(database, table_name, column)
    traces = database.desurvey()
    mapped = units.map_codes(table[column])
    events = _events(spec)
    scored = tuple(dict.fromkeys(u for event in events for u in event.units))

    results = [
        cross_validate(
            table,
            mapped,
            traces,
            stack_builder(events),
            layer="lith",
            scored=scored,
            folds=args.folds,
            seed=args.seed,
            step=args.step,
        )  # fmt: skip
    ]
    faults = spec.get("faults")
    if faults:
        results[0].notes.append(
            "the rock units are tested as one model for all holes; the per-fault-block fit "
            "of the full build is not repeated in the cross-validation"
        )
    skipped = ""
    if faults and faults.get("blocks") and faults.get("surfaces"):
        if any(s.get("replace_with") for s in faults["surfaces"]):
            skipped = (
                "fault blocks not cross-validated: a fault surface is imported "
                "(replace_with), so it does not depend on the holes"
            )
        else:
            from bakumine.core.geology.faults import block_codes

            blocks_at = faults["blocks"]
            block_table = _interval_table(database, blocks_at["table"], blocks_at["column"])
            blocks = block_codes(block_table[blocks_at["column"]], blocks_at["pattern"])
            zone_units = [str(u) for u in faults.get("fault_zone_units") or []]
            zones = table[mapped.isin(zone_units).to_numpy()] if zone_units else None
            surfaces = [
                {
                    "name": s["name"],
                    "between": s["between"],
                    "prior": None
                    if not s.get("prior")
                    else (s["prior"]["dip"], s["prior"]["dip_azimuth"]),
                }
                for s in faults["surfaces"]
            ]
            builder = fault_network_builder(
                surfaces, fault_zone=zones, min_picks=int(faults.get("min_picks") or 3)
            )
            results.append(
                cross_validate_fault_blocks(
                    block_table,
                    blocks,
                    traces,
                    builder,
                    folds=args.folds,
                    seed=args.seed,
                    step=args.step,
                )  # fmt: skip
            )

    for n, result in enumerate(results):
        print(("\n" if n else "") + result.format_text())
    if skipped:
        print("\n" + skipped)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([r.samples.assign(layer=r.layer) for r in results], ignore_index=True)
    rows = rows[["layer", *[c for c in rows.columns if c != "layer"]]]
    rows.to_csv(out, index=False)
    _wrote(out, f"{len(rows)} scored pieces of the left-out holes")
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "bakumine.geology-cv",
            "version": 1,
            "layers": {r.layer: r.to_dict() for r in results},
            "notes": [skipped] if skipped else [],
        }
        summary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        _wrote(summary, "cross-validation figures and provenance entries")
    return 0
