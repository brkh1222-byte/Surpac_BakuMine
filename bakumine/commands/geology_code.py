"""``bakumine geology code``: code a block model from a geology model.

    bakumine geology code MODEL --geology FOLDER [--layers fault_block,lith,weathering]
        [--no-subblock] [--max-level N] [--density NAME | --no-density] [--distance NAME]
        [--display NAME --ore-attribute NAME] [--write-steps FILE] [--merge] [--out FOLDER]

Writes the attributes ``lith``, ``weathering``, ``fault_block`` (categories) and
``density`` (research.md 5.7.6), prints the volume and tonnes per rock unit and the three
provenance tables, and stores the record of the run as
``FOLDER/coding/<model name>.provenance.json``. The ore attribute and all other attributes
of the model are left as they are.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote

HELP = "Code a block model from a geology model: rock unit, weathering, fault block, density"

DESCRIPTION = (
    HELP + ". Blocks cut by a geology boundary are sub-blocked first (labels at the 8 "
    "corners and the centre of each block: a body thinner than a block can be missed). "
    "The weathering-zone density replaces the rock density where the dictionary gives one."
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.description = DESCRIPTION
    parser.add_argument("model", help="block model folder")
    parser.add_argument("--geology", required=True, metavar="FOLDER", help="geology model folder")
    parser.add_argument("--layers", default="fault_block,lith,weathering", metavar="NAMES",
                        help="layers to code, in order (default fault_block,lith,weathering; "
                             "a layer the geology model does not have is skipped)")  # fmt: skip
    parser.add_argument("--no-subblock", action="store_true",
                        help="do not split the blocks cut by a geology boundary")  # fmt: skip
    parser.add_argument("--max-level", type=int, metavar="N",
                        help="deepest sub-block level (default: to the minimum size)")  # fmt: skip
    parser.add_argument("--density", default="density", metavar="NAME",
                        help="density attribute to write, t/m3 (default density)")  # fmt: skip
    parser.add_argument("--no-density", action="store_true", help="do not code a density")
    parser.add_argument("--distance", metavar="NAME",
                        help="also write the distance to the nearest logged hole, m")  # fmt: skip
    parser.add_argument("--display", metavar="NAME",
                        help="also write a display attribute: the rock unit, then ORE where "
                             "--ore-attribute is ORE")  # fmt: skip
    parser.add_argument("--ore-attribute", metavar="NAME",
                        help="the existing ore attribute (read only), for --display")  # fmt: skip
    parser.add_argument("--write-steps", metavar="FILE",
                        help="write the coding as a `blockmodel code --steps` file")  # fmt: skip
    parser.add_argument("--merge", action="store_true",
                        help="merge sibling sub-blocks equal in all attributes")  # fmt: skip
    parser.add_argument("--out", metavar="FOLDER",
                        help="model folder to write (default: update MODEL in place)")  # fmt: skip
    parser.set_defaults(func=_guarded(run))


def _geology_name(folder: Path) -> str:
    name = folder.resolve().name
    return name[: -len(".geology")] if name.endswith(".geology") and len(name) > 8 else name


def run(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.storage import load_model, save_model
    from bakumine.core.geology.coding import (
        code_geology,
        coding_provenance_path,
        unit_tonnage,
        write_coding_provenance,
    )
    from bakumine.core.geology.model import GeologyModel
    from bakumine.core.geology.provenance import Provenance

    folder = Path(args.geology)
    if not (folder / "geology.json").is_file():
        raise CommandError(f"{folder} is not a geology folder (no geology.json)")
    if args.ore_attribute and not args.display:
        raise CommandError("--ore-attribute is only used together with --display")
    layers = [name.strip() for name in args.layers.split(",") if name.strip()]
    if not layers:
        raise CommandError("--layers names no layer")
    target = Path(args.out or args.model)
    model_name = target.resolve().name
    coding_provenance_path(folder, model_name)  # a name that cannot be a file name: fail now
    geology = GeologyModel.load(folder)
    model = load_model(args.model)
    name = _geology_name(folder)
    result = code_geology(
        model,
        geology,
        geology_name=name,
        layers=layers,
        subblock=not args.no_subblock,
        max_level=args.max_level,
        density=None if args.no_density else args.density,
        distance=args.distance,
        display=args.display,
        ore_attribute=args.ore_attribute,
        merge=args.merge,
    )
    for layer, reason in result.skipped.items():
        print(f"layer {layer} not coded: {reason}")
    n_split = sum(r.n_split for r in result.results)
    print(
        f"coded {', '.join(result.attributes.values()) or 'no layer'} from the geology model "
        f"{geology.name!r}: {n_split} blocks split, {result.n_merged} sibling sets merged, "
        f"{model.n_blocks} blocks"
    )
    for layer, attribute in result.attributes.items():
        if layer == "lith":
            continue
        rows = result.volumes[result.volumes["layer"] == layer]
        print(f"\n{attribute}: volume per category (m3)")
        for row in rows.itertuples():
            print(f"  {row.category or '(none)':<12} {row.volume:>16.0f}")
    if "lith" in result.attributes:
        coded_density = None if args.no_density else args.density
        table = unit_tonnage(model, lith=result.attributes["lith"], density=coded_density or "")
        print("\nVolume and tonnes per rock unit")
        print(f"  {'unit':<12} {'blocks':>10} {'volume m3':>16} {'tonnes':>16} {'t/m3':>7}")
        for row in table.itertuples():
            density = "" if row.density != row.density else f"{row.density:.3f}"
            print(
                f"  {row.unit or '(none)':<12} {row.blocks:>10d} {row.volume:>16.0f} "
                f"{row.tonnes:>16.0f} {density:>7}"
            )
        print(
            f"  {'total':<12} {int(table['blocks'].sum()):>10d} {table['volume'].sum():>16.0f} "
            f"{table['tonnes'].sum():>16.0f}"
        )
    record = Provenance(object_kind="geology coding", name=model_name)
    record.extend(result.provenance)
    print("\n" + record.format_text() + "\n")
    header = save_model(model, target)
    _wrote(header.path, f"{header.n_blocks} blocks, {len(header.attributes)} attributes")
    path = write_coding_provenance(folder, model_name, result.provenance)
    _wrote(path, f"{len(result.provenance)} provenance entries")
    if args.write_steps:
        steps = Path(args.write_steps)
        steps.write_text(json.dumps(result.steps, indent=2) + "\n", encoding="utf-8")
        _wrote(
            steps,
            f"{len(result.steps['steps'])} steps; replay with `bakumine blockmodel code MODEL "
            f"--steps {steps.name} --geometry-dir DIR`, where DIR holds {name}.geology",
        )
    return 0
