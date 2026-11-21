"""``bakumine classify``: resource classes of an estimated block model from a parameter file.

The rules are those of :mod:`bakumine.core.estimation.classification` (JSON, see its
docstring). ``--composites`` is needed only for the drillhole-support criteria
(``min_holes``, ``max_hole_distance``) when the parameter file has ``hole_support``: at run
level (``radius``, ``nth``, optional ``sample_filter`` ``{"column", "equals" | "in"}``) and /
or in a class, with that class's own radius and n-th hole.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote

HELP = "Classify the estimated blocks of a block model (Measured / Indicated / ..., B / C / P)"


EXAMPLE = """\
parameter file (JSON), a complete example:

  {
    "attribute": "au",
    "output": "res_class",
    "hole_support": {"radius": 50, "sample_filter": {"column": "domain", "equals": "ORE"}},
    "categories": [
      {"label": "Measured",  "criteria": {"max_pass": 1, "min_samples": 8, "min_holes": 3}},
      {"label": "Indicated", "criteria": {"max_pass": 2, "max_hole_distance": 75},
                             "hole_support": {"radius": 75, "nth": 2}},
      {"label": "Inferred"}
    ],
    "smoothing": null
  }

attribute   the estimated grade; its diagnostics (<attribute>_pass, _ns, _dist, _ndist, _kv,
            _slope, _ke) were written by `bakumine estimate`
categories  in order; a block gets the label of the FIRST class whose criteria it meets all
            of; a class without criteria takes every estimated block that is left
criteria    max_pass, min_samples, max_mean_distance, max_nearest_distance,
            max_kriging_variance, min_slope, min_efficiency, min_holes, max_hole_distance
            (the last two need hole_support and --composites)
a class     may also have "where" (a block expression such as rock == "ORE") and its own
            "hole_support" (radius: holes are counted within it; nth: max_hole_distance is
            the distance to the nth nearest hole, default 3)
smoothing   null, or {"iterations": 1, "min_agree": 4, "max_same": 0}
output      the category attribute written (default res_class); report by class with
            `bakumine blockmodel report MODEL --group-by res_class`
"""


def configure(parser: argparse.ArgumentParser) -> None:
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = EXAMPLE
    parser.add_argument("model", help="block model folder")
    parser.add_argument(
        "--params", required=True, help="classification parameter JSON (example below)"
    )
    parser.add_argument(
        "--composites", help="composites table (CSV or Parquet) for the drillhole support"
    )
    parser.add_argument("--out", help="model folder to write (default: update MODEL in place)")
    parser.add_argument("--summary", metavar="JSON", help="also write the summary as JSON")
    parser.set_defaults(func=_guarded(run))


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"classification of {summary['attribute']} into {summary['output']}: "
        f"{summary['n_classified']} of {summary['n_estimated']} estimated blocks classified, "
        f"{summary['n_unclassified']} met no rule, {summary['n_not_estimated']} not estimated"
        + (f", {summary['n_smoothed']} changed by smoothing" if summary["n_smoothed"] else "")
    ]
    width = max(len(c["label"]) for c in summary["classes"])
    for c in summary["classes"]:
        lines.append(f"  {c['label']:<{width}}  {c['blocks']:>10} blocks  {c['volume']:>16.1f} m3")
    if summary["written"]:
        lines.append(f"drillhole support written to {', '.join(summary['written'])}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.storage import is_model_folder
    from bakumine.core.estimation.classification import (
        ClassificationParams,
        classify_model_folder,
    )

    for path in (args.params, args.composites):
        if path and not Path(path).is_file():
            raise CommandError(f"{path} not found")
    if not is_model_folder(args.model):
        raise CommandError(f"{args.model} is not a block model folder")
    params = ClassificationParams.load(args.params)
    summary = classify_model_folder(args.model, params, args.composites, out=args.out).to_dict()
    print(format_summary(summary))
    _wrote(Path(args.out or args.model), f"class attribute {summary['output']}")
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _wrote(path, "classification summary")
    return 0


def replay(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """CLI steps of a logged ``classify`` job (``bakumine project replay``)."""
    from bakumine.commands.replay import ReplayError, _in_project, _run, _work, _write

    params = (entry.get("replay") or {}).get("params")
    if params is None:
        raise ReplayError(f"job {entry['id']}: the classification parameters were not logged")
    file = _work(entry["id"], "classification.json")
    argv = ["classify", _in_project(entry["outputs"][0]), "--params", file]
    composites = next((o for o in entry["inputs"] if o["kind"] == "composite"), None)
    if composites is not None:
        argv += ["--composites", _in_project(composites)]
    return [_write(file, json.dumps(params, indent=2, ensure_ascii=False)), _run(*argv)]


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "classify": (HELP, configure),
}
