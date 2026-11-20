"""``bakumine estimate-plan``: write the estimation parameter file from flags.

    bakumine estimate-plan --out plan.json --attribute au --method ok \\
        --domain ORE --constraint 'rock == "ORE"' --sample-filter domain=MIN \\
        --dip 69 --dip-azimuth 216 --ranges 100,55,10 \\
        --composites composites.parquet --top-cut 31 \\
        --nugget-fraction 0.25 --sill-fraction 0.75 --structure spherical \\
        --pass 0.8:4:10:3 --pass 1.2:4:10:3 --pass 4:1:10 --discretisation 3,3,3

The file is what :meth:`bakumine.core.estimation.block_estimation.EstimationPlan.save`
writes for one domain, built by :func:`build_plan`:

* **orientation**: ``--dip / --dip-azimuth [/ --pitch]`` (the plane of the body, see
  :mod:`bakumine.core.estimation.orientation`) or ``--angles bearing,plunge,dip`` of the
  project's rotation convention. Not needed when the three ranges are equal.
* ``--ranges a,b,c``: along the major, semi-major and minor axis, metres. They are the
  ranges of every variogram structure (``--structure-ranges`` once per structure to
  differ) and the radii of the search ellipsoid (``--search-ranges`` to differ).
* **variogram** (kriging only): ``--structure TYPE`` once per nested structure (default
  one spherical); ``--nugget`` and ``--sill v[,v...]`` in the squared units of the grade,
  or ``--nugget-fraction`` / ``--sill-fraction f[,f...]`` as fractions of the **variance of
  the composites**: the sample variance (n - 1) of the value column of ``--composites``,
  over the rows the sample filter keeps that have a value, after the top cut.
* ``--pass factor:min:max[:max_per_hole]``, repeated in the order the passes run: the
  radius factor of the search ellipsoid, the minimum and maximum number of samples and
  optionally the most samples from one hole. Passes are named ``p1, p2, ...``.
* ``--constraint`` is a block expression (``rock == "ORE"``), ``--sample-filter
  column=value`` (several values: ``column=A,B``) keeps those composites, ``--top-cut``
  caps the sample values of the domain.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote

HELP = "Write an estimation parameter file (plan JSON) from flags"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="plan JSON to write")
    parser.add_argument("--attribute", required=True, help="block attribute to estimate")
    parser.add_argument("--value-column", help="composite column (default: the attribute)")
    parser.add_argument("--method", default="ok", choices=("nn", "idw", "ok", "sk"),
                        help="estimator of every pass (default: ok)")  # fmt: skip
    parser.add_argument("--domain", default="domain", help="name of the domain")
    parser.add_argument("--constraint", metavar="EXPR",
                        help='blocks of the domain: rock == "ORE" (default: all)')  # fmt: skip
    parser.add_argument("--sample-filter", metavar="COLUMN=VALUE[,VALUE]",
                        help="composites of the domain (default: all)")  # fmt: skip
    parser.add_argument("--top-cut", type=float, help="cap the sample values at this grade")
    ori = parser.add_argument_group("orientation (a plane or angles)")
    ori.add_argument("--dip", type=float, help="dip of the body, degrees from horizontal")
    ori.add_argument("--dip-azimuth", type=float, help="direction the body dips towards")
    ori.add_argument("--pitch", type=float, default=0.0,
                     help="rake of the major axis in the plane, from strike")  # fmt: skip
    ori.add_argument("--angles", metavar="BEARING,PLUNGE,DIP")
    parser.add_argument("--ranges", required=True, metavar="A,B,C",
                        help="ranges along major, semi-major and minor axis, metres")  # fmt: skip
    parser.add_argument("--search-ranges", metavar="A,B,C",
                        help="radii of the search ellipsoid (default: --ranges)")  # fmt: skip
    var = parser.add_argument_group("variogram (kriging)")
    var.add_argument("--structure", action="append", metavar="TYPE",
                     help="spherical, exponential or gaussian; repeat to nest")  # fmt: skip
    var.add_argument("--structure-ranges", action="append", metavar="A,B,C",
                     help="ranges of one structure, once each (default: --ranges)")  # fmt: skip
    var.add_argument("--nugget", type=float, help="nugget variance")
    var.add_argument("--sill", metavar="V[,V]", help="partial sill of each structure")
    var.add_argument("--nugget-fraction", type=float,
                     help="nugget as a fraction of the composite variance")  # fmt: skip
    var.add_argument("--sill-fraction", metavar="F[,F]",
                     help="partial sills as fractions of the composite variance")  # fmt: skip
    var.add_argument("--composites", metavar="FILE",
                     help="composites table (CSV or Parquet) for the variance")  # fmt: skip
    parser.add_argument("--pass", dest="passes", action="append", required=True,
                        metavar="FACTOR:MIN:MAX[:MAX_PER_HOLE]",
                        help="one search pass; repeat in the order the passes run")  # fmt: skip
    parser.add_argument("--discretisation", metavar="NX,NY,NZ",
                        help="points per block (default: 4,4,2)")  # fmt: skip
    parser.add_argument("--power", type=float, help="inverse-distance exponent (default: 2)")
    parser.add_argument("--mean", type=float, help="known mean of simple kriging")
    parser.set_defaults(func=_guarded(run))


def _numbers(text: Any, what: str, count: int | None = None) -> list[float]:
    if isinstance(text, str):
        parts: list[Any] = [p for p in text.replace(";", ",").split(",") if p.strip()]
    else:
        parts = list(text) if isinstance(text, list | tuple) else [text]
    try:
        values = [float(p) for p in parts]
    except (TypeError, ValueError):
        values = []
    if not values or (count is not None and len(values) != count):
        need = f"{count} numbers" if count else "numbers"
        raise ValueError(f"{what} expects {need} separated by commas, got {text!r}")
    return values


def parse_sample_filter(text: str) -> dict[str, Any]:
    """``"domain=MIN"`` -> equals; ``"domain=MIN,OX"`` -> in. Numbers are numbers, the rest
    text (``code="1"`` keeps a number as text)."""
    column, sep, rest = text.partition("=")
    values = [v.strip() for v in rest.split(",") if v.strip()]
    if not sep or not column.strip() or not values:
        raise ValueError(f"--sample-filter expects COLUMN=VALUE[,VALUE], got {text!r}")

    def value(v: str) -> Any:
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            return v[1:-1]
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v

    parsed = [value(v) for v in values]
    if len(parsed) == 1:
        return {"column": column.strip(), "equals": parsed[0]}
    return {"column": column.strip(), "in": parsed}


def parse_pass(text: str) -> dict[str, Any]:
    """``"0.8:4:10:3"`` -> radius factor 0.8, 4 to 10 samples, at most 3 per hole."""
    parts = text.split(":")
    try:
        if len(parts) not in (3, 4):
            raise ValueError
        out: dict[str, Any] = {
            "factor": float(parts[0]),
            "min_samples": int(parts[1]),
            "max_samples": int(parts[2]),
        }
        if len(parts) == 4:
            out["max_per_hole"] = int(parts[3])
    except ValueError:
        raise ValueError(
            f"--pass expects FACTOR:MIN:MAX[:MAX_PER_HOLE] (e.g. 1.5:4:12:3), got {text!r}"
        ) from None
    return out


def composite_variance(
    table: Any, column: str, sample_filter: Any = None, top_cut: float | None = None
) -> tuple[float, int]:
    """Sample variance (n - 1) of the composites of a domain after the top cut, and n."""
    import numpy as np

    from bakumine.core.estimation.block_estimation import _as_filter, _filter_mask

    if column not in table.columns:
        raise ValueError(f"the composites have no column {column!r}")
    values = table.loc[_filter_mask(table, _as_filter(sample_filter)), column]
    values = values.to_numpy(dtype=np.float64, na_value=np.nan)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError(f"fewer than two composites with a value of {column!r}: no variance")
    if top_cut is not None:
        values = np.minimum(values, top_cut)
    return float(values.var(ddof=1)), int(len(values))


def build_plan(spec: Mapping[str, Any], composites: Any = None) -> tuple[Any, dict[str, Any]]:
    """The :class:`EstimationPlan` of ``spec`` and what was derived on the way.

    ``spec`` has the keys of the command's options (underscores): ``attribute, method,
    domain, constraint, sample_filter, top_cut, dip, dip_azimuth, pitch, angles, ranges,
    search_ranges, structures, structure_ranges, nugget, sill, nugget_fraction,
    sill_fraction, passes, discretisation, power, mean, value_column``. ``composites`` is
    the composite table, needed for the fractions only. Returns ``(plan, info)`` with
    ``info = {"angles", "variance", "n_composites", "nugget", "sills"}``.
    """
    from bakumine.core.blockmodel.constraints import AttributeConstraint
    from bakumine.core.estimation.block_estimation import (
        ColumnFilter,
        EstimationDomain,
        EstimationPass,
        EstimationPlan,
    )
    from bakumine.core.estimation.orientation import angles_from_plane
    from bakumine.core.estimation.rotation import Ellipsoid
    from bakumine.core.estimation.search import SearchParams
    from bakumine.core.estimation.variogram import Structure, VariogramModel

    attribute = spec.get("attribute")
    if not attribute:
        raise ValueError("give the attribute to estimate")
    method = spec.get("method") or "ok"
    column = spec.get("value_column") or attribute
    ranges = tuple(_numbers(spec.get("ranges"), "ranges", 3))

    plane = spec.get("dip") is not None or spec.get("dip_azimuth") is not None
    if plane and spec.get("angles") is not None:
        raise ValueError("give either a plane (dip, dip azimuth) or angles, not both")
    if plane:
        if spec.get("dip") is None or spec.get("dip_azimuth") is None:
            raise ValueError("a plane needs both the dip and the dip azimuth")
        angles = angles_from_plane(
            float(spec["dip"]), float(spec["dip_azimuth"]), float(spec.get("pitch") or 0.0)
        )
    elif spec.get("angles") is not None:
        angles = tuple(_numbers(spec["angles"], "angles", 3))
    elif len(set(ranges)) == 1:
        angles = (0.0, 0.0, 0.0)
    else:
        raise ValueError("give the orientation: dip and dip azimuth, or angles")

    sample_filter = spec.get("sample_filter")
    if isinstance(sample_filter, str):
        sample_filter = parse_sample_filter(sample_filter)
    if isinstance(sample_filter, Mapping):
        sample_filter = ColumnFilter.from_dict(sample_filter)
    top_cut = None if spec.get("top_cut") is None else float(spec["top_cut"])

    info: dict[str, Any] = {"angles": tuple(angles), "variance": None, "n_composites": None}
    variogram = None
    if method in ("ok", "sk"):
        types = list(spec.get("structures") or ["spherical"])
        per_structure = [tuple(_numbers(r, "structure ranges", 3))
                         for r in spec.get("structure_ranges") or []]  # fmt: skip
        if per_structure and len(per_structure) != len(types):
            raise ValueError("give the structure ranges once per structure, or not at all")
        fractions = spec.get("nugget_fraction") is not None or spec.get("sill_fraction") is not None
        absolute = spec.get("nugget") is not None or spec.get("sill") is not None
        if fractions == absolute:
            raise ValueError(
                "give the nugget and the sill(s) either as values (nugget, sill) or as "
                "fractions of the composite variance (nugget_fraction, sill_fraction)"
            )
        key = "sill_fraction" if fractions else "sill"
        if spec.get(key) is None:
            raise ValueError(f"give the {key.replace('_', ' ')} of each structure")
        sills = _numbers(spec[key], key.replace("_", " "))
        if len(sills) != len(types):
            raise ValueError(f"{len(types)} structure(s) but {len(sills)} sill(s)")
        nugget = float(spec.get("nugget_fraction" if fractions else "nugget") or 0.0)
        if fractions:
            if composites is None:
                raise ValueError("fractions of the variance need the composites")
            variance, n = composite_variance(composites, column, sample_filter, top_cut)
            info.update(variance=variance, n_composites=n)
            nugget, sills = nugget * variance, [s * variance for s in sills]
        variogram = VariogramModel(
            nugget,
            tuple(
                Structure(kind, sill, per_structure[i] if per_structure else ranges, angles)
                for i, (kind, sill) in enumerate(zip(types, sills, strict=True))
            ),
        )
        info.update(nugget=nugget, sills=sills)
    elif any(spec.get(k) is not None for k in ("nugget", "sill", "nugget_fraction",
                                                "sill_fraction", "structures")):  # fmt: skip
        raise ValueError(f"method {method!r} has no variogram: leave out nugget, sill, structure")
    if method == "sk" and spec.get("mean") is None:
        raise ValueError("simple kriging needs the mean")

    search = ranges if spec.get("search_ranges") is None else tuple(
        _numbers(spec["search_ranges"], "search ranges", 3)
    )
    ellipsoid = Ellipsoid.from_radii(search, angles)
    discretisation = spec.get("discretisation")
    if discretisation is not None:
        discretisation = tuple(int(v) for v in _numbers(discretisation, "discretisation", 3))
    passes = []
    rows = [parse_pass(p) if isinstance(p, str) else dict(p) for p in spec.get("passes") or []]
    if not rows:
        raise ValueError("give at least one pass (factor:min:max[:max_per_hole])")
    for number, row in enumerate(rows, start=1):
        passes.append(
            EstimationPass(
                method,
                ellipsoid,
                SearchParams(
                    max_samples=row["max_samples"],
                    min_samples=row["min_samples"],
                    max_per_hole=row.get("max_per_hole"),
                ),
                radius_factor=row["factor"],
                power=2.0 if spec.get("power") is None else float(spec["power"]),
                variogram=variogram,
                mean=spec.get("mean") if method == "sk" else None,
                discretisation=discretisation,
                name=f"p{number}",
            )
        )
    constraint = spec.get("constraint")
    domain = EstimationDomain(
        str(spec.get("domain") or "domain"),
        passes,
        block_constraint=AttributeConstraint(constraint) if constraint else None,
        sample_filter=sample_filter,
        capping=top_cut,
    )
    return EstimationPlan(attribute, [domain], value_column=column), info


def run(args: argparse.Namespace) -> int:
    from bakumine.commands.estimate import _read_table
    from bakumine.core.blockmodel.constraints import ConstraintError

    spec = {
        key: getattr(args, key)
        for key in ("attribute", "value_column", "method", "domain", "constraint",
                    "sample_filter", "top_cut", "dip", "dip_azimuth", "pitch", "angles",
                    "ranges", "search_ranges", "structure_ranges", "nugget", "sill",
                    "nugget_fraction", "sill_fraction", "passes", "discretisation", "power",
                    "mean")
    }  # fmt: skip
    spec["structures"] = args.structure
    composites = _read_table(args.composites) if args.composites else None
    try:
        plan, info = build_plan(spec, composites)
    except (ValueError, TypeError, ConstraintError) as exc:
        raise CommandError(str(exc)) from None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plan.save(out)
    domain = plan.domains[0]
    bearing, plunge, dip = info["angles"]
    print(f"angles (bearing, plunge, dip): {bearing:g}, {plunge:g}, {dip:g}")
    if info["variance"] is not None:
        print(f"composite variance {info['variance']:.6g} from {info['n_composites']} "
              f"composites{'' if domain.capping is None else ' after the top cut'}")  # fmt: skip
    if info.get("sills") is not None:
        sills = ", ".join(f"{s:.6g}" for s in info["sills"])
        print(f"variogram: nugget {info['nugget']:.6g}, sill(s) {sills}")
    _wrote(out, f"{plan.attribute}, domain {domain.name}, {len(domain.passes)} "
                f"{domain.passes[0].method} pass(es)")  # fmt: skip
    return 0


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "estimate-plan": (HELP, configure),
}
