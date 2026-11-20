"""A project folder for the viewer's browser tests of block display (web/e2e).

    uv run python scripts/e2e_display_project.py PROJECT_DIR

Runs the synthetic gate (``scripts/gate_synthetic.py``: import, composite, create, code,
estimate) with two sub-block levels and puts its outputs into a new project with
``bakumine project init`` / ``add``: the drillholes, the topography, the ore solid and the
estimated model. That model has ``rock`` = AIR / WASTE / ORE and ``au`` (with ``au_pass``
and the other diagnostics) in the ORE blocks alone, strictly inside the model, and more
blocks below the topography than the viewer is sent in full
(``bakumine.api.blocks.AUTO_SHELL_BLOCKS``): the case in which a model coloured by grade
used to open empty.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from gate_synthetic import SYNTH, cli, run


def build(folder: Path) -> None:
    work = folder.with_name(folder.name + "-gate")
    for path in (folder, work):
        if path.exists():
            shutil.rmtree(path)
    run(work, depth=2)
    cli("project", "init", folder, "--name", "Display")
    cli("project", "add", folder, "database", work / "db", "--name", "Drillholes")
    cli("project", "add", folder, "dtm", SYNTH / "topo.obj", "--name", "Topography", "--id", "topo")
    cli("project", "add", folder, "solid", SYNTH / "ore_solid.obj", "--name", "Ore solid",
        "--id", "ore-solid")  # fmt: skip
    cli("project", "add", folder, "model", work / "model", "--name", "Gate model", "--id", "model")
    shutil.rmtree(work)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(Path(sys.argv[1]))
