# BakuMine Block Modeller

Drilling data in, validated 3D block model out. `plan.md` is the task list and source of truth for progress; `research.md` holds the theory and formulas. Read the relevant `research.md` section before writing numerical code.

## Commands

- `make install` — `uv sync` + `npm install` in `web/` (run once per fresh checkout or worktree)
- `make check` — ruff + pytest + web typecheck + vitest. Must be green before any commit.
- `make dev` — API on :8000 and web dev server (5173, or the next free port)
- `uv run bakumine <command>` — CLI; every pipeline step must be runnable from here
- `uv run pytest tests/path/test_x.py -k name` — single test

Python is pinned to 3.12 through `uv`. Do not use the system Python (3.14).

## Layout

- `bakumine/core/drillholes` — import, validate, desurvey, composite
- `bakumine/core/geometry` — strings, DTM, solids, mesh operations
- `bakumine/core/blockmodel` — definition, octree sub-blocks, attributes, constraints
- `bakumine/core/estimation` — search, NN, IDW, variogram models, kriging
- `bakumine/core/io` — CSV, Surpac `.str`/`.dtm`, block CSV, DXF, OBJ
- `bakumine/core/report` — volume / tonnes / grade
- `bakumine/api` — FastAPI app; `bakumine/cli.py` — CLI (add subcommands with `bakumine.cli.register`)
- `web/` — Vite + React + TypeScript + three.js viewer
- `bakumine/blast`, `web/src/blast` — Blast BakuMine (bench blast design and simulation, a separate product tab): `plan_blast.md` is its task list, `research_blast.md` its theory. Rule 7 does not apply to it; it only reads block-modeller objects
- `data/synthetic` — generated benchmark deposit with known truth; `data/reference` — outputs of independent implementations
- `tests/` — mirrors the package layout

## Rules

1. **Clean room.** Implement from `research.md` and published literature. Never decompile or copy Surpac code or assets, and never parse the binary `.mdl` format. Interop is by text exports only.
2. **Tests define done.** No task is complete without passing acceptance tests under `make check`.
3. **Independent numerical checks.** Test every numerical routine against an analytic case or a second implementation (scipy, trimesh, gstools, PyKrige, GSLIB), not only against itself. Oracle libraries are dev dependencies only.
4. **Coordinates.** Internally `x = east, y = north, z = up`, metres, float64. Surpac's `Y, X, Z` order is converted at the I/O boundary only.
5. **Missing stays missing.** Absent assays are NaN, never 0, unless the user asks.
6. **One task, one commit:** `P<phase>.T<n>: <summary>`. Do not push.
7. **Scope.** No statistics module, plotting module, pit design, scheduling or macro language.
8. **Report honestly.** State what was not verified. A failing test is reported as failing.
