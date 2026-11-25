# BakuMine Block Modeller — execution plan

Build BakuMine's own block modeller: drilling data in, validated 3D block model out, viewed in an interactive 3D scene. Functional target is the modelling chain of GEOVIA Surpac. Theory and formulas are in `research.md`; every agent reads the relevant section before coding.

Budget: 48 hours of continuous agent work. This file is the single source of truth for what is done and what is next.

---

## 0. Ground rules

1. **Clean room.** Implement from `research.md` and published literature. Never decompile, disassemble or copy Surpac binaries, code, icons or UI assets. Never parse the binary `.mdl` format. Interop is through text exports only (`.str`, `.dtm`, block CSV, DXF).
2. **Tests are the definition of done.** A task is ticked only when its acceptance tests exist, pass under `make check`, and are committed. No test, no tick.
3. **Numbers are checked against an independent source.** Every numerical routine is tested against an analytic case or a second implementation (GSLIB, PyKrige, gstools, trimesh, scipy), not only against itself.
4. **Coordinates.** Internally always `x = east, y = north, z = up`, metres, float64. Surpac's `Y, X, Z` order is converted at the I/O boundary only.
5. **No silent defaults in geology.** Missing assays stay missing (NaN); they are never set to 0 unless the user asks.
6. **One task, one commit**, message `P<phase>.T<n>: <summary>`. Do not push.
7. **Stay in scope.** No statistics module, plotting module, pit design, scheduling or macro language.

---

## 1. Architecture

| Layer | Choice | Reason |
|-------|--------|--------|
| Core engine | Python 3.12 (pinned with `uv`; system 3.14 is too new for numba), numpy, scipy, numba, pandas, pyarrow | Fast to write, easy to verify, vectorised and JIT-compiled where hot |
| Geometry helpers | `scipy.spatial` (Delaunay, cKDTree), `triangle` (constrained Delaunay), `trimesh` (test oracle only) | |
| API | FastAPI + uvicorn, binary array responses (Arrow IPC) | Viewer needs millions of values without JSON overhead |
| Viewer / UI | Vite + React + TypeScript + three.js (react-three-fiber) | Runs locally in a browser, no install for users |
| Storage | Project folder: `project.json` + Parquet tables (holes, composites, blocks) + mesh files | Open, inspectable, fast |
| Tests | pytest (+ hypothesis for geometry), vitest, Playwright for viewer smoke tests | |

```
bakumine/
  core/
    drillholes/   # import, validate, desurvey, composite
    geometry/     # strings, DTM, solids, mesh ops, point-in-solid
    blockmodel/   # definition, octree sub-blocks, attributes, constraints, storage
    estimation/   # search, NN, IDW, variogram models, kriging
    io/           # csv, surpac_str, surpac_dtm, block_csv, dxf, obj
    report/       # volume / tonnes / grade
  api/            # FastAPI app
  cli.py          # `bakumine <command>` — every pipeline step runnable headless
web/              # viewer + workflow UI
data/synthetic/   # generated benchmark deposit with known truth
data/reference/   # outputs from independent implementations
tests/
```

Every pipeline step must be callable from the CLI without the UI. The UI is a client of the same functions.

---

## 2. Orchestrator protocol (the loop)

Each loop iteration, the orchestrating session does exactly this:

1. Read this file. Find the earliest phase with unticked tasks whose dependencies are ticked.
2. Run `make check`. If it fails on `main`, fixing it is the only job of this iteration.
3. Select up to **4** unblocked tasks that touch different directories. Spawn one subagent per task with `isolation: "worktree"`, using the brief template in §3.
4. When an agent returns: read its diff, run `make check` in its worktree, and run the task's acceptance tests yourself. Do not trust the agent's report.
5. Pass → merge to `main`, tick the box, add one line to the progress log (§8). Fail → send the failure output back to the same agent once; if it fails again, log it under Blockers and move on.
6. At each phase gate (§5), run the gate checks before starting dependent phases.
7. If a task is blocked on a human decision (see §7), log it under Blockers, pick the stated default, and continue.
8. Stop the loop only when every box in Phases 0–7 is ticked and the final gate passes, or when every remaining task is blocked.

Never mark a task done because time is short. An unticked box with a note is correct; a ticked box with failing tests is not.

---

## 3. Subagent brief template

```
You are implementing one task of the BakuMine Block Modeller.

Task: <ID and title, copied from plan.md>
Read first: plan.md §0–§1, research.md §<sections>, and <existing files this task touches>.
Deliverable: <files / functions / CLI command>
Acceptance tests (you must write them and they must pass): <copied from plan.md>
Constraints: clean-room rule; internal coords x=east,y=north,z=up; no new dependencies
without stating why; do not edit files outside <directories>.
Finish by running `make check` and reporting: files changed, tests added, the exact test
output, and anything you could not verify. Report failures as failures.
```

---

## 4. Phases and tasks

Estimated hours are wall-clock with parallel agents.

### Phase 0 — Bootstrap (H0–H2) · sequential

- [x] **P0.T1 Repo skeleton.** `uv` project pinned to Python 3.12, package layout from §1, `web/` Vite app, `.gitignore`, `Makefile` with `make check` (ruff + pytest + tsc + vitest), `make dev` (API + web). *Accept:* `make check` green on an empty test; `make dev` serves a page.
- [x] **P0.T2 CLAUDE.md.** Project conventions from §0, commands, directory map. *Accept:* file exists, under 80 lines.
- [x] **P0.T3 Synthetic deposit generator.** `bakumine synth` creates: a known ore solid (dipping, plunging lens with a pinch), a known anisotropic grade field (unconditional simulation with known variogram), a topography DTM, 60–100 deviated drillholes with collar/survey/assay/geology CSVs sampled from the truth, and the true block grades. Seeded and deterministic. *Accept:* files written; truth solid volume known analytically or from a fine voxel count; rerun with the same seed gives identical files.
- [x] **P0.T4 Public dataset.** Add one small real, openly licensed drillhole dataset under `data/public/` with its licence noted. *Accept:* loads via pandas; licence file present.

### Phase 1 — Drillhole data (H2–H12) · parallel with Phase 2 and 5A

Read `research.md` §2–§4.

- [x] **P1.T1 Table import + column mapping.** Read collar/survey/assay/geology from CSV/XLSX with a user column map; unit and coordinate-order options; store as Parquet. *Accept:* synthetic and public datasets import; Y/X swap option tested.
- [x] **P1.T2 Validation report.** All checks listed in `research.md` §2.5, returned as structured errors/warnings per hole. *Accept:* a deliberately corrupted dataset triggers each check exactly once.
- [x] **P1.T3 Desurvey.** Minimum curvature (default), balanced tangential, tangent; dip sign convention selectable (Surpac negative-down default); `xyz_at(hole, depth)` with arc interpolation; straight extension past the last station. *Accept:* vertical hole exact; straight inclined hole exact; quarter-circle hole within 1 mm of the analytic arc; results match an independent implementation (e.g. `wellpathpy`) to 1e-6 m.
- [x] **P1.T4 Interval merge.** Combine assay and geology intervals into one segment table per hole (split at every boundary). *Accept:* total length preserved; property test with random intervals.
- [x] **P1.T5 Compositing.** Downhole fixed length; by bench; within-domain (by geology code or by solid, once P2.T6 exists); best-fit option; min-length rule; optional density weighting; NaN handling. *Accept:* metal accumulation (Σ grade × length) conserved to 1e-9 per hole; hand-worked example matches; no composite crosses a domain boundary in domain mode.
- [x] **P1.T6 Grade capping** as a compositing/estimation pre-step. *Accept:* unit test.
- [x] **P1.T7 CLI.** `bakumine import`, `validate`, `desurvey`, `composite`. *Accept:* end-to-end on synthetic data from the shell.

### Phase 2 — Geometry: strings, surfaces, solids (H2–H14) · parallel

Read `research.md` §5.

- [x] **P2.T1 Mesh core.** Vertex/face arrays, weld duplicates, edge map, normals, bounding boxes, BVH or uniform-grid triangle index. *Accept:* unit tests on cube, tetrahedron.
- [x] **P2.T2 Mesh validation.** Closed, manifold, consistent orientation, degenerate/duplicate faces, self-intersection; auto-fix orientation. *Accept:* each defect injected into a cube is detected; results agree with `trimesh` on 20 random meshes.
- [x] **P2.T3 Volume, area, centroid.** *Accept:* cube, sphere (within tessellation error), torus; equals `trimesh` to 1e-9 relative.
- [x] **P2.T4 DTM from points and strings.** Delaunay in plan, breaklines honoured, optional boundary string, `z_at(x, y)` batch query. *Accept:* plane and paraboloid reproduced; breakline edges present in output; 1M-point query < 2 s.
- [x] **P2.T5 Point-in-solid, batch.** Ray casting with column acceleration for regular grids; robust on edges/vertices. *Accept:* matches `trimesh.contains` on 1M random points for 5 shapes including a concave one and one with a hole; 10M grid centroids < 20 s.
- [x] **P2.T6 Solid from sections.** Triangulate between consecutive closed strings (shortest-diagonal stitching), end caps, simple bifurcation; output passes P2.T2. *Accept:* slices of a known ellipsoid rebuild to within 2% volume; synthetic deposit sections rebuild to a valid solid.
- [x] **P2.T7 Mesh–plane section.** Cut a mesh with an arbitrary plane, return polylines. *Accept:* cube and sphere sections exact.
- [x] **P2.T8 I/O.** Surpac `.str` and `.dtm` text read/write, DXF (3DFACE, POLYLINE) read/write, OBJ read/write. *Accept:* round-trip is lossless; parser is written to the layout in `research.md` §5.6 and marked provisional until checked against a real Surpac export (see §7).
- [x] **P2.T9 (stretch) Implicit solid from drillholes.** RBF on signed distance to contacts + marching cubes. *Accept:* synthetic ore solid recovered with IoU > 0.85.

### Phase 3 — Block model core (H12–H22) · needs P2.T1–T5

Read `research.md` §6.

- [x] **P3.T1 Definition and indexing.** Origin, extent, user block size, rotation (bearing/dip/plunge); `ijk ↔ centroid ↔ world`; containing-block lookup; auto-extent from meshes + margin. *Accept:* round-trip on random points for rotated and unrotated models.
- [x] **P3.T2 Block table with octree sub-blocks.** Flat columnar store (centroid, size, level, parent id, Morton key); split and merge; dense fast path when no sub-blocks. *Accept:* total volume invariant under any split/merge sequence (property test); 5M blocks fit in < 2 GB.
- [x] **P3.T3 Attributes.** Typed columns, background values, add/delete/rename, calculated attributes from a safe expression parser (no `eval`). *Accept:* unit tests including a malicious expression rejected.
- [x] **P3.T4 Constraints.** Primitives: DTM above/below, solid inside/outside, attribute comparison, plane, closed string, coordinate range. Boolean expression tree (AND/OR/NOT). Save/load as JSON. *Accept:* each primitive tested; composite expression equals manual mask.
- [x] **P3.T5 Sub-blocking on geometric constraints.** Split boundary blocks to the minimum size while coding. *Accept:* volume of blocks inside the synthetic solid within 1% of mesh volume at 1/8 sub-blocks; error decreases monotonically with depth.
- [x] **P3.T6 Partial percentages.** Fraction of block inside a solid by lattice sampling. *Accept:* sphere volume via partials within 0.5%.
- [x] **P3.T7 Assign value / code blocks.** Fill an attribute under a constraint. *Accept:* air/ore/waste coding of the synthetic model correct vs truth > 99% by volume.
- [x] **P3.T8 Persistence and interchange.** Save/load as Parquet + JSON header; import/export block CSV in Surpac's export layout (centroid Y, X, Z, sizes, attributes). *Accept:* lossless round-trip; 5M blocks save < 10 s.
- [x] **P3.T9 Report.** Volume, tonnes, grade, metal under a constraint, grouped by attribute, bench and cut-off; CSV output. *Accept:* hand-computed 3×3×3 example.

**Gate A (after Phases 1–3):** synthetic data → composites → coded block model, run from the CLI in one script; volume check passes; `/code-review high` on the whole core with findings fixed.

### Phase 4 — Estimation (H18–H30) · needs P1.T5, P3.T2

Read `research.md` §7. This phase decides whether the product is credible; it gets the most verification.

- [x] **P4.T1 Rotation conventions.** Rotation matrices for `surpac_zxy_lrl` (default), `gslib`, `datamine_zxz`, with a documented test per convention: an ellipsoid with bearing 30°, plunge 20°, dip 10° must put its major axis at the hand-computed world vector. *Accept:* tests pass; conventions selectable everywhere an ellipsoid is used.
- [x] **P4.T2 Anisotropic search.** Transform to isotropic space, cKDTree; radius, min/max samples, max per hole, octant limits, max vertical distance. *Accept:* equals brute force on 10k random queries for every option combination.
- [x] **P4.T3 Nearest neighbour.** *Accept:* equals brute force; global mean equals declustered (cell/polygonal) mean within tolerance.
- [x] **P4.T4 IDW.** Any power, anisotropic, exact at coincident samples, block discretisation optional. *Accept:* hand example; brute-force match; 1M blocks × 20k composites < 60 s.
- [x] **P4.T5 Variogram models.** Nugget + nested spherical / exponential / gaussian with per-structure anisotropy; covariance form; JSON schema. *Accept:* values match `gstools` at 1000 random lags.
- [x] **P4.T6 Ordinary and simple kriging.** Point and block (discretisation nx×ny×nz); weights, estimate, kriging variance, sample count, mean distance, optional negative-weight correction; duplicate-sample handling; numba-parallel. *Accept:* weights sum to 1 (OK); exact interpolation at a sample with zero nugget; a 4-sample hand-solved system matches to 1e-10; estimates and variances match GSLIB `kt3d` (or PyKrige `OrdinaryKriging3D`) to 1e-6 on the synthetic set; 1M blocks, max 24 samples, < 5 min.
- [x] **P4.T7 Multi-pass estimation per domain.** Run a list of passes under a constraint, composites filtered by domain, fill only unestimated blocks, store pass number. *Accept:* no block uses samples from another domain; pass attribute consistent.
- [x] **P4.T8 Experimental variogram (minimal).** Directional, lag/tolerance parameters, output table + a least-squares auto-fit to seed kriging parameters. No UI beyond a simple chart. *Accept:* recovers the synthetic field's range within 20% and nugget ratio within 0.1.
- [x] **P4.T9 Validation tools.** Swath table (blocks vs composites along X/Y/Z), global mean comparison vs NN, leave-one-out cross-validation. *Accept:* run on synthetic; OK beats IDW beats NN in RMSE against truth, and this is asserted in a test.
- [x] **P4.T10 (stretch) Dynamic anisotropy.** Per-block bearing/dip/plunge attributes drive the ellipsoid. *Accept:* folded synthetic case RMSE lower than fixed ellipsoid.

**Gate B:** full synthetic run. Estimated model vs truth: OK correlation with true block grades > 0.75 inside the ore domain; global tonnes and metal within 5% of truth; all P4 reference comparisons green.

### Phase 5 — 3D viewer (5A: H2–H14 in parallel with mock data; 5B: H24–H36 on real output)

Use the `frontend-design` skill for the UI and `dataviz` for colour ramps and legends.

- [x] **P5.T1 Scene.** Orbit/pan/zoom, Z-up, orthographic/perspective toggle, grid, axes with real coordinates, vertical exaggeration, large-coordinate handling (recentre to local origin to avoid float32 jitter). *Accept:* Playwright screenshot test; a point at E 500 000 / N 5 000 000 renders without jitter.
- [x] **P5.T2 Drillholes.** Traces as tubes/lines coloured by any interval attribute, collar labels, hover/pick shows hole, depth, values. *Accept:* 500 holes at 60 fps.
- [x] **P5.T3 Meshes.** DTMs and solids, solid/wireframe/transparent modes, per-object visibility and colour. *Accept:* 1M triangles interactive.
- [x] **P5.T4 Blocks.** Colour by attribute with editable legend (ranges or continuous ramp); render only exposed faces of the filtered set; sub-blocks at true size; pick a block to see all attributes. *Accept:* 2M blocks loaded, filter change < 1 s, ≥ 30 fps while orbiting.
- [x] **P5.T5 Constraint filter in viewer.** Attribute ranges, inside/outside solid, above/below DTM, bench range; same JSON as P3.T4, evaluated server-side. *Accept:* viewer mask equals CLI mask.
- [x] **P5.T6 Section tools.** Clipping plane at any orientation with thickness (slab), step forward/back, section view locks camera normal to plane; drillholes, mesh outlines and blocks all clip together. *Accept:* screenshot test on synthetic section matches expected composition.
- [x] **P5.T7 Section digitising.** Draw/edit closed strings on the active section, snap to drillhole intercepts, save as strings, send to P2.T6 to build a solid. *Accept:* digitise 5 sections of the synthetic deposit by script → valid solid.
- [x] **P5.T8 Data transport.** Arrow/binary endpoints for holes, meshes, blocks, with level-of-detail for very large models. *Accept:* 2M blocks transferred and displayed < 5 s on localhost.

### Phase 6 — Application workflow (H30–H40) · needs Gates A, B and 5B

- [x] **P6.T1 Project.** Create/open a project folder; object tree (database, composites, strings, DTMs, solids, block models, constraints). *Accept:* reopen restores everything.
- [x] **P6.T2 Import wizard.** Column mapping, preview, validation report with click-to-hole. *Accept:* public dataset imported through the UI by a Playwright script.
- [x] **P6.T3 Compositing dialog.** *Accept:* result equals CLI.
- [x] **P6.T4 Block model dialogs.** Create (with auto-extent), attributes, constraints builder, code blocks with sub-blocking. *Accept:* result equals CLI.
- [x] **P6.T5 Estimation dialog.** Method, attribute, domain constraint, ellipsoid (shown live in the 3D scene), variogram parameters, passes; background job with progress and cancel. *Accept:* result equals CLI; cancel leaves the model unchanged.
- [x] **P6.T6 Report panel.** Volume/tonnes/grade table with export. *Accept:* equals CLI.
- [x] **P6.T7 Job log and undo.** Every operation logged with its parameters (reproducible as a CLI script); undo for attribute-changing operations via snapshot. *Accept:* replaying the log on a fresh project gives an identical model.

### Phase 7 — Validation, performance, delivery (H40–H48)

- [x] **P7.T1 End-to-end test.** One Playwright test drives the UI from raw CSVs to an estimated, reported model on synthetic data and asserts Gate B numbers. *Accept:* green, < 10 min.
- [x] **P7.T2 Real-data run.** Same on the public dataset; sanity checks from `research.md` §9 recorded. *Accept:* swath and global mean checks pass.
- [ ] ~~CUT: no inputs~~ **P7.T3 Surpac comparison** (only if the inputs in §7 are supplied). Import Surpac's `.str/.dtm` and block CSV for the same data; compare block by block. *Accept:* report of differences; coded volume within 1%, IDW grades within 1e-3 relative where search parameters are identical. If inputs are absent, leave unticked with that reason.
- [x] **P7.T4 Performance pass.** Profile the 5M-block case; fix the top three hotspots. *Accept:* targets in P4.T4, P4.T6, P5.T4 met, numbers logged below.
- [x] **P7.T5 Review.** `/code-review high` and `/security-review` on the full diff; fix confirmed findings. *Accept:* no open important findings.
- [x] **P7.T6 User guide + one-command start.** `README.md` with install, `make dev`, and a walkthrough of the synthetic project with screenshots. *Accept:* a fresh clone reaches the 3D model by following the README only.
- [x] **P7.T7 Honest status report.** `STATUS.md`: what works, what is verified against which reference, what is provisional, what is not done. *Accept:* every unticked box and every **[verify]** item in `research.md` is listed.

---

### Phase 8 — Proposed after the Surpac comparison research (2026-11-21) · NOT started, awaiting the owner's go

From a web review of public Surpac material (course catalogues, tutorials, the 1996 "Appendix D - File Formats"). Tasks marked *(rule 7)* need the scope rule in CLAUDE.md relaxed first.

- [x] **P8.T1 Surpac `.dtm` I/O to the published format.** The 1996 appendix (https://www.cse.unr.edu/~fredh/papers/working/vr-mining/string.html) contradicts our provisional reader/writer: vertex numbers count the null (separator) records, line 1 is `<name>.str,<checksum>;...`, trisolation flags include `direction=solid`, no zero record before `END`. *Accept:* a hand-written fixture following the appendix's box example (not copied verbatim) reads to a valid closed solid of the right volume; multi-segment solids round-trip.
- [ ] **P8.T2 Domain statistics and top-cut analysis** *(rule 7)*. Histogram, log-probability, CV, mean and CV against the cap, per domain. *Accept:* numbers match numpy/scipy on the synthetic deposit.
- [ ] **P8.T3 Interactive variogram fitting.** Downhole + three directional experimental variograms with a model the user adjusts; auto-fit is only the seed. *Accept:* fitted model reproduces the synthetic deposit's known variogram within tolerance.
- [ ] **P8.T4 Swath and grade-tonnage charts** in the report panel *(rule 7 for the charts; the tables exist)*.
- [ ] **P8.T5 Kriging neighbourhood analysis.** Slope of regression, kriging efficiency, negative weights against min/max samples and discretisation.
- [ ] **P8.T6 Section plot export (PDF/SVG)** *(rule 7)*: holes, assays, outlines, blocks, title block.
- [ ] **P8.T7 Solid / DTM booleans.** Clip a solid to topography, intersect and outersect domains. *Accept:* volumes against trimesh/analytic cases.
- [ ] **P8.T8 Search ellipsoid preview in 3D.**
- [ ] **P8.T9 Indicator kriging** for skewed grades.
- [ ] **P8.T10 Solids from sections:** ring holes, one-to-many splits, grade of a digitised segment.
- [ ] **P8.T11 Trend surface to dynamic-anisotropy attributes** (trend DTM -> dip / dip direction per block).
- [ ] *(Babbitt part done 2026-11-21; other datasets not reachable)* **P8.T12 Retest on public data with a wireframe and a published result:** the Babbitt domain wireframe (`Mpz.stl`, pygslib repo, MIT) and the tutorial's published means; candidates not yet reachable: Northern Quarries iron (Tasmania MRT), Wolfram Camp (Queensland open data).
- [x] **P8.T14 Topography from elevation rasters.** `bakumine surface from-raster` (GeoTIFF incl. Copernicus / ALOS / SRTM tiles, ESRI ASCII grid; resampled bilinearly onto a grid in the model CRS; adjacent tiles joined), vertical tie to the collars by the median offset, and `bakumine surface check` (collar z - surface z, suspect collars) for any surface. *Accept:* a plane is reproduced exactly; a lon/lat raster matches scipy's interpolator after projection; a planted datum shift and bad collar are found.
- [x] **P8.T15 Merge a precise survey with a DEM:** `surface from-points --raster FILE [--drop-within M] [--raster-z-shift median|none|M]` adds the DEM grid nodes around the survey points, strings and collars, tied to them by the median height difference; nodes within M (default: grid spacing) of survey data are left out.
- [x] **P8.T16 Fetch Copernicus GLO-30 tiles** for the model area: `surface from-raster | from-points --fetch copernicus30 [--dem-cache DIR]`, off unless given; the tiles over the model box (edges sampled in longitude / latitude) are downloaded once into a cache, tiles over sea reported missing; the help and the output say the tile names go to the public bucket.
- [x] **P8.T17 Elevation files in the viewer:** Surfaces panel "Topography from an elevation file" (upload GeoTIFF / ASCII grid, file and model CRS, area, spacing, vertical shift, collar check with suspect collars) and elevation files accepted in "Topography from points" as the DEM around the data; endpoint `POST /ops/surfaces/from-raster`, `raster` block in from-points, job-log replay to the CLI.
- [ ] **P8.T13 Mineralised shell at a geological boundary.** On the real gold project the report grade is reproduced by a shell boundary below the reporting cut-off, not by the reporting cut-off applied per hole: let the shell cut-off differ from the reporting cut-off in the workflow panels, and add class rules by drill spacing as national reports state them.

---

### Phase 9 — Geology model: rock units, weathering surfaces, faults beside the ore (owner's go 2026-11-25)

Final product: topography + accurate blocks + the underground layer by layer (possibly offset by faults).
Theory: `research.md` 5.7. File formats, signatures, file ownership per task and the synthetic truth:
`geology_contracts.md`. Every geology build writes a provenance record: real data used / assumed by the
software / inputs still needed.

- [x] **P9.T1 Theory and contracts.** `research.md` 5.7 and `geology_contracts.md`.
- [x] **P9.T2 Synthetic layered deposit with truth** (`bakumine synth --preset layered | layered-small`).
- [ ] **P9.T3 Unit dictionary and provenance record**, `geology units init | check`.
- [ ] **P9.T4 2-D kriging with inequality data** (exact picks, lower and upper bounds), checked against PyKrige and a quadratic-programme solver.
- [ ] **P9.T5 Viewer: 20 category colours and colour maps per code.**
- [ ] **P9.T6 Unit picks per hole with bounds.**
- [ ] **P9.T7 Multi-unit rock model (largest field), grid sampling, meshes, save / load.**
- [ ] **P9.T8 Fault surfaces and fault blocks** (inferred from logged blocks, or imported).
- [ ] **P9.T9 Cover / weathering surfaces**, `geology surface`.
- [ ] **P9.T10 Sequence mode, intrusion overprints, LoopStructural comparison** (dev dependency).
- [ ] **P9.T11 Hole-out cross-validation for geology.**
- [ ] **P9.T12 Geology constraint, project kind `geology`, block coding** (`lith`, `weathering`, `fault_block`, `density`).
- [ ] **P9.T13 Rock model per fault block, throw diagnostic.**
- [ ] **P9.T14 Build spec, `geology build`, API, job-log replay, scene manifest.**
- [ ] **P9.T15 Viewer Geology group and panel** (with the provenance tables).
- [ ] **P9.T16 Synthetic gate:** the whole chain through the CLI against the targets in `geology_contracts.md` 14.
- [ ] **P9.T17-T19 Real deposits** (git-ignored data and notes; nothing of them in tracked files).
- [ ] **P9.T20 STATUS, README, research update.**

## 5. Dependency map

```
P0 ──┬── P1 (drillholes) ───────────────┬── P4 (estimation) ──┐
     ├── P2 (geometry) ── P3 (blocks) ──┘         │           ├── P6 (app) ── P7
     └── P5A (viewer on mock data) ──────── P5B (viewer on real data) ┘
                              Gate A after P1–P3 · Gate B after P4
```

Parallel width: 3–4 agents from H2 to H36. Phases 0 and 7 are mostly sequential.

---

## 6. Time plan

| Hours | Work | Checkpoint |
|-------|------|------------|
| 0–2 | Phase 0 | `make check` green, synthetic deposit exists |
| 2–14 | Phases 1, 2, 5A in parallel | Composites; valid solids; viewer shows mock data |
| 12–22 | Phase 3 | **Gate A** — coded block model from CLI |
| 18–30 | Phase 4 | **Gate B** — estimates match references |
| 24–36 | Phase 5B | 2M real blocks in the viewer with sections |
| 30–40 | Phase 6 | Whole workflow from the UI |
| 40–48 | Phase 7 | End-to-end test, README, STATUS |

If behind schedule, cut in this order: P2.T9, P4.T10, P5.T7, P6.T7 undo, P4.T8 auto-fit. Never cut tests, Gate A or Gate B.

---

## 7. Inputs needed from BakuMine (non-blocking; defaults apply)

| # | Input | Used by | Default if absent |
|---|-------|---------|-------------------|
| 1 | A Surpac licence, or files exported from one: a `.str` + `.dtm` pair, a block model CSV export, and the estimation parameters used | P2.T8, P7.T3, all **[verify]** items | Formats stay "provisional"; equivalence shown only against GSLIB-class references |
| 2 | A real BakuMine drillhole dataset (collar, survey, assay, geology) | P1, P7.T2 | Public dataset + synthetic deposit |
| 3 | Commodity, typical block size and bench height | Defaults in dialogs | Gold, 10 × 10 × 5 m |
| 4 | UI language | Phase 6 | English, with strings in one file for later translation |

---

## 8. Progress log

One line per merged task: `YYYY-MM-DD HH:MM  P#.T#  commit  note`.

2026-11-19 16:01  P0.T1  12214e3  skeleton; `make check` green (re-run by orchestrator). Vite uses 5174 if 5173 is taken. CLI subcommands register via `bakumine.cli.register`.
2026-11-19 16:08  P0.T4  77bee15  Babbitt Cu-Ni (399 holes, from MIT-licensed pygslib repo); `make check` green. Quirks: units are feet, dip is POSITIVE down, 70 holes have a sentinel survey at depth 90000, no geology table. Upstream (NRRI TR-2003/21) has no explicit licence: fine for local testing, confirm before redistributing.
2026-11-19 16:20  P1.T3  db13542  desurvey (min curvature / balanced / tangent). Matches wellpathpy to 1e-6 on 50 holes; orchestrator's own quarter-circle check: 1e-9 m. Runs on Babbitt. API: `desurvey(collar, survey, method=, dip_convention=) -> HoleTraces`, `.xyz_at(hole_id, depths)`, `.stations()`.
2026-11-19 16:35  P0.T3  9a859c8  `bakumine synth`; default dataset in data/synthetic/default (80 holes, 22 476 assays, truth blocks 10x10x5 m, true ore volume 2 055 405 m3, 5.86 Mt @ 2.01 g/t). Orchestrator checks: trimesh says watertight, volume -0.06% of truth; merged desurvey reproduces truth sample XYZ to 5e-7 m. Schemas are listed in truth.json `files`. Phase 0 complete.
2026-11-19 16:52  P2.T1-T3  180aad2  `Mesh` (immutable), `TriangleGridXY`, `TriangleBVH`, `validate()`, `fix_orientation()`, volume/area/centroid; agrees with trimesh to 1e-9. Orchestrator check: synthetic ore solid validates, volume -0.057% of truth. Indexes return AABB candidates only; exact ray tests belong to P2.T4/T5. Import from modules, not the package `__init__`.
2026-11-19 17:12  P5.T1  9535af0  viewer shell: Z-up scene, views, ortho/perspective, vertical exaggeration, shared scene origin (`web/src/scene/origin.ts`), zustand store, layer recipe in `web/src/scene/layers.tsx`. `make check` and `make e2e` (9 tests) re-run green by orchestrator; screenshots inspected. Not tested: real mouse/trackpad input, jitter while the camera moves. `npx playwright install chromium` is needed once per machine.
2026-11-19 17:12  P1.T1/T2/T4  22a89b0  `import_drillholes` + `DrillholeDatabase` (Parquet), `validate()` (18 check codes), `merge_intervals`. Babbitt imports from raw; synthetic import test runs on main (no skip). Defaults to note: below-detection = half the limit; -99/-999 sentinels apply to attribute columns only.
2026-11-19 17:25  P4.T1/T5/T2  dde480f  `Ellipsoid` (rotation.py), `VariogramModel` (+ njit covariance), `SampleSearch.query -> SearchResult` (padded indices/distances, `estimated` mask). Variograms match gstools; search matches brute force (also re-checked by orchestrator). Local frame is (semi-major, major, minor). `datamine_zxz` NOT implemented (convention not known with confidence). `surpac_zxy_lrl` and `gslib` are derived from recalled documentation, unverified against the real programs: positive plunge = major axis UP. 1M targets x 20k samples: 0.5 s plain, 8 s with octant + per-hole limits.
2026-11-19 20:17  --  --  Usage limit stopped all four agents at about 17:40; resumed at 20:17 from their worktrees (P2.T5/T4/T7, P1.T6/T7, P4.T6, P5.T8/T2/T3). About 2.5 h of wall-clock lost; expect this again roughly every 5 h of full-width work.
2026-11-19 20:30  P4.T3/T4/T6  59d93d1  `nearest_neighbour`, `inverse_distance`, `krige` (OK/SK, point/block). OK matches PyKrige and gstools to 1e-6 (measured 1e-14); orchestrator's own variogram-form solve agrees to 1e-15. Synthetic ore blocks vs truth: NN RMSE 1.417 / r 0.71, IDW2 0.921 / 0.84, OK 0.879 / 0.86. Block kriging and nested models are checked only against a dense numpy solver; kt3d nugget treatment is from memory, not run against GSLIB. `lagrange` is covariance-form (sign opposite to research.md).
2026-11-19 20:45  P1.T5/T6/T7  a7c4ccc  `composite()` (downhole / bench / domain, best-fit, density weighting), `cap_grades()`, CLI `import | validate | desurvey | composite`. Orchestrator ran the CLI chain on the synthetic CSVs: 11 410 two-metre domain composites, metal accumulation conserved exactly (3086.592 in and out). `<attr>_coverage` is a length, not a fraction. Domain source is pluggable (callable) for the later inside-solid case. No third-party compositing oracle exists; checks are hand arithmetic + conservation + truth. Phase 1 complete.
2026-11-19 21:00  P2.T5/T4/T7  8aa764a  `points_in_solid`, `grid_in_solid` (+ per-column entry/exit intervals), `DTM.z_at/above_below`, `section()`. Exact plan-view predicates (symbolic perturbation); 0 of 5M points differ from trimesh (agent's full run, 13 min, gated by BAKUMINE_FULL_ORACLE=1). Orchestrator checks: truth-block ORE codes agree 99.997%, AIR 99.97%, collars exactly on topo, 70M-cell voxel volume within 0.003% of mesh volume in 0.03 s. `grid_in_solid` origin = first centroid, not model corner. Default test suite now takes about 75 s.
2026-11-19 21:20  P2.T8  2512087  `bakumine/core/io`: Surpac `.str` / `.dtm` text, DXF, OBJ, PLY, dispatcher `read_mesh / write_mesh / read_strings`. Orchestrator round-tripped the ore solid through .dtm/.str: valid, volume within 6e-7. PROVISIONAL: the Surpac layouts are unverified against a real export (14 listed assumptions in the module docstrings; layout constants are isolated; `describe_surpac_file(path)` is the debugging tool for the first real file). Binary .str/.dtm are detected and rejected.
2026-11-19 21:40  P4.T8/T9  a18c7e2  `experimental_variogram`, `downhole_variogram`, `fit_variogram`; `swath_table`, `global_comparison`, cell declustering, `cross_validate` (leave-one-out and leave-one-hole-out), `compare_to_truth`. T8 CAVEAT: the plan's recovery criterion (range within 20%, nugget ratio within 0.1) passes only on assay-support samples with a 10 deg cone (nugget 0.121 vs 0.15, major range 116 vs 120 m). It FAILS with a 22.5 deg cone (range about 91 m) and the nugget fails on 2 m composites (0.024); these are 3 strict xfails, not loosened. Semi-major range is not recovered (92-115 vs 70 m). Auto-fit is a seed only. T9: OK <= IDW <= NN holds with true and fitted variograms (OK RMSE 0.879 true / 0.880 fitted); hole-out CV is far harsher than LOO (OK r 0.585 vs 0.871).
2026-11-19 22:05  P5.T2/T3 (+T8 part)  51be947  API `/api/v1/scene` manifest + drillhole and mesh payloads; viewer loads the synthetic demo (topography, ore solid, 80 holes coloured by au or rock, legend, hover/select, mesh display modes). `make check` + `make e2e` (30 tests) re-run green by orchestrator; two screenshots inspected. P5.T8 stays UNTICKED: block transport and the 2M-block criterion are still to do (with P5.T4). Transport is a custom binary container "BKMA" (JSON header + raw arrays), not Arrow, because drillhole payloads hold arrays of different lengths; float64 world coords on the wire. Meshes over 300k triangles are excluded from cursor picking. Not tested: real mouse input, non-Chromium browsers.
2026-11-19 22:40  P3.T1/T2/T3  582be7b  `BlockModelDefinition(origin, block_size, shape, angles, convention, matrix, max_depth)`, `BlockModel` (split / merge / locate / dense_view / attributes / `calculate` with a whitelisted `ast` expression engine). `make check` green on main: 1001 Python tests, 60 s. Merged ahead of the agent's final report (it was cut off by a network error after these commits and resumed for P3.T4), so its conventions summary is still to come.
2026-11-19 23:55  P3.T4  419e45e  constraints: DTM / solid / attribute expression / plane / polygon / range / block-id, `combine("a and (b or not c)", named)`, JSON via resolvers, `classify_boxes` (INSIDE / OUTSIDE / STRADDLE; exact triangle-box tests, conservative by sampling, not proven). Orchestrator: `ore and not air` on the 10x10x5 truth grid selects 4 125 blocks. Conventions: flat = (i*ny + j)*nz + k; sub-blocks = parent + level + integer sub-index; minimum faces in, maximum faces out; model rotation assumed to follow the ellipsoid convention [verify].
2026-11-19 23:55  P2.T6  ee876d8  `solid_from_sections` (min-area DP stitching, caps, 1<->2 bifurcation, tie lines), `solid_from_strings`, `sections_from_mesh`. Orchestrator: ore solid re-sectioned at 25 m rebuilds valid, volume -0.77%. At 50 m spacing the error is -2.8% (outside the 2% target; test asserts < 5% there). Holes in rings and 1->3 splits raise errors. Never run on real digitised strings.
2026-11-19 23:50  --  --  Second usage-limit pause, about 22:50 to 23:50; P5.T6 and P2.T9 resumed.
2026-11-20 00:15  P2.T9  2f49f3c  `implicit_solid(db, table, column, code)` (signed-distance RBF + own marching tetrahedra) and `marching_tetrahedra`. Orchestrator, default settings with automatic anisotropy and no truth input, own 400k-point sample: IoU 0.907 with the true ore solid, volume ratio 0.9985, valid, 0.2 s. DISCLOSURE from the agent: the default sampling rules (cap and spacing as fractions of the median ore intercept) were chosen after a sensitivity sweep against the truth; first defaults gave IoU 0.80. Isotropic gives 0.59, so anisotropy is essential. Suits a single tabular body; untested on real data, folded or multi-limb domains. Phase 2 complete.
2026-11-20 00:40  P5.T6  9d6ef23  section tools: slab clipping applied centrally in `<SceneContent>` (new layers get it for free; shader materials need `clipping: true`), W-E / S-N / plan presets, draw-a-line, through-hole, stepping, locked square-on section view with real E/N and elevation axes, mesh outlines cut in a Web Worker (1M triangles in about 5 ms). `make check` + `make e2e` (42 tests) re-run green by orchestrator; two screenshots inspected. Not done (optional): cap/fill of cut solids, key-plan inset. Section-view e2e covers W-E only; dipping/oblique sections are unit-tested and screenshot-only.
2026-11-20 11:56  --  --  Weekly usage limit stopped all four agents at about 00:50; resumed at 11:56 (P3.T5-T7, P3.T8/T9, P4.T7, P5.T8/T4/T5). About 11 h lost. `make dev` left running for the user at http://localhost:5173/.

2026-11-20 12:30  P4.T7  81d2657  Multi-pass per-domain block estimation + JSON plan file. Synthetic Gate B core: OK corr 0.861, RMSE OK 0.880 < IDW 0.911 < NN 1.321, tonnes +0.35 %, metal -1.41 %. Orchestrator: estimation tests 308 pass; OK RMSE matches own earlier check (0.879). No CLI yet (waits on P3.T8 storage). z-swath corr only 0.655.
2026-11-20 12:45  P3.T5-T7  8665258 4dd5bed 481ab09  Sub-blocking to depth 3 on ore solid: +0.0034 % vs mesh; net error NOT strictly monotone (+0.0100 % d1, +0.0119 % d2, noise level) so test asserts shrinking straddle bound instead. Partials n>=2 within 0.5 % on sphere. Coding agreement 99.96 % centroid / 99.88 % random points at d3.
2026-11-20 12:45  P3.T8/T9  8b5027b 1edb102  Parquet+JSON model folder (5.6M blocks: save 0.84 s, load 0.46 s, lossless), block CSV (provisional Surpac-like layout, 9 unverified assumptions), report + `bakumine blockmodel` CLI. Truth totals: tonnes -3e-5, grade/metal -1.1 % (synth block grid vs truth.json node averaging; cause not verified).
2026-11-20 12:50  fix  51b319a  Full make check after merges: 1 hypothesis failure in desurvey property test (180-degree dogleg; test measure invalid for documented fallback). Test fixed; code unchanged. 1246 tests pass.
2026-11-20 13:20  P5.T8/T4/T5  b366734 efed50d 58c5a8b  Blocks endpoint (BKMA, no coords on wire, 2-16 B/block), instanced blocks layer with section caps and picking, constraint filter panel (UI counts = server = Python masks). Shell LOD: 2M model first view 16 ms, orbit 2 ms/frame; all 2M as instances 26 fps (plan said 30: NOT met for lod=full). Rotated model on screen and opacity<1 not inspected. Orchestrator: make check 1276 + 118 vitest green after merge; e2e (54) run by agent only.
2026-11-20 13:40  scope  --  Token budget: cut P4.T10 (stretch), P5.T7, P6.T7 undo (job log kept if cheap), P7.T3 (no Surpac inputs supplied). Remaining tasks run 2 agents at a time, Sonnet for routine work, targeted tests while working + one make check at the end.
2026-11-20 14:05  Gate A/B CLI  e314a1a 4b67efc  `bakumine estimate`, `blockmodel create|code`, scripts/gate_synthetic.py. Orchestrator reran script: ORE volume -0.047 %, tonnes -0.047 %, metal -2.4 %, OK corr 0.812 (per parent, volume-weighted). Gate A numbers PASS (code review part pending in P7.T5); Gate B PASS.
2026-11-20 14:30  P6.T1/T6  afc4089 9e56b7b  Project folder (project.json, atomic), `bakumine project init|add|list`, API serves BAKUMINE_PROJECT; report endpoint + CSV + sidebar Report panel, equals CLI report. Orchestrator: api+cli 73 pass, vitest 123 pass after resolving cli.py conflict. Report panel NOT yet seen in a browser (no Playwright test).
2026-11-20 15:00  P7.T2 (partial)  d1459d6  scripts/run_babbitt.py: ft->m, 3.048 m composites, 100x100x20 m model, manual variogram (auto-fit R2 0.12 rejected), OK 3 passes. Global mean OK vs NN +7.8 % (target 5 %: NOT met; NN biased by hole-end picks), swath corr x 0.33 / y 0.58 / z 0.89, LOHO xval corr 0.29. Test asserts 10 % + 2 of 3 swaths. Left UNTICKED. No domain wireframe, density 2.9 assumed.
2026-11-20 15:20  scope  --  User: "dont save tokens, just build". Cuts reverted (P4.T10, P5.T7, P6.T7 undo back in scope; P7.T3 still blocked on Surpac inputs). Up to 4 agents, default model.
2026-11-20 15:50  P7.T4  405a3f4  scripts/bench_perf.py (5M blocks, 20k composites). Orchestrator rerun: IDW 1M 0.54 s (<60), OK 1M 0.70 s (<300), 5M 3-pass 3.6 s, sub-block+code d3 4.5 s, save 0.64 / load 0.49 s. P5.T4 exposed-face path 2 ms/frame (from P5 log). No code changed; nothing missed a target. Random-composite data is easier than real drilling.
2026-11-20 16:10  P4.T10  cfa5e83  Dynamic anisotropy (per-block angles, grouped calls of array estimators; kriging uses block's own anisotropy). Folded case: IDW RMSE 0.804 -> 0.263, OK 0.777 -> 0.190 (corr 0.71 -> 0.98). Constant angles bit-identical to fixed. Orchestrator: estimation tests 326 pass. Slow if every block has distinct angles (angle_step mitigation, not benchmarked).
2026-11-20 16:45  P6.T2-T5, P7.T1  9ebeebc 59cad85  Import / composite / block model / estimation dialogs (background job, progress, cancel) over /api/v1/ops; endpoints equal CLI (rtol 1e-6, rotation matrix round-off). E2E UI run raw CSVs -> report: tonnes -0.05 %, metal -2.4 %, 4 s. Orchestrator: make check 1313 + vitest 127 green, make e2e 55 passed. Not done: 3D ellipsoid preview, solid/DTM import dialog, capping options in UI. NOTE: synth truth.json plunge (+10) is -10 in the UI/rotation.py convention: sign convention to resolve in STATUS.
2026-11-20 17:20  P7.T2  b008a6f  Babbitt rework: footwall contact + sampled-column domain, basal/upper zones, 99.5 pct caps, fitted variograms (R2 0.89/0.88), 50x50x15 m blocks. Orchestrator rerun: OK vs NN +2.3 %, OK vs declustered +3.3 % (basal +4.4 %), swath corr x 0.84 / y 0.91 / z 0.96, LOHO xval corr 0.39. Domain is inferred from grades/gaps, not logged geology; density 2.9 assumed.
2026-11-20 17:20  P7.T5 (part 1)  51ecbab..1c5ae17  Core review: 10 confirmed bugs fixed with failing-first tests (project path traversal + concurrent registry loss, block CSV missing-token clash, assay column silently read as text, DXF 2D polyline z, int64 overflow, round() extremes, .str header). Estimation core: read only, no bug found. Gate A code-review part done.
2026-11-20 17:20  P5.T7  de539db a4afe36  Section digitising: draw/edit/snap strings on sections, save (.str), build solid. 5 synthetic sections -> valid watertight solid, volume -0.76 %. Orchestrator: after merging all, make check 1348 + vitest 136 green, make e2e 56 passed.
2026-11-20 17:50  P6.T7  acbf049  Job log (jobs.jsonl, append+fsync), undo via column/folder snapshots (last 5, checksum-guarded), `bakumine project replay|export-log|export-geometry`, Job log panel. Replay of synthetic chain into a fresh project: bit-identical model; undo after code and estimate bit-identical. Orchestrator: make check 1356 + vitest 140 green. Limits: constraint-JSON report filters and non-default import options not replayable.
2026-11-20 18:10  P7.T6  b73947d  README (install, make dev, browser + CLI walkthroughs, project folders, Babbitt, formats, testing), 5 screenshots via `make screenshots`, Makefile API_PORT. Fresh clone verified by agent (make install, make dev, API + viewer, gate script). Orchestrator: make check 1356 + vitest 140 green; looked at 01-scene-3d.png.
2026-11-20 19:10  P7.T5 (part 2)  security merge ..7a8fc5b, 9d9a5f5  API/web review: 8 confirmed findings fixed (composite-name path write, DNS rebinding + cross-origin writes -> Host/Origin guard, upload delete wildcards + 1 GB limit, malformed input 500s, unbounded report/model/job sizes, concurrent numba kernels SIGABRT -> serialised parallel_njit, concurrent string/solid saves, vite proxy Host). Merge interaction with job log fixed in test stub. Orchestrator: make check 1382 + vitest 140 green, make e2e 56 passed.
2026-11-20 19:10  P7.T7  8e7336d + update  STATUS.md: what works with references and numbers, 5 [verify] items, 18 .str/.dtm and 9 block-CSV assumptions, plunge sign mismatch, known gaps, P7.T3 inputs needed. Orchestrator updated review state and test counts.
2026-11-20 19:10  DONE  --  60 of 61 tasks ticked. P7.T3 open: needs Surpac exports (plan §7).
2026-11-20 22:40  client project fixes  56fbc07..(merge fix)  After the first real-data test (client gold project, data kept out of git): viewer (blocks hidden when colouring a large model by grade; project opens on grade), `bakumine intercepts`, `bakumine classify`, `bakumine surface from-points|import`, `bakumine solid implicit` (auto anisotropy no longer follows the holes: plane-normal error 27 deg -> 0.8 deg on a 100 m thick synthetic body), nested variogram structures + dip/dip-azimuth input, import of multi-sheet / two-row-header / Mongolian-Russian workbooks, sample-number depths, QC rows, special values, DMS collars, merging, `bakumine extract-tables` (pdfplumber), import recipes. Orchestrator: make check 1584 + vitest 171 green, make e2e 60 passed. New panels beyond the e2e paths not checked by eye.
2026-11-21 05:10  client project fixes, round 2  ..04fe20f  From the full CLI rerun on the client project (figures kept with the client data, outside git). Import: PDF continuation-page row loss fixed (pdfplumber found no cell above an unruled first row), per-file options (`--source-option FILE:KEY=VALUE`), hole filters, total/empty rows dropped, element headers with units, duplicate intervals, conflicts per column, sample-number depths checked against hole depth. Modelling: auto anisotropy for stacked lenses (between-hole indicator correlation; synthetic pole error <= 4.6 deg; on the real data it REFUSES to choose, best candidate 59 deg off: orientation must be given), cavities valid, classify per-class hole support + sample filter, intercept modes strict/average/moving-average, `bakumine estimate-plan`, estimate timing. Orchestrator: make check 1724 + vitest 181 green, make e2e 60 passed.
2026-11-21 06:00  client project fixes, round 3  ..cd60a4a  Final CLI rerun (one import, 285 s; figures kept with the client data, outside git). Then fixed: extract-tables joining consecutive tables with near-identical headers, sample-number depths per file and never over stated from/to, near-duplicate intervals (--interval-tolerance) with overlaps reported at import. Real import now reproduces the workaround database through the CLI alone (0 overlaps). Orchestrator: make check 1736 + vitest 181 green, make e2e 60 passed. The real-data script was not rerun after round 3.
2026-11-21 17:30  Import panel + rerun  (this commit)  Import panel: interval tolerance field, per-file slack of the sample-number rule, overlap report and refused sample-number rows shown after import (API returns `overlaps`). Real-data script rerun end to end without hand workarounds: database and resource figures identical to the previous run. make check 1,738 pytest + 184 vitest, make e2e 60 passed.
2026-11-21 20:10  P8.T1  (this commit)  `.dtm` reader/writer to the published 1996 format appendix: vertex numbers count the null records of the `.str`, header `<name>.str,` + `0, ..., END` line, flags `neighbours, closed, direction=solid`, no record of zeros, and triangles clockwise seen from outside (found by reading the appendix's box: volume was -100^3 before the swap). The appendix's own example reads as a closed box of 100^3 with matching neighbours (read from a scratch copy, not committed). Files of the old provisional writer are recognised and still read. Not checked against a current Surpac export. Also from the real-data follow-up: composites below the stated hole depth are located (d688cb0).
2026-11-21 21:00  P8.T12 (Babbitt)  25801bb (cherry-picked)  Public domain wireframe `Mpz.stl` (binary STL reader added, read-only) + the pygslib tutorial's parameters. Against the published numbers: composites 54189 = 54189; domain mean +0.01 %; declustered mean -0.12 %; NN block mean +0.12 %; OK -0.86 % (+0.16 % without the 3785 blocks whose neighbourhood holds coincident twin-hole samples, which GSLIB leaves unestimated: inferred from its source, PyGSLIB not run); ID2 +1.23 % (ours weights by anisotropic distance; +0.17 % with isotropic weights). Coded volume vs mesh -0.02 %. Wireframe is not closed as published (repaired at run time). Gaps noted: no CLI for block fraction in a solid / composites in a solid, no unit scale on `surface import`, web upload does not take `.stl`. Orchestrator: make check 1768 + vitest 184 green.
2026-11-22 00:20  Final review  06065ac..88d3ce3  Three independent review passes (core code since 27f5e94, a first-time user following README / --help, the web workflow driven in a browser). No wrong numbers on clean input. Fixed: 7 core import / intercept defects (thousands commas, hole id once per hole, keep_first losing values, moving-average 'including', source index cleaned as a value, stacked numbers in PDF cells, true thickness abort); command-line usability (classify help with a full example, estimate --overwrite, create refuses to replace a model, coding file in help, STL precision warning); 9 API / viewer findings (re-estimate control, classify refusing to overwrite foreign attributes, bounded hole-id templates / job log / classify support / implicit grids, 500s on names, stale report selections after undo, dark-theme contrast, wide report table, sub-block depth 0..3 everywhere). Added: `blockmodel fraction`, `composite --flag-solid`, `surface import --scale`, STL upload. client project rerun after the fixes: database and resource identical. Orchestrator: make check 1832 + vitest 190 green, make e2e 60 passed. Open: digitised strings/solids not in the job log; CLI estimate does not stop on a sample filter that matches nothing.

### Blockers

(none) — GitHub's initial commit was merged into local `main` on 2026-11-19 (26aa553) at the user's request. Nothing has been pushed.

### Measured performance

- 1M targets x 20k samples, max 24 neighbours, 10 threads: IDW point 0.7 s, IDW block 4x4x2 1.8 s, point OK 1.3 s, block OK 4x4x2 1.6 s (agent-measured, one machine).
- `grid_in_solid`: 10M centroids in 0.01 s; 70M in 0.03 s (orchestrator). DTM `z_at` 1M points 0.012 s.
- Viewer (headless Chromium, Apple GPU, agent-measured): 500 holes / 100k intervals 2.2 ms per frame; plus 1M triangles shaded 2.8 ms, shaded+edges 12.5 ms.
- Search alone: 0.5 s plain; 8.3 s with octant + per-hole limits.
