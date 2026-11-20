# BakuMine Block Modeller: status

Status as of 2026-11-20, `main` at `b7e2aeb` (P7.T6 logged). Sources: `plan.md` §4 and the §8
progress log, `research.md`, the module docstrings in `bakumine/core/io/` and
`bakumine/core/estimation/rotation.py`. Numbers are copied from §8 unless marked otherwise. Where
§8 says that something was not verified, this file says the same.

## 1. Summary

The full chain runs end to end from the CLI and from the browser UI: raw collar / survey / assay /
geology CSVs, then validation, desurvey, compositing, a sub-blocked block model coded against a
solid and a DTM, multi-pass NN / IDW / OK / SK estimation per domain, and a volume / tonnes /
grade report. The viewer shows drillholes, meshes and up to 2M blocks, with sections and
digitising. Every numerical routine is checked against an analytic case, an oracle library or the
synthetic deposit's known truth. **Nothing has been compared with Surpac itself.** No Surpac
licence or Surpac-exported files were available, so every Surpac file layout and every Surpac
convention below is an unverified assumption.

| Check | Criterion | Measured | Source |
|---|---|---|---|
| Gate A: coded ORE volume vs true solid | within 1 % | -0.047 % | `scripts/gate_synthetic.py`, orchestrator rerun (§8 Gate A/B) |
| Gate B: OK vs true block grades, ORE | corr > 0.75 | 0.812 (per parent, volume-weighted); 0.861 per block in the P4.T7 core test | §8 Gate A/B, P4.T7 |
| Gate B: tonnes vs truth | within 5 % | -0.047 % (CLI), -0.05 % (UI e2e) | §8 Gate A/B, P7.T1 |
| Gate B: metal vs truth | within 5 % | -2.4 % (CLI and UI) | §8 Gate A/B, P7.T1 |
| Gate B: RMSE ranking vs truth | OK < IDW < NN | 0.880 < 0.911 < 1.321 g/t | §8 P4.T7 |
| Babbitt (real data): OK vs NN global mean | within 5 % | +2.3 % | §8 P7.T2 (b008a6f) |
| Babbitt: OK vs declustered mean | within 5 % | +3.3 % overall, +4.4 % basal zone | §8 P7.T2 |
| Babbitt: swath correlation x / y / z | > 0.5 (test) | 0.84 / 0.91 / 0.96 | §8 P7.T2 |
| Babbitt: leave-one-hole-out cross-validation | (none) | corr 0.39 | §8 P7.T2 |

Gate A also required `/code-review high` on the core: done in P7.T5 (10 confirmed bugs fixed).
The API / web security review is done too (8 confirmed findings fixed, see §5).

## 2. What works, and how it is verified

| Stage | What it does | Verified against | Measured |
|---|---|---|---|
| Drillhole import (P1.T1) | CSV / XLSX, column map, ft to m, Y/X swap option, below-detection rule (default: half the limit), -99 / -999 sentinels on attribute columns, Parquet store | Synthetic and Babbitt import; Y/X swap unit test | Babbitt: 399 holes import from raw |
| Validation (P1.T2) | 18 check codes per hole (research.md §2.5) | Corrupted dataset triggers each check exactly once | 0 errors on synthetic |
| Desurvey (P1.T3) | Minimum curvature (default), balanced tangential, tangent; dip sign selectable (default negative down); `xyz_at` with arc interpolation | Analytic vertical, inclined and quarter-circle holes; `wellpathpy` oracle on 50 holes | wellpathpy 1e-6 m; quarter circle 1e-9 m; reproduces synthetic sample XYZ to 5e-7 m |
| Interval merge (P1.T4) | Split assay and geology intervals at every boundary | Property test (random intervals), total length kept | |
| Compositing (P1.T5) | Downhole / bench / domain, best-fit, min-length rule, density weighting, NaN kept | Hand-worked example; metal accumulation conserved; no composite crosses a domain | 11 410 two-metre domain composites on synthetic, Σ grade × length 3086.592 in and out. No third-party compositing oracle exists |
| Grade capping (P1.T6) | Cap before compositing or estimation | Unit test | |
| Mesh core, validation, measures (P2.T1–T3) | Weld, edge map, normals, BVH / grid index, closed / manifold / orientation / degenerate / self-intersection checks, orientation fix, volume / area / centroid | trimesh on 20 random meshes; analytic cube, sphere, torus | Agrees with trimesh to 1e-9 relative; synthetic ore solid volume -0.057 % of truth |
| DTM (P2.T4) | Plan Delaunay, breaklines, boundary, `z_at`, above / below | Plane and paraboloid reproduced; breakline edges present | 1M `z_at` queries 0.012 s; collars exactly on topography |
| Point-in-solid (P2.T5) | Ray casting with column acceleration, exact plan predicates | `trimesh.contains` on 5 shapes | 0 of 5M points differ (full oracle run gated by `BAKUMINE_FULL_ORACLE=1`, 13 min, agent-run); 10M centroids 0.01 s, 70M 0.03 s; 70M-cell voxel volume within 0.003 % of mesh volume |
| Mesh–plane section (P2.T7) | Arbitrary plane cut to polylines | Cube and sphere exact | |
| Solid from sections (P2.T6) | Min-area DP stitching, end caps, 1↔2 bifurcation, tie lines | Ellipsoid slices; re-sectioned synthetic ore solid | 25 m spacing -0.77 % volume; 50 m spacing -2.8 % (outside the 2 % target; test asserts < 5 % there) |
| Implicit solid (P2.T9) | Signed-distance RBF + marching tetrahedra | Synthetic true ore solid (IoU) | IoU 0.907, volume ratio 0.9985, 0.2 s. Default sampling rules were tuned against the truth (first defaults gave IoU 0.80) |
| Block model definition (P3.T1) | Origin, block size, shape, rotation (angles + convention, or matrix), ijk ↔ centroid ↔ world, auto-extent | Round trip on random points, rotated and unrotated | |
| Sub-blocks (P3.T2, P3.T5) | Octree split / merge to depth 3 on solid and DTM contacts, Morton keys, dense fast path | Volume-invariance property test; synthetic ore solid volume | Depth 3: +0.0034 % vs mesh volume. Error is not strictly monotone with depth (+0.0100 % d1, +0.0119 % d2); the test asserts a shrinking straddle bound instead |
| Partials (P3.T6) | Fraction inside a solid by lattice sampling | Analytic sphere | Within 0.5 % for n ≥ 2 |
| Attributes and expressions (P3.T3) | Typed columns, background values, calculated attributes via a whitelisted `ast` engine (no `eval`) | Unit tests incl. rejected malicious expressions | |
| Constraints (P3.T4) | DTM, solid, attribute expression, plane, polygon, range, block id; AND / OR / NOT; JSON | Each primitive tested; composite = manual mask | `ore and not air` on the 10×10×5 truth grid selects 4 125 blocks. `classify_boxes` is conservative by sampling, not proven |
| Coding (P3.T7) | Assign value under a constraint, with sub-blocking | Synthetic truth codes | 99.96 % agreement by centroid, 99.88 % by random points at depth 3; truth-block ORE 99.997 %, AIR 99.97 % |
| Storage (P3.T8) | Parquet + JSON header folder | Lossless round trip | 5.6M blocks: save 0.84 s, load 0.46 s |
| Report (P3.T9) | Volume, tonnes, grade, metal by attribute, bench, cut-off; CSV | Hand-computed 3×3×3 example; synthetic truth totals | Tonnes -3e-5 relative; grade / metal -1.1 % (see §4) |
| Rotation (P4.T1) | `surpac_zxy_lrl` (default) and `gslib`; generic sequences and `from_axes` | Hand-computed vectors; gslib against a separate transcription of the published `setrot` formulas | Internal consistency only; see §3 |
| Search (P4.T2) | Anisotropic cKDTree; radius, min / max, max per hole, octants, max vertical distance | Brute force on 10k random queries for every option combination | 1M targets × 20k samples: 0.5 s plain, 8.3 s with octant + per-hole limits |
| NN, IDW (P4.T3, P4.T4) | Any power, anisotropic, exact at coincident samples, block discretisation | Brute force; hand example; declustered mean | IDW 1M blocks 0.54 s (target < 60 s) |
| Variogram models (P4.T5) | Nugget + nested spherical / exponential / gaussian, per-structure anisotropy, JSON | `gstools` at 1000 random lags | Match |
| OK / SK (P4.T6) | Point and block, variance, negative-weight correction, duplicates, numba-parallel | 4-sample hand system; PyKrige `OrdinaryKriging3D` and gstools | OK matches PyKrige / gstools to 1e-14 (criterion 1e-6); orchestrator's own solve 1e-15. 1M blocks OK 0.70 s (target < 300 s). Block kriging and nested models are checked only against a dense numpy solver; GSLIB `kt3d` was never run |
| Multi-pass per domain (P4.T7) | Pass list under a constraint, domain-filtered composites, pass number stored, JSON plan | No cross-domain samples; pass attribute consistent; synthetic truth | 5M blocks 3-pass 3.6 s |
| Experimental variogram, auto-fit (P4.T8) | Directional, downhole, least-squares fit as a seed | Synthetic field with known variogram | Assay support, 10° cone: nugget ratio 0.121 vs 0.15, major range 116 vs 120 m. Fails in other settings (3 strict xfails, §4) |
| Validation tools (P4.T9) | Swath tables, global mean vs NN, cell declustering, LOO and leave-one-hole-out CV, compare to truth | Synthetic truth | OK ≤ IDW ≤ NN asserted (OK RMSE 0.879 true / 0.880 fitted variogram); LOO OK r 0.871, hole-out r 0.585 |
| Dynamic anisotropy (P4.T10) | Per-block bearing / plunge / dip | Folded synthetic case; constant angles vs fixed ellipsoid | IDW RMSE 0.804 → 0.263; OK 0.777 → 0.190 (corr 0.71 → 0.98); constant angles bit-identical to fixed |
| I/O (P2.T8, P3.T8) | `.str`, `.dtm`, DXF (3DFACE, POLYLINE / polyface), OBJ, PLY, block CSV | Own round trips only | Ore solid via `.dtm/.str`: valid, volume within 6e-7. **Surpac layouts provisional (§3)** |
| API and transport (P5.T8) | FastAPI; binary "BKMA" container (JSON header + raw arrays, not Arrow) | API tests; e2e | 2M-block model: first view 16 ms with shell LOD, 2–16 B per block on the wire |
| Viewer (P5.T1–T6) | Z-up scene, large-coordinate recentring, drillholes, meshes, instanced blocks with picking, constraint filter, slab sections, locked section view | Playwright screenshots (inspected by orchestrator); filter counts = server = Python masks | 500 holes 2.2 ms/frame; 1M triangles 2.8 ms shaded, 12.5 ms with edges; 2M blocks orbit 2 ms/frame (shell); headless Chromium, Apple GPU, agent-measured |
| Section digitising (P5.T7) | Draw / edit / snap strings on sections, save `.str`, build solid | 5 synthetic sections by script | Valid watertight solid, volume -0.76 % |
| UI dialogs (P6.T2–T6) | Import wizard, compositing, block model create / code, estimation (background job, progress, cancel), report panel | Endpoints equal CLI (rtol 1e-6); P7.T1 e2e drives raw CSVs → report | e2e: tonnes -0.05 %, metal -2.4 %, 4 s |
| Project, job log, undo (P6.T1, P6.T7) | `project.json` folder, object tree, append-only `jobs.jsonl`, replay, undo via snapshots (last 5, checksum-guarded) | Reopen restores; replay into a fresh project | Replay of the synthetic chain: bit-identical model; undo after code and after estimate: bit-identical |
| CLI (P1.T7 and later) | `import, validate, desurvey, composite, blockmodel create / code / info / export-csv / import-csv / report, estimate, project init / add / list / replay / export-log / export-geometry, synth` | `tests/cli/`, gate script | |
| Performance (P7.T4) | `scripts/bench_perf.py`, 5M blocks, 20k composites | Plan targets | All targets met; no code changed. Random composites are easier than real drilling |

## 3. Provisional: not verified against Surpac

### 3.1 The [verify] items in research.md

research.md marks five Surpac-specific statements **[verify]** (lines 33, 42, 98, 157, 308). It
also uses the tag in its introduction (line 5) and in the open-risk note (line 358), which refer
to these five. None has been checked against Surpac.

| # | research.md quote | Current assumption in code |
|---|---|---|
| V1 | "Surpac stores coordinates in **Y, X, Z order** (northing first) **[verify]**" (§2.1) | `.str`, `.dtm` and block CSV read and write Y, X, Z, swapped at the I/O boundary only (`COORD_COLUMNS = ("y", "x", "z")` in `surpac_str.py`). Drillhole import has a `yxz` option, off by default |
| V2 | "Surpac's convention: **dip is negative downward**, so a vertical hole is −90° **[verify]**" (§2.2) | `desurvey(..., dip_convention="negative_down")` is the default; positive-down is selectable (Babbitt uses it) |
| V3 | "Which method Surpac applies for "curved" holes is **[verify]**" (§3) | Minimum curvature is the default; balanced tangential and tangent are available. Not known which one matches Surpac |
| V4 | "### 5.6 File formats (text interchange) **[verify against real exports]**" (`.str`, `.dtm`, block CSV layouts) | See 3.2 and 3.3 |
| V5 | "Surpac's own convention is documented as **"Surpac ZXY LRL"**: rotate about Z (bearing), then X (plunge), then Y (dip), with left-, right-, left-hand rules respectively; GSLIB, Vulcan, Datamine and Isatis each differ **[verify]**" (§7.5) | See 3.4 |

### 3.2 `.str` and `.dtm` layouts

Written from research.md §5.6 only; no Surpac file or manual was used. Binary `.str` / `.dtm` are
detected and rejected. `describe_surpac_file(path)` is the tool for inspecting the first real
file. All layout constants are isolated at the top of each module. (§8 P2.T8 counted 14
assumptions; the current docstrings list 8 for `.str` and 10 for `.dtm`, including the
binary-variant note in each.)

| File | Assumption (unverified) |
|---|---|
| `.str` | 1. Line 1 header `location, date, purpose, memo`; anything after the third comma kept as memo |
| `.str` | 2. Line 2 axis record `0, Y1, X1, Z1, Y2, X2, Z2` |
| `.str` | 3. Records `string_no, Y, X, Z, d1, d2, …`; first coordinate is northing |
| `.str` | 4. A record with string number 0 and zero coordinates ends a segment |
| `.str` | 5. `0, 0, 0, 0, END` ends the file (case-insensitive) |
| `.str` | 6. Descriptions are comma-separated, no quoting, so they cannot contain a comma |
| `.str` | 7. Writer layout (`", "`, fixed decimals, trailing empty description) is our choice; whether Surpac needs fixed column widths is unknown |
| `.str` | 8. Binary variant not supported |
| `.dtm` | 1. Line 1 is a free-text header, not interpreted |
| `.dtm` | 2. Extra lines before the first `OBJECT` are kept as a preamble |
| `.dtm` | 3. `OBJECT, id` then `TRISOLATION, id, neighbours=…, validated=…, closed=…`; which flags use yes/no vs true/false is a guess |
| `.dtm` | 4. Triangle record `id, v1, v2, v3, n1, n2, n3`; ids 1-based, restart per trisolation |
| `.dtm` | 5. Vertex indices 1-based, counting point records of the `.str` across all segments, excluding header, axis and separators |
| `.dtm` | 6. `n1` across edge v1–v2, `n2` across v2–v3, `n3` across v3–v1; 0 = none |
| `.dtm` | 7. A zero record ends each trisolation; `END` ends the file |
| `.dtm` | 8. Writer puts all vertices of a mesh in one `.str` segment with one string number; Surpac's choice unknown |
| `.dtm` | 9. Winding written counter-clockwise from outside; whether Surpac expects the opposite is unknown |
| `.dtm` | 10. Binary variant not supported |

### 3.3 Block CSV layout (9 assumptions, `bakumine/core/io/block_csv.py`)

| # | Assumption (unverified) |
|---|---|
| 1 | Column order: centroid Y, X, Z, then sizes along Y, X, Z, then attributes |
| 2 | Column names `ycentre, xcentre, zcentre, ysize, xsize, zsize`; the reader accepts many synonyms |
| 3 | Default writes two free-text header lines and one line of column names; reader skips leading non-data lines |
| 4 | Comma delimiter (reader also accepts semicolon, tab, whitespace) |
| 5 | Sizes are the block's own size along the model axes; for a rotated model, centroids are world coordinates and sizes are along the rotated axes |
| 6 | One row per leaf block; no rows for split parents |
| 7 | Surpac's missing-value token is unknown; it is an option (default: empty field), never 0 |
| 8 | Row order not relied on |
| 9 | Text attributes written bare, quoted only when they contain the delimiter |

### 3.4 Rotation conventions

| Item | Status |
|---|---|
| `surpac_zxy_lrl` (default) | Implemented from the one-line statement in research.md §7.5. Derived, not checked: the sequence is intrinsic; **positive plunge = major axis up**; positive dip lifts the semi-major axis' right-hand end; `major = (sin b cos p, cos b cos p, sin p)`. Tests check internal consistency and hand-computed vectors only |
| `gslib` | From the published `setrot` formulas, recalled from the book; cross-checked against a separate transcription, never against a GSLIB executable |
| `datamine_zxz` | **Not implemented.** Handedness and major-axis assignment could not be stated with confidence. Use `rotation_from_sequence` or `Ellipsoid.from_axes` |
| Block model rotation | Uses the ellipsoid convention unchanged (`definition.py`, marked **[verify]**). Assumed: Surpac applies model bearing / dip / plunge with the same axes, order and signs as its ellipsoid, pivoting about the minimum corner. Not compared with a rotated Surpac model. Workaround: define rotated models with `matrix` |
| Plunge sign mismatch (§8 P6.T2-T5 note) | `data/synthetic/default/truth.json` gives the true anisotropy as plunge **+10** with `major = (…, -sin p)`, i.e. positive plunge = down. `rotation.py` and the UI use positive = up, so the same field is entered as plunge **-10** (README walkthrough: angles `35, -10, 50`). Both are internally consistent; which sign matches Surpac is unknown (V5). Not resolved in code: the two files keep opposite sign conventions |

## 4. Known gaps and limitations

| Area | Limitation (from §8 unless stated) |
|---|---|
| Variogram auto-fit (P4.T8) | Plan criterion passes only on assay-support samples with a 10° cone. 3 strict xfails, not loosened: 22.5° cone gives major range ≈ 91 m (truth 120 m), on samples and on composites; nugget on 2 m composites 0.024 (truth 0.15). Semi-major range not recovered (92–115 m vs 70 m). Auto-fit is a seed only |
| Babbitt auto-fit | First run: R² 0.12, rejected; manual variogram used. Rework uses per-zone fits with R² 0.89 / 0.88 |
| Synthetic z-swath | z-swath correlation only 0.655 (P4.T7) |
| Report grade vs truth | Grade / metal -1.1 % vs truth.json; attributed to synthetic block grid vs truth.json node averaging. **Cause not verified.** README walkthrough: 1.96 vs 2.01 g/t |
| Gate B metal | -2.4 % (within the 5 % criterion) |
| Kriging references | Block kriging and nested models checked only against a dense numpy solver. GSLIB `kt3d` never run; its nugget treatment is from memory. `lagrange` output is covariance-form (sign opposite to research.md) |
| Solid from sections | 50 m spacing -2.8 % (outside the 2 % target). Holes in rings and 1→3 splits raise errors. Never run on real hand-digitised strings |
| Implicit solid | Default sampling tuned against the synthetic truth; isotropic mode IoU 0.59. Untested on real data, folded or multi-limb domains |
| Sub-blocking | Volume error not strictly monotone with depth (noise level) |
| `classify_boxes` | Conservative by sampling; not proven |
| Viewer frame rate | All 2M blocks as instances (`lod=full`): 26 fps, below the 30 fps target. Shell LOD meets it (2 ms/frame) |
| Viewer, not inspected | Rotated block model never looked at on screen; opacity < 1 not inspected (instances are not depth-sorted). Real mouse / trackpad input and camera-motion jitter not tested. Only Chromium tested. Dipping / oblique sections: unit tests and screenshots only (e2e covers W-E) |
| Viewer, not done | Cap / fill of cut solids, key-plan inset (P5.T6 optional); 3D ellipsoid preview in the estimation dialog; solid / DTM import dialog (CLI `project add` only); capping options in the UI |
| Report panel | No dedicated Playwright test (§8 P6.T1/T6); it now appears in the README screenshot run (`make screenshots`) |
| Picking | Meshes over 300k triangles are excluded from cursor picking |
| Dynamic anisotropy | Slow if every block has distinct angles; `angle_step` mitigation not benchmarked |
| Job log replay | Constraint-JSON report filters and non-default import options are not replayable |
| DXF | 2D polyline z fix (P7.T5) tested on constructed files only, not on real DXF exports |
| Babbitt data | Units feet, dips positive down, 70 holes with a sentinel survey at 90 000. No geology table: the domain (footwall contact, basal / upper zones) is inferred from grades and gaps, not logged geology. Density 2.9 t/m³ assumed. LOHO xval corr 0.39. Upstream (NRRI TR-2003/21) has no explicit licence: confirm before redistributing |
| Performance numbers | One machine, mostly agent-measured; random composites are easier than real drilling |
| Drillhole import defaults | Below detection = half the limit; -99 / -999 sentinels on attribute columns only |
| Transport | Custom "BKMA" container, not Arrow as planned in §1 |

## 5. Not done, and the review outcome

| Task | State | Reason |
|---|---|---|
| P7.T3 Surpac comparison | Not started (marked CUT: no inputs) | No Surpac licence or exports supplied (plan.md §7 input 1). Needed: a `.str` + `.dtm` pair exported from Surpac, a block model CSV export, and the estimation parameters used, all for the same data set. To answer the [verify] items as well, the block model should be coded and IDW-estimated from drillholes we also have, and ideally use a rotated model and a rotated ellipsoid (V5, 3.4) and include desurveyed hole coordinates for a curved hole (V2, V3); these additions are this report's suggestion, not part of §7 |
| P7.T5 Review | Done (ticked) | Core review: 10 confirmed bugs fixed with failing-first tests (51ecbab..1c5ae17); estimation core read only, no bug found. API / web security review: 8 confirmed findings fixed (composite name path write, DNS rebinding / cross-origin writes now refused by a Host + Origin guard, upload delete wildcards and size limit, 500s on malformed input, unbounded report / model / job sizes, concurrent numba kernels aborting the process, concurrent string/solid saves, dev proxy Host). Unconfirmed and left open: a read between `save_model`'s two renames, XLSX zip/XML bombs, delay of UI kernels behind the numba lock, memory blow-ups inferred but not run |

## 6. Reproducing the numbers

```sh
make install                                   # once per checkout
make check                                     # ruff + pytest + web typecheck + vitest
make e2e                                       # Playwright (once: cd web && npx playwright install chromium); includes P7.T1 UI run
uv run python scripts/gate_synthetic.py        # Gate A / B: ORE volume, tonnes, metal, OK correlation
uv run python scripts/run_babbitt.py           # Babbitt: global means, swaths, LOHO cross-validation
uv run python scripts/bench_perf.py            # 5M blocks / 20k composites timings -> data/bench_perf.json
BAKUMINE_FULL_ORACLE=1 uv run pytest tests/geometry/test_inside.py   # full trimesh point-in-solid oracle (~13 min)
uv run pytest tests/estimation/test_kriging.py     # PyKrige / gstools / hand-solved kriging
uv run pytest tests/estimation/test_experimental.py  # variogram recovery incl. the 3 strict xfails
uv run pytest tests/cli/test_babbitt_run.py        # Babbitt acceptance assertions
```

## 7. Test inventory

| Suite | Count | Source |
|---|---|---|
| pytest (`make check`) | 1382 passed, 3 skipped | orchestrator run after the P7.T5 security merge |
| of which strict xfail | 3 | `tests/estimation/test_experimental.py` |
| vitest | 140 passed | orchestrator run after the P7.T5 security merge |
| Playwright e2e | 56 passed | orchestrator `make e2e` after the P7.T5 security merge |
