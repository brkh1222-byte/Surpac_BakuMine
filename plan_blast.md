# Blast BakuMine: plan

Owner request (2026-11-25): a separate product tab, **Blast BakuMine**, next to the block modeller. It designs an
open-pit bench blast on the 3D site model (topography + block model + geology) and simulates it: where the rock
goes, how much damage, how strong the shock / vibration, all shown in 3D. Results feed the feasibility study.
Reference products: Maptek BlastLogic (main), Vulcan blast hole layout, Deswik.OPDB.

Branch: `blast-bakumine` (local only; **not pushed**; `main` stays the block modeller).
Theory and formulas: `research_blast.md`. Explanation for the owner: Claude Doc "How open-pit blasting works".

## 0. Rules for this module

1. Same rules as `CLAUDE.md` (clean room from published literature, tests define done, independent numerical
   check for every formula, x east / y north / z up, SI units inside, missing stays NaN, no silent defaults).
2. The scope rule 7 of the block modeller ("no pit design, scheduling") does not apply here: blasting is this
   module's subject. The module must not change block-modeller behaviour; it only reads its objects.
3. Units inside: metres, kilograms, seconds, pascals. Delays are given in milliseconds at the I/O boundary only
   (fields named `*_ms`); PPV is reported in mm/s (`*_mm_s`).
4. **Engineering-model honesty.** The models are the empirical / analytical ones used in blast engineering
   practice (Kuz-Ram, scaled distance, Holmberg-Persson, Richards-Moore, ballistic throw). They are not a
   hydrocode. Every result names its model and its site constants; defaults for site constants are published
   typical values and are labelled "uncalibrated" until a site calibration is supplied.
5. One task, one commit `B<n>: <summary>` on the branch. No push.

## 1. What the user gets

1. **Design**: pick a bench polygon on the model, get a drill pattern (burden, spacing, staggered / square),
   hole depth with subdrill, charge and stemming per hole, powder factor, drilled metres, explosive mass.
2. **Timing**: row-by-row, echelon / V, or hole-by-hole delays; firing time per hole; maximum instantaneous
   charge (MIC) in an 8 ms window; burden relief.
3. **Predictions**:
   - fragmentation curve (Kuz-Ram: x50, uniformity n, P80, oversize / fines fractions);
   - ground vibration PPV at any point and on the surface (scaled-distance law, per-delay charge) with limits;
   - air overpressure (cube-root scaling, dBL);
   - flyrock maximum range by mechanism and the clearance zone (Richards and Moore);
   - near-field damage in the block model (Holmberg-Persson PPV, damage classes), back-break depth;
   - throw and muckpile: where the broken rock lands (ballistic fragments, swell, repose slope).
4. **3D animation** on the real model: holes firing in sequence, stress wavefronts expanding at the P-wave
   speed and coloured by PPV, blocks changing to damaged / broken, fragments flying and settling into the
   muckpile, PPV map draped on the topography, flyrock envelope, monitor points.
5. **Report** (JSON + text) for the feasibility study: drill and blast parameters, consumption, predicted
   fragmentation, environmental effects against limits.

## 2. Architecture

```
bakumine/blast/
  model.py          dataclasses: Rock, Explosive, Hole, BlastDesign, Monitor; JSON I/O          (B1, orchestrator)
  design.py         rules of thumb, pattern layout on a bench polygon, charging, powder factor   (B2)
  timing.py         initiation sequences, firing times, MIC, burden relief                       (B2)
  fragmentation.py  Kuz-Ram (Kuznetsov, Cunningham, Rosin-Rammler), rock factor                  (B3)
  vibration.py      PPV scaled distance, charge per delay, airblast, limits                      (B3)
  flyrock.py        Richards-Moore ranges, launch velocity, ballistic trajectories               (B3)
  damage.py         Holmberg-Persson near field PPV, damage classes, block attribute             (B4)
  waves.py          wave speeds, arrival times, wavefront sources for the viewer                 (B4)
  throw.py          fragment launch, ballistic flight, landing, muckpile heightmap               (B4)
  simulate.py       design + site -> result bundle (schema in section 3)                         (B5)
  bench.py          demo bench: cut a bench (floor + face) into a topography                     (B5)
bakumine/commands/blast.py     CLI: blast design | simulate | report | demo                      (B5)
bakumine/api/routes_blast.py   /api/v1/blast/*                                                   (B5)
web/src/blast/                 Blast BakuMine tab (own scene, player, panels)                    (B6)
tests/blast/                   mirrors bakumine/blast
```

Shared contract = `bakumine/blast/model.py` + the result bundle (section 3). Agents own disjoint files.

## 3. Result bundle (`bakumine-blast-result`, version 1)

JSON written by `simulate`, read by the viewer. Arrays are plain lists; coordinates are world x, y, z.

```
{ "format": "bakumine-blast-result", "version": 1,
  "design":  BlastDesign.to_dict(),
  "summary": { volume_m3, tonnes, explosive_kg, powder_factor_kg_m3, powder_factor_kg_t, drilled_m, holes,
               x50_m, uniformity_n, p80_m, oversize_fraction, mic_kg, duration_ms,
               flyrock_max_m, clearance_m, backbreak_m, models: {name: {constants..., calibrated: bool}} },
  "timeline": { "t_end_ms": T, "holes": [ { id, row, t_fire_ms, collar:[x,y,z], toe:[x,y,z],
               charge_top:[x,y,z], charge_bottom:[x,y,z], charge_kg, diameter_m } ] },
  "waves":   { "cp_m_s", "cs_m_s", "k", "beta",
               "sources": [ { hole, t_ms, centre:[x,y,z], charge_kg } ] },
               # viewer: radius(t) = cp * (t - t_ms); PPV(R) = k * (R / sqrt(charge_kg)) ** -beta  [mm/s]
  "ppv_surface": { origin:[x0,y0], spacing, shape:[nx,ny], z:[...], ppv_mm_s:[...], arrival_ms:[...] },  # row-major [ix*ny+iy]
  "damage":  { "classes": ["intact","damaged","fractured","crushed"], "thresholds_mm_s": [...],
               "blocks": { centre:[[x,y,z]...], size:[[dx,dy,dz]...], ppv_mm_s:[...], cls:[int...],
                           t_break_ms:[...], blasted:[bool...] } },
  "fragmentation": { sizes_m:[...], passing:[...], x50_m, n, xc_m },
  "throw":   { "g": 9.81, "particles": { p0:[[x,y,z]...], v0:[[vx,vy,vz]...], t0_ms:[...], t_land_ms:[...],
               p_land:[[x,y,z]...], p_rest:[[x,y,z]...], size_m:[...], kind:[0 muck | 1 flyrock] } },
               # viewer: p(t) = p0 + v0*dt - g/2*dt^2 (z) for t0 <= t <= t_land, then eased to p_rest
  "muckpile": { origin, spacing, shape, z_before:[...], z_after:[...], cut_m:[...] },   # cut_m only when a face was cut (B11)
  "flyrock": { face_burst_m, cratering_m, stemming_ejection_m, max_range_m, clearance_m, safety_factor },
  "airblast": { k, a, rings: [ { radius_m, pa, dbl } ] },
  "monitors": [ { name, xyz, distance_m, charge_kg, scaled_distance, ppv_mm_s, limit_mm_s, ok,
                  airblast_dbl, airblast_limit_dbl, airblast_ok } ],
  "envelope": { "window_ms": 8, "mic_kg", "bins": [ { t_ms, kg, holes:[ids] } ] } }
               # charge fired per 8 ms window (vibration.charge_per_delay): t_ms = start of the window,
               # kg = explosive in it, holes = their ids; the tallest bin is the MIC (mic_kg)
```

Added by B5: `envelope` (above); `summary.notes` (plain-language caveats) and further report totals in
`summary` (rows, burden_m, spacing_m, stemming_m, subdrill_m, throw_com_m, muckpile_volume_m3, ...). Values
that are not finite are written as `null`.

## 4. Tasks

| Id | Task | Owner | Depends | Acceptance (independent check) |
|---|---|---|---|---|
| B1 | Plan, data model, research skeleton | orchestrator | - | model JSON round trip; charge mass = pi/4 d^2 L rho by hand |
| B2 | `design.py`, `timing.py` | agent | B1 | published worked examples for burden rules; hole count / volume of a rectangle analytic; firing times of a row-by-row and a V pattern by hand; MIC by brute force |
| B3 | `fragmentation.py`, `vibration.py`, `flyrock.py` | agent | B1 | Kuz-Ram against a published worked example (Cunningham) and Rosin-Rammler identities (P(x50)=0.5); PPV against USBM numbers and log-linear regression recovery with numpy; flyrock range against Richards-Moore example and against scipy ODE ballistic integration |
| B4 | `damage.py`, `waves.py`, `throw.py` | agent | B1 | Holmberg-Persson against scipy quad of the integral and the far-field point-charge limit; wave speeds from elastic constants; ballistic landing against scipy solve_ivp; muckpile volume = swell x blasted volume to 1 %, slopes <= repose |
| B5 | `simulate.py`, `bench.py`, CLI, API | agent | B2-B4 | bundle validates against section 3; CLI == API; demo runs on `data/synthetic/default` |
| B6 | Web tab "Blast BakuMine" | agent | B1 (fixture), B5 | vitest for the animation math (wave radius, particle position, colour scales); tab loads the demo bundle and plays; typecheck |
| B7 | Research doc `research_blast.md` + Claude Doc | orchestrator + agent | - | formulas cited; doc filled |
| B8 | Integration on a real model (git-ignored), screenshots, STATUS | orchestrator | B5, B6 | simulation on one client deposit runs; honest limits listed |
| B9 | Maptek BlastLogic feature study and gap analysis (`research_blast.md` part 4) | orchestrator + agent | - | public sources only, a source per claim, unverified items listed |
| B10 | Satellite imagery draped on the blast terrain (`blast/imagery.py`, `/blast/imagery`, viewer) | orchestrator | B6 | tile numbers against the published formula and pyproj's Web Mercator; viewer interpolation within 0.05 px of exact; nothing fetched unless asked; vitest for the texture coordinates |
| B11 | Realistic terrain look (`web/src/blast/scene/terrain.tsx`, `muckpile.cut_m`, `imagery.fetch_best`): relief shading, fine relief and grain, rock on cut / broken ground, sky and haze, zoom fallback | orchestrator | B10 | shade = 1 on flat ground and the analytic value on a 30 degree slope; haze law; cut depth against a plane; a zoom answered with one repeated picture is skipped; seen in the browser on a real site |
| B12 | Site view (`web/src/blast/scene/realism.ts`, `Throw`, `Dust`, `Equipment`, `Scatter`, `topography.ts`): fragment clusters of tumbling blocks, soil layer, dust, two excavators outside the clearance radius with shadows, stones and shrubs, safe-zone camera, site / engineering presets | orchestrator | B11 | fragment sizes invert the Rosin-Rammler curve; dust path against a Runge-Kutta solution of its equation; excavators beyond clearance and behind the face; height lookup exact on a plane; seen playing in the browser |
| B14 | Site view, second pass (`scene/Effects.tsx`, `Equipment.tsx`): sun shadows of fragments, walls, scatter and machines on the terrain; excavators from extruded profiles; collar flash; camera tremble and rumble with physical arrival times; soft dust edge; camera kept above ground | orchestrator | B12 | tremble envelope by hand (arrival d / cp, decay, sum, distance); seen in the browser |
| B15 | Site view, third pass: dust shadows on the ground, idle motion of the machines, gravel / sand / grass detail near the camera, measured check of the rumble signal | orchestrator | B14 | idle pose bounds; rumble rendered off-line (peak, duration, low-frequency share) and logged when played; seen in the browser without shader errors |
| B16 | Site view, fourth pass: fine procedural relief of the ground near the blast and the machines (display only), everything standing on it | orchestrator | B15 | relief bounds and continuity; patch weight by hand (edge fade, hole); seen in the browser |
| B17 | Site view, fifth pass: clouds and sun in the sky, rough rock faces, larger dust cloud, firmer colours | orchestrator | B16 | seen in the browser without shader errors; existing tests |
| B18 | Site view, sixth pass: far terrain from a project surface named 'backdrop' out to the horizon, cloud shade on the ground | orchestrator | B17 | seen in the browser without shader errors; existing tests |
| B19 | Site view, seventh pass: drone-level ground from above (detail synthesised at every viewing height from the picture's colour: soil grain, gravel, sand, stones, separate shrubs with shadows, grass) | orchestrator | B18 | seen from above at three heights in the browser without shader errors; existing tests |
| B20 | Site view, eighth pass: loose rock, grit and dust tone on the floor of the cut | orchestrator | B19 | seen from above in the browser without shader errors; existing tests |
| B21 | Site view, ninth pass: camera shake removed (owner's request); ragged edges of ground cover, satellite veil off, plants in groups with large bushes, rills on slopes | orchestrator | B20 | seen from above in the browser without shader errors; tests without the tremble |
| B22 | Site view, tenth pass (owner's review): no round plants or stones; irregular soft plant cover in the picture's proportions, irregular rubble on the cut floor, 3-D shrubs and tufts removed | orchestrator | B21 | seen from above at two heights in the browser without shader errors; existing tests |

Parallel waves: W1 = B2, B3, B4, B6 (fixture-driven), B7 research. W2 = B5. W3 = B6 wiring to the API, B8.
Agents do not commit and do not run the full `make check`; they run their own tests and ruff on their files.
The orchestrator integrates, runs `make check` and commits.

## 5. Not in this build (stated limits)

- No hydrocode / discrete-element rock breakage (LS-DYNA, HSBM, PFC): fragment sizes come from Kuz-Ram, the
  motion from ballistics with a launch-velocity model, the waves from attenuation laws. The pictures are an
  engineering visualisation of those models, not a first-principles simulation.
- Site constants (K, beta for vibration; rock factor A; flyrock k; launch-velocity constant) are typical
  published values until calibrated with monitored blasts.
- No electronic-detonator scatter statistics, no wave-interference (signature-hole) vibration model, no
  presplit / wall-control design, no ore-movement (dig-line) model beyond the muckpile heightmap. Candidates
  for later tasks.
- No regulatory compliance statement: limits shown are the published guideline values named in the output.
