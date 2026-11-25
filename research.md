# How Surpac-style block modelling works

Research notes for the BakuMine block modeller. Scope: only the chain that turns drilling data into a 3D model of the deposit. Statistics, plotting, pit design and scheduling are out of scope.

**How to read this document.** The geostatistics and geometry here are standard, published theory and are stated with confidence. Statements about Surpac's *own* conventions (sign of dip, rotation order, file layouts) are marked **[verify]**: they come from memory of public training material and third-party importers, not from GEOVIA's reference manual, and each one has a matching test task in `plan.md`.

**Clean-room rule.** We reimplement the published methods and interoperate through Surpac's documented text exports. We do not decompile Surpac binaries, copy its code, or parse its proprietary binary `.mdl` format.

---

## 1. The idea in one page

A mine cannot see the orebody. It only has a few thousand thin cylinders of rock (drill core) sampled along lines through the ground. Block modelling answers: *given those lines, what is the grade and rock type of every cubic metre in between?*

The pipeline is always the same six steps:

| # | Step | Input | Output |
|---|------|-------|--------|
| 1 | Measure | Drill rig, surveyors, lab | Collar, survey, assay, geology tables |
| 2 | Desurvey | Collar + survey | 3D path of every hole |
| 3 | Composite | 3D paths + assays | Equal-length samples with XYZ and grade |
| 4 | Interpret | Composites + geology | Wireframe solids (ore domains) and surfaces (topography) |
| 5 | Build blocks | Wireframes + block definition | Grid of blocks, each coded with its domain |
| 6 | Estimate | Composites + coded blocks | Grade (and density) in every block |

After step 6, volume × density × grade gives tonnes and metal. That reporting step is trivial arithmetic, but it is the best check that steps 1–6 are right.

---

## 2. What is measured on the ground

### 2.1 Collar survey
The collar is the point where the hole enters the ground. A surveyor fixes it with RTK GPS or a total station in the mine grid: **Northing (Y), Easting (X), Elevation (Z)**. Accuracy is centimetres. Surpac stores coordinates in **Y, X, Z order** (northing first) **[verify]**, which is the reverse of most software and a classic import bug.

### 2.2 Downhole survey
Holes bend. A probe (magnetic single-shot, multi-shot, or gyroscope) is lowered and records, at stations every 10–50 m:

- **depth** along the hole (metres from collar),
- **azimuth** (compass bearing of the hole direction, 0–360° clockwise from north),
- **dip** (inclination from horizontal).

Surpac's convention: **dip is negative downward**, so a vertical hole is −90° **[verify]**. Magnetic azimuths must be corrected to grid north (declination + grid convergence) before use.

### 2.3 Logging and sampling
The core is laid in trays. A geologist logs intervals (`from`, `to`) of rock type, alteration and structure. The core is cut and sampled in intervals (typically 1–2 m), and a lab assays each sample for grade (g/t Au, % Cu, …). Density (specific gravity) is measured on selected pieces by water immersion.

### 2.4 Topography
The ground surface is surveyed (drone photogrammetry, LiDAR or ground survey) as a cloud of XYZ points or contour strings. It caps the model: there are no blocks in the air.

### 2.5 The resulting database
Four tables, linked by `hole_id`:

| Table | Fields | Kind |
|-------|--------|------|
| `collar` | hole_id, y, x, z, max_depth, hole_path | one row per hole |
| `survey` | hole_id, depth, dip, azimuth | point data |
| `assay` | hole_id, depth_from, depth_to, grade fields | interval data |
| `geology` | hole_id, depth_from, depth_to, rock code | interval data |

Surpac requires `collar` and `survey`; the rest are optional user tables. `hole_path` says whether the trace is straight or curved.

Validation that must run before anything else: duplicate hole IDs, missing collars, `from ≥ to`, overlapping intervals, intervals past `max_depth`, survey at depth 0 missing, dip/azimuth out of range, sudden deviation between stations (> ~10°/30 m is suspect).

---

## 3. Desurveying: turning survey readings into a 3D path

The survey gives direction at a few depths. Desurveying integrates those directions into XYZ coordinates along the hole.

Let inclination from vertical be `I` and azimuth `A`. With Surpac dip `δ` (negative down): `I = 90° + δ` (so δ = −90° gives I = 0, straight down).

### 3.1 Tangent method (crude)
Use the upper station's direction for the whole interval. Fast, and biased whenever the hole curves.

### 3.2 Balanced tangential
Average the two stations' direction vectors:

```
ΔN = ΔMD/2 · (sin I₁ cos A₁ + sin I₂ cos A₂)
ΔE = ΔMD/2 · (sin I₁ sin A₁ + sin I₂ sin A₂)
ΔZ = −ΔMD/2 · (cos I₁ + cos I₂)
```

### 3.3 Minimum curvature (the industry standard)
Assume the hole follows a circular arc between stations. Compute the dogleg angle β between the two direction vectors and apply a ratio factor to the balanced-tangential result:

```
cos β = cos(I₂ − I₁) − sin I₁ sin I₂ · (1 − cos(A₂ − A₁))
RF    = (2/β) · tan(β/2)          (RF = 1 when β < 1e-7)

ΔN = ΔMD/2 · (sin I₁ cos A₁ + sin I₂ cos A₂) · RF
ΔE = ΔMD/2 · (sin I₁ sin A₁ + sin I₂ sin A₂) · RF
ΔZ = −ΔMD/2 · (cos I₁ + cos I₂) · RF
```

To find the XYZ of an arbitrary depth (a sample midpoint), interpolate along the arc, not along the chord: slerp the direction vector between stations and integrate. Below the last station, extend straight on the last direction.

We implement minimum curvature as default and the tangent methods for comparison. Which method Surpac applies for "curved" holes is **[verify]**; the difference on normal holes is centimetres.

---

## 4. Compositing: making samples comparable

Assay intervals have unequal lengths. A 0.3 m sample and a 2 m sample must not carry equal weight in estimation ("support" must be equal). Compositing re-cuts every hole into intervals of one fixed length `L`.

Length-weighted mean for each composite:

```
g_comp = Σ(gᵢ · lᵢ) / Σ lᵢ
```

where `lᵢ` is the length of sample `i` that falls inside the composite. If density varies strongly, weight by `lᵢ · ρᵢ`.

Variants (all present in Surpac):

- **Downhole** — fixed length measured along the hole. The default for grade estimation.
- **By bench / elevation** — intervals bounded by horizontal levels; used for open-pit bench models.
- **Within a domain** — compositing restarts at each geology or wireframe boundary, so no composite mixes ore and waste. This is the one to use for estimation.
- **Best fit** — adjusts length slightly inside each domain intercept so there is no short remnant.

Rules: record the composite's real length; discard or merge composites shorter than a threshold (typically 50–75% of `L`); never fill unsampled gaps with zero unless the user says absent means barren. Each composite gets the XYZ of its midpoint from the desurveyed path.

Output: a point cloud `(x, y, z, grade…, domain, hole_id, length)`. This is the only data the estimator sees.

---

## 5. Geological interpretation: wireframes

Estimation must not smear ore grade into waste. So the geologist first draws where each rock unit is, and the model is built inside those shapes.

### 5.1 Strings
Surpac's primitive is the **string**: an ordered list of 3D points with a string number, open or closed. On each vertical cross-section through the drilling, the geologist digitises a closed string around the ore intercepts.

### 5.2 Surfaces (DTM)
A DTM is a triangulated surface that has one Z for each XY: topography, a weathering boundary, a fault. It is built by **2D Delaunay triangulation** of the points in plan view, with strings honoured as **breaklines** (constrained Delaunay), so ridges and crests are not cut across.

#### 5.2.1 Topography from an elevation raster
Sources, most precise first: a ground survey (RTK GNSS / total station, centimetres), drone photogrammetry or LiDAR (centimetres to decimetres, exported as points or a grid), the contour drawings of a topographic map (half the contour interval), commercial stereo-satellite surfaces (1-3 m), free global DEMs (Copernicus GLO-30, ALOS AW3D30, SRTM/NASADEM: 1 arc second, about 30 m; Copernicus states an absolute vertical error below 4 m at 90 % confidence), and the collar elevations alone (exact at the holes only). A global DEM is a surface model (it includes dumps and buildings) of the date of its images (Copernicus: 2011-2015), so later pits and dumps are missing.

A raster is a grid of heights `z[row, col]` at cell centres. GeoTIFF georeferencing (OGC GeoTIFF 1.1): `ModelPixelScale (sx, sy)` and a tie point `(i, j) -> (X, Y)` give `X = X_tie + (col - i) * sx`, `Y = Y_tie - (row - j) * sy`; with `RasterPixelIsArea` the tie point is the corner of the cell, so its centre is half a cell further, with `RasterPixelIsPoint` it is the centre. ESRI ASCII grids give the lower-left corner (`xllcorner`) or centre (`xllcenter`) and list the rows from north to south.

The model works in a projected CRS (UTM); a geographic DEM (longitude, latitude) is resampled onto a regular grid in that CRS: every node is transformed to longitude / latitude and the height is interpolated **bilinearly** between the four surrounding cell centres, which reproduces a plane exactly and keeps the surface continuous. The grid is then triangulated cell by cell.

Heights of a global DEM are above a geoid model (EGM2008 for Copernicus); a national survey uses its own height system, and the two can differ by metres. The surveyed collars tie them: `dz = collar z - DEM z` per collar; the **median** of `dz` is the vertical shift (robust to a few wrong collars), and collars far from it after the shift point to a wrong collar elevation, a dump or pit made after the DEM, or a DEM error.

A precise survey of the deposit area and a DEM around it are joined by triangulating them together: the survey points, strings and collars are kept as they are, the DEM grid nodes are shifted by the median of `survey z - DEM z` (the survey is the height reference) and every node closer than a distance `d` in plan to a survey point or breakline segment is left out (default `d` = the grid spacing), so no coarse DEM node sits next to a precise point and makes a spike. Where the two disagree by more than the DEM error after the shift (a pit or dump made after the DEM images), the survey wins within `d` of its data; the surface between the last survey point and the first kept node is the straight triangle between them.

Copernicus GLO-30 is published as 1° x 1° Cloud Optimised GeoTIFF tiles named after their south-west corner (`Copernicus_DSM_COG_10_N50_00_E015_00_DEM`) in a public bucket; there is no tile over open sea. The tiles for a model come from the longitude / latitude box of its plan rectangle, with the edges sampled, because a straight UTM edge is curved in longitude / latitude (a line of constant northing reaches its highest latitude on the central meridian).

### 5.3 Solids
A solid is a closed triangulated mesh. Classic Surpac workflow: triangulate between the closed strings on successive sections ("between segments"), then cap the two ends. The between-section step is a ring-stitching problem: walk both polygons and at each step add the triangle with the shorter diagonal. Bifurcations (one polygon on a section becoming two on the next) are handled by splitting the parent ring.

A modern alternative is **implicit modelling**: fit a radial-basis-function to signed distance from the contacts in the drillholes and extract the zero isosurface with marching cubes. Surpac itself is explicit (section-based); implicit is listed as a stretch goal because it makes solids from drilling without hand-digitising.

### 5.4 Solid validation
A solid is usable only if it is: closed (every edge shared by exactly two triangles), free of self-intersection, free of duplicate or degenerate triangles, and consistently oriented. Surpac refuses to use an unvalidated solid for volume or constraints.

### 5.5 Formulas on meshes

Volume of a closed mesh (divergence theorem, signed tetrahedra from the origin):

```
V = | (1/6) · Σ_triangles  v₁ · (v₂ × v₃) |
```

Point inside solid: cast a ray from the point (use +Z); count triangle crossings; odd = inside. For speed, index triangles in a 2D grid or BVH, and exploit columns: all block centroids in one XY column share one ray.

Above/below DTM: locate the triangle under the point in plan, interpolate Z by barycentric coordinates, compare.

### 5.6 File formats (text interchange) **[verify against real exports]**
- `.str` — text. Header line (location, date, purpose, memo); an axis record; then one point per line as `string_no, Y, X, Z, d1, d2…`; a line of zeros ends each segment; `0, 0, 0, 0, END` ends the file.
- `.dtm` — text. Holds only topology: triangles as index triples into the paired `.str` file, grouped into `OBJECT` / `TRISOLATION` blocks, with neighbour indices. A `.dtm` is meaningless without its `.str`.
- `.mdl` — the block model. Proprietary binary. **We do not read it.** Interchange is by Surpac's block-model CSV export (centroid Y, X, Z, block sizes, attributes).
- We also support DXF and OBJ/PLY for meshes, since every other package exports them.

### 5.7 Geology model: several units, weathering surfaces, faults

The ore solid (5.3) is one binary domain. A geology model describes the whole volume under the topography: every point gets one rock unit, one weathering zone and one fault block. It is built from the drillhole logs alone, with the same signed-distance interpolation as the implicit solid, plus 2-D surfaces for the weathering profile and planes or surfaces for the faults. File formats and function signatures are in `geology_contracts.md`. Items marked **[unverified]** are written from memory and have not been checked against the cited source; tests must not rely on them without an independent check.

#### 5.7.1 Units and the log dictionary

Raw logged codes are first *folded* (leading and trailing spaces removed, inner runs of spaces made one, upper case), then mapped through a per-deposit dictionary to **units**. Every unit has a **kind**, which decides what the modeller does with its intervals:

| kind | modelled as | its intervals, for the rock-unit fields |
|---|---|---|
| `lithology` | a rock unit (5.7.2, 5.7.3) | the unit |
| `intrusion` | a rock unit that overprints others in age order (5.7.3) | the unit in its own field; unknown for the host fields |
| `cover` | the base surface of the cover (5.7.4); overprints the rock units above it | unknown |
| `weathering` | the base surface of a weathering zone (5.7.4) | unknown |
| `fault-zone` | evidence of a fault position (5.7.5) | unknown |
| `other` | logged rock that is not modelled (too few metres) | unknown |
| `missing` | no information (core loss, not logged) | unknown |

"Unknown" means what it already means for the implicit solid: no constraint point is placed in the interval and **no contact is created at its ends** (5.3 and `implicit.extract_constraints`). A unit that is logged instead of the rock (an oxidation code written in place of the rock type) therefore hides the rock contact underneath rather than inventing one. For a weathering base the difference between `other` and `missing` matters: a logged rock of any kind below the base makes the base exact, a core-loss interval makes it a bound (5.7.4).

#### 5.7.2 Several rock units at once: largest signed-distance field

For each unit `i` of a group `G` (the units modelled together, inside one fault block) take, at points along every hole, the signed down-hole distance to the nearest contact of `i` in that hole, positive inside `i`, capped at `C`:

    d_i(p) = +min(dist to nearest i-contact, C)   p logged as i
           = -min(dist to nearest i-contact, C)   p logged as another unit of G
           (no point)                             p unknown (not in G, or kind other / missing / ...)

and interpolate it with the RBF of 5.3 (`fit_field`): `f_i(p)`. Every point takes the unit with the largest field:

    L(p) = argmax_i f_i(p)                     (ties: the unit listed first)

Properties, used as tests:

- **Partition.** Every point gets exactly one unit; there are no gaps and no overlaps, whatever the fields.
- **The logs are reproduced at the constraint points.** At a point logged as `i`, `d_i >= 0` and `d_j <= 0` for every other `j`, and an exact (global) RBF reproduces the data, so `L = i` there (contact points, where several values are 0, excepted).
- **Two units reduce to the single solid.** If only two units are logged, and both fields use the same points, then `d_1 = -d_2` at every point. The interpolant is linear in the data, so `f_1 = -f_2` and the boundary `f_1 = f_2` is the zero level `f_1 = 0` of the binary solid of unit 1.
- **Common scale.** All fields of a group use the same cap `C`, the same point spacing, the same kernel and the same anisotropy. With caps that differ, the unit with the larger cap wins the comparison between holes for no geological reason. The default `C` is the smallest of the units' median run lengths, counting only runs that touch a logged contact (the median over all runs is dominated by long barren holes in the host: a test sphere went from IoU 0.88 to 0.75). `extract_constraints` computes its own defaults per call, so the group passes `cap`, `sample_spacing` and `far_spacing` explicitly.

The boundary of unit `i` is where its **margin** changes sign:

    g_i(p) = f_i(p) - max_{j != i} f_j(p)        g_i > 0  <=>  L(p) = i

The display mesh of unit `i` is the zero level of `g_i` (marching tetrahedra, 5.3). On a grid edge where `i` and `j` are the two largest fields at both ends, `g_i = -g_j` at both nodes, so the two units' meshes get the same vertex on that edge. Where three units meet inside one cell the meshes may leave or overlap a sliver smaller than a cell; the block coding does not use the meshes and is not affected.

This "indicator by largest signed distance" approach treats units like categories rather than like an ordered sequence. It needs no age order, which suits intrusive bodies, dykes and mixed hosts. It does not stop a unit from thinning or thickening unrealistically between holes; for a clear layer stack use 5.7.3. Far from every hole all `d_i` are `-C`, so the fields extrapolate their drift and the unit there is weakly determined. Each block gets the distance to the nearest logged hole (`geo_distance`), and a `max_extrapolation` can leave such points without a unit. (The signed-distance idea for domain boundaries: Wilde & Deutsch 2012; implicit RBF modelling: Carr et al. 2001, Cowan et al. 2003. The largest-field rule over several units is this project's choice.)

**Fast evaluation.** Evaluating every RBF at millions of block centroids and sub-block corners is too slow (units x points x nodes). Instead the fields are sampled once on a regular world-aligned grid. Default spacing: the model's smallest sub-block edge, coarsened with a warning above a node limit. Between nodes they are interpolated **linearly on the six Kuhn tetrahedra of each cell**, the same subdivision the isosurface uses. For local cell coordinates `(f_x, f_y, f_z)` in `[0, 1)^3`, sorted so that `f_a >= f_b >= f_c`, the point lies in the tetrahedron `0, e_a, e_a + e_b, e_a + e_b + e_c` with weights

    w_0 = 1 - f_a,   w_1 = f_a - f_b,   w_2 = f_b - f_c,   w_3 = f_c
    v(p) = w_0 v[0] + w_1 v[e_a] + w_2 v[e_a + e_b] + w_3 v[e_a + e_b + e_c]

Labels are then `argmax_i` of the interpolated margins `g_i`, with `g` computed at the nodes. A point is then inside the mesh of unit `i` exactly when its interpolated `g_i > 0`, so blocks and meshes agree except in the triple-junction slivers. An `exact` switch evaluates the RBFs directly (for tests).

#### 5.7.3 Layer stacks and intrusions

**Sequence field.** For a stack whose order is known (top to bottom `U_0, ..., U_{m-1}`) one scalar field, the *stratigraphic position* `phi`, replaces the `m` signed-distance fields. Layers then cannot cross or leave gaps, and their thickness varies smoothly. Data values:

    phi = T_k = t_1 + ... + t_k     on the contact U_k / U_{k+1}   (T_0 = 0 on the contact U_0 / U_1)
    t_k = median true thickness of U_k = median over complete intercepts of  L |cos theta|

Here `L` is the down-hole length of the intercept and `theta` the angle between the hole and the pole of the layering. Inside a unit whose top and base are both logged, `phi` varies linearly with depth between the two contact values. Where only one contact is known (the hole starts or ends in the unit), `phi = T +- (down-hole distance x |cos theta|)`, capped at the unit's thickness (for the top and bottom units, at the median of the `t_k`). The pole is the user's dip and dip azimuth, or the direction of least spread of the contact points after each contact's own mean is subtracted (principal components, as in 5.3). The field is fitted with `fit_field` in the anisotropy of the layering, and

    L(p) = U_0 if phi < T_0;   U_k if T_{k-1} <= phi < T_k;   U_{m-1} if phi >= T_{m-2}

This is the "value constraint" form of a potential field, a simplification of the cokriging of potential increments and gradients of Lajaunie et al. (1997) and Calcagno et al. (2008), and of the generalised RBF interpolation of Hillier et al. (2014). Grose et al. (2021, LoopStructural) interpolate the same kind of value data with other interpolators, which is why LoopStructural serves as the second implementation in the tests. Holes whose units appear out of the stated order are reported, not silently used. Their points are still used, with the values their own contacts imply.

**Intrusions.** An intrusion unit is modelled as its own binary field (5.3, the `intrusion` unit against everything known) and **overprints** the result where its field is positive. Overprints are applied from the oldest to the youngest, so a later event wins. This is the same rule as the coding steps of 6.6. Inside the host fields the intrusion's intervals are unknown: what was there before the intrusion is not observed, so no host contact is placed at an intrusion's margin.

**Order of events for one fault block.** Host group (largest field or sequence) -> intrusions, oldest first -> cover above its base surface (5.7.4) -> AIR above the topography. The weathering zones and the fault blocks are separate layers (5.7.6), not rock units.

#### 5.7.4 Weathering and cover surfaces: depth below the topography with bounds

Weathering and cover follow the ground surface, so the **depth below the topography** is interpolated, not the elevation. With the topography `z_T(x, y)` (5.2.1) and a pick at `(x, y, z)` on a hole:

    delta = z_T(x, y) - z          (vertical depth below the surface at the pick, not at the collar)
    surface z(x, y) = z_T(x, y) - max(delta*(x, y), 0)

**Picks per hole** for the base of a set of units `S` (oxidation codes, soil codes, or a numeric rule such as `sulphur < 1000 ppm` applied to assay intervals). Hole intervals are classed IN (in `S`), OUT (logged, another unit, including kind `other`) or UNKNOWN (NaN, `missing`, or an unlogged gap). In `deepest` mode the base is taken at the bottom `b` of the deepest IN run; in `first_run` mode, at the bottom of the first IN run from the top of the logging.

| situation below `b` | pick |
|---|---|
| an OUT interval abuts `b` | exact: `delta(b)` |
| the hole ends at `b` | lower bound: `delta >= delta(b)` |
| UNKNOWN from `b` to `b + u`, then OUT | between: `delta(b) <= delta <= delta(b + u)` |
| UNKNOWN to the end of the hole | lower bound: `delta >= delta(b)` |
| no IN at all, logging starts OUT at `a_0`, policy `absent = zero` | upper bound: `delta <= delta(a_0)` |
| no IN at all, policy `absent = unknown` (default) | no information |

`absent = unknown` is the default for weathering because a weathering zone is often simply not logged; reading the absence as "no oxidation" would pull the surface up to the topography.

**Kriging in dual form.** With equality data `z_e` at `x_e`, ordinary kriging with a global neighbourhood can be written as an interpolant:

    f(x) = sum_j b_j C(x - x_j) + c            [ C    1 ] [ b ]   [ z_e ]
                                                [ 1^T  0 ] [ c ] = [ 0   ]

`C` is the covariance of the variogram model (7.4), evaluated on plan lags `(dx, dy, 0)`. A linear drift replaces the constant `c` by `c_0 + c_1 x + c_2 y` and the row of ones by the monomials (universal kriging). The thin-plate RBF is the same form, with `C(r) = r^2 log r` and a linear drift. The dual form and the usual kriging weights give the same estimate (Royer & Vieira 1984) **[unverified: equivalence recalled, not re-derived here]**. The default variogram is spherical, nugget 0 (the picks are honoured exactly), sill = variance of the exact depths, range = 3 x the median distance from a pick to its nearest neighbour.

**Inequality data.** The interpolant above minimises the norm `b^T C b` among all functions of this form that honour the data **[unverified as stated for the conditionally positive definite case; the tests check the result against a general quadratic-programme solver]**. With lower bounds `l` at `x_l` and upper bounds `u` at `x_u` the problem becomes a convex quadratic programme (Dubrule & Kostov 1986):

    minimise   (1/2) b^T K b             over b (one coefficient per data point and bound point) and c
    subject to f(x_e) = z_e,   f(x_l) >= l,   f(x_u) <= u,   P^T b = 0

Here `K` is the covariance between all data and bound points and `P` holds the drift monomials. From the Lagrangian, the coefficient `b_j` of a bound point is its multiplier: `b_j >= 0` at an active lower bound, `b_j <= 0` at an active upper bound, and `b_j = 0` at an inactive bound (derived here, **[unverified against the paper]**). This gives a **primal active-set method** (Nocedal & Wright 2006, ch. 16):

1. Start from the equality-only surface, with its values `v` at the bound points clipped into the bounds; the bounds it violated form the working set `A`.
2. Solve with `A` as equalities, giving values `v*` at the bound points. If `v*` violates a bound outside `A`, move from `v` towards `v*` only as far as the first bound met (`v <- v + alpha (v* - v)`, `0 <= alpha < 1`), add that bound to `A` and solve again.
3. If nothing is violated, take `v = v*` and check the signs of `b_j` for `j` in `A`. If one has the wrong sign, remove the worst one from `A` and solve again. Otherwise stop.

This is the primal active-set method (Nocedal & Wright, algorithm 16.3): every iterate honours all bounds and the objective never increases, so with a strictly convex programme (`K` positive definite on `P^T b = 0`) it ends in a finite number of steps, barring degenerate ties. (The simpler scheme "add the most violated bound, re-solve" has infeasible iterates and no such guarantee, although it gave the same surfaces in trials.) A cap on the number of iterations is still kept; reaching it is an error, not a silent result. A "between" pick contributes one lower and one upper bound at the same location.

**Nested surfaces never cross.** The surfaces of one profile are listed top to bottom (cover base, oxide base, transition base). Each is interpolated on its own, then the depths are made non-decreasing at every grid node:

    delta_1' = max(delta_1, 0),   delta_k' = max(delta_k, delta_{k-1}')

This is a correction after the fact, not a joint estimate. Where two surfaces would cross, the deeper one is pushed down to touch the shallower one, and the zone between them vanishes there. The number of nodes changed is reported.

#### 5.7.5 Faults

**Fault blocks from codes.** A fault block is known on an interval when a column carries it, for example as a prefix of the ore-lens code (a regular expression with one group, e.g. `^(FB\d)-`). Usually only the mineralised intervals carry it. Where a hole passes from block `a` to block `b`, the fault lies somewhere in the gap between the last `a` interval and the first `b` interval, so the crossing is **interval-censored**. Pick: the middle of a `fault-zone` interval inside the gap if there is one, otherwise the middle of the gap. Uncertainty `s` = half the length of that interval or gap. Weight `w = 1 / max(s, s_min)^2`.

**Fault plane through the picks** (at least three picks, not on one line). Weighted principal components:

    c = sum w_i p_i / sum w_i,      M = sum w_i (p_i - c)(p_i - c)^T
    n = eigenvector of M with the smallest eigenvalue, turned so that n_z >= 0
        (for a vertical plane, so that n points to azimuth [0, 180))
    dip = arccos(n_z),   dip azimuth = atan2(n_x, n_y) mod 360

With a prior orientation (the user's dip and dip azimuth) only the position is fitted: `n = n_0`, `c . n_0 = sum w_i (p_i . n_0) / sum w_i`.

**Separating plane** (fewer picks, or picks on one line). The interval midpoints of block `a` should lie on one side and those of block `b` on the other. This is a soft-margin linear support vector machine (Cortes & Vapnik 1995) in coordinates centred on the points and divided by their extent:

    minimise  (1/2) |w|^2 + lambda sum_i xi_i    subject to  y_i (w . p_i + beta) >= 1 - xi_i,  xi_i >= 0
              y_i = +1 for block b, -1 for block a;  plane: w . p + beta = 0

With a prior orientation `n_0` the plane is fixed by one number: the offset in the middle of the gap between `max(n_0 . p_a)` and `min(n_0 . p_b)` when the points separate. Otherwise it is the offset with the fewest points on the wrong side, and the middle of that range. Every fault reports its misfit: the points on the wrong side, by hole.

**Imported faults** replace inferred ones of the same name or block pair: a mesh (DXF / OBJ / STL), a set of points (plane fit as above), or a plane through a point given by dip and dip azimuth. The side of a point relative to a mesh is computed in the mesh's own frame. Rotate so that `z'` is the mean normal of the mesh, then use the height above the surface (`DTM.height_above`, 5.2). Outside the mesh's footprint in that frame the best-fit plane of the mesh is used.

**Sign convention.** The normal points up (`n_z >= 0`). A point with `n . (p - c) > 0` is on the **positive side**, which for a non-vertical fault is the hanging wall.

**Fault blocks in space.** `k` faults split space into cells labelled by the sign vector `(s_1, ..., s_k)`. Each cell that contains labelled points takes the majority block of those points, and conflicts are reported. A cell without labelled points takes the block of the nearest labelled point. A fault may be declared to **terminate** on another fault (`fault F2 exists only on the + side of F1`); where it does not exist its sign is ignored, so the cells on either side merge.

**Rock units per fault block.** Every interval is assigned to the block of its midpoint (from the geometry above, not only the coded intervals). The rock-unit fields of 5.7.2 and 5.7.3 are fitted for each block from that block's intervals only, and `L(p) = L_{block(p)}(p)`. This is what lets a contact jump at a fault. A block with fewer than `min_holes` holes uses the fields fitted on all blocks together, with a warning in the provenance record.

**Throw diagnostic.** For a reference contact (two units) and a fault:

1. Find the modelled contact points on each side, within a band (default 20 to 60 m) from the fault. These are the label changes along vertical lines on a regular plan grid.
2. Fit a plane `z = a + b x + c y` to each side.
3. Throw = median over fault-plane points inside the drilled area of `z_footwall - z_hangingwall` (vertical separation of the contact at the fault).

The vertical separation equals the true throw (vertical component of the slip) only when the beds are not tilted along the slip direction. In general it also depends on the dip of the beds and the direction of slip (Fossen 2016; **[unverified: chapter not checked]**). The synthetic deposit is built so that the two are equal (`geology_contracts.md`).

#### 5.7.6 Coding the blocks

A geology model gives three independent categorical layers, each coded into its own block attribute:

| attribute | categories | order of the steps (later wins) |
|---|---|---|
| `fault_block` | block names | one step per block, from the fault cells |
| `lith` | rock units, cover unit, `AIR` | host units -> intrusions (oldest first) -> cover above its base -> `AIR` above the topography |
| `weathering` | zones top to bottom, `AIR` | deepest zone everywhere -> each shallower zone above its base surface -> `AIR` |
| `density` (float) | t/m3 | default density -> per rock unit -> per weathering zone where given (not in cover blocks) -> NaN in `AIR` |

The existing ore attribute (`rock = ORE / WASTE / AIR`, from the ore solids) is not touched. Ore does not overwrite the rock unit; a display attribute can combine them (`geo`: the rock unit, then `ORE` where `rock == "ORE"`).

Every layer is coded through a new **geology constraint**: `mask` = the label of the block centroid is in a given set. `classify_boxes` labels the 8 corners and the centre of each box: all in the set gives INSIDE, none gives OUTSIDE, otherwise STRADDLE. Unlike the solid and DTM constraints this test is **not conservative**: a body thinner than a box can pass between its sample points, so a dyke thinner than a parent block can be missed by the sub-blocking. Its measured effect on the synthetic dyke is one of the acceptance numbers. The weathering and cover surfaces are also written as DTMs, so the exact `DtmConstraint` can be used for them instead. Sub-blocking on "the label differs between the sample points of the box" refines along all boundaries of a layer in one pass. It splits the same boxes as one step per category with sub-blocking, so both give identical models.

#### 5.7.7 What the geology model says about itself

Every build writes a provenance record with three lists:

- **measured**: the site data used and what for (log intervals per unit, picks per surface, fault picks, density samples);
- **assumed**: what the software chose on its own (default variograms and caps, anisotropy, inferred fault planes, stand-in densities, extrapolation beyond the drilling, reading an unlogged zone as unknown);
- **missing**: inputs still needed and where they would come from (fault surfaces, age order of intrusions, density by rock, logs of a zone in holes that lack them).

The model is validated by hole-out cross-validation: refit without a fold of holes and compare the predicted unit with the logged unit along the left-out holes, in metres. On the synthetic layered deposit it is also validated against the exact truth.

---

## 6. The block model itself

### 6.1 Definition
A block model is a regular 3D grid plus a table of attributes. It is fully defined by:

- **origin** — the minimum corner (Y, X, Z),
- **extent** — size of the model along each axis,
- **user block size** — parent block dimensions (dy, dx, dz), usually tied to drill spacing (¼–½ of it in plan) and bench height (dz),
- **rotation** — bearing, dip, plunge, for orebodies not aligned with the grid; fixed at creation,
- **sub-blocking** — the smallest block allowed.

Centroid of parent block `(i, j, k)`:

```
local = ((i + ½)·dx, (j + ½)·dy, (k + ½)·dz)
world = origin + R · local
```

`R` is the model rotation matrix; identity for an unrotated model.

### 6.2 Sub-blocking
Parent blocks are too coarse at a contact: a 10 m block straddling the ore boundary is neither ore nor waste. Surpac subdivides such blocks as an **octree**: each split halves the block in every direction, so sub-block sizes are `user size / 2ⁿ`. Blocks away from any boundary stay whole. Blocks inside homogeneous regions may also be merged into super-blocks for storage. The result follows the wireframe closely while storing far fewer blocks than a uniformly fine grid.

Consequences for our data structure: blocks are not a dense 3D array. Store a flat table of `(centroid, size, attributes)` with an octree or Morton-code index. Dense arrays remain the fast path for models without sub-blocks.

### 6.3 Attributes
Named columns on the block table: grade (float), density (float), rock code (integer or string), plus calculated attributes defined by an expression of others. Every attribute has a **background value** used where nothing was assigned (e.g. grade −99 or 0, density of waste).

### 6.4 Constraints
A constraint is a logical test that selects blocks. This is the core mechanism of Surpac block modelling; every operation (fill, estimate, report, display) takes a constraint. Primitive types:

| Type | Test (on the block centroid) |
|------|------|
| DTM | above / below a surface |
| 3DM (solid) | inside / outside a closed mesh |
| Block attribute | `grade > 0.5`, `rock = "OX"` |
| Plane | above / below / in front of a plane |
| Closed string | inside / outside a polygon projected along an axis |
| Block index / coordinate range | Z between two levels |

Primitives are combined with AND / OR / NOT into an expression, e.g. `(inside ore_solid) AND (below topo) AND NOT (inside dyke)`. Constraints can be saved and reused.

When a geometric constraint is applied with sub-blocking enabled, blocks cut by the boundary are split until the minimum size, and the test is then made on each sub-block's centroid.

### 6.5 Partial percentages
Alternative or supplement to sub-blocking: store in each block the fraction of its volume inside a solid. Computed by testing an `n × n × n` lattice of points in the block, or exactly by clipping the mesh against the block. Used for precise volume reporting.

### 6.6 Typical build sequence
1. Create the empty model from extents that cover the wireframes plus margin.
2. Add attributes with background values.
3. Code blocks: `rock = "AIR"` above topography; `rock = "ORE1"` inside solid 1, etc., with sub-blocking.
4. Assign density per rock code.
5. Estimate grade into each domain using only that domain's composites.
6. Validate; report.

---

## 7. Estimation: filling blocks with grade

For each block (in a domain), find nearby composites (of the same domain) and compute a weighted average:

```
z*(block) = Σ λᵢ · z(xᵢ)
```

Methods differ only in how the weights `λᵢ` are chosen.

### 7.1 Assign value
Constant per constraint. Used for rock codes and default density.

### 7.2 Nearest neighbour (NN)
`λ = 1` for the closest sample, 0 otherwise. Equivalent to Voronoi polyhedra. No smoothing; unusable for local grade but its global mean is the declustered mean of the data, which makes it the standard check on other estimates.

### 7.3 Inverse distance weighting (IDW)
```
λᵢ = (1/dᵢᵖ) / Σⱼ (1/dⱼᵖ)
```
`p` = 2 is usual; higher `p` approaches NN, lower approaches a plain average. `d` is the **anisotropic** distance (§7.5). A sample at `d = 0` takes all the weight. Simple and robust, but ignores clustering (ten samples side by side count ten times) and uses an arbitrary `p`.

### 7.4 Kriging

Kriging chooses the weights that minimise expected squared error, given a model of how grade similarity decays with distance: the variogram.

**Variogram.** Half the mean squared difference between samples separated by vector `h`:

```
γ(h) = 1/(2N(h)) · Σ [z(xᵢ) − z(xᵢ + h)]²
```

It is fitted with a model, a nugget `c₀` plus one or more nested structures with sill `c` and range `a`:

```
Spherical     γ(h) = c₀ + c·[1.5(h/a) − 0.5(h/a)³]   for h ≤ a;  c₀ + c  beyond
Exponential   γ(h) = c₀ + c·[1 − exp(−3h/a)]
Gaussian      γ(h) = c₀ + c·[1 − exp(−3h²/a²)]
```

Nugget = variation at zero distance (sampling error + micro-scale variability). Range = distance beyond which samples are uncorrelated. The user said no statistics module, so variogram *modelling tools* are out of scope; but the estimator must accept variogram parameters as input, and we provide a basic experimental-variogram calculator so those parameters can be obtained without Surpac.

**Ordinary kriging (OK).** Unknown local mean, weights forced to sum to 1. For `n` samples, solve the `(n+1)` linear system:

```
| γ₁₁ … γ₁ₙ 1 |   | λ₁ |   | γ̄₁V |
|  ⋮      ⋮  ⋮ | · |  ⋮ | = |  ⋮  |
| γₙ₁ … γₙₙ 1 |   | λₙ |   | γ̄ₙV |
|  1  …  1  0 |   | μ  |   |  1  |
```

`γᵢⱼ` = variogram between samples i and j; `γ̄ᵢV` = average variogram between sample i and the block; `μ` = Lagrange multiplier.

Kriging variance (a measure of data configuration quality, not of true error):

```
σ²_K = Σ λᵢ · γ̄ᵢV + μ − γ̄VV
```

**Block kriging.** A block is a volume, not a point. `γ̄ᵢV` is computed by averaging the variogram from sample i to a lattice of discretisation points inside the block (e.g. 4 × 4 × 2). `γ̄VV` is the average between all pairs of those points. This is what makes the estimate valid for the block's volume.

**Simple kriging (SK).** Known global mean `m`, no constraint row: `z* = m + Σ λᵢ (zᵢ − m)`. Solved with covariances `C(h) = sill − γ(h)`.

Why kriging beats IDW: it declusters automatically (redundant samples share weight), it honours anisotropy and nugget from the data, and it returns a variance. Practical hazards: negative weights (screen effect), ill-conditioned matrices from duplicate sample locations (remove duplicates, add a tiny diagonal), and over-smoothing.

Numerically, solve in covariance form for a positive-definite sample block and use an LU or Cholesky-based solver; batch blocks with the same sample count.

### 7.5 Anisotropy and the search ellipsoid

Grade continuity is rarely equal in all directions: long along strike, shorter down dip, shortest across the vein. This is expressed as an ellipsoid:

- three **angles** orienting it: **bearing** (azimuth of the major axis), **plunge** (inclination of the major axis), **dip** (rotation of the semi-major axis about the major axis),
- **major radius** (max search distance),
- two **ratios**: major/semi-major and major/minor.

Anisotropic distance: rotate the sample–block vector into the ellipsoid's axes `(u, v, w)`, then stretch the short axes:

```
[u v w]ᵀ = R⁻¹ · (x_sample − x_block)
d_aniso  = √( u² + (v · r_semi)² + (w · r_minor)² )
```

with `r_semi = major/semi-major`, `r_minor = major/minor`. A sample is inside the search if `d_aniso ≤ major radius`. The same transform is applied to the variogram ranges.

Implementation consequence: pre-transform all composites into this stretched space once, build a KD-tree there, and every search becomes an ordinary spherical nearest-neighbour query.

The rotation order and handedness differ between packages. Surpac's own convention is documented as **"Surpac ZXY LRL"**: rotate about Z (bearing), then X (plunge), then Y (dip), with left-, right-, left-hand rules respectively; GSLIB, Vulcan, Datamine and Isatis each differ **[verify]**. Getting this wrong silently produces a plausible but wrong model, so it gets a dedicated test, and the convention is a selectable parameter.

**Dynamic anisotropy** (newer Surpac): bearing/dip/plunge stored per block as attributes, so the ellipsoid follows a folded orebody. Stretch goal.

### 7.6 Search parameters
Each controls which samples enter a block's estimate:

- minimum and maximum number of samples (block left unestimated below the minimum),
- maximum samples per drillhole (stops one hole dominating),
- octant / quadrant search (max samples per sector, forces samples from all sides),
- maximum vertical search distance,
- multiple passes: a tight first pass, then wider passes that fill only still-empty blocks; store the pass number as an attribute.

Also stored per block as diagnostics: number of samples, average distance, distance to nearest sample, kriging variance. These later drive resource classification.

### 7.7 Grade capping
Extreme assays (a single 300 g/t sample) are cut to a top value before estimation, or their influence radius is restricted. A simple pre-estimation parameter.

---

## 8. Reporting (minimal, used as validation)

```
Volume = Σ block volumes inside the constraint  (× partial percentage if used)
Tonnes = Σ Vᵢ · ρᵢ
Grade  = Σ (Vᵢ · ρᵢ · gᵢ) / Σ (Vᵢ · ρᵢ)
Metal  = Tonnes · Grade
```

Grouped by rock code, bench, and grade cut-off.

---

## 9. How we know the model is right

1. **Volume check** — volume of sub-blocked blocks inside a solid vs the solid's analytic mesh volume. Target: within 1% at sensible sub-block size.
2. **Global mean check** — mean estimated grade vs NN (declustered) mean. Should agree within a few percent.
3. **Swath check** — mean grade of blocks vs composites in slices along X, Y, Z; the block curve should follow the composite curve, smoother.
4. **Visual check** — sections showing block colours against drillhole grades. This is the 3D viewer's main job.
5. **Reference check** — same inputs through an independent implementation (GSLIB `kt3d`, PyKrige, gstools) must give the same numbers to rounding; and, if a Surpac licence is available, the same inputs through Surpac with the block CSV compared block by block.
6. **Synthetic truth** — generate a known orebody and grade field, drill it virtually, run the pipeline, and compare the model with the truth. This is the only test where the right answer is known.

---

## 10. What "as good as Surpac" means for this project

In scope (the modelling chain): drillhole database import and validation; desurvey; compositing; DTM and solid creation, validation and import; block model creation with rotation and octree sub-blocking; attributes and constraints; NN / IDW / OK / SK estimation with anisotropic multi-pass search; volume–tonnes–grade report; interactive 3D viewer with drillholes, wireframes, coloured blocks, slicing and constraint filtering; Surpac text-format interchange.

Out of scope: variogram modelling UI, statistics and plotting modules, pit design and optimisation, scheduling, survey tools, macros (TCL/SCL), the proprietary `.mdl` binary, conditional simulation, indicator and multivariate kriging.

Open risk: without a Surpac licence and real exported files, the **[verify]** items can only be checked against third-party documentation, and "matches Surpac" can only be shown indirectly, through agreement with GSLIB-class references.

---

## 11. References

Theory (standard texts):
- Isaaks & Srivastava, *An Introduction to Applied Geostatistics* (1989) — variograms, kriging, search strategy.
- Deutsch & Journel, *GSLIB: Geostatistical Software Library and User's Guide* (1998) — reference algorithms (`kt3d`, `gamv`) and rotation conventions.
- Journel & Huijbregts, *Mining Geostatistics* (1978) — block kriging, support.
- Rossi & Deutsch, *Mineral Resource Estimation* (2014) — the full practical workflow, compositing, validation.
- Sinclair & Blackwell, *Applied Mineral Inventory Estimation* (2002).
- Sawaryn & Thorogood, "A Compendium of Directional Calculations Based on the Minimum Curvature Method", SPE 84246 (2005) — desurvey formulas.

Geology model (5.7). Bibliographic details written from memory; titles, volumes and pages are **[unverified]** until checked:

- Carr, Beatson, Cherrie, Mitchell, Fright, McCallum, Evans, "Reconstruction and representation of 3D objects with radial basis functions", SIGGRAPH (2001) — RBF implicit surfaces.
- Cowan, Beatson, Ross, Fright, McLennan, Evans, Carr, Lane, Bright, Gillman, Oshust, Titley, "Practical implicit geological modelling", 5th International Mining Geology Conference, AusIMM (2003).
- Lajaunie, Courrioux, Manuel, "Foliation fields and 3D cartography in geology: principles of a method based on potential interpolation", Mathematical Geology 29(4) (1997) — potential field, interface and gradient data.
- Calcagno, Chilès, Courrioux, Guillen, "Geological modelling from field data and geological knowledge, Part I", Physics of the Earth and Planetary Interiors 171 (2008) — potential field with geological rules.
- Hillier, Schetselaar, de Kemp, Perron, "Three-dimensional modelling of geological surfaces using generalized interpolation with radial basis functions", Mathematical Geosciences 46 (2014).
- Grose, Ailleres, Laurent, Jessell, "LoopStructural 1.0: time-aware geological modelling", Geoscientific Model Development 14 (2021) — used as a test oracle only (MIT).
- Wilde & Deutsch, "Kriging and simulation in presence of stationary domains: developments in boundary modelling", Geostatistics Oslo 2012 (2012) — signed distance for domain boundaries.
- Royer & Vieira, "Dual formalism of kriging", in *Geostatistics for Natural Resources Characterization*, NATO ASI (1984).
- Dubrule & Kostov, "An interpolation method taking into account inequality constraints: I. Methodology", Mathematical Geology 18(1) (1986); Kostov & Dubrule, part II — kriging with inequality data as a quadratic programme.
- Journel, "Constrained interpolation and qualitative information — the soft kriging approach", Mathematical Geology 18(3) (1986).
- Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer (2006), ch. 16 — active-set methods for quadratic programmes.
- Cortes & Vapnik, "Support-vector networks", Machine Learning 20 (1995) — soft-margin separating plane.
- Fossen, *Structural Geology*, 2nd ed., Cambridge (2016) — throw, heave, separation.

Surpac-specific and format sources consulted:
- [GEOVIA: Top 5 Tips in Surpac (dynamic anisotropy)](https://www.3ds.com/fileadmin/PRODUCTS-SERVICES/GEOVIA/PDF/3DS-GEOVIA-SurpacTop5Tips.pdf)
- [GEOVIA community: Variable sub-celling in Surpac](https://3dswym.3dexperience.3ds.com/post/geovia-user-community/surpac-tip-improve-your-estimations-with-variable-sub-celling_v6b_6VKOSxmgWQH5IHl0QA)
- [GEOVIA community: SDM, a format for the future](https://3dswym.3dexperience.3ds.com/post/geovia-user-community/sdm-a-format-for-the-future_a8NsLaURRgyFiv26JDkhbA)
- [Kirra docs: Surpac DTM/STR import](https://brentbuffham.github.io/kirra-docs/importing/surpac-dtm-str.html)
- [Maptek Vulcan: Import Surpac data](https://help.maptek.com/vulcan/2026/topics/File/Import/File_Import_Surpac_DesignStringsV2.htm)
- [Seequent Leapfrog: Importing block models (Surpac rotation)](https://help.seequent.com/Geo/2025.3/en-GB/Content/block-models/import-block-models.htm)
- [Seequent Leapfrog: Octree sub-blocked models](https://help.seequent.com/Geo/2024.1/en-GB/Content/block-models/octree-models.htm)
- [Deswik white paper: Block model knowledge for mining engineers](https://www.deswik.com/whitepapers/block-model-knowledge-for-mining-engineers-an-introduction)
- [Maptek GeologyCore: Ellipsoid orientation conventions](https://help.maptek.com/geologycore/2023.2/topics/menus-and-tools/modelling/ellipsoid.htm)
- Surpac training manuals (Geological Database, Block Modelling, Solids, Surfaces) — public copies on Scribd; the authoritative format reference is the Surpac Online Help, "Appendix D – File Formats".
