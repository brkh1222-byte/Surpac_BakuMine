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
