# Blast design and timing: theory (B2)

Formulas implemented in `bakumine/blast/design.py` and `bakumine/blast/timing.py`, with their sources.
To be merged into `research_blast.md`. SI units: metres, kilograms, kg/m3; delays in milliseconds.

Symbols: `d` hole (charge) diameter, `H` bench height, `B` burden, `S` spacing, `T` stemming, `J` subdrill,
`L` hole length, `rho_e`, `rho_r` explosive and rock density, `SGe = rho_e / 1000`, `SGr = rho_r / 1000`.

## 1. Burden

### 1.1 Konya (`burden_konya`)

Konya and Walter (1990) give, in imperial units,

    B [ft] = (2 SGe / SGr + 1.5) De [in]

with `De` the diameter of the explosive. Conversion: 1 ft = 0.3048 m, 1 in = 25.4 mm, so
`B [m] = 0.3048 / 25.4 * (...) * De [mm] = 0.012 (2 SGe / SGr + 1.5) De [mm]` (0.3048 / 25.4 = 0.012 exactly):

    B = 0.012 (2 SGe / SGr + 1.5) De        B in m, De in mm

Check (hand arithmetic, not a number quoted from the book): ANFO 0.85, rock 2.70, De = 4 in = 101.6 mm:
`(0.6296 + 1.5) * 4 = 8.519 ft = 2.596 m`.

### 1.2 Ash (`burden_ash`)

Ash (1963): `B = Kb d` (same unit), `Kb` between 20 and 40; `Kb = 30` for average conditions (rock of density
about 2.7 with an explosive of density about 1.2 to 1.3), about 25 for a light explosive such as ANFO and 35
for a dense one in the same rock. Values outside 20 to 40 are refused. Ash's other ratios, as summarised by
Hustrulid (1999): spacing `Ks = S / B` 1 to 2, subdrill `Kj = J / B` about 0.3 (0.2 to 0.5), stemming
`Kt = T / B` about 0.7 (0.5 to 1.0), hole length `Kh = H / B` 1.5 to 4.

### 1.3 From a target powder factor (`burden_from_powder_factor`)

Derived here (no source needed beyond the definitions). With `S = m B`, `T = kt B`, `J = kj B` and the linear
charge density `w = pi/4 d^2 rho_e`, one vertical hole carries `Q = w (H + J - T)` and breaks `B S H`:

    q = w (H + (kj - kt) B) / (m B^2 H)
    q m H B^2 - w (kj - kt) B - w H = 0
    B = [ w (kj - kt) + sqrt( w^2 (kj - kt)^2 + 4 q m H^2 w ) ] / (2 q m H)

(the positive root). Tested against `scipy.optimize.brentq` on the first equation.

## 2. Spacing, stemming, subdrill, stiffness (`suggest`)

- **Spacing.** Staggered: `S = 1.15 B`. This is the equilateral-triangle pattern, `S / B = 2 / sqrt(3) = 1.1547`,
  which covers the plan area most evenly; it is the staggered ratio recommended by Jimeno, Jimeno and Carcedo
  (1995) and by Hustrulid (1999). Square: `S = B` (default). The range offered for square / rectangular patterns
  is 1.0 to 1.25; `S = 1.25 B` is the ratio of Langefors and Kihlstrom (1963). `spacing_ratio` overrides.
  Not implemented: Konya's spacing rules that depend on `H / B` and on instantaneous / delayed rows
  (`S = (H + 2B) / 3`, `2B`, `(H + 7B) / 8`, `1.4B`).
- **Stemming.** `T = 0.7 B` (Konya and Walter 1990; Ash's `Kt`). Published range 0.7 to 1.0 B; 0.7 is the
  default, `stemming_ratio` overrides.
- **Subdrill.** `J = 0.3 B` (Konya and Walter 1990; Ash's `Kj`).
- **Stiffness ratio** `H / B` (Konya and Walter 1990): 1 = poor (severe airblast, flyrock, vibration, coarse
  fragmentation; redesign), 2 = fair, 3 = good, 4 = excellent, above 4 no further benefit. `suggest` and
  `stiffness_warning` return a text when `H / B < 2` or `> 4`.
- **Charge** (vertical hole): `L = H + J`, charge length `L - T` (0 with a warning when not positive),
  `Q = pi/4 d^2 rho_e (L - T)` (fully coupled bulk explosive), powder factor `q = Q / (B S H)` in kg/m3 and
  `Q / (B S H rho_r / 1000)` in kg/t.

## 3. Layout on a bench polygon (`layout`)

Conventions (ours; geometry only):

- `f = (sin az, cos az)` is the plan unit vector of the face azimuth (direction the face looks, clockwise
  from north); `u = (cos az, -sin az)` is `f` turned 90 degrees clockwise = left to right for an observer
  behind the blast looking out over the face. Polygon vertices are taken to `(s, t) = (p . u, p . f)`.
- The face edge is `t_max`. Row `r` (0 = front) lies at `t = t_max - (r + 1) B`; there are
  `floor((t_max - t_min) / B)` rows, so a row exactly on the back edge is kept (the back row is the back limit
  of the blast and a rectangle `n B` deep has `n` rows, each owning a strip `B` wide).
- Holes in a row at `s = s_min + S / 2 + k S`; in a staggered pattern odd rows are shifted by `S / 2`. Holes
  inside the polygon or on its boundary (1e-6 m) are kept: even-odd ray casting plus a point-to-edge distance
  test. A rectangle `m S` wide has `m` holes in even rows and, staggered, `m + 1` in odd rows (edge holes).
- Names: row letter (A, B, ... Z, AA, ...) + position from the left starting at 1. `Hole.index` is that
  position minus 1.
- Collar elevation: the flat `top`, or `top.z_at(x, y)`. No surface (NaN) or collar at / below the floor:
  the hole is dropped and listed in `notes["dropped_holes"]`.
- Hole: `L = (z_collar - z_floor) / sin(dip) + J`, toe `= collar + L (cos(dip) f, -sin(dip))`. Vertical:
  toe at `z_floor - J`. Inclined holes (dip below horizontal < 90) are parallel to an inclined face: the
  toe is displaced toward the face (Jimeno et al. 1995 describe inclined drilling parallel to the face).
  Burden and spacing are plan distances at the collar (the burden perpendicular to an inclined hole is
  `B sin(dip)`); stemming and subdrill are lengths along the hole.
- Charge from `T` to `L`. A hole with `L <= T` is kept uncharged (`charge_top = charge_bottom = L`) and
  listed in `notes["uncharged_holes"]`.
- `bench_height` of the design: `top - floor` (flat) or mean `z_collar - z_floor` (surface).

## 4. Totals (`summary`)

`volume = polygon area (shoelace) x bench height` (from `BlastDesign.volume`); `tonnes = volume rho_r / 1000`;
powder factor `= explosive mass / volume` (kg/m3) and `/ tonnes` (kg/t); specific drilling
`= drilled metres / volume`; stiffness `= H / B`. With a sloping top the volume uses the mean bench height
at the collars: an approximation, not a surface-to-floor integral.

## 5. Delay patterns

All patterns are `t = n Th + r Tr`, `Th` inter-hole delay, `Tr` inter-row delay (> 0), `r` the row from the
face and `n` the number of spacings along the row from the initiation hole (a front-row hole at `s0`). These
are the standard surface-delay layouts (row-by-row, echelon, V) described by Konya and Walter (1990),
Jimeno et al. (1995) and Hustrulid (1999); the rounding rules for staggered rows are ours:

- row-by-row: `Th = 0` (whole row at once), or `Th > 0` = echelon from the left;
- echelon from a front corner: `n = ceil(d / S)` for a hole `d >= 0` inside of the corner hole,
  `n = floor(|d| / S)` outside it;
- V-cut from the centre of the front row (the front-row hole nearest the row's middle, the left one of a
  tie): `n = floor(|s - s0| / S)`;
- hole-by-hole: echelon from the left, refused if two holes get the same time.

Property (proved by cases, tested): in a square pattern a hole fires `Tr` after the hole straight ahead; in a
staggered pattern, of the two holes ahead (at `+- S / 2`), the one nearer the initiation hole fires earlier
for any `Th >= 0`, `Tr > 0`, and both do when `Tr > Th`.

## 6. Maximum instantaneous charge (`max_instantaneous_charge`)

The charge weight per delay `W` in the scaled-distance law `PPV = K (R / sqrt(W))^-beta` is the largest
explosive mass detonated within any 8 ms: Duvall, Johnson, Meyer and Devine (1963) found that charges
separated by delays of 8 ms or more act as separate charges for vibration amplitude, and the rule is in
US regulation (30 CFR 816.67: "maximum weight of explosives to be detonated within any 8-millisecond
period"; also used in Siskind et al. 1980). Implemented as the maximum over half-open windows
`[t_i, t_i + 8 ms)` starting at each firing time (sorted times, cumulative sums, binary search), so two
charges exactly 8 ms apart are separate delays. Tested against an O(n^2) scan.

## 7. Burden relief (`burden_relief`)

`relief = (first firing time of row r+1 - first firing time of row r) / B` in ms per metre of burden; the
median over the row pairs is reported. Guide (Konya and Walter 1990, row-to-row time constant `TR`, SI form
as given in Konya and Walter 1991): 6.5 ms/m violent, excessive airblast, backbreak; 8 ms/m pile high and
close to the face, moderate airblast and backbreak; 11.5 ms/m average pile height, airblast and backbreak;
16.5 ms/m scattered pile with minimum backbreak. Flags: below 6.5, above 16.5, or within 6.5 to 16.5.
**These four values are quoted from memory of the table and were not checked against the printed source.**

## References

- Ash, R.L. (1963). The mechanics of rock breakage (parts 1 to 4). *Pit and Quarry*, 56.
- Duvall, W.I., Johnson, C.F., Meyer, A.V.C. and Devine, J.F. (1963). *Vibrations from instantaneous and
  millisecond-delayed quarry blasts*. US Bureau of Mines Report of Investigations 6151.
- Hustrulid, W. (1999). *Blasting Principles for Open Pit Mining*, vol. 1. Balkema, Rotterdam.
- Jimeno, C.L., Jimeno, E.L. and Carcedo, F.J.A. (1995). *Drilling and Blasting of Rocks*. Balkema, Rotterdam.
- Konya, C.J. and Walter, E.J. (1990). *Surface Blast Design*. Prentice Hall, Englewood Cliffs.
- Konya, C.J. and Walter, E.J. (1991). *Rock Blasting and Overbreak Control*. US Federal Highway
  Administration, FHWA-HI-92-001.
- Langefors, U. and Kihlstrom, B. (1963). *The Modern Technique of Rock Blasting*. Wiley, New York.
- Siskind, D.E., Stagg, M.S., Kopp, J.W. and Dowding, C.H. (1980). *Structure response and damage produced by
  ground vibration from surface mine blasting*. US Bureau of Mines Report of Investigations 8507.
- US Code of Federal Regulations, 30 CFR 816.67, Use of explosives: control of adverse effects.

Not verified against the printed sources (written from memory of the literature): the Ash `Kb` values 25 / 35
for light / dense explosives, the attribution of `S = 1.25 B` to Langefors and Kihlstrom, the attribution of
`S = 1.15 B` to Jimeno et al. and Hustrulid (the value itself is geometry), and the burden-relief table.
