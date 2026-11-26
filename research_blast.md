# Blast BakuMine: theory and formulas

Every formula the blast module implements, with its published source. Items marked unverified were written from memory and not checked against the printed source; site constants are uncalibrated typical values.

Parts: 1 design and timing, 2 fragmentation, vibration, airblast and flyrock, 3 damage, waves and throw.

---

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


---

# Blast effects: fragmentation, vibration, air overpressure, flyrock

Theory for `bakumine/blast/fragmentation.py`, `vibration.py` and `flyrock.py` (task B3). To be merged into
`research_blast.md`. Units: metres, kilograms, seconds, pascals unless a formula states otherwise.
All models here are empirical. Their constants are site constants: the defaults are published typical
values and are reported as **uncalibrated**.

Confidence marks: **[A]** formula or value as I know it from the primary source; **[B]** widely quoted,
taken from secondary literature, not re-checked against the primary source in this build (no access to
the documents). Section 5 lists what was not verified.

## 1. Fragmentation: the Kuz-Ram model

### 1.1 Mean fragment size (Kuznetsov) [A]

Kuznetsov (1973) related the mean fragment size to the specific charge and the charge per hole for TNT.
Cunningham (1983, 1987) rewrote it for any explosive with the relative weight strength (ANFO = 100,
TNT = 115):

```
x50 [cm] = A * K^(-0.8) * Q^(1/6) * (115 / RWS)^(19/20)
```

- `A` rock factor (1.2)
- `K` powder factor, kg of explosive per m3 of rock
- `Q` explosive mass per hole, kg
- `RWS` relative weight strength of the explosive, ANFO = 100

`kuznetsov_x50` returns metres (`x50_cm / 100`).

### 1.2 Rock factor [B]

Cunningham (1987) derived `A` from the blastability index of Lilly (1986):

```
A   = 0.06 * (RMD + RDI + HF)
RMD = 10 (powdery / friable) | JF (vertically jointed) | 50 (massive)
JF  = JPS + JPA
JPS = 10 (joint spacing < 0.1 m) | 20 (0.1 m to oversize) | 50 (oversize to drill pattern size)
JPA = 20 (dip out of the face) | 30 (strike perpendicular to the face) | 40 (dip into the face)
RDI = 25 * rho - 50            rho in t/m3
HF  = E / 3   if E < 50 GPa,   else UCS / 5   (E in GPa, UCS in MPa)
```

`A` ranges from about 0.8 to 22. In Cunningham's form the joint factor **replaces** RMD for a jointed
rock mass. Some secondary sources write `A = 0.06 (RMD + JF + RDI + HF)`; with Cunningham's class
values that would count the joints twice, so the code uses the form above. Cunningham (2005) adds a
joint-condition factor on JPS and a correction factor `C(A)` for site calibration; these are not
implemented.

`rock_factor_from_rock` is an estimate from the `Rock` fields (joint spacing gives JPS, JPA = 30 as the
middle class because the model has no joint orientation). `Rock.rock_factor` wins when set.

### 1.3 Uniformity index (Cunningham 1987) [A]

```
n = (2.2 - 14 B/d) * sqrt((1 + S/B) / 2) * (1 - W/B) * (|BCL - CCL| / L + 0.1)^0.1 * (L / H)
```

- `B` burden, m; `S` spacing, m; `d` hole diameter, **mm**
- `W` standard deviation of drilling accuracy, m
- `BCL`, `CCL` bottom and column charge length, m; `L = BCL + CCL` total charge length, m
- `H` bench height, m
- staggered pattern: `n * 1.1`

The code takes `d` in metres and converts. For a hole with one explosive the code uses `BCL = 0`,
`CCL = L`, so the charge term is `1.1^0.1 = 1.0096` (see section 5). `n <= 0` (burden far too large for
the diameter, or `W >= B`) is outside the model and returned as NaN.

### 1.4 Size distribution (Rosin and Rammler 1933) [A]

```
P(x) = 1 - exp(-0.693 * (x / x50)^n)          fraction passing size x
xc   = x50 / 0.693^(1/n)                      characteristic size, P(xc) = 1 - 1/e = 0.632
x(p) = x50 * (-ln(1 - p) / 0.693)^(1/n)       size with fraction p passing (P80: p = 0.8)
```

0.693 is `ln 2` rounded; the code uses `ln 2` so that `P(x50) = 0.5` exactly. The distribution is the
Weibull distribution with shape `n` and scale `xc` (used as the independent check with
`scipy.stats.weibull_min`).

Known limit: Kuz-Ram underestimates the fines (Cunningham 2005; Ouchterlony 2005).

### 1.5 Swebrec function (Ouchterlony 2005) [A]

```
P(x) = 1 / (1 + [ln(xmax / x) / ln(xmax / x50)]^b)        0 < x <= xmax
```

`xmax` largest fragment, `b` curve undulation. `P(x50) = 0.5`, `P(xmax) = 1`. Given as an alternative
curve; the prediction of `xmax` and `b` from the design (the KCO model) is not implemented.

## 2. Ground vibration

### 2.1 Scaled-distance law [A]

```
PPV [mm/s] = K * (R / sqrt(W))^(-beta)
```

`R` distance in m, `W` explosive mass per delay in kg. Square-root scaling follows from the cylindrical
charge geometry of blast holes (Duvall and Fogelson 1962; Nicholls, Johnson and Duvall 1971; Siskind et
al. 1980; Dowding 1985). `K` and `beta` are fitted to monitored blasts of the site.

Presets (all uncalibrated):

| Preset | K | beta | Source |
|---|---|---|---|
| `AS2187_AVERAGE` (default) | 1140 | 1.6 | AS 2187.2-2006 Appendix J, "average conditions", free face, 50 % exceedance [A] |
| `ISEE_TYPICAL` | 1143 | 1.6 | ISEE Blasters' Handbook: `PPV [in/s] = 160 (D [ft] / sqrt(W [lb]))^-1.6` [B] |
| `ISEE_UPPER_BOUND` | 1729 | 1.6 | ISEE Blasters' Handbook (after Oriard): `242 (D / sqrt(W))^-1.6` [B] |

Unit conversion: `1 m/kg^0.5 = 3.28084 / sqrt(2.20462) = 2.2096 ft/lb^0.5`, so
`K_metric = K_imperial * 25.4 * 2.2096^(-beta)`; for 160 this gives 1143, which agrees with the 1140 of
AS 2187.2.

The law is a far-field law: it grows without bound as `R -> 0`. The near field is handled by the
Holmberg-Persson model (`damage.py`).

### 2.2 Charge per delay

Charges that fire less than 8 ms apart are counted as one delay (the 8 ms convention goes back to
Duvall, Johnson, Meyer and Devine 1963, USBM RI 6151, and is used in Bulletin 656 and the OSM rules)
[B]. `charge_per_delay` forms one group per window `[t, t + window)` starting at each firing time; the
largest group is the maximum instantaneous charge (MIC). Holes exactly 8 ms apart are separate delays.

`ppv_at` has two modes:

- `"mic"`: `W` = MIC, `R` = distance to the nearest charge centre. The classical conservative number.
- `"per_hole"`: `max over groups g of K (R_g / sqrt(W_g))^-beta`, `R_g` = distance to the charge
  centroid of the group. This is an engineering envelope, not a published model; it does not model
  wave superposition.

### 2.3 Site calibration [A]

Least squares in log-log space (Dowding 1985):

```
log10 PPV = log10 K - beta * log10(R / sqrt(W))
s   = sqrt(SSE / (n - 2))            standard error of the residuals, log10 units
K95 = K * 10^(1.645 * s)             95 % upper-bound line, same slope
```

`K95` assumes normal residuals in log space and ignores the uncertainty of the fitted line itself (no
prediction-interval widening).

### 2.4 Guideline limits (PPV, mm/s)

| Key | Value | Source |
|---|---|---|
| `usbm_ri8507_plaster_low_freq` | 12.7 (0.50 in/s) | Siskind et al. 1980, RI 8507: plaster, below 40 Hz [A] |
| `usbm_ri8507_drywall_low_freq` | 19.05 (0.75 in/s) | RI 8507: drywall, below 40 Hz [A] |
| `usbm_ri8507_above_40hz` | 50.8 (2.0 in/s) | RI 8507: above 40 Hz [A] |
| `osm_0_300ft` / `osm_301_5000ft` / `osm_beyond_5000ft` | 31.75 / 25.4 / 19.05 | OSM 30 CFR 816.67(d) distance table: 1.25 / 1.00 / 0.75 in/s [A] |
| `din4150_industrial` / `_residential` / `_sensitive` | 20 / 5 / 3 | DIN 4150-3:1999 Table 1, foundation, 1-10 Hz (rising to 40-50 / 15-20 / 8-10 at 50-100 Hz) [A] |
| `as2187_sensitive_95pct` / `_max` | 5 / 10 | AS 2187.2-2006 Appendix J, human comfort, sensitive sites [B] |
| `as2187_commercial` | 25 | AS 2187.2-2006 Appendix J, occupied non-sensitive sites [B] |

The limits are frequency dependent (RI 8507 Appendix B curve, DIN lines); the table holds the value of
the band named in the key. The code does not predict the vibration frequency.

## 3. Air overpressure

```
P [kPa] = K_a * (R / W^(1/3))^(-a)
dBL     = 20 * log10(P / 20e-6 Pa)
```

Cube-root scaling (spherical spreading in air). AS 2187.2-2006 Appendix J gives `a = 1.45` and
`K_a = 10 to 100` for confined blasthole charges [B]. The default `K_a = 100` is the upper end of that
range (conservative). `W` is the MIC; `R` is measured from the nearest collar. Weather (inversions,
wind) and topographic shielding are not modelled; they can change the level by 10 dB or more.

Check values: 2 Pa = 100 dBL; 20 Pa = 120 dBL; a doubling of distance lowers the level by
`20 * 1.45 * log10 2 = 8.7 dB`.

Guideline limits (dBL, peak linear):

| Key | Value | Source |
|---|---|---|
| `as2187_sensitive_95pct` / `_max` | 115 / 120 | AS 2187.2-2006 Appendix J (and ANZEC 1990: 115 dBL, 5 % of blasts up to 120) [B] |
| `as2187_commercial` | 125 | AS 2187.2-2006 Appendix J [B] |
| `as2187_damage` | 133 | AS 2187.2-2006 Appendix J [B] |
| `usbm_ri8485_0_1hz` / `_2hz` / `_6hz` | 134 / 133 / 129 | Siskind, Stachura, Stagg and Kopp 1980, RI 8485; OSM 30 CFR 816.67(b), by the high-pass frequency of the measuring system [A] |

## 4. Flyrock

### 4.1 Richards and Moore (2004) [A]

Launch velocity of a fragment from a charge with linear density `m` (kg/m) behind a confinement `c` (m):

```
v0 = k * (sqrt(m) / c)^1.3
```

Maximum horizontal range on level ground, `g = 9.81 m/s2`:

```
face burst          L = (k^2 / g) * (sqrt(m) / B)^2.6                    c = burden B
cratering           L = (k^2 / g) * (sqrt(m) / SH)^2.6                   c = stemming height SH
stemming ejection   L = (k^2 / g) * (sqrt(m) / SH)^2.6 * sin(2 theta)    theta = hole angle
```

Face burst and cratering are the ballistic maximum `v0^2 / g` (launch at 45 degrees). Stemming
ejection travels along the hole axis; `theta` is the inclination of the hole from the horizontal, so a
vertical hole gives zero horizontal range. `k` = 13.5 (soft rock) to 27 (hard rock); the default is 27.
The paper's safety factors on `L`: 2 for equipment, 4 for people.

Convention in `predict`: face burst for the holes of row 0 with their burden; cratering and stemming
ejection for every charged hole with `Hole.stemming` (measured along the hole); `m = Hole.linear_density`.
The design burden is used, not the surveyed face burden: an under-burdened face hole (the main cause of
face-burst flyrock) needs face profiling data that the model does not have.

### 4.2 Ballistics [A]

Without drag, from height `h0` above the landing level, launch angle `a` above the horizontal:

```
R = v0 cos(a) / g * (v0 sin(a) + sqrt(v0^2 sin(a)^2 + 2 g h0))       (= v0^2 sin(2a) / g for h0 = 0)
t = (vz + sqrt(vz^2 + 2 g (z0 - z_ground))) / g
```

With quadratic drag on a sphere of diameter `d` and density `rho` (standard aerodynamics, e.g. Hoerner
1965):

```
dv/dt = -g ez - c |v| v,     c = 3 rho_air Cd / (4 rho d),     Cd = 0.47,  rho_air = 1.225 kg/m3
```

integrated with the classical fourth-order Runge-Kutta method. `Cd = 0.47` is the smooth-sphere value
at Reynolds numbers of about 1e3 to 2e5; angular rock fragments have higher and variable coefficients,
so drag results are indicative. Ground contact: the flight is stepped, the first step in which the
height above the ground changes sign is bracketed, and the contact is found by bisection. Checked
against `scipy.integrate.solve_ivp` with a terminal event.

## 5. Not verified, and choices made

- No formula was checked against a published worked example (Cunningham's or Richards and Moore's
  examples were not available). Tests check the code against hand arithmetic of the formulas above and
  against independent implementations (scipy), i.e. they verify the implementation, not the
  transcription of the formulas from the papers.
- Rock factor: the form `0.06 (RMD + RDI + HF)` with `RMD = JF` for jointed rock, and the class values,
  are from memory of Cunningham (1987, 2005) [B].
- Uniformity index: `BCL = 0, CCL = L` for a single-explosive hole is a choice. Some implementations
  set the charge term to 1 or use `BCL = CCL`; the difference is 1.0096 against 0.794 at the extreme.
- `ISEE_TYPICAL` / `ISEE_UPPER_BOUND` (160 and 242 in imperial units): widely quoted, edition and page
  not checked [B].
- AS 2187.2-2006 Appendix J values (K = 1140, B = 1.6; K_a 10-100, a = 1.45; comfort limits 5 / 10 /
  25 mm/s and 115 / 120 / 125 / 133 dBL): from secondary sources, table numbers not checked [B].
- The 8 ms criterion source (RI 6151) [B].
- The `"per_hole"` PPV envelope and measuring the airblast distance from the nearest collar are
  engineering conventions of this code, not published models.

## 6. References

- Australian and New Zealand Environment Council (ANZEC) 1990. Technical basis for guidelines to
  minimise annoyance due to blasting overpressure and ground vibration.
- Cunningham, C.V.B. 1983. The Kuz-Ram model for prediction of fragmentation from blasting. Proc. 1st
  Int. Symp. on Rock Fragmentation by Blasting, Lulea, Sweden, 439-453.
- Cunningham, C.V.B. 1987. Fragmentation estimations and the Kuz-Ram model: four years on. Proc. 2nd
  Int. Symp. on Rock Fragmentation by Blasting, Keystone, Colorado, 475-487.
- Cunningham, C.V.B. 2005. The Kuz-Ram fragmentation model: 20 years on. Proc. 3rd EFEE World
  Conference on Explosives and Blasting, Brighton, 201-210.
- DIN 4150-3:1999. Erschuetterungen im Bauwesen, Teil 3: Einwirkungen auf bauliche Anlagen (Structural
  vibration, Part 3: Effects of vibration on structures). Deutsches Institut fuer Normung.
- Dowding, C.H. 1985. Blast Vibration Monitoring and Control. Prentice-Hall.
- Duvall, W.I. and Fogelson, D.E. 1962. Review of criteria for estimating damage to residences from
  blasting vibrations. US Bureau of Mines RI 5968.
- Duvall, W.I., Johnson, C.F., Meyer, A.V.C. and Devine, J.F. 1963. Vibrations from instantaneous and
  millisecond-delayed quarry blasts. US Bureau of Mines RI 6151.
- Hoerner, S.F. 1965. Fluid-Dynamic Drag. Published by the author.
- International Society of Explosives Engineers (ISEE). Blasters' Handbook (17th ed. 1998, 18th ed.
  2011).
- Kuznetsov, V.M. 1973. The mean diameter of the fragments formed by blasting rock. Soviet Mining
  Science 9(2), 144-148.
- Lilly, P.A. 1986. An empirical method of assessing rock mass blastability. Proc. Large Open Pit
  Mining Conference, AusIMM, Newman, 89-92.
- Nicholls, H.R., Johnson, C.F. and Duvall, W.I. 1971. Blasting vibrations and their effects on
  structures. US Bureau of Mines Bulletin 656.
- Office of Surface Mining Reclamation and Enforcement 1983. 30 CFR 816.67, Use of explosives: control
  of adverse effects.
- Ouchterlony, F. 2005. The Swebrec function: linking fragmentation by blasting and crushing. Mining
  Technology (Trans. Inst. Min. Metall. A) 114, A29-A44.
- Richards, A.B. and Moore, A.J. 2004. Flyrock control: by chance or design. Proc. 30th Annual
  Conference on Explosives and Blasting Technique, ISEE, New Orleans.
- Rosin, P. and Rammler, E. 1933. The laws governing the fineness of powdered coal. Journal of the
  Institute of Fuel 7, 29-36.
- Siskind, D.E., Stachura, V.J., Stagg, M.S. and Kopp, J.W. 1980. Structure response and damage produced
  by airblast from surface mining. US Bureau of Mines RI 8485.
- Siskind, D.E., Stagg, M.S., Kopp, J.W. and Dowding, C.H. 1980. Structure response and damage produced
  by ground vibration from surface mine blasting. US Bureau of Mines RI 8507.
- Standards Australia 2006. AS 2187.2-2006, Explosives: Storage and use, Part 2: Use of explosives.
  Appendix J: Ground vibration and airblast overpressure.


---

# Blast physics: waves, near-field damage, throw and muckpile (task B4)

Theory behind `bakumine/blast/waves.py`, `damage.py` and `throw.py`. To be merged into `research_blast.md`.
Units: metres, kilograms, seconds, pascals; PPV in mm/s and delays in ms at the boundary.

Every formula below is the one the code implements. Where a constant or a citation detail was written from
memory of the literature and not checked against the primary source during this task, it is marked
**[unverified]**. Nothing here was checked against a monitored blast.

## 0. What this is not

These are engineering models: closed-form or one-dimensional relations fitted to field observations, of the
kind used in blast design practice. They are **not** a hydrocode and not a discrete-element simulation. There
is no equation of state of the detonation products, no stress tensor, no crack growth, no gas flow in
fractures, no wave reflection at the free face, no wave interference between holes, no collisions between
fragments, no air drag. The wavefronts in the viewer are spheres expanding at one constant speed, the damage
classes are thresholds on one scalar (PPV), and the flying rock is a cloud of independent points with an
assumed launch speed. The site constants (Holmberg-Persson `K, alpha, beta`; damage multiples; all launch
parameters) are published typical values or plain assumptions and are labelled `calibrated=False`. The
pictures are a visualisation of these models; they must not be read as a first-principles prediction.

## 1. Waves (`waves.py`)

### 1.1 Elastic wave speeds

Isotropic linear elastic solid with Young's modulus `E`, Poisson's ratio `nu`, density `rho`:

    cp = sqrt( E (1 - nu) / (rho (1 + nu) (1 - 2 nu)) )        P wave (longitudinal)
    cs = sqrt( E / (2 rho (1 + nu)) )                          S wave (shear)
    cs / cp = sqrt( (1 - 2 nu) / (2 (1 - nu)) )

Equivalent to `cp = sqrt((lambda + 2 mu) / rho)`, `cs = sqrt(mu / rho)` with the Lame constants. Standard
elastodynamics (e.g. Kolsky, H., 1963, *Stress Waves in Solids*, Dover; Jaeger, J.C., Cook, N.G.W. and
Zimmerman, R.W., 2007, *Fundamentals of Rock Mechanics*, 4th ed., Blackwell). A measured `Rock.p_wave`
replaces `cp`; `cs` then comes from the ratio. Note: `E` from a static test is lower than the dynamic
modulus, so a computed `cp` tends to be lower than a measured one; a measured value should be preferred.

Check: `E = 50 GPa, nu = 0.25, rho = 2700` gives `cp = 4714.05 m/s`, `cs = 2721.66 m/s` (by hand in the tests).

Arrival time of the P wave from a source fired at `t0`: `t = t0 + |x - x_s| / cp` (straight ray, one speed).

### 1.2 Detonation and borehole pressure

    P_d = rho_e * VOD^2 / 4          detonation (Chapman-Jouguet) pressure
    P_b = P_d / 2                    borehole (explosion) pressure, fully coupled charge

`P_d = rho_e VOD^2 / (gamma + 1)` with the adiabatic exponent of the products `gamma = 3` gives the factor 4;
the borehole pressure after expansion to the hole volume is commonly taken as half of it. Sources: Cook, M.A.
(1958), *The Science of High Explosives*, Reinhold; Persson, P.-A., Holmberg, R. and Lee, J. (1994), *Rock
Blasting and Explosives Engineering*, CRC Press. These are approximations for an ideal detonation; commercial
explosives in small diameters detonate non-ideally and reach less. Decoupled charges are not modelled.

Check: ANFO 850 kg/m3, 4000 m/s: `P_d = 3.4 GPa`, `P_b = 1.7 GPa`.

### 1.3 Crushed zone radius

Two published simple models, selectable; they differ by a factor of about four for the default rock, which
reflects the real disagreement in the literature.

* `model="esen"` (default). Esen, S., Onederra, I. and Bilgin, H.A. (2003), "Modelling the size of the
  crushed zone around a blasthole", *Int. J. Rock Mech. Min. Sci.* 40(4), 485-495:

      CZI = P_b^3 / (K * sigma_c^2),   K = E_d / (1 + nu_d),   r_c = 0.812 * r_0 * CZI^0.219

  `r_0` hole radius, `sigma_c` uniaxial compressive strength, `K` rock stiffness from the dynamic modulus and
  Poisson ratio. The code uses `Rock.youngs_gpa` and `Rock.poisson` as given (not necessarily dynamic), and
  `P_b` from 1.2 (the paper's borehole pressure `rho_e VOD^2 / 8` for `gamma = 3` is the same number).
  **[unverified]** the constants 0.812 and 0.219 are as remembered from the paper.
* `model="pressure_ratio"`: `r_c = r_0 * sqrt(P_d / sigma_c)` with the detonation pressure, as used for the
  fines model of Kanchibotla, S.S., Valery, W. and Morrell, S. (1999), "Modelling fines in blast fragmentation
  and its impact on crushing and grinding", *Proc. Explo '99*, AusIMM, Kalgoorlie. **[unverified]** that the
  paper uses the detonation (not the borehole) pressure; page numbers not checked.

The result is never smaller than the hole radius.

### 1.4 Wave sources for the viewer

One source per charged hole: the centre of the charge column, the firing time `delay_ms`, the charge mass. The
viewer draws `radius(t) = cp (t - t_ms)`. Untimed designs are refused.

## 2. Near-field damage (`damage.py`)

### 2.1 Holmberg-Persson PPV of an extended charge

Far from a charge of mass `W` the peak particle velocity follows `PPV = K W^alpha / R^beta`. Near a blast hole
the charge is a column. Holmberg, R. and Persson, P.-A. (1978), "The Swedish approach to contour blasting",
*Proc. 4th Conf. on Explosives and Blasting Technique*, Society of Explosives Engineers, New Orleans, and
(1979) "Design of tunnel perimeter blasthole patterns to prevent rock damage", *Proc. Tunnelling '79*,
Institution of Mining and Metallurgy, London, split the column into elements `q dx` (`q` = linear charge
density, kg/m) and add them:

    PPV = K * ( q * I )^alpha
    I   = integral over the charged length of  dx / ( r0^2 + (x - x0)^2 )^( beta / (2 alpha) )

`r0` = perpendicular distance of the point from the hole axis, `x0` = its position along the axis. The code
takes both from the hole's collar and direction, so inclined holes need nothing special. Constants for Swedish
hard bedrock: `K = 700, alpha = 0.7, beta = 1.5` with PPV in mm/s, `W` in kg, `R` in m (the commonly quoted
values; uncalibrated for any other site).

Evaluation. With `x - x0 = r0 tan(theta)` and `p = beta / (2 alpha)`:

    I = r0^(1 - 2p) * integral_{theta1}^{theta2} cos(theta)^(2p - 2) d theta,   theta_i = atan((x_i - x0) / r0)

integrated by 32-node Gauss-Legendre quadrature (smooth integrand for `p >= 1`). For `beta = 2 alpha`
(`p = 1`) this is the closed form of the papers, `I = (atan((x2 - x0)/r0) - atan((x1 - x0)/r0)) / r0`.
`r0` is clamped to the hole radius. Far field: for `R >> L`, `I -> L / R^(2p)`, so
`PPV -> K (q L)^alpha / R^beta`, the point-charge law.

Checks in the tests: `scipy.integrate.quad` of the integral in the original variable (3 sets of constants,
vertical and inclined hole), the arctan closed form, the point-charge limit at 2000 m (1e-4), monotone decay.

Known weaknesses, not corrected: (i) the elements are added as if their peaks arrived at the same time and in
the same direction, which overestimates PPV near a long column; (ii) the sum of element contributions is not
a physically consistent superposition of velocities. See Blair, D. and Minchinton, A. (1996), "On the damage
zone surrounding a single blasthole", *Proc. Fragblast-5*, Montreal, Balkema; Hustrulid, W. and Lu, W. (2002),
"Some general design concepts regarding the control of blast-induced damage during rock slope excavation",
*Proc. Fragblast-7*, Beijing **[unverified]** (venue and exact title of the second from memory).

Several holes. `combine="max"` (default): the field is the maximum of the single-hole PPV; holes are never
summed. `combine="window"`: the holes whose firing times lie within `window_ms` (default 8 ms) after a hole
are treated as one charge, `PPV = K (sum of q I)^alpha`, and the largest group counts. This extension of the
element sum across holes is this module's convention, by analogy with "charge per delay"; it is not from the
Holmberg-Persson papers.

### 2.2 Critical PPV and damage classes

A plane P wave with particle velocity `v` carries the strain `epsilon = v / cp` and the stress
`sigma = E v / cp`. Tensile failure when `sigma = sigma_t`:

    PPV_c = sigma_t * cp / E

(Persson, Holmberg and Lee 1994; also in Jimeno, C.L., Jimeno, E.L. and Carcedo, F.J.A., 1995, *Drilling and
Blasting of Rocks*, Balkema.) Default rock: `10 MPa * 4714 m/s / 50 GPa = 0.943 m/s = 943 mm/s`, inside the
700-1000 mm/s usually quoted from Holmberg and Persson for the onset of damage in Swedish bedrock
**[unverified]**.

Classes (0 intact, 1 damaged, 2 fractured, 3 crushed; a class starts strictly above its threshold):

| class | threshold | default rock |
|---|---|---|
| damaged | `0.4 PPV_c` | 377 mm/s |
| fractured | `PPV_c` | 943 mm/s |
| crushed | `4 PPV_c` | 3771 mm/s |

The multiples 0.4 and 4 are the ratios of the steps of the rock-independent table of Bauer, A. and Calder,
P.N. (1978), *Open Pit and Blast Seminar*, Mining Engineering Department, Queen's University, Kingston,
Ontario: below 250 mm/s no fracturing of intact rock; 250-635 mm/s minor tensile slabbing; 635-2540 mm/s
strong tensile and some radial cracking; above 2540 mm/s complete break-up of the rock mass (10, 25, 100
in/s). Identifying `PPV_c` with their 635 mm/s step is **a choice of this module**, not a published rule; the
names of the classes ("crushed" for their "complete break-up") are ours too. The table itself is available as
`BAUER_CALDER`. Rock-specific tables exist (Mojtabai, N. and Beattie, S.G., 1996, "Empirical approach to
prediction of damage in bench blasting", *Trans. Inst. Min. Metall. Sect. A* 105) but their numbers are not
used, because they could not be quoted with confidence. Thresholds carry `calibrated=False`.

### 2.3 Back-break

From the mid-charge point of every last-row hole the PPV (all holes, maximum) is followed horizontally away
from the free face; the back-break is the distance where it falls to the "fractured" threshold (bracket by
doubling, then bisection to 1 mm). 0 when even the hole wall is below the threshold. This is only the reach
of the PPV criterion: joints, gas penetration and the confinement of the last row, which control real
back-break, are absent, and with the uncalibrated Swedish constants the value is large (about 7 m for a
150 mm ANFO hole in the default rock). Treat it as an index until `K, alpha, beta` are fitted on site.

### 2.4 Block attribute

For each block centre: `ppv_mm_s` (2.1), `cls` (2.2), `blasted` = centre inside the design polygon in plan
(even-odd ray casting, implemented locally) and between `floor_z` and the bench top (collar elevation of the
nearest hole in plan), and `t_break_ms` = firing time of the nearest charge column + distance to that column
/ `cp`, for blocks that are blasted or at least fractured; NaN otherwise and NaN for untimed holes.
"Breaks when the P wave arrives" is a display convention: real breakage takes several wave transits and the
gas phase.

## 3. Throw and muckpile (`throw.py`)

### 3.1 Launch velocity: parametrised, not derived

No burden-velocity formula is implemented. Published relations exist (power laws of burden over charge
per metre fitted to high-speed films; Gurney-type energy relations), but their constants could not be stated
with confidence in this clean-room task, so none is claimed. Instead the launch is described by parameters
(`ThrowParams`, `calibrated=False`):

* `face_velocity` = 15 m/s for the front row. Face velocities of roughly 10-30 m/s are the range commonly
  reported from high-speed photography of bench blasts (Chiappetta, R.F., Bauer, A., Dailey, P.J. and
  Burchell, S.L., 1983, "The use of high-speed motion picture photography in blast evaluation and design",
  *Proc. 9th Conf. on Explosives and Blasting Technique*, Society of Explosives Engineers, Dallas; summarised
  in Jimeno, Jimeno and Carcedo 1995). **[unverified]** the range is quoted as "typical", not from a table of
  the primary source.
* `row_factor` = 0.8: row `r` moves at `face_velocity * 0.8^r` (later rows are more confined). Assumption.
* Elevation angle from 5 degrees at the toe to 25 degrees at the crest, linear in height. Assumption.
* Scatter: speed x `(1 + 0.15 N(0,1))` (not below 0.1), elevation and azimuth + `5 deg * N(0,1)`. Assumption.
* `size_exponent` = 0: optional slowing of fragments larger than x50 by `(x50 / x)^exponent`.
* Direction: `(sin az, cos az)` in plan for the face azimuth `az` clockwise from north.
* Start time: `t0 = delay + d / cp (+ response_ms_per_m * d)`, `d` = distance to the charge column of the
  nearest hole. Real burden response times are longer than the wave travel time; the extra term is 0 by
  default because no value could be cited.

The distinct flyrock model (Richards and Moore) lives in `flyrock.py`; this module's particles are all
`kind = 0` (muck).

### 3.2 Fragment size

Rosin-Rammler distribution (Rosin, P. and Rammler, E., 1933, "The laws governing the fineness of powdered
coal", *J. Inst. Fuel* 7, 29-36) in the Kuz-Ram form `P(x) = 1 - exp(-ln 2 (x / x50)^n)`; sampled by the
inverse `x = x50 (-ln(1 - u) / ln 2)^(1/n)`, `u` uniform. `x50` and `n` come from the fragmentation model.

### 3.3 Flight and landing

No drag: `p(t) = p0 + v0 t - (g / 2) t^2 ez`, `g = 9.81 m/s2`. Landing = first time the path meets the ground:
the flight is sampled at 64 time steps up to a time at which the particle is certainly down, the first step
ending on or below the ground brackets the root, 60 bisections locate it. The ground is a bilinear heightmap
of the surface with the blasted block removed (inside the polygon: `floor_z`). Limits: a ridge thinner than
one time step can be flown through; no bounce, no rolling, no collisions; drag matters for small fragments
and is ignored. Check: `scipy.integrate.solve_ivp` with a terminal event on a level, on two inclined planes
and on a heightmap with a bench step; closed form on level ground.

### 3.4 Muckpile

Each of the `n` particles carries `V_blast * swell / n` (`swell` = broken / in-situ volume, `Rock.swell`) and
puts it into the grid cell it lands in. The heap is then relaxed to the angle of repose `phi`: for each pair
of neighbouring cells (8 neighbours, distance `d` or `d sqrt 2`) whose surface difference exceeds
`tan(phi) * distance`, a share (`damping` = 0.25) of the excess moves from the upper to the lower cell, never
more than the loose rock the upper cell holds; the fixed ground does not move. Sweeps over disjoint pair
groups, cycling through the 8 symmetries of the grid, until the largest transfer is below 0.1 mm. This is the
talus ("thermal erosion") relaxation of Musgrave, F.K., Kolb, C.E. and Mace, R.S. (1989), "The synthesis and
rendering of eroded fractal terrains", *Computer Graphics (SIGGRAPH '89)* 23(3), 41-50, a relative of the
sandpile automaton. Volume is conserved to rounding; the resting shape is one of many stable ones and is
slightly order-dependent (a 60 m3 column relaxes to a heap 3.10 m high, analytic cone 3.19 m, centre within
0.15 cell). `p_rest` of a particle is its landing position in plan on the final surface; particles are not
moved with the sliding rock.

Outputs: `z_before` (given surface), `z_after = ground with block removed + thickness`; where the given
surface is unknown (NaN) the level `floor_z` is used internally and both stay NaN. Summary: throw of the
centre of mass, largest throw, pile height, `swell_check = pile volume / blasted volume` (must equal the
swell), largest slope excess, particles landed outside the grid, volume on unknown ground.

Not modelled: the real mechanics of muckpile formation (the face slumping as a body, rows piling against each
other, cast distance depending on powder factor and delay timing). The delay sequence changes only the start
times here, not the shape of the pile.

---

## Part 4: Maptek BlastLogic feature study and gap analysis (public sources only, 2026-11-25)

**Headline finding:** BlastLogic itself has no drill-pattern design. Its help says outright that it "does not support drill or drillhole design"; patterns come from Vulcan or CSV/DXF. Pattern generation only arrives through the BlastMCF add-on (cloud genetic-algorithm optimiser, 2025+). BlastLogic's real weight is charge rules, tie-up design, field capture and reconciliation, plus four prediction models that are close to what Blast BakuMine already has.

**How this was done:** read-only, nothing in the repo touched. Most facts come from the public BlastLogic 2026 help site and three public help PDFs (Integration Service, Tablet, OData). Help pages were read through a summarising fetcher, so wording is paraphrased and fine detail could be off. Equations on the help pages are images and were not read.

URL shorthand: `H/` = `https://help.maptek.com/blastlogic/2026/topics/`, `V/` = `https://help.maptek.com/vulcan/2025.1/Content/topics/Drill_and_Blast/`. Confidence: High = stated in help text; Med = marketing/news; Inferred = my reading.

---

## A. Feature inventory

### 1. Design
- **No pattern design in BlastLogic core.** Designs are imported from CSV, DXF or Vulcan design databases and can be re-synced from the source file; holes match by ID or collar XYZ. High. `H/key-concepts/drill-and-blast-design.htm`, `H/menus-and-tools/home/data.htm`
- **Vulcan Drill and Blast module does the layout** (separate product). High. `V/00Index/DrillandBlast_00INDEX.htm`
  - Pattern inside a boundary polygon with a reference line setting where rows start.
  - Rows that progressively follow the free face, with min/max burden limits (defaults 3.5 m and 14 m) and smoothing. `https://help.maptek.com/vulcan/2024.0.0/Content/topics/Drill_and_Blast/Create/DrillandBlast_CreateHolesToFreeFace.htm`
  - Holes at points, along a line, or between a collar line and a toe line.
  - Legacy layout tool: square or staggered, bearing, dip from vertical, standoff above a target horizon, minimum drill length. `https://help.maptek.com/vulcan/2024.4/Content/topics/OpenPit/Blast_Hole_Layout/OpenPit_01LAY.htm`
  - Naming: alphabetic or numeric rows, echelon numbering, continuous forward/back, by object, zero-fill, prefix/suffix. `https://help.maptek.com/vulcan/2024.4/Content/topics/Drill_and_Blast/Create/DrillandBlast_Create_Holes_Along_Collar_and_Toe.htm`
  - Manual editing: insert, delete (also by polygon), drag, translate, rotate, double stitching, split/merge blasts, reverse row direction, recalculate after a surface change.
  - Checks: hole clearance (charged sections too close), hole separation report, compare holes.
  - Templates live in a specification file with five parts: loading styles, pattern, reporting, posting, display. Parameter-level content not found.
- **Face-profile burden (in Vulcan, not BlastLogic).** Design is compared with a triangulated real face; the face is coloured below/valid/above the burden range, a profile line is drawn in front of each hole, minimum burden is plotted down the hole, and the hole is nudged in position, dip, bearing. An "Offset to Face" report covers front-row holes. High. `https://help.maptek.com/vulcan/2024.4/Content/topics/Drill_and_Blast/Edit/DrillandBlast_Minimum_Burden_Adjustment.htm`
- **Buffer / trim / presplit row types:** not found as named features. The collar-and-toe-line tool is the likely presplit route (Inferred).
- **BlastMCF pattern generation.** Inputs: boundary polygon, upper surface, lower surface or RL or fixed depth, or a blast solid. High (2026 page + 2022 parameter page). `H/blastlogic-system-component-documentation/blastmcf.htm`, `https://help.maptek.com/blastlogic/2022/topics/menus-and-tools/blastmcf/optimisation.htm`
  - Bit-diameter list, subdrill (downhole or vertical), angle and bearing.
  - Pattern type: square, rectangular, equilateral, staggered.
  - Burden and spacing as min/max or ratio ranges; minimum hole-to-boundary distance; alignment point and row direction.
  - Parameter list is from the 2022 help; the 2026 page is less detailed.
- **Per-hole depth to a surface.** Charge depth and standoff are set against a reference surface or target RL, per row if wanted. High. `H/menus-and-tools/charging/Design/design-charge-depths.htm`
- **Coal / strata.** No separate coal mode. High. `H/menus-and-tools/home/hole.htm`, `H/quick-reference/glossary.htm`
  - Downhole "intervals" (labelled zones) are built from surfaces, from distance to a surface (seam or highwall offsets), from a drillhole database, from CSV, or from drill-measurement conditions.
  - Charge rules can query labelled intervals.
  - "Protective cap" (rock left above a seam or ore) has its own tolerance and report.

### 2. Charging
- **Charge rules are JavaScript scripts,** either hand-written or generated from a load table. High. `H/menus-and-tools/charging/Design/create-charge-rule.htm`
  - Load table: hole-length bands, each with a deck list.
  - Six deck sizing methods: variable ratio, fixed length, fixed volume, fixed mass, target powder factor, length above water.
  - Typed parameters (product, product family, number, boolean, select, interval label, load table), optionally editable on the tablet.
- **Scripting interface exposes hole conditions.** High. `H/quick-reference/charge-rule-api/object-interfaces.htm`
  - Hole: length, diameter, angle, burden, spacing, standing water length, wet-sides length, temperature (current and previous, with age in days), custom properties, labelled intervals, neighbours within a radius, local water level, approximate rock volume.
  - Actions: fill an interval, fill to a mass, fill packaged product, add a primer at a depth, add evenly spaced primers.
  - Products: density, water and damp resistance, delay, and gassed-emulsion density by depth (toe and average in-hole).
  - Built-in air and drill-cuttings products, so air decks and stemming decks are ordinary decks.
  - Hardness is not a built-in field; it would arrive as a custom property or interval (Inferred).
- **Product catalogue:** groups and families, density-by-depth tables, density tolerance, unit cost, stock level, import/export. Energy, RWS or VOD fields not found. High. `H/menus-and-tools/home/setup-group/blast-product-catalogue.htm`
- **Charge plans and sheets:** rule applied per hole gives a plan with preview; manual deck and primer editing in a row/echelon profile view; charge sheets and dip sheets go to tablet or paper. High. `H/key-concepts/profile-view.htm`, `H/process-overview/process-overview.htm`
- **Powder factor:** explosive mass per rock volume, from design volume ((drilled length + standoff) x cos(angle) x burden x spacing) or from hole length; reported for reference design, target, plan and actual. High. `H/quick-reference/glossary.htm`, `H/menus-and-tools/analysis/analysis-report.htm`
- **Energy display:** only two places found. The timing envelope plots detonation energy (MJ) per time window, and the fragmentation surface weights by each hole's "explosive energy density". No 3D energy-halo display found. High for what exists. `H/menus-and-tools/analysis/timing.htm`
- **Coloured surfaces from holes:** any metric, optionally one metric minus or divided by another (design vs actual). High. `H/menus-and-tools/geometry/surfaces.htm`

### 3. Timing
- **Tie-up editor** with pyrotechnic, electronic and auto-detected hybrid modes. High. `H/menus-and-tools/charging/Design/create-tie-up-design.htm`
  - Pyrotechnic: connector catalogue (single or composite), four tie modes (row-by-row, hole-by-hole, auto-fill along a control line, dummy holes), initiation point, connector length limits.
  - Electronic: timing lines and diamond "roses", each segment with its own burden relief (ms/m) plus fixed delay; per-hole time table; timing horizons for decks (deck delay, detonator delay, firing order).
  - Optimise: enforce minimum time between detonations, or between wave arrivals at named locations.
  - Harness wire layout with limits on wire length and detonators per run.
  - Tie-ups can span several blasts; states are Draft, Design, Actual.
- **Analysis tools.** High. `H/menus-and-tools/analysis/timing.htm`
  - Timing envelope: detonations, holes, mass, energy, PPV, pressure per window (usually 8 ms; 1-2 ms for electronic), optionally as seen at a point using a propagation speed.
  - Timing contours.
  - Burden relief surface: detonation time divided by distance from the free face.
  - First-movement arrows (method not given).
  - Animation with optional wavefront rings.
- **Vendor initiation systems:** only DetNet DigiShot (special-hole flag) and BME AXXIS (log-file round trip) are named; also generic CSV, tagger report and PDF. Export/import of actuals and charge plans "for SHOTplus" exists. Orica i-kon, Dyno, Davey Bickford: not found.

### 4. Field data and reconciliation
- **As-drilled vs design.** High. `H/menus-and-tools/drilling/validation.htm`, `H/menus-and-tools/home/setup-group/site.htm`
  - Auto-association by search radius with hole-ID fallback; redrill and ad hoc hole handling; green/red/orange validation.
  - Tolerances per diameter: short/long depth, collar XY, collar Z, setup error (angle and bearing at a set depth), dip depth, charge depth, backfill.
- **Drill navigation** through the Integration Service, which polls vendor databases (SQL Server, Oracle, PostgreSQL, ODBC, CSV). High. `http://help-files.maptek.com/blastlogic/2026/BlastLogic Integration Service Help Documentation.pdf`
  - Named systems: Atlas Copco Surface Manager, Caterpillar Aquila and Terrain, FLANDERS ARDVARC, Leica IMS/Tritronics and Jigsaw/Jmineops, Modular ProVision, ROCKMA, Wenco.
  - IREDES: not used by BlastLogic. Vulcan exports IREDES and a Tritronics file.
- **Drill-monitoring (MWD) data:** depth intervals with site-defined measures; viewed down the hole and in profile; intervals created from rule conditions; round trip through Maptek Eureka/GeologyCore for lithology boundaries. No hardness index formula. High / Med. `https://www.maptek.com/forge/june_2020/measurement-while-drilling-data/`
- **Dipping:** depth, water, wet sides, temperature, custom properties; backfill entry with check-dip percentage. High.
- **Tablet:** Windows app, offline with a sync queue, GNSS nearest-hole selection. High. `http://help-files.maptek.com/blastlogic/2026/BlastLogic Tablet Help Documentation.pdf`
  - Modules: drilling, dipping, backfilling, charging (load from plan, top-up, ad hoc decks and primers, cup density), blasts (mark fired), inventory.
  - Charge plans recalculate in the field from new dip data.
- **As-loaded vs design:** loaded and reconciled decks, charge placement accuracy, charge reconciliation, density measurement and protective-cap reports. High. `H/menus-and-tools/charging/charging-report.htm`
- **QA/QC reports.** High. `H/menus-and-tools/drilling/drilling-report.htm`, `H/menus-and-tools/pre-charging/pre-charging-report.htm`
  - Drilling: depth, collar and setup accuracy, drill calibration, productivity.
  - Pre-charging: water, pattern condition, time-dependent fallback, backfill compliance, depth accuracy.
  - Also hole status and blast summary.
- **Architecture:** central server, publish step, snapshots with a reference design, roles and permissions, hole history with comments and anomalies. High.
  - OData feeds for Excel/Power BI. `http://help-files.maptek.com/blastlogic/2026/BlastLogic OData Service Help Documentation.pdf`
  - HTTP API and Python package (`maptekblastlogic`). `H/blastlogic-system-component-documentation/blastlogic-server/blastlogic-extend/blastlogic-extend.htm`
  - Server database engine not found.
- **Explosive trucks:** no vendor named in the help. A truck is a pick-list item; the API can write loaded decks, and its key example is named "Blast Truck Integration" (Inferred integration path). A search snippet mentioned a Sasol IBIS truck link, but the fetched article did not confirm it.

### 5. Post-blast
- **Fragmentation measurement** is in PointStudio (scan-based rock segmentation, S-curves), compared with BlastLogic's prediction. Med. `https://www.maptek.com/forge/september_2019/spatial-measurements-for-drill-blast/`
- **"Spatial measurements":** back-break, heave, cratering imported and tied to holes for heat maps. Med.
- **Vibration and overpressure observations:** table per monitoring station, manual or CSV, linked to a tie-up so MIC and source location are derived. High. `H/menus-and-tools/analysis/Modelling/modelling-observation-history.htm`
- **Site-law regression:** not described. The models have a "use vibration observations" mode with filters, which suggests a fit (Inferred).
- **Muckpile shape, blast movement, ore loss/dilution, dig-line adjustment:** not found in BlastLogic.
- **Wall damage model:** not found (only back-break import).

### 6. Prediction models
- **Vibration (PPV).** Site constant and exponent per location, referenced to AS 2187.2-2006; standard square-root scaled-distance form assumed (Inferred, equation is an image). High for the logic. `H/menus-and-tools/analysis/Modelling/modelling-vibration-modelling.htm`
  - MIC is found per grid point by arrival time (distance / propagation speed + firing time) within the window, summing whole deck masses.
  - "Conservative" uses the nearest contributing charge; "non-conservative" uses a mass-weighted effective distance.
  - Outputs: heat map and a table at locations.
- **Overpressure.** Same structure with cube-root scaling, wind speed and direction, 340 m/s default, dB re 20 µPa, 5 dB contours, limit breaches shown red. High. `.../modelling-overpressure-modelling.htm`
- **Fragmentation.** Kuz-Ram with the Swebrec function replacing Rosin-Rammler. High. `.../modelling-fragmentation-modelling.htm`
  - Rock factor from Cunningham-style inputs: rock mass description, joint condition/spacing/angle, density influence (RDI = 25 x density - 50, from the 2022 page), UCS, Young's modulus, site correction factor.
  - Timing factors (rock compression velocity, delay scatter) change the curve.
  - Outputs: S-curve, maximum size, per-hole fragmentation surface.
- **Flyrock.** Choice of McKenzie (2009) or Richards and Moore (2006). High. `.../modelling-fly-rock-modelling.htm`
  - Cratering and face burst only; rifling explicitly not modelled.
  - 2D circles, or 3D trajectories intersected with topography; safety factor; exclusion polygons.
  - Face burst uses a user-supplied face triangulation.
- **Cost model:** drilling, explosives, primers, tie-up; planned vs actual. High.
- **BlastMCF** combines cost, powder factor, PPV, fragmentation oversize and 2D flyrock as objectives or constraints. High.
- **"Hardness modelling"** appears only in a 2017 brochure line; no model found in current help. Low.

### 7. Data in/out
- **In:** CSV, DXF, Vulcan design database, Vulcan triangulations and grids, GeologyCore drillhole file, product catalogue and charge-rule files, attachments. High. `H/menus-and-tools/home/data.htm`
- **Out:** the same, plus loaded decks, drill measurements, PDF/CSV reports.
- **Block model import: not found.** Geology reaches holes as intervals; Vulcan rock-type intervals map across. `H/key-concepts/blastlogic-and-vulcan-integration.htm`
- **Fleet management:** only through drill-navigation databases. Maptek Evolution or Drive links: not found.

### 8. Reporting for cost or feasibility
- **Cost report:** drilling, explosive, primer and tie-up costs, planned vs actual, per hole and total. High. `H/menus-and-tools/analysis/analysis-report.htm`
  - Defaults per the OData manual: drilling cost per metre by diameter, fixed drilling cost per hole, charging cost per hole, bulk explosive cost per kg, primer cost per deck.
  - Per-volume or per-mass cost display is described for BlastMCF only (needs rock density).
- **Blast statistics feed:** hole count, volume, design/actual drill metres, designed/loaded/reconciled explosive mass, powder factor, redrills, out-of-tolerance counts. High.
- **Inventory:** planned usage vs stock; actual usage of bulk ingredients and accessories. High. `H/menus-and-tools/inventory/inventory-report.htm`

### Competitors: what they offer that BlastLogic does not (brief; partly from general knowledge)
- **Orica SHOTPlus / BlastIQ / OREPro 3D / FRAGTrack:** blast movement and ore-control modelling, a physics-based movement and damage model, camera fragmentation measurement. Source: `https://www.orica.com/blastiq` and trade press. The SHOTPlus page returned 403.
- **Hexagon:** in-hole blast movement monitors with dig-line generation, AI muckpile block model, Split photo fragmentation. `https://hexagon.com/products/hexagon-blast-movement-monitoring`
- **Deswik.OPDB:** pattern design inside the planning package, hole templates that re-fit to the next bench's surfaces, blast clearance, publish to drills. `https://www.deswik.com/products/opdb`
- **Dyno Nobel:** Differential Energy (varying emulsion density down the hole; heading only, no detail found), Nobel Fire with a fracture-density fragmentation model, ViewShot 3D timing. `https://www.dynonobel.com/our-services/software-services`
- **O-Pitblast:** borehole deviation measurement, greenhouse-gas estimate, rules-of-thumb guide. `https://www.o-pitblast.com/`. Signature-hole vibration and photo fragmentation not confirmed (the search errored).

---

## B. Gap table

| Feature | BlastLogic | Ours today | Value: (i) feasibility / (ii) operations | Size | Input needed |
|---|---|---|---|---|---|
| Cost model | Drill, explosive, primer, tie-up costs, plan vs actual | None | (i) Very high / (ii) High | S | Unit costs, rock density |
| Deck charging and rule engine | Scripted rules, six deck sizing methods, air and stemming decks, multiple primers | One column | (i) High / (ii) Very high | M-L | Product table, hole conditions |
| Product catalogue | Density, density-by-depth, water resistance, cost | Minimal | (i) High / (ii) High | S | Supplier data sheets |
| Wet/dry product choice | Water and wet-sides length, length-above-water decks | None | (i) Med (assumed wet fraction) / (ii) High | S after decks | Hydrogeology, dips |
| Geology-driven design | Downhole intervals from surfaces; rules read them | None | (i) Very high / (ii) High | M | Our block model, rock-mass data |
| Rock factor from rock-mass inputs | Cunningham inputs, Swebrec option, correction factor | Kuz-Ram | (i) High / (ii) Med | S | UCS, E, jointing, density |
| Scenario optimiser | BlastMCF: constraints and objectives across pattern, charge, timing | Single design | (i) Very high / (ii) Med | L | Limits, costs, monitors |
| Pattern options | Diameter list, four pattern types, angled holes, hole depth to a surface, free-face rows | Staggered/square | (i) High / (ii) High | M | Bench surfaces, pit design |
| Row types (buffer/trim/presplit) | Not found as named features | None | (i) High for wall control / (ii) High | M | Wall design |
| Manual hole editing | Vulcan: insert, delete, move, rename | None | (i) Low / (ii) High | M | None |
| Face-profile burden | Vulcan: face triangulation vs holes | None | (i) None / (ii) High | M | Face scan |
| Tie-up drawing and accessory count | Connectors, timing lines/roses, harness, deck horizons | Auto patterns only | (i) Med (counts feed cost) / (ii) High | M-L | Accessory catalogue |
| Timing envelope and optimiser | Mass/energy/PPV per window; minimum separation | MIC in 8 ms | (i) Med / (ii) Med | S | None |
| Arrival-time MIC per receiver | Per grid point, propagation speed | Firing-time window | (i) Med / (ii) Med | S | Propagation speed |
| Overpressure with wind, dB limits | Yes | Rings | (i) Med / (ii) Med | S | Site constants, wind |
| Flyrock in 3D on topography, McKenzie option | Yes | Richards-Moore ranges | (i) Med / (ii) Med | S-M | Topography |
| Monitoring observation store | Table tied to tie-ups | Fit function only | (i) Low / (ii) High | S | Monitor records |
| Field capture (drill, dip, load) | Tablet, server, tolerances | None | (i) None / (ii) Very high | L | Field crews, devices |
| Drill-navigation / MWD link | Seven vendor systems | None | (i) None / (ii) High | L | Vendor databases |
| QA/QC reports | Around 15 reports | JSON report | (i) Low / (ii) High | M | Field data |
| Inventory | Magazines, usage vs stock | None | (i) Low / (ii) Med | M | Stock records |
| Blast movement / dilution | Not in BlastLogic (Orica, Hexagon have it) | Throw and muckpile only | (i) Med-High / (ii) Very high | L | Block model, movement data |

---

## C. Ranked recommendations

### Top 10 for a feasibility study (no field data yet)
1. **Cost model** (per hole, per BCM, per tonne): the chapter's main deliverable and the cheapest to build.
2. **Product catalogue** with density, water resistance and unit cost: prerequisite for items 1, 3 and 4.
3. **Decks:** stemming, air decks, multiple explosive decks, primers per deck. Realistic consumption and correct MIC per deck.
4. **Design domains from our own block model:** rock type and hardness per hole or zone driving burden, spacing, powder factor and rock factor. BlastLogic lacks this; it is our natural advantage.
5. **Rock factor calculator** from UCS, modulus, jointing and density, with a Swebrec option: defensible fragmentation from geotechnical data that a study actually has.
6. **Consumption roll-up over the mine life:** drill metres, explosive tonnes, accessory counts per period and domain. Needs pit volumes by domain.
7. **Pattern options:** diameter choice, angled holes, depth to a floor surface, equilateral/rectangular patterns.
8. **Wall-control rows** (buffer, trim, presplit) with their own charge rules, tied to our Holmberg-Persson damage model.
9. **Constraint check and scenario comparison** (PPV, airblast, flyrock, oversize, cost side by side); a small grid search before any genetic algorithm.
10. **Wet-hole assumption** (wet fraction selects product and cost), plus overpressure in dB with limits and arrival-time MIC as small model upgrades.

### Top 5 for an operating mine
1. **As-drilled and dipping import with tolerances:** CSV first, vendor links later.
2. **Condition-driven charge rules** that recalculate from dip, water and temperature.
3. **As-loaded reconciliation** with powder factor and cost, plan vs actual.
4. **Face-profile burden and manual hole editing.**
5. **Monitoring database with site-law refit, then a blast-movement / dig-line model.** The second half is where BlastLogic is weakest and the grade-control value is highest.

---

## D. Not verified from public sources
- **Exact equations** for vibration, overpressure and fragmentation (images on the help pages). The scaled-distance forms and the Kuznetsov variant are assumed standard.
- **Site-law regression:** whether BlastLogic fits the constants from observations, and how.
- **First-movement method** and how the free face is defined for burden relief.
- **Product catalogue fields** for energy/RWS/VOD and the full list of product types; how detonation energy (MJ) is computed.
- **Release notes:** behind a login. Version history here comes from news items and manual revision tables only.
- **Vulcan specification contents:** parameter detail of pattern and loading-style templates, and any named buffer/presplit row types.
- **Block model import, Evolution/Drive links, IREDES import, a named explosive-truck integration:** none found in BlastLogic documentation. Absence from the pages read is not proof of absence.
- **Server database engine,** audit-trail detail, tablet hardware.
- **PointStudio blast tools** beyond fragmentation analysis.
- **Failed sources:** the miningmagazine.com article on electronic timing and the Orica SHOTPlus page returned 403; the O-Pitblast feature search errored. Competitor notes are therefore partly from general knowledge and marketing pages.
- **Marketing claims** (9-10% shovel gain, 20% less commodity loss) are vendor statements with no independent evidence.


---

## Part 5: The realistic look of the terrain (task B11)

Display only. Nothing here is measured and nothing changes the surface, a volume or a result. Code:
`web/src/blast/scene/terrain.tsx`; the server only adds `muckpile.cut_m`.

- **Relief shading.** A vertical photograph has the brightness of flat ground. The picture is multiplied by
  `clamp(1 + 0.85 (n . s - s_z), 0.5, 1.35)`, with `n` the unit normal of the surface and `s` the unit
  direction to the sun, so flat ground (n = up) keeps the factor 1. The sun is fixed at azimuth 135 degrees
  (south-east), 45 degrees up: north of the tropics a mid-morning satellite pass has the sun there, so the
  added shading agrees with the shadows already in the photograph. The true sun angle of the picture is not
  known (the tile service does not give the acquisition time). Strength and limits are choices.
- **Fine relief from the picture.** The luminance of the picture times 1.6 m is read as a height and the
  normal is tilted by its gradient (bump mapping, Blinn 1978; evaluated with screen-space derivatives after
  Mikkelsen 2010). Dark is not really low: this is a visual cue, not a terrain model.
- **Grain.** Fractal value noise (4 octaves) modulates the picture by up to +/- 15 % nearer than 450 m, where
  one pixel of the picture (about 0.9 m here) covers many pixels of the screen.
- **Rock on cut and broken ground.** `cut_m` = natural ground minus the ground after the bench cut, inside the
  excavated area. Nodes with more than 0.2 m of cut (fully from 1 m), nodes on slopes steeper than 35 degrees
  (fully from 60 degrees) and nodes where the muckpile moved the surface are drawn as procedural rock (bands
  along z, mottling, grit) instead of the photograph of the old surface. The rock colours are choices.
- **Sky and haze.** A gradient sky; colours fade to the horizon colour by `1 - exp(-d / 12 km)` (about 8 % at
  1 km).
- **Zoom without imagery.** The tile service answers a zoom it has no photograph for with one fixed picture.
  A block of two or more tiles that are all the same bytes is taken as that picture and the next lower zoom is
  fetched (`imagery.fetch_best`). On the test site zoom 18 has no imagery and zoom 17 (0.86 m per pixel) is
  used.

Not done: no higher-resolution elevation (the surface is still the project's topography), no imagery or
terrain beyond the first topography surface, no shadows cast by the terrain, no vegetation or equipment.

References: J. F. Blinn, "Simulation of wrinkled surfaces", SIGGRAPH 1978. M. Mikkelsen, "Bump mapping
unparametrized surfaces on the GPU", Journal of Graphics, GPU and Game Tools 15(1), 2010. (Both cited from
memory; page numbers not checked.)


---

## Part 6: The site view (task B12)

What a person standing at the site would see. Display only: no result of the simulation changes. Code:
`web/src/blast/scene/` (`realism.ts` holds the rules and their tests, `Throw.tsx`, `Dust.tsx`, `Equipment.tsx`,
`Scatter.tsx`, `topography.ts`).

From the simulation (not invented here): the path, launch time and landing of every parcel of rock; the predicted
size distribution; the firing time of every hole; the clearance radius; the ground before and after.

Added by the viewer:

- **Fragments.** throw.py moves a few thousand parcels, each standing for many tonnes. A parcel is drawn as up to
  10 fragments. The first has the parcel's own size and path. The others take a size from the predicted
  Rosin-Rammler curve, `x = xc (-ln(1 - u))^(1/n)`, and a velocity that differs from the parcel's by up to 10 % of
  its speed, so the cluster spreads in flight; the height difference fades over the last fifth of the flight so
  the fragments meet the ground where the parcel does. This is a drawing of scatter, not a second simulation:
  fragments do not collide and do not have their own landing point on the terrain. Each fragment is an irregular
  42-vertex block that tumbles at 2 to 14 rad/s (assumed, faster for small pieces) and stops when it lands.
- **Soil.** Parcels that start within 1.5 m of the bench top are drawn as soil: brown clods of 3 to 16 cm with a
  wider spread (22 % of the speed) and more lift. The 1.5 m is assumed: the blast does not read a cover thickness
  from the geology model yet, and the simulation gives soil the same launch velocity as rock.
- **Dust.** Puffs start at every hole collar when it fires (a column and a ground surge, strength varied per
  hole), at parcels when they launch and where they land. A puff obeys `dv/dt = (w - v) / tau` with `tau` = 1.1 s
  and a wind `w` of (1.6, 0.7) m/s plus 0.5 m/s rise, so
  `x(t) = x0 + v0 tau (1 - e^(-t/tau)) + w (t - tau (1 - e^(-t/tau)))`; its radius grows as `r0 + k sqrt(t)`; it
  fades over 5 to 12 s. The form is the standard linear-drag solution; every number in it is assumed. Dust
  amounts are not computed from the rock or the explosive. The player runs 7 s past the last landing so the cloud
  can drift.
- **Excavators.** Two generic 30-tonne-class machines (boxes and cylinders, not a product model) stand behind the
  blast, 25 m outside the clearance radius of the flyrock model, 18 m apart, on the topography. "Behind" is the
  side opposite the free face. They cast shadows on a patch of ground around them.
- **Stones and shrubs.** About 7000 of them within 280 m of the blast and 110 m of the machines, at fixed
  pseudo-random places, none on the cut bench. Invented scenery for depth and scale.
- **Sharpening.** The picture plus 0.9 of its difference from the mean of four neighbours 1.5 pixels away.
- **Height lookup.** Machines and scatter stand on the project topography: bilinear in the grid cell when the
  surface is a complete regular grid (exact on a plane), else an inverse-distance mean of nearby vertices.

Second pass (task B14):

- **Shadows.** Two weak lights from the sun's direction own shadow maps: one over 460 m around the blast (4096
  pixels, about 11 cm each), one over 80 m around the machines. Fragments, the bench walls and the muckpile,
  stones, shrubs and the machines cast shadows. The terrain shows them through a copy of its mesh that draws only
  the shadow, so its picture and relief shading are not changed. The sun direction is the assumed one of part 5.
- **Excavators.** Rebuilt from extruded side profiles with bevelled edges: track loops with grouser bars and
  rollers, house with engine cover and counterweight, cab with glass, gooseneck boom, stick, bucket with teeth,
  four hydraulic rams, handrail, exhaust. Still generic, still not moving.
- **Flash.** A glow of 110 ms at each collar when its hole fires. Strength chosen for effect.
- **Tremble.** Removed again in the ninth pass: the owner did not want the camera to move on its own.
- **Sound.** A rumble made in the browser (filtered noise over a falling 62 to 28 Hz tone) starts when the player
  time passes `t_first + d / 340 m/s`, so at the machines it comes more than two seconds after the picture. It
  is a generic rumble, not a prediction of the airblast, and plays at normal speed also in slow motion.
- **Dust edge.** The lower edge of a puff thins out, which hides most of the hard line where it met the ground.
- **Camera.** In the site view the camera cannot go below 1.6 m above the topography.

Third pass (task B15):

- **Dust shadow.** Every puff also draws a thin dark disc on the ground: its centre slid along the sun's ray down
  to the height the puff started from, with the puff's radius. The discs add up to the shadow of the cloud. The
  ground under the cloud is taken as level at the start height, so on a slope the shadow is slightly misplaced.
- **Machines move.** While the blast plays the upper structure of each excavator swings up to 7 degrees and its
  bucket curls up to 9 degrees, slowly, as sines of the player time with a phase per machine. Invented idle
  motion; boom and stick stay put.
- **Ground up close.** Within 150 m the picture gets patches of gravel (greyer, rougher) and sand (paler, with
  ripples), small bright stones, and about 7000 dry grass tufts of 10 to 35 cm, half of them around the machines.
  All from fixed noise; none of it is surveyed.
- **Sound, measured.** The rumble rendered off-line (`rumbleCheck`): peak 0.94 of full scale at the loudest
  setting, above a tenth of its peak for 1.25 s, 78 % of its energy below 200 Hz. In the browser it played once
  when the player time passed the computed arrival (2.44 s at 828 m), with the audio context running. How it
  sounds on a given loudspeaker was not judged by ear.

Fourth pass (task B16):

- **Fine relief of the ground.** Within 260 m of the blast and 90 m of the machines the surveyed surface is
  resampled on a 2.5 m and a 1 m grid and low mounds and hollows are added: value noise 18 m across (0.24 m), 6 m
  across (0.11 m) and 2 m across (0.05 m), never negative, fading to nothing at the edge of the area and within
  12 m of the bench and muckpile. The elevation model has one height per 30 m and shows none of this; the relief
  is invented so that ground seen from eye height is not flat facets. Stones, shrubs, grass, machines and the
  camera stand on the refined ground. No volume, no result and no stored surface uses it.

Fifth pass (task B17): fair-weather clouds (fractal noise on a flat layer, lit on the sun's side), a sun disc with
glow, rough relief on rock faces (noise read as height, lit), a larger dust cloud (18 puffs per hole, rising
faster), and a slightly firmer picture (contrast 1.08 about mid grey, saturation 1.12). All choices for looks.

Sixth pass (task B18):

- **Far terrain.** A project surface whose name contains "backdrop" (the elevation model over 12 km at 60 m
  spacing, with the same vertical shift as the topography) is drawn around the detailed terrain, so the view has
  the real hills out to the horizon. It has no picture: it takes the mean colour of the satellite mosaic in broad
  invented patches, the same relief shading, and the haze. It is not drawn inside the detailed terrain.
- **Cloud shade.** Soft patches a few hundred metres across darken the ground by up to 20 %. They are fixed noise
  on the ground and are not the shadows of the clouds drawn in the sky.

Seventh pass (task B19): drone-level ground from above.

The satellite picture resolves about 0.9 m, so from above the ground looked blurred. The picture is now used only
for the colour at a metre and more; everything finer is synthesised in the terrain shader, at every viewing
height. A detail of size l is drawn only while it covers a few pixels (weight `smoothstep(1.5 p, 5 p, l)` with p
the metres of ground per pixel), which keeps the ground crisp without shimmer.

- Bare ground: tone drifting over 3 and 8 m, patches of gravel and sand, grain at 45, 15 and 6 cm, dark and pale
  stones, sand ripples.
- Plants: the picture is read as "plants grow here" where it is greener than grey or dark. The plane is cut into
  cells of 1.7 m (shrubs) and 0.42 m (grass tufts); a cell holds a plant with a probability of 5 % on bare ground
  rising to 60 % where plants grow, at a random place and size. Each shrub darkens the ground 0.55 m away from
  the sun (its shadow).
- The same details form a height (shrubs 0.5 m, stones a few centimetres) that the sun models.

All of it is invented from fixed noise and from the colour of the picture: no plant, stone or patch is where a
real one is. It is what a drone photograph of such ground looks like, not a drone photograph of this ground.

Eighth pass (task B20): the floor of the cut. Level rock gets loose blocks (cells of 1.1 m, 42 % filled) and chips
(0.33 m, 55 %) with their shadows, grit at 15 and 5.5 cm and dust changing tone over about 6 m, with the same
rule for what is drawn at which viewing height. Invented, like the rest of the fine ground detail.

Ninth pass (task B21):

- **No camera shake.** The camera tremble of the second pass is removed (code, test and switch). The layer switch
  that held it is now "Sound" only.
- **Ragged edges.** Up close the picture changes colour in soft blurs a metre wide. The point it is read at is
  shifted by up to about a metre with two noises (wavelengths 0.6 and 2.2 m), which turns each soft edge into a
  ragged one, as the edge of ground cover is. The shift fades out when a metre is under a few pixels.
- **Veil off.** A small bluish offset is subtracted and contrast and saturation are raised a little (6 and 8 %),
  for the difference between a picture taken through the whole atmosphere and one taken from a hundred metres.
  The amounts are chosen by eye.
- **Plants in groups.** The chance of a plant varies over tens of metres; a few large bushes (cells of 4.6 m)
  stand where plants are dense.
- **Rills.** On slopes, thin darker lines run down the slope (noise stretched along the fall line).

Tenth pass (task B22), after the owner's review ("random green pits", "green circle like stuff"): every round
shape is gone. Plants are no longer separate discs in cells but irregular soft clumps (fractal noise read through
a second noise, blended in at about half strength, almost flat), covering 2 % of bare ground up to 50 % where the
picture is green, so the proportion of soil and green of the picture is kept. The floor of the cut has irregular
rubble patches and chips in place of round blocks. The 3-D ball shrubs and cone tufts are removed; small stones
stay. The cell-plant rule of the seventh pass and the large bushes of the ninth are no longer used.

Eleventh pass (task B23), after the owner's review ("texture looks painted"): the ground detail is turned down to
what a photograph shows. The shift of the reading point is 0.4 and 0.55 m (was 1.0 and 1.3), sharpening 0.55,
grain 17 % (was 34 %); rills and sand ripples are removed (they drew thin lines over everything); small stones
and tufts change colour only and are no longer lit as relief (lit, each became a dark crescent). What remains:
tone drifting over metres, gravel and sand patches, soft grain, faint stones, irregular soft plant cover.

Twelfth pass (task B24), after the owner's review ("from an extended view it just looks painted"): from a few
hundred metres the fine detail is below a pixel and the magnified picture showed as soft blots with raised
contrast and embossing. Now: irregular texture 1 to 3 m across (fractal noise, 12 to 16 %) fills that range; a
per-pixel sensor grain of 3.5 % is laid over the ground; the picture's brightness is read as relief only up
close; sharpening, contrast and saturation are lower (1.02 to 1.04).

Thirteenth pass (task B25), after the owner's review ("wide view still painted"): the picture itself is soft and
renders plants as smooth bluish-green smears. For the wide view only: irregular bush specks one to two metres
across with a shadow beside them, following the plant cover, drawn when a pixel covers more than about 0.1 to
0.26 m and fading out as the camera comes close; stronger mottling at 0.9, 1.8 and 5 m; plant areas turned from
the picture's bluish green to olive (38 % towards olive at full cover).

Fourteenth pass (task B26), after the owner's review ("sound needs work"): the sound is rebuilt and starts with the
first hole. It no longer waits for distance / 340 m/s: the player shows the detonation in slow motion, and a
sound arriving seconds later read as a fault. It is made of a crack per hole at its own firing time (at most 48,
bright noise bursts of 90 ms), a boom falling from 70 to 24 Hz with a lower second thump, a rumble that darkens
over five seconds, an echo (delays of 0.31 and 0.53 s with feedback through a low-pass), and faint irregular
clatter of falling rock between 0.7 and 4.2 s; 6.5 s in all. It plays at normal speed whatever the player speed.
It is a generic blast sound chosen for effect, not a prediction of the airblast. The off-line measurement
(`rumbleCheck`) was not rerun for this version: the browser link was lost.

Not done: the flash does not light the dust or the ground; boom and stick of the machines do not move; the terrain is still
the 30 m elevation model resampled to 10 m, so close-up ground detail is the picture's 0.9 m pixels plus the
added grain and scatter.
