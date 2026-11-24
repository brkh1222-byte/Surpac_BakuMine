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
