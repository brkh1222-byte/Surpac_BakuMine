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
