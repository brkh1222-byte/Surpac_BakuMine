# Babbitt Cu-Ni drillhole dataset

Real exploration drilling from the Babbitt (also called Mesaba) copper-nickel deposit,
Partridge River intrusion, Keweenawan Duluth Complex, St. Louis County, north-eastern
Minnesota, USA. Commodity: Cu-Ni (+ S; a handful of Fe assays). Magmatic sulphide
mineralisation, mostly near the base of the intrusion.

## Where it came from

| | |
|---|---|
| Retrieved | 2026-11-19, anonymous HTTPS download |
| Repository | https://github.com/opengeostat/pygslib (PyGSLIB, by Adrian Martinez Vargas) |
| Commit | `c26c3bff20c7cc16583c94fb850e17600b9a610d` (head of `master`, committed 2024-10-03) |
| Path | `doc/source/Tutorial_1/Babbitt/` (data for the PyGSLIB resource-estimation tutorial) |
| Licence | MIT, see `LICENSE` |

Download URL pattern:
`https://raw.githubusercontent.com/opengeostat/pygslib/c26c3bff20c7cc16583c94fb850e17600b9a610d/doc/source/Tutorial_1/Babbitt/<file>`

The PyGSLIB tutorial (`doc/source/Tutorial.md` in the same commit) states: "The informing
data is from the BABBITT zone of the KEWEENAWAN DULUTH COMPLEX" and links NRRI report
TR-2003/21. That report is:

> Patelke, Richard L. (2003). *Exploration Drill Hole Lithology, Geologic Unit,
> Copper-Nickel Assay, and Location Database for the Keweenawan Duluth Complex,
> Northeastern Minnesota.* NRRI Technical Report NRRI/TR-2003/21, Natural Resources
> Research Institute, University of Minnesota Duluth. https://hdl.handle.net/11299/187133

Its abstract describes it as compiling "virtually all publically available drill hole
location data, lithological logging data, copper-nickel assay data, and rock quality data"
for the Duluth Complex. The files here are the Babbitt subset as prepared by the PyGSLIB
author; we did not re-derive them from the NRRI CD-ROM files.

### Licence evidence and its limit

- GitHub reports the repository licence as `MIT`; `LICENSE.txt` at the pinned commit is
  the MIT text, "Copyright (C) 2015 Adrian Martinez Vargas" (copied verbatim into `LICENSE`).
- `README.md` at the same commit: "This software may be modified and distributed under the
  terms of the MIT and GPL licenses."
- The MIT grant covers "this software and associated documentation files"; the data sits
  under `doc/` in that repository.
- Not verified: the University Digital Conservancy record of the original NRRI report
  carries no explicit rights/licence statement in its metadata (checked 2026-11-19), and
  the original NRRI page linked by the tutorial returns HTTP 403. So redistribution rests
  on the MIT licence of the repository that ships the data, not on a licence issued by
  NRRI itself.

### Files and checksums (SHA-256 of `raw/`, byte-identical to the download)

| File | Bytes | SHA-256 |
|---|---|---|
| `raw/collar_BABBITT.csv` | 13437 | `2dbce5d751c283d0d71d9fc2afbebace62d74d6e3db2c92af9cbcac7b7f62c32` |
| `raw/survey_BABBITT.csv` | 48545 | `5f41cfaf7adf4cdd7f9652e2db6d53520c5565489123ff6cec0fc0cf6a300c09` |
| `raw/assay_BABBITT.csv` | 1475941 | `121956eb0e158af5d6c13e66ea1d57b66bee36e9577c20d80b624b1a88140f2f` |
| `raw/Mpz.stl` | 3187534 | `a324327f6dd4802ad95c6570cce8488ea397311459a48ea2a75c20f6cd5e867b` |

### Domain wireframe `raw/Mpz.stl`

| | |
|---|---|
| Retrieved | 2026-11-21, anonymous HTTPS download |
| URL | `https://raw.githubusercontent.com/opengeostat/pygslib/c26c3bff20c7cc16583c94fb850e17600b9a610d/doc/source/Tutorial_1/Babbitt/Mpz.stl` |
| Commit | `c26c3bff20c7cc16583c94fb850e17600b9a610d` (same as the tables; still the head of `master` on 2026-11-21) |
| Licence | MIT, as for the tables (see `LICENSE` and the limit stated above) |

The mineralised-zone solid the PyGSLIB tutorial uses as its estimation domain (the tutorial
text loads it as `domain.stl`). Facts checked on the file:

- **Binary STL**, 63 749 triangles (84 + 50 x 63 749 bytes), 31 873 distinct vertices,
  float32 coordinates: at eastings of 2.3 million the resolution is 0.25 ft.
- **Feet, same grid as the holes**: X 2 289 334 to 2 302 998, Y 415 146 to 424 122,
  Z -645.8 to 1690. Not stated in the file; established by the result: 46 % of the 10 ft
  composites and 79 % of the Cu metal lie inside it, and the mean Cu of the composites inside
  (0.2415 %, unassayed = 0 as in the tutorial) equals the tutorial's published 0.2415 %.
- The surface is stepped (it looks extracted from a block or grid model), volume
  62.63 x 10^9 ft3 = 1.773 x 10^9 m3.
- **Not a closed solid as published**: one triangle is missing (a horizontal hole of
  3227 ft2 at Z = 717.6 ft near X 2 289 400, Y 416 100) and three zero-thickness fins (pairs
  of coincident triangles with opposite winding) hang on the surface. Removing the six fin
  triangles and adding the missing one gives a closed, outward-oriented surface of 63 744
  triangles; one vertex where two sheets touch remains non-manifold.
  `scripts/run_babbitt_wireframe.py` does this repair at run time; the file here is the
  unmodified download.
- It reaches 54 ft south of the tutorial's block model (origin Y = 415 200 ft): 0.4 % of
  its volume lies outside the model.

The tutorial's published results and our reproduction of them are in
`scripts/run_babbitt_wireframe.py` and `tests/cli/test_babbitt_wireframe.py`.

## Tables

`raw/` holds the unmodified originals. `collar.csv`, `survey.csv` and `assay.csv` are
produced by `convert.py`, which only renames columns and adds a derived `depth` to the
collar; every other cell is copied as the exact source text (no rescaling, no sign change).

All lengths, depths, coordinates and elevations are in **feet** (inferred, see Quirks).

### collar.csv: 399 holes (raw: `collar_BABBITT.csv`)

| Column | Raw name | Unit | Description |
|---|---|---|---|
| `hole_id` | `BHID` | text | Hole identifier (e.g. `B1-001`, `RMC-65246`, `34873`) |
| `easting` | `XCOLLAR` | ft | Collar easting, 2 288 231 to 2 306 237 |
| `northing` | `YCOLLAR` | ft | Collar northing, 413 723 to 425 039 |
| `elevation` | `ZCOLLAR` | ft | Collar elevation, 1528 to 1633.3 |
| `depth` | (derived) | ft | Hole length = largest assay `TO` of the hole; 35 to 3574, total 541 230 ft |

### survey.csv: 2628 stations, 399 holes (raw: `survey_BABBITT.csv`)

| Column | Raw name | Unit | Description |
|---|---|---|---|
| `hole_id` | `BHID` | text | Hole identifier |
| `depth` | `AT` | ft | Downhole depth of the station; every hole has a station at 0 |
| `dip` | `DIP` | degrees | Inclination, **positive downward** (see below); 32 to 90 |
| `azimuth` | `AZ` | degrees | Azimuth clockwise from grid north, 0 to 359 |

Column order differs from raw (`BHID,AT,AZ,DIP`).

### assay.csv: 35 616 intervals, 399 holes (raw: `assay_BABBITT.csv`)

| Column | Raw name | Unit | Description |
|---|---|---|---|
| `hole_id` | `BHID` | text | Hole identifier |
| `from` | `FROM` | ft | Interval start depth |
| `to` | `TO` | ft | Interval end depth |
| `cu` | `CU` | wt % | Copper; 23 685 values, 0 to 24.4, mean 0.40 |
| `ni` | `NI` | wt % | Nickel; 23 439 values, 0 to 5.2, mean 0.10 |
| `s` | `S` | wt % | Sulphur; 23 545 values, 0 to 37.11, mean 1.16 |
| `fe` | `FE` | wt % | Iron; only 24 values, 10.1 to 15.7 |

Blank grade = not assayed (not zero).

There is no geology/lithology table in this dataset.

## Dip sign convention

**Dips are positive downward.** All 2628 dip values lie between +32 and +90; there is no
negative value. 348 of 399 holes start at dip 90 (vertical, drilled down). This is the
PyGSLIB/GSLIB-style convention and the opposite of the Surpac/Datamine-style convention
where a downward hole has a negative dip. A loader that expects negative-down must negate
`dip`. The CSVs here keep the source sign.

## Coordinate system

Not stated at the source. The values are consistent with **Minnesota State Plane, North
zone, NAD27, US survey feet (EPSG:26791)**: with that zone's false easting of 2 000 000 ft
at 93.1 W and origin latitude 46.5 N, the town of Babbitt (about 47.65 N, 91.9 W) falls
near E 2 295 000 ft, N 420 000 ft, which is where the collars are. Treat this as an
inference, not a documented fact; the vertical datum is unknown (elevations of
1528 to 1633 ft match local ground level above sea level). For modelling, use the
coordinates as a local grid in feet.

## Known quirks

- **Units are feet, not metres**, and are not stated in the files. Inferred from the US
  location, state-plane magnitudes, elevations and the dominant 10 ft assay interval.
- **Sentinel survey stations at depth 90000.** 70 holes have exactly two stations: one at
  0 and one at `AT = 90000` repeating the same dip and azimuth. It means "straight hole to
  end of hole", not a real measurement at 90 000 ft. No other station lies deeper than
  its hole.
- **No hole depth in the raw collar.** `depth` in `collar.csv` is derived from the assay
  table. Each hole's intervals start at 0 and are contiguous (no gaps, no overlaps), so
  max `TO` is the logged length.
- **Long unassayed intervals.** Unsampled stretches are single rows with blank grades
  (e.g. hole `34873`, 0 to 2515 ft). Interval length ranges from 0.1 to 3574 ft; median 10.
- **Float32 artefacts** in grades: `0.0299999993` means 0.03.
- **Numeric-looking hole IDs** (`34873`) occur alongside text ones; read `hole_id` as a
  string (`dtype={"hole_id": str}`) so a subset of the file is not parsed as integers.
- **Extreme grades** (Cu up to 24.4 %, S up to 37.1 %) are present as delivered: narrow
  massive-sulphide intervals, not checked against the original logs.
- Survey rows are sorted by depth within a hole with no duplicate depths; hole IDs are
  unique in the collar and all three tables cover exactly the same 399 holes.
- The upstream tutorial drops NI, S and FE and sets unassayed CU to 0 before estimating;
  that is a tutorial shortcut and is not applied here.
