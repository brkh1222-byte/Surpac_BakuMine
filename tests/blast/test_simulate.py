"""Result bundle of ``bakumine.blast.simulate``: the contract of plan_blast.md section 3 (and
web/src/blast/types.ts), and independent checks of the numbers the orchestration passes on
(hand formulas, brute force, the model functions called directly)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from bakumine.blast import bench, timing, vibration
from bakumine.blast.model import BlastDesign
from bakumine.blast.simulate import (
    MAX_BLOCKS,
    SimulationRequest,
    defaults,
    plain,
    simulate,
    validate_bundle,
)
from bakumine.blast.throw import Heightmap

# web/src/blast/types.ts, interface by interface
TOP = ["format", "version", "design", "summary", "timeline", "waves", "ppv_surface", "damage",
       "fragmentation", "throw", "muckpile", "flyrock", "airblast", "monitors",
       "envelope"]  # fmt: skip
SUMMARY = ["volume_m3", "tonnes", "explosive_kg", "powder_factor_kg_m3", "powder_factor_kg_t",
           "drilled_m", "holes", "x50_m", "uniformity_n", "p80_m", "oversize_fraction", "mic_kg",
           "duration_ms", "flyrock_max_m", "clearance_m", "backbreak_m", "models"]  # fmt: skip
HOLE = ["id", "row", "t_fire_ms", "collar", "toe", "charge_top", "charge_bottom", "charge_kg",
        "diameter_m"]  # fmt: skip
MONITOR = ["name", "xyz", "distance_m", "charge_kg", "scaled_distance", "ppv_mm_s", "limit_mm_s",
           "ok", "airblast_dbl", "airblast_limit_dbl", "airblast_ok"]  # fmt: skip
SECTIONS = {
    "timeline": ["t_end_ms", "holes"],
    "waves": ["cp_m_s", "cs_m_s", "k", "beta", "sources"],
    "ppv_surface": ["origin", "spacing", "shape", "z", "ppv_mm_s", "arrival_ms"],
    "damage": ["classes", "thresholds_mm_s", "blocks"],
    "fragmentation": ["sizes_m", "passing", "x50_m", "n", "xc_m"],
    "throw": ["g", "particles"],
    "muckpile": ["origin", "spacing", "shape", "z_before", "z_after"],
    "flyrock": ["face_burst_m", "cratering_m", "stemming_ejection_m", "max_range_m",
                "clearance_m", "safety_factor"],
    "airblast": ["k", "a", "rings"],
    "envelope": ["window_ms", "bins", "mic_kg"],
}  # fmt: skip
BLOCKS = ["centre", "size", "ppv_mm_s", "cls", "t_break_ms", "blasted"]
PARTICLES = ["p0", "v0", "t0_ms", "t_land_ms", "p_land", "p_rest", "size_m", "kind"]

ANFO_DENSITY = 850.0


def small(**changes):
    """A quick request: a 30 x 14 m block, 0.2 m holes, few particles."""
    request = defaults()
    request.update(
        polygon=bench.rectangle((100.0, 200.0), 30.0, 14.0, 180.0), diameter=0.2, particles=400
    )
    request.update(changes)
    return request


@pytest.fixture(scope="module")
def demo():
    return simulate(defaults())


@pytest.fixture(scope="module")
def quick():
    return simulate(small(seed=3))


@pytest.fixture(scope="module")
def design(demo):
    return BlastDesign.from_dict(demo["design"])


class Plane:
    """Ground that rises to the north: z = 500 + 0.05 (y - 200)."""

    bounds = np.array([[-200.0, -100.0, 485.0], [400.0, 500.0, 515.0]])

    def z_at(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        return 500.0 + 0.05 * (y - 200.0)


# ------------------------------------------------------------------------------ contract


def test_bundle_is_valid_plain_json(demo):
    assert validate_bundle(demo) == []
    text = json.dumps(demo, allow_nan=False)
    assert json.loads(text) == demo
    assert demo["format"] == "bakumine-blast-result" and demo["version"] == 1


def test_bundle_has_every_key_of_the_web_contract(demo):
    assert set(TOP) <= set(demo)
    assert set(SUMMARY) <= set(demo["summary"])
    for name, keys in SECTIONS.items():
        assert set(keys) <= set(demo[name]), name
    assert set(demo["damage"]["blocks"]) == set(BLOCKS)
    assert set(demo["throw"]["particles"]) == set(PARTICLES)
    assert all(set(h) == set(HOLE) for h in demo["timeline"]["holes"])
    assert all(set(m) == set(MONITOR) for m in demo["monitors"]) and demo["monitors"]
    assert all(set(s) == {"hole", "t_ms", "centre", "charge_kg"} for s in demo["waves"]["sources"])
    assert all(set(r) == {"radius_m", "pa", "dbl"} for r in demo["airblast"]["rings"])
    assert all(set(b) == {"t_ms", "kg", "holes"} for b in demo["envelope"]["bins"])
    # numbers the viewer needs are numbers, not null
    for key in SUMMARY[:-1]:
        assert isinstance(demo["summary"][key], int | float), key
    assert all(isinstance(v, float) for v in demo["flyrock"].values())
    assert all(isinstance(m["calibrated"], bool) for m in demo["summary"]["models"].values())
    assert all(m["calibrated"] is False for m in demo["summary"]["models"].values())
    assert any("not calibrated" in note for note in demo["summary"]["notes"])
    assert any("launch velocity" in note for note in demo["summary"]["notes"])


def test_lengths_are_consistent(demo):
    for name, keys in (("ppv_surface", ["z", "ppv_mm_s", "arrival_ms"]),
                       ("muckpile", ["z_before", "z_after"])):  # fmt: skip
        nx, ny = demo[name]["shape"]
        assert all(len(demo[name][k]) == nx * ny for k in keys)
    assert max(demo["ppv_surface"]["shape"]) <= 150
    blocks = demo["damage"]["blocks"]
    assert len({len(blocks[k]) for k in BLOCKS}) == 1 and 0 < len(blocks["cls"]) < MAX_BLOCKS
    particles = demo["throw"]["particles"]
    assert {len(particles[k]) for k in PARTICLES} == {3000}
    assert set(particles["kind"]) == {0}
    ids = [h["id"] for h in demo["timeline"]["holes"]]
    assert len(ids) == len(set(ids)) == demo["summary"]["holes"]
    assert {s["hole"] for s in demo["waves"]["sources"]} == set(ids)
    assert len(demo["fragmentation"]["sizes_m"]) == len(demo["fragmentation"]["passing"])


def test_validate_bundle_reports_problems(demo):
    broken = json.loads(json.dumps(demo))
    del broken["waves"]["cp_m_s"]
    assert validate_bundle(broken) == ["waves.cp_m_s is missing"]
    broken = json.loads(json.dumps(demo))
    broken["ppv_surface"]["z"].pop()
    broken["throw"]["particles"]["kind"].pop()
    broken["summary"]["backbreak_m"] = None
    broken["waves"]["sources"][0]["hole"] = "nope"
    problems = " | ".join(validate_bundle(broken))
    for text in ("ppv_surface.z has", "throw.particles.kind has", "summary.backbreak_m",
                 "'nope' is not a timeline hole"):  # fmt: skip
        assert text in problems
    assert validate_bundle({"format": "x"})[0].startswith("format is")
    assert validate_bundle(None) == ["the bundle is not an object"]


def test_plain_replaces_what_json_cannot_hold():
    out = plain({"a": np.array([1.0, np.nan, np.inf]), "b": (np.int64(2), np.bool_(True)),
                 "c": math.nan, "d": np.array([[1, 2]]), "e": np.float64(1.5)})  # fmt: skip
    assert out == {"a": [1.0, None, None], "b": [2, True], "c": None, "d": [[1, 2]], "e": 1.5}
    assert type(out["b"][0]) is int and type(out["b"][1]) is bool


# ------------------------------------------------------------------------------ physics


def test_explosive_mass_and_powder_factor_by_hand(demo):
    total = 0.0
    for h in demo["timeline"]["holes"]:
        length = math.dist(h["charge_top"], h["charge_bottom"])
        mass = math.pi / 4.0 * h["diameter_m"] ** 2 * length * ANFO_DENSITY
        assert h["charge_kg"] == pytest.approx(mass, rel=1e-12)
        total += mass
    s = demo["summary"]
    assert s["explosive_kg"] == pytest.approx(total, rel=1e-12)
    volume = 60.0 * 35.0 * 10.0  # the demo block
    assert s["volume_m3"] == pytest.approx(volume, rel=1e-12)
    assert s["powder_factor_kg_m3"] == pytest.approx(total / volume, rel=1e-12)
    assert s["tonnes"] == pytest.approx(volume * 2.7, rel=1e-12)
    assert s["powder_factor_kg_t"] == pytest.approx(total / (volume * 2.7), rel=1e-12)
    # Konya and Walter: B = 0.012 (2 SGe / SGr + 1.5) De[mm]; S = 1.15 B; T = 0.7 B; J = 0.3 B
    burden = 0.012 * (2 * 0.85 / 2.7 + 1.5) * 165.0
    assert s["burden_m"] == pytest.approx(burden, rel=1e-12)
    assert s["spacing_m"] == pytest.approx(1.15 * burden, rel=1e-12)
    h = demo["timeline"]["holes"][0]
    assert h["collar"][2] - h["charge_top"][2] == pytest.approx(0.7 * burden, rel=1e-9)
    assert 990.0 - h["toe"][2] == pytest.approx(0.3 * burden, rel=1e-9)
    assert s["drilled_m"] == pytest.approx(s["holes"] * (10.0 + 0.3 * burden), rel=1e-12)


def _landing_ground(demo):
    """The ground the rock lands on, rebuilt from the bundle: ``muckpile.z_before`` with the
    blasted block taken out down to the floor."""
    pile = demo["muckpile"]
    nx, ny = pile["shape"]
    z = np.array(pile["z_before"], dtype=float).reshape(nx, ny)
    gx = pile["origin"][0] + pile["spacing"] * np.arange(nx)
    gy = pile["origin"][1] + pile["spacing"] * np.arange(ny)
    poly = np.array(demo["design"]["polygon"])
    (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)  # the demo block: a rectangle
    inside = ((gx > x0) & (gx < x1))[:, None] & ((gy > y0) & (gy < y1))[None, :]
    z[inside] = demo["design"]["floor_z"]
    return Heightmap(tuple(pile["origin"]), pile["spacing"], z), inside


def test_particles_fly_ballistically_and_land_on_the_ground(demo):
    p = {k: np.array(v, dtype=float) for k, v in demo["throw"]["particles"].items()}
    g = demo["throw"]["g"]
    dt = (p["t_land_ms"] - p["t0_ms"])[:, None] / 1000.0
    assert (dt >= 0).all()
    end = p["p0"] + p["v0"] * dt - 0.5 * g * dt**2 * np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(end, p["p_land"], atol=1e-6)
    ground, _ = _landing_ground(demo)
    z_ground = ground.z_at(p["p_land"][:, 0], p["p_land"][:, 1])
    flew = dt[:, 0] > 0
    np.testing.assert_allclose(z_ground[flew], p["p_land"][flew, 2], atol=1e-6)
    # the few that start under the gridded wall of the block do not fly: they stay put
    assert (~flew).sum() == demo["summary"]["particles_not_thrown"] < 0.01 * len(flew)
    np.testing.assert_array_equal(p["p_land"][~flew], p["p0"][~flew])
    assert (z_ground[~flew] >= p["p0"][~flew, 2]).all()
    # the rock is thrown out over the face (azimuth 180 = toward -y), and rests on the pile
    assert np.mean(p["p_land"][:, 1] - p["p0"][:, 1]) < -5.0
    np.testing.assert_allclose(p["p_rest"][:, :2], p["p_land"][:, :2])
    assert (p["p_rest"][:, 2] >= p["p_land"][:, 2] - 1e-9).all()
    # start inside the block, between floor and top
    assert (p["p0"][:, 2] >= 990.0).all() and (p["p0"][:, 2] <= 1000.0).all()


def test_muckpile_volume_is_swell_times_blasted_volume(demo):
    ground, _ = _landing_ground(demo)
    pile = demo["muckpile"]
    after = np.array(pile["z_after"], dtype=float).reshape(pile["shape"])
    thickness = after - ground.z
    assert thickness.min() >= -1e-9
    volume = thickness.sum() * pile["spacing"] ** 2
    assert volume == pytest.approx(1.3 * 21000.0, rel=1e-9)
    assert demo["summary"]["muckpile_volume_m3"] == pytest.approx(volume, rel=1e-9)
    # before the blast the bench is intact: top behind the face, floor in front
    before = np.array(pile["z_before"], dtype=float)
    assert set(np.unique(before)) == {990.0, 1000.0}


def test_ppv_surface_is_the_site_law_and_decays(demo, design):
    surf = demo["ppv_surface"]
    nx, ny = surf["shape"]
    gx = surf["origin"][0] + surf["spacing"] * np.arange(nx)
    gy = surf["origin"][1] + surf["spacing"] * np.arange(ny)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")  # index ix * ny + iy
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.array(surf["z"], dtype=float)])
    ppv = np.array(surf["ppv_mm_s"], dtype=float)
    centres = np.array([s["centre"] for s in demo["waves"]["sources"]])
    fire = np.array([s["t_ms"] for s in demo["waves"]["sources"]])
    k, beta, mic = demo["waves"]["k"], demo["waves"]["beta"], demo["summary"]["mic_kg"]
    rng = np.random.default_rng(1)
    for i in rng.choice(len(nodes), 40, replace=False):
        d = np.linalg.norm(centres - nodes[i], axis=1)
        assert ppv[i] == pytest.approx(k * (d.min() / math.sqrt(mic)) ** -beta, rel=1e-10)
        arrival = np.min(fire + d / demo["waves"]["cp_m_s"] * 1000.0)
        assert surf["arrival_ms"][i] == pytest.approx(arrival, rel=1e-10)
    # draped on the ground of the demo bench
    assert set(np.unique(nodes[:, 2])) == {990.0, 1000.0}
    # decreasing away from the pattern on average
    distance = np.linalg.norm(
        nodes[:, :2] - np.array(demo["design"]["polygon"]).mean(axis=0), axis=1
    )
    rings = [ppv[(distance >= lo) & (distance < lo + 30.0)].mean() for lo in (40, 70, 100, 130)]
    assert all(a > b for a, b in zip(rings, rings[1:], strict=False))
    assert vibration.ppv_at(nodes[:5], design).ppv_mm_s == pytest.approx(ppv[:5], rel=1e-12)


def test_monitors_equal_the_vibration_module(demo, design):
    (monitor,) = demo["monitors"]
    point = np.array([monitor["xyz"]])
    direct = vibration.ppv_at(point, design)
    assert monitor["ppv_mm_s"] == pytest.approx(float(direct.ppv_mm_s[0]), rel=1e-12)
    air = vibration.airblast_at(point, design)
    assert monitor["airblast_dbl"] == pytest.approx(float(air.dbl[0]), rel=1e-12)
    # and the law by hand
    r, w = monitor["distance_m"], monitor["charge_kg"]
    assert w == pytest.approx(demo["summary"]["mic_kg"])
    assert monitor["scaled_distance"] == pytest.approx(r / math.sqrt(w), rel=1e-12)
    assert monitor["ppv_mm_s"] == pytest.approx(1140.0 * (r / math.sqrt(w)) ** -1.6, rel=1e-12)
    assert monitor["ok"] is (monitor["ppv_mm_s"] <= monitor["limit_mm_s"])
    assert monitor["airblast_ok"] is (monitor["airblast_dbl"] <= monitor["airblast_limit_dbl"])
    # rings: P [kPa] = 100 (R / W^(1/3))^-1.45, dBL = 20 log10(P / 20 uPa)
    for ring in demo["airblast"]["rings"]:
        pa = 1000.0 * 100.0 * (ring["radius_m"] / w ** (1 / 3)) ** -1.45
        assert ring["pa"] == pytest.approx(pa, rel=1e-12)
        assert ring["dbl"] == pytest.approx(20.0 * math.log10(pa / 20e-6), rel=1e-12)
    assert [r["radius_m"] for r in demo["airblast"]["rings"]] == [50, 100, 200, 500, 1000]


def test_envelope_is_the_charge_per_delay(demo, design):
    env = demo["envelope"]
    mic, ids = timing.max_instantaneous_charge(design)
    assert env["window_ms"] == 8 and env["mic_kg"] == pytest.approx(mic, rel=1e-12)
    assert demo["summary"]["mic_kg"] == env["mic_kg"]
    assert max(b["kg"] for b in env["bins"]) == pytest.approx(mic, rel=1e-12)
    # brute force: every bin holds exactly the holes firing in [t, t + 8 ms)
    holes = demo["timeline"]["holes"]
    for b in env["bins"]:
        inside = [h for h in holes if b["t_ms"] <= h["t_fire_ms"] < b["t_ms"] + 8.0]
        assert sorted(b["holes"]) == sorted(h["id"] for h in inside)
        assert b["kg"] == pytest.approx(sum(h["charge_kg"] for h in inside), rel=1e-12)
    assert {h for b in env["bins"] for h in b["holes"]} == {h["id"] for h in holes}
    assert demo["summary"]["duration_ms"] == max(h["t_fire_ms"] for h in holes)
    assert demo["timeline"]["t_end_ms"] >= max(demo["throw"]["particles"]["t_land_ms"])


def test_damage_classes_follow_the_thresholds(demo):
    dmg = demo["damage"]
    assert dmg["classes"] == ["intact", "damaged", "fractured", "crushed"]
    limits = np.array(dmg["thresholds_mm_s"])
    # PPV_c = sigma_t cp / E = 10e6 * cp / 50e9 m/s; classes at 0.4, 1 and 4 times it
    critical = 10e6 * demo["waves"]["cp_m_s"] / 50e9 * 1000.0
    np.testing.assert_allclose(limits, [0.4 * critical, critical, 4.0 * critical], rtol=1e-12)
    blocks = dmg["blocks"]
    ppv = np.array(blocks["ppv_mm_s"])
    np.testing.assert_array_equal(blocks["cls"], (ppv[:, None] > limits[None, :]).sum(axis=1))
    centre, blasted = np.array(blocks["centre"]), np.array(blocks["blasted"])
    poly = np.array(demo["design"]["polygon"])
    inside = (
        (centre[:, 0] > poly[:, 0].min()) & (centre[:, 0] < poly[:, 0].max())
        & (centre[:, 1] > poly[:, 1].min()) & (centre[:, 1] < poly[:, 1].max())
        & (centre[:, 2] >= 990.0) & (centre[:, 2] <= 1000.0)
    )  # fmt: skip
    np.testing.assert_array_equal(blasted, inside)
    # blasted volume in blocks ~ the block (cubes cut by the outline: within a block layer)
    volume = np.prod(np.array(blocks["size"])[blasted], axis=1).sum()
    assert volume == pytest.approx(21000.0, rel=0.12)
    # every blasted block breaks, at or after the first shot; no block is in the air
    t_break = np.array([np.nan if t is None else t for t in blocks["t_break_ms"]])
    assert np.isfinite(t_break[blasted]).all() and np.nanmin(t_break) >= 0.0
    front = centre[:, 1] < poly[:, 1].min()  # in front of the face (azimuth 180)
    assert front.any() and (centre[front, 2] < 990.0).all()
    assert (centre[:, 2] < 1000.0).all()
    assert 0.0 < demo["summary"]["backbreak_m"] < 30.0


def test_fragmentation_is_rosin_rammler(demo):
    frag = demo["fragmentation"]
    sizes, passing = np.array(frag["sizes_m"]), np.array(frag["passing"])
    want = 1.0 - np.exp(-math.log(2.0) * (sizes / frag["x50_m"]) ** frag["n"])
    np.testing.assert_allclose(passing, want, atol=1e-12)
    s = demo["summary"]
    assert (s["x50_m"], s["uniformity_n"]) == (frag["x50_m"], frag["n"])
    assert 1.0 - math.exp(-math.log(2.0) * (s["p80_m"] / s["x50_m"]) ** frag["n"]) == pytest.approx(
        0.8
    )
    assert s["oversize_fraction"] == pytest.approx(
        math.exp(-math.log(2.0) * (1.0 / s["x50_m"]) ** frag["n"]), rel=1e-9
    )
    # the particle sizes are drawn from that curve
    assert np.median(demo["throw"]["particles"]["size_m"]) == pytest.approx(frag["x50_m"], rel=0.1)


def test_flyrock_and_summary_agree(demo):
    fly, s = demo["flyrock"], demo["summary"]
    assert fly["max_range_m"] == max(
        fly["face_burst_m"], fly["cratering_m"], fly["stemming_ejection_m"]
    )
    assert fly["clearance_m"] == pytest.approx(fly["safety_factor"] * fly["max_range_m"])
    assert (s["flyrock_max_m"], s["clearance_m"]) == (fly["max_range_m"], fly["clearance_m"])
    # Richards and Moore cratering: L = k^2 / g (sqrt(m) / stemming)^2.6
    m = math.pi / 4.0 * 0.165**2 * ANFO_DENSITY
    crater = 27.0**2 / 9.81 * (math.sqrt(m) / s["stemming_m"]) ** 2.6
    assert fly["cratering_m"] == pytest.approx(crater, rel=1e-9)


# ------------------------------------------------------------------------------ requests


def test_deterministic_for_a_seed(quick):
    a, b = quick, simulate(small(seed=3))
    assert a == b


def test_request_round_trip_and_site_constants(quick):
    request = SimulationRequest.from_dict(small(unknown_key=1))
    assert SimulationRequest.from_dict(request.to_dict()) == request
    json.dumps(defaults(), allow_nan=False)
    site = {"vibration": {"k": 500.0, "beta": 1.4}, "flyrock_k": 13.5,
            "hp": {"k": 400.0, "alpha": 0.8, "beta": 1.6}}  # fmt: skip
    wall = {"name": "Wall", "xyz": [100.0, 400.0, 1000.0]}
    base, out = (
        quick,
        simulate(small(site=site, burden=5.0, stemming=4.0, subdrill=1.0, seed=4, monitors=[wall])),
    )
    assert (out["waves"]["k"], out["waves"]["beta"]) == (500.0, 1.4)
    models = out["summary"]["models"]
    assert models["flyrock"]["k"] == 13.5 and models["damage"]["alpha"] == 0.8
    assert (
        models["design"]["burden"] == "given"
        and base["summary"]["models"]["design"]["burden"] == "rule"
    )
    assert out["design"]["burden"] == 5.0 and out["design"]["spacing"] == pytest.approx(5.75)
    h = out["timeline"]["holes"][0]
    assert h["collar"][2] - h["charge_top"][2] == pytest.approx(4.0)
    assert out["design"]["floor_z"] - h["toe"][2] == pytest.approx(1.0)
    assert out["flyrock"]["cratering_m"] < base["flyrock"]["cratering_m"]
    # another seed: other particles; a monitor without limits has no verdict
    assert out["throw"]["particles"]["v0"] != base["throw"]["particles"]["v0"]
    (monitor,) = out["monitors"]
    assert monitor["limit_mm_s"] is None and monitor["ok"] is None
    assert monitor["airblast_limit_dbl"] is None and monitor["airblast_ok"] is None
    assert monitor["ppv_mm_s"] > 0


@pytest.mark.parametrize("kind", ["echelon", "v_cut", "hole_by_hole"])
def test_every_timing_type_runs(kind, quick):
    assert quick["design"]["notes"]["timing"]["scheme"] == "row_by_row"
    out = simulate(small(timing={"type": kind, "inter_hole_ms": 17, "inter_row_ms": 42}))
    assert out["design"]["notes"]["timing"]["scheme"] == kind
    times = [h["t_fire_ms"] for h in out["timeline"]["holes"]]
    assert min(times) == 0.0
    if kind == "hole_by_hole":
        assert len(set(times)) == len(times)
        assert out["summary"]["mic_kg"] == pytest.approx(
            max(h["charge_kg"] for h in out["timeline"]["holes"])
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pattern": "hex"}, "pattern must be"),
        ({"diameter": 165}, "diameter is in metres"),
        ({"bench_height": 0}, "bench_height must be > 0"),
        ({"explosive": "TNT"}, "explosive must be"),
        ({"polygon": [[0, 0], [1, 1]]}, "at least three"),
        ({"timing": {"type": "spiral"}}, "timing.type"),
        ({"timing": {"type": "hole_by_hole", "inter_hole_ms": 20, "inter_row_ms": 40}}, "same"),
        ({"rock": {"density": -1}}, "rock density"),
        ({"monitors": [{"name": "a", "xyz": [1, 2]}]}, "xyz must be"),
        ({"particles": 0}, "particles must be"),
        ({"stemming": 0}, "stemming must be > 0"),
        ({"stemming": 50}, "no hole has room for a charge"),
        ({"burden": 0.2, "spacing": 0.2, "polygon": bench.rectangle((0, 0), 300, 300, 0)}, "limit"),
        ({"site": {"vibration": {"k": -1}}}, "site.vibration.k"),
    ],
)
def test_invalid_requests_say_why(changes, message):
    with pytest.raises(ValueError, match=message):
        simulate(small(**changes))


def test_request_must_be_an_object():
    with pytest.raises(ValueError, match="JSON object"):
        simulate([1, 2])


# ------------------------------------------------------------------------------ site


def test_demo_site_and_polygons():
    site = bench.demo_site()
    poly = np.array(site.polygon)
    assert BlastDesign("x", [], polygon=[tuple(p) for p in poly]).plan_area == pytest.approx(2100.0)
    # azimuth 180: the face is the south edge; top on and behind it, floor in front
    south = poly[:, 1].min()
    assert (
        site.ground.z_at(500.0, south) == 1000.0 and site.ground.z_at(500.0, south + 20) == 1000.0
    )
    assert site.ground.z_at(500.0, south - 0.01) == 990.0 and site.floor_z == 990.0
    (x0, y0), (x1, y1) = site.bounds
    assert south - y0 >= 120.0 and y1 - poly[:, 1].max() >= 60.0
    assert poly[:, 0].min() - x0 >= 60.0 and x1 - poly[:, 0].max() >= 60.0
    # a rotated face: 60 m along the face, 35 m behind it, face looking east
    east = np.array(bench.rectangle((0.0, 0.0), 60.0, 35.0, 90.0))
    assert np.ptp(east[:, 0]) == pytest.approx(35.0) and np.ptp(east[:, 1]) == pytest.approx(60.0)
    centred = np.array(bench.default_polygon(Plane()))
    np.testing.assert_allclose(centred.mean(axis=0), [100.0, 200.0])
    front = bench.front_of_polygon([0.0, 40.0, 0.0, 0.0, 0.0], [10.0, 10.0, -18.0, 0.0, 200.0],
                                   east, 90.0, 150.0)  # fmt: skip
    # points: inside the block, ahead of it, beside it, behind... x = east is the throw axis
    assert front.tolist() == [False, True, False, False, False]


def test_cut_bench_only_lowers_the_front():
    poly = bench.rectangle((100.0, 200.0), 30.0, 14.0, 180.0)
    ground = Plane()
    floor = 490.0
    cut = bench.cut_bench(ground, poly, floor, 180.0)
    nx, ny = cut.shape
    gx = cut.origin[0] + cut.spacing * np.arange(nx)
    gy = cut.origin[1] + cut.spacing * np.arange(ny)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    before = ground.z_at(xx, yy)
    assert (cut.z <= before + 1e-12).all()
    # the apron: the block's width (x 85..115), from the face (y 193) 150 m to the south;
    # nodes exactly on its outline may fall either way
    front = (xx > 85.001) & (xx < 114.999) & (yy < 192.999) & (yy > 43.001)
    outside = (xx < 85.0) | (xx > 115.0) | (yy > 193.0) | (yy < 43.0)
    np.testing.assert_array_equal(cut.z[front], np.minimum(before[front], floor))
    np.testing.assert_array_equal(cut.z[outside], before[outside])
    assert front.sum() > 4000
    assert (cut.z[front] == floor).any()
    # grid reaches 150 m in front and 60 m around; ground below the floor is not raised
    assert gy[0] <= 193.0 - 210.0 + 1e-9 and gy[-1] >= 207.0 + 60.0 - 1e-9
    low = bench.cut_bench(480.0, poly, floor, 180.0)
    assert (low.z == 480.0).all()


def test_blast_on_a_ground_surface_cuts_a_free_face():
    out = simulate(small(polygon=None, diameter=0.3, particles=300), ground=Plane())
    assert validate_bundle(out) == []
    d = out["design"]
    np.testing.assert_allclose(np.mean(d["polygon"], axis=0), [100.0, 200.0])  # default polygon
    assert d["floor_z"] == pytest.approx(500.0 - 10.0, abs=0.05)  # mean ground - bench height
    for h in out["timeline"]["holes"]:  # collars on the ground
        assert h["collar"][2] == pytest.approx(500.0 + 0.05 * (h["collar"][1] - 200.0), abs=1e-9)
    notes = " ".join(out["summary"]["notes"])
    assert "Free face excavated in front of the block" in notes and "No polygon given" in notes
    assert out["summary"]["models"]["site"]["free_face_cut_m"] > 5.0
    # the muckpile's "before" surface has the floor in front of the block
    pile = out["muckpile"]
    grid = Heightmap(tuple(pile["origin"]), pile["spacing"],
                     np.array(pile["z_before"], dtype=float).reshape(pile["shape"]))  # fmt: skip
    assert float(grid.z_at(100.0, 170.0)) == pytest.approx(d["floor_z"], abs=1e-6)
    assert float(grid.z_at(100.0, 200.0)) == pytest.approx(500.0, abs=0.05)
    # volume: polygon area x mean bench height
    assert out["summary"]["volume_m3"] == pytest.approx(2100.0 * d["bench_height"], rel=1e-12)
    assert d["bench_height"] == pytest.approx(10.0, abs=0.3)


def test_blocks_of_a_block_model_are_used_and_capped():
    request = small(particles=200)
    edges = np.arange(60.0, 140.0, 1.0)
    xx, yy, zz = np.meshgrid(edges, np.arange(180.0, 220.0, 1.0), np.arange(985.5, 1006.0, 1.0),
                             indexing="ij")  # fmt: skip
    centres = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    sizes = np.ones_like(centres)
    out = simulate(request, blocks=(centres, sizes))
    blocks = out["damage"]["blocks"]
    assert len(blocks["cls"]) == MAX_BLOCKS and np.array(blocks["size"]).max() == 1.0
    assert out["summary"]["models"]["damage"]["blocks"]["source"] == "block model"
    assert any("evenly picked" in note for note in out["summary"]["notes"])
    c = np.array(blocks["centre"])
    assert c[:, 2].max() < 1000.0  # the model's blocks above the bench top are air: left out
    assert {tuple(row) for row in c.tolist()} <= {tuple(row) for row in centres.tolist()}
    # a model somewhere else: a generated grid, and the note says so
    away = simulate(request, blocks=(centres + [5000.0, 0.0, 0.0], sizes))
    assert away["summary"]["models"]["damage"]["blocks"]["source"] == "generated grid"
    assert any("no blocks around the blast" in note for note in away["summary"]["notes"])
    assert np.array(away["damage"]["blocks"]["size"]).min() == 2.5
