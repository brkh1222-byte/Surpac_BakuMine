"""P3.T3 acceptance: typed attributes and the safe expression engine."""

import warnings

import numpy as np
import pytest

from bakumine.core.blockmodel.attributes import MISSING_CODE, RESERVED_NAMES, Attribute
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.expressions import (
    MAX_LENGTH,
    CategoryValues,
    ExpressionError,
    ExpressionWarning,
    parse,
)

ORIGIN = (500_400.0, 5_300_250.0, 825.0)


def make(shape=(4, 3, 2), depth=1, **kw):
    return BlockModel(BlockModelDefinition(ORIGIN, (10.0, 12.0, 5.0), shape, max_depth=depth, **kw))


@pytest.fixture
def model():
    m = make()
    rng = np.random.default_rng(5)
    m.add_attribute("au", background=np.nan, unit="g/t", decimals=3, description="gold")
    m.add_attribute("density", dtype=np.float32, background=2.7)
    m.add_attribute("zone", "int", background=-1, dtype=np.int16)
    m.add_attribute("rock", "category", background="WASTE")
    m.split([3, 7, 11])
    n = m.n_blocks
    au = rng.lognormal(0.0, 1.0, n)
    au[rng.random(n) < 0.2] = np.nan
    m.set("au", au)
    m.set("zone", rng.integers(0, 4, n))
    m.set("rock", rng.choice(["ORE", "WASTE", "AIR"], n))
    m.set("rock", None, rng.random(n) < 0.1)
    return m


# ------------------------------------------------------------------ add / delete / rename


def test_add_list_delete_rename():
    m = make()
    spec = m.add_attribute("au", unit="g/t", decimals=2, description="gold grade")
    assert isinstance(spec, Attribute) and spec is m.attribute("au")
    assert (spec.kind, spec.dtype, spec.unit, spec.decimals) == ("float", np.float64, "g/t", 2)
    m.add_attribute("sg", dtype="float32", background=2.7)
    m.add_attribute("zone", "int")
    m.add_attribute("rock", "category")
    assert m.attribute_names == ["au", "sg", "zone", "rock"]
    assert list(m.attributes) == m.attribute_names and "au" in m and "nope" not in m
    assert m["sg"].dtype == np.float32 and m["zone"].dtype == np.int32
    assert m["rock"].dtype == np.int32
    with pytest.raises(ValueError, match="already exists"):
        m.add_attribute("au")

    m["au"][:] = np.arange(m.n_blocks)
    m.rename_attribute("au", "gold")
    assert m.attribute_names == ["gold", "sg", "zone", "rock"]  # position kept
    np.testing.assert_array_equal(m["gold"], np.arange(m.n_blocks))
    assert m.attribute("gold").name == "gold" and m.attribute("gold").unit == "g/t"
    with pytest.raises(KeyError, match="no attribute 'au'"):
        m["au"]
    with pytest.raises(ValueError, match="already exists"):
        m.rename_attribute("gold", "sg")
    with pytest.raises(ValueError):
        m.rename_attribute("gold", "volume")
    m.rename_attribute("gold", "gold")

    nbytes = m.nbytes
    m.delete_attribute("sg")
    assert m.attribute_names == ["gold", "zone", "rock"] and m.nbytes == nbytes - 4 * m.n_blocks
    with pytest.raises(KeyError):
        m.delete_attribute("sg")
    with pytest.raises(KeyError):
        m.get("sg")


@pytest.mark.parametrize(
    "name", ["", "1au", "_au", "__class__", "au grade", "au-1", "x", "volume", "min", "where",
             "and", "AND", "Not", "lambda", "nan", "a" * 65, 5, None, "au.x", "import"],
)  # fmt: skip
def test_invalid_or_reserved_names_rejected(name):
    m = make()
    with pytest.raises(ValueError):
        m.add_attribute(name)
    assert m.attribute_names == []


def test_names_valid_in_python_grammar_are_accepted():
    m = make()
    for name in ("As", "AU", "au_ppm_2", "Z", "X", "type", "In"):  # case-sensitive
        m.add_attribute(name)
    assert {"x", "y", "z", "dx", "dy", "dz", "volume"} <= RESERVED_NAMES


def test_kind_and_dtype_validation():
    m = make()
    for bad in (
        dict(kind="string"),
        dict(kind="float", dtype=np.int32),
        dict(kind="int", dtype=np.float64),
        dict(kind="category", dtype=np.int64),
        dict(kind="int", background=1.5),
        dict(kind="int", dtype=np.int8, background=300),
        dict(kind="category", background=3),
        dict(kind="float", categories=["A"]),
        dict(kind="category", categories=["A", "A"]),
        dict(decimals=-1),
    ):
        with pytest.raises(ValueError):
            m.add_attribute("a", **bad)
    assert m.attribute_names == []


# ------------------------------------------------------------------ backgrounds


def test_backgrounds_at_creation_reset_and_in_new_sub_blocks():
    m = make()
    m.add_attribute("au")  # default background NaN = missing
    m.add_attribute("sg", background=2.7)
    m.add_attribute("zone", "int", background=-99)
    m.add_attribute("count", "int")
    m.add_attribute("rock", "category", background="WASTE")
    m.add_attribute("ox", "category")  # default background missing
    assert np.isnan(m["au"]).all() and m.is_missing("au").all()
    np.testing.assert_array_equal(m["sg"], 2.7)
    np.testing.assert_array_equal(m["zone"], -99)
    np.testing.assert_array_equal(m["count"], 0)
    assert (m.get("rock", decode=True) == "WASTE").all()
    assert (m["ox"] == MISSING_CODE).all() and m.is_missing("ox").all()
    assert (m.get("ox", decode=True) == None).all()  # noqa: E711
    assert not m.is_missing("zone").any()
    # sub-blocks of blocks where nothing was assigned hold the background
    m.set("sg", 3.1, [1])
    m.set("rock", "ORE", [1])
    result = m.split([0, 1])
    kids0 = np.flatnonzero(result.source == 0)
    kids1 = np.flatnonzero(result.source == 1)
    assert np.isnan(m["au"][kids0]).all()
    np.testing.assert_array_equal(m["sg"][kids0], 2.7)
    np.testing.assert_array_equal(m["zone"][kids0], -99)
    assert (m.get("rock", kids0, decode=True) == "WASTE").all()
    assert (m["ox"][kids0] == MISSING_CODE).all()
    # ... and where something was assigned they inherit it
    np.testing.assert_array_equal(m["sg"][kids1], 3.1)
    assert (m.get("rock", kids1, decode=True) == "ORE").all()
    # an attribute added after splitting covers the sub-blocks too
    m.add_attribute("late", background=5.0)
    assert len(m["late"]) == m.n_blocks and (m["late"] == 5.0).all()
    # reset
    m.reset("sg", kids1[:3])
    np.testing.assert_array_equal(m["sg"][kids1], [2.7, 2.7, 2.7, 3.1, 3.1, 3.1, 3.1, 3.1])
    m.reset("rock")
    assert (m.get("rock", decode=True) == "WASTE").all()
    assert m.attribute("rock").categories == ["WASTE", "ORE"]


# ------------------------------------------------------------------ get / set


def test_get_set_by_mask_and_ids():
    m = make()
    m.add_attribute("au", background=0.0)
    m.add_attribute("zone", "int")
    n = m.n_blocks
    mask = np.arange(n) % 3 == 0
    m.set("au", 1.5, mask)
    np.testing.assert_array_equal(m["au"], np.where(mask, 1.5, 0.0))
    m.set("au", np.arange(mask.sum()) + 10.0, mask)
    np.testing.assert_array_equal(m.get("au", mask), np.arange(mask.sum()) + 10.0)
    m.set("au", [7.0, 8.0], [5, 2])
    assert m["au"][5] == 7.0 and m["au"][2] == 8.0
    np.testing.assert_array_equal(m.get("au", [2, 5]), [8.0, 7.0])
    m.set("au", np.nan, [4])
    np.testing.assert_array_equal(np.flatnonzero(m.is_missing("au")), [4])
    assert m.get("au") is m["au"]  # no copy
    m.set("au", np.arange(n, dtype=float))
    np.testing.assert_array_equal(m["au"], np.arange(n))
    # integers: whole numbers only, range-checked, never silently truncated
    m.set("zone", [3.0, 4.0], [0, 1])
    np.testing.assert_array_equal(m["zone"][:2], [3, 4])
    with pytest.raises(ValueError, match="whole"):
        m.set("zone", 1.5)
    with pytest.raises(ValueError, match="whole"):
        m.set("zone", np.nan, [0])
    with pytest.raises(ValueError, match="fit"):
        m.set("zone", 2**40, [0])
    with pytest.raises(TypeError):
        m.set("au", "high", [0])
    with pytest.raises(ValueError):
        m.set("au", [1.0, 2.0, 3.0], [0, 1])
    with pytest.raises(IndexError):
        m.set("au", 1.0, [n])
    with pytest.raises(ValueError):
        m.get("au", np.ones(n + 1, dtype=bool))
    with pytest.raises(TypeError):
        m.get("au", [0.5])


def test_categorical_encode_decode():
    m = make()
    spec = m.add_attribute("rock", "category", dtype=np.int8, categories=["AIR"])
    n = m.n_blocks
    assert (m["rock"] == MISSING_CODE).all()
    values = np.array(["ORE", "WASTE", None, "AIR"], dtype=object)[np.arange(n) % 4]
    m.set("rock", values)
    assert m["rock"].dtype == np.int8
    # AIR keeps code 0; new strings get the next codes in sorted order of first assignment
    assert spec.categories == ["AIR", "ORE", "WASTE"]
    np.testing.assert_array_equal(m["rock"][:4], [1, 2, MISSING_CODE, 0])
    decoded = m.get("rock", decode=True)
    assert decoded.dtype == object
    assert all(a == b for a, b in zip(decoded, values, strict=True))  # None round-trips
    np.testing.assert_array_equal(m.is_missing("rock"), np.arange(n) % 4 == 2)
    # scalar assignment by mask; new category; codes of existing categories never change
    before = m["rock"].copy()
    m.set("rock", "DYKE", [0, 1])
    assert spec.categories == ["AIR", "ORE", "WASTE", "DYKE"]
    np.testing.assert_array_equal(m["rock"][2:], before[2:])
    assert list(m.get("rock", [0, 1, 2], decode=True)) == ["DYKE", "DYKE", None]
    m.set("rock", None, [0])
    assert m["rock"][0] == MISSING_CODE
    # numpy string arrays and integer codes are accepted; bad codes are not
    m.set("rock", np.array(["ORE", "AIR"]), [0, 1])
    m.set("rock", np.array([3, -1]), [2, 3])
    assert list(m.get("rock", [0, 1, 2, 3], decode=True)) == ["ORE", "AIR", "DYKE", None]
    with pytest.raises(ValueError, match="dictionary"):
        m.set("rock", np.array([9]), [0])
    with pytest.raises(TypeError):
        m.set("rock", [1.5], [0])
    with pytest.raises(TypeError):
        m.set("rock", ["A", 3], [0, 1])
    # the dictionary is limited by the code dtype
    small = make()
    small.add_attribute("c", "category", dtype=np.int8)
    with pytest.raises(ValueError, match="more categories"):
        small.set("c", [f"c{i}" for i in range(200)], np.arange(200) % small.n_blocks)
    # spec JSON round trip
    back = Attribute.from_dict(spec.to_dict())
    assert back == spec
    nan_spec = Attribute("au", unit="g/t", decimals=2)
    assert nan_spec.to_dict()["background"] is None
    assert np.isnan(Attribute.from_dict(nan_spec.to_dict()).background)


# ------------------------------------------------------------------ expressions


def test_calculated_attributes_equal_direct_numpy(model):
    m = model
    au, sg, zone = m["au"], m["density"].astype(np.float64), m["zone"].astype(np.float64)
    c, s = m.centroids(), m.sizes()
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    vol = s.prod(axis=1)
    rock = m.get("rock", decode=True)
    cases = {
        "au * density * volume": au * sg * vol,
        "(au + 1) ** 2 / 4 - -au": (au + 1) ** 2 / 4 + au,
        "2 ** 3 ** 2": np.full(m.n_blocks, 512.0),  # right-associative, as in Python
        "+au - 3": au - 3,
        "min(au, 1.5, zone)": np.minimum(np.minimum(au, 1.5), zone),
        "max(au, density)": np.maximum(au, sg),
        "abs(z - 830) + sqrt(volume)": np.abs(z - 830) + np.sqrt(vol),
        "log(au) + log10(au) + exp(-au)": np.log(au) + np.log10(au) + np.exp(-au),
        "floor(au) + ceil(au) + round(au) + round(au, 2)": (
            np.floor(au) + np.ceil(au) + np.round(au) + np.round(au, 2)
        ),
        "where(au > 1, au, 0)": np.where(au > 1, au, 0.0),
        "iif(isnan(au), -1, au)": np.where(np.isnan(au), -1.0, au),
        "where(zone == 2 or zone == 3, 1, where(rock == 'ORE', 2, 3))": np.where(
            (zone == 2) | (zone == 3), 1.0, np.where(rock == "ORE", 2.0, 3.0)
        ),
        "x - 500000 + (y - 5300000) * 2 + z": x - 500000 + (y - 5300000) * 2 + z,
        "dx * dy * dz - volume": s[:, 0] * s[:, 1] * s[:, 2] - vol,
        "dx / 2 + dy + dz": s[:, 0] / 2 + s[:, 1] + s[:, 2],
        "pi * 2": np.full(m.n_blocks, 2 * np.pi),
        "1e-3 + 0.5": np.full(m.n_blocks, 0.501),
        "(au > 1) * 2 + true": (au > 1) * 2.0 + 1.0,
    }
    for text, expected in cases.items():
        got = m.evaluate(text)
        assert got.dtype == np.float64 and got.shape == (m.n_blocks,), text
        np.testing.assert_array_equal(got, expected, err_msg=text)
    conditions = {
        "au > 0.5": au > 0.5,
        "0.5 <= au < 2": (au >= 0.5) & (au < 2),
        'au > 0.5 and rock == "ORE"': (au > 0.5) & (rock == "ORE"),
        "au > 0.5 AND rock = 'ORE'": (au > 0.5) & (rock == "ORE"),  # Surpac-style spelling
        "rock != 'ORE' Or NOT (zone >= 2)": (rock != "ORE") | ~(zone >= 2),
        "'AIR' == rock": rock == "AIR",
        "rock == 'NOT_IN_DICTIONARY'": np.zeros(m.n_blocks, dtype=bool),
        "rock != 'NOT_IN_DICTIONARY'": np.ones(m.n_blocks, dtype=bool),
        "isnan(rock)": rock == None,  # noqa: E711
        "isnan(au) or z > 830": np.isnan(au) | (z > 830),
        "not au": ~((au != 0) & ~np.isnan(au)),  # a missing number is not true
        "au != 1": au != 1,  # IEEE: NaN != 1
        "True and zone": zone != 0,
        "x > 500420 and dx < 10": (x > 500420) & (s[:, 0] < 10),
    }
    for text, expected in conditions.items():
        got = m.evaluate(text)
        assert got.dtype == bool and got.shape == (m.n_blocks,), text
        np.testing.assert_array_equal(got, expected, err_msg=text)
    assert (m.evaluate("dx < 10").sum(), m.evaluate("volume == 600").sum()) == (24, 21)

    # by mask / rows
    mask = m.evaluate("zone >= 2")
    np.testing.assert_array_equal(m.evaluate("au * 2", mask), (au * 2)[mask])
    np.testing.assert_array_equal(m.evaluate("x + volume", [4, 2]), (x + vol)[[4, 2]])
    assert m.evaluate("3", [1, 2]).tolist() == [3.0, 3.0]

    # calculate: new attribute, existing attribute, subset, kinds
    assert m.calculate("metal", "au * density * volume", unit="g") == 0
    np.testing.assert_array_equal(m["metal"], au * sg * vol)
    assert m.attribute("metal").description == "= au * density * volume"
    assert m.attribute("metal").unit == "g"
    m.calculate("metal", "0", m.evaluate("isnan(au)"))
    np.testing.assert_array_equal(m["metal"], np.where(np.isnan(au), 0.0, au * sg * vol))
    m.calculate("high", "au > 1", kind="int", dtype=np.int8)
    np.testing.assert_array_equal(m["high"], (au > 1).astype(np.int8))
    m.calculate("part", "au * 2", mask, background=-1.0)
    np.testing.assert_array_equal(m["part"], np.where(mask, au * 2, -1.0))
    m.calculate("f32", "density * 2", dtype=np.float32)
    assert m["f32"].dtype == np.float32
    np.testing.assert_array_equal(m["f32"], (sg * 2).astype(np.float32))
    with pytest.raises(ValueError, match="whole"):
        m.calculate("bad", "au", kind="int")  # NaN / fractions do not fit an integer
    assert "bad" not in m
    with pytest.raises(ValueError, match="category"):
        m.calculate("rock", "1")
    with pytest.raises(ValueError, match="category"):
        m.calculate("new_rock", "1", kind="category")
    with pytest.raises(ValueError, match="exists"):
        m.calculate("metal", "1", unit="t")
    with pytest.raises(ExpressionError):
        m.calculate("oops", "nothing + 1")
    assert "oops" not in m and "new_rock" not in m


MALICIOUS = [
    "__import__('os')",
    "__import__('os').system('echo hi')",
    "().__class__",
    "().__class__.__bases__[0].__subclasses__()",
    "au.__class__",
    "au.real",
    "au[0]",
    "au[0:2]",
    "lambda: 1",
    "(lambda: au)()",
    "[a for a in (1, 2)]",
    "{a for a in (1, 2)}",
    "(a for a in (1, 2))",
    "{1: 2}",
    "[1, 2]",
    "(1, 2)",
    "eval('1')",
    "exec('1')",
    "open('/etc/passwd')",
    "getattr(au, 'real')",
    "print(au)",
    "sum(au)",
    "len(au)",
    "abs",
    "abs + 1",
    "min(au)",
    "sqrt(au, au)",
    "where(au > 1, au)",
    "max(au, key=abs)",
    "max(*au)",
    "sqrt(**au)",
    "round(au, zone)",
    "au if au > 1 else 0",
    "au := 3",
    "(au := 3)",
    "au; 1",
    "au = ",
    "import os",
    "f'{au}'",
    "b'x' == rock",
    "None",
    "1j * au",
    "...",
    "au ^ 2",
    "au // 2",
    "au % 2",
    "au @ au",
    "au << 2",
    "au | zone",
    "~au",
    "au is zone",
    "au in zone",
    "au not in zone",
    "await au",
    "yield au",
    "*au",
    "_private + 1",
    "__builtins__",
    "__name__",
    "unknown_name + 1",
    "au + Au",
    "'text'",
    "'a' + 'b'",
    "au == 'ORE'",
    "rock == 3",
    "rock > 'ORE'",
    "rock + 1",
    "rock == rock",
    "'A' == 'A'",
    "rock",
    "sqrt(rock)",
    "where(rock == 'ORE', 'a', 'b')",
    "not rock",
    "rock and au",
    "",
    "   ",
    "au +",
    "(au",
    "au))",
    "au\n+ 1",
    "au\x00",
    "1 +" * 400 + "1",  # too long
    "-" * 300 + "au",  # too deep
    "not " * 150 + "au",  # too deep
    "max(" + ",".join(["au"] * 300) + ")",  # too many terms
    " + ".join(["au"] * 200),  # too many terms
    "+".join(["1"] * 300),  # too many terms / deep
    "sqrt(" * 150 + "au" + ")" * 150,
    3,
    None,
]


@pytest.mark.parametrize("text", MALICIOUS, ids=lambda t: repr(t)[:40])
def test_malicious_or_invalid_expressions_are_rejected(model, text):
    before = {name: model[name].copy() for name in model.attribute_names}
    with pytest.raises(ExpressionError) as excinfo:
        model.evaluate(text)
    message = str(excinfo.value)
    assert message and len(message) < 600  # a clear message, not a dump
    with pytest.raises(ExpressionError):
        model.calculate("result", text)
    assert "result" not in model
    for name, values in before.items():
        np.testing.assert_array_equal(model[name], values)


def test_rejection_messages_are_clear():
    def message(text, names=None):
        with pytest.raises(ExpressionError) as excinfo:
            parse(text, names)
        return str(excinfo.value)

    assert "attribute access" in message("au.real")
    assert "subscripting" in message("au[0]")
    assert "lambda" in message("lambda: 1")
    assert "unknown function '__import__'" in message("__import__('os')")
    assert "unknown function 'open'" in message("open('f')")
    assert "leading underscore" in message("_x + 1")
    assert "use ** for power" in message("au ^ 2")
    assert f"longer than {MAX_LENGTH}" in message("1+" * 600 + "1")
    assert "nested deeper" in message("-" * 300 + "au")
    assert "more than" in message("max(" + ",".join(["a"] * 300) + ")")
    assert parse(" + ".join(["au"] * 60)).names == {"au"}  # long sums are fine
    assert parse("(" * 60 + "au" + ")" * 60).names == {"au"}  # parentheses do not nest the tree
    assert "unknown name(s) 'cu'" in message("au + cu", names=["au"])
    assert "takes 2 to" not in message("min(au)") and "at least 2" in message("min(au)")
    assert "1 to 2" in message("round(au, 1, 2)")
    assert "is a function" in message("abs + 1")
    # nothing is executed by parsing: the tree is only inspected
    assert parse("au + 1", names=["au"]).names == {"au"}
    assert parse("where(x > 0, au, nan) + pi").names == {"x", "au"}


def test_huge_powers_do_not_hang_or_overflow_python_integers():
    m = make()
    assert m.evaluate("9 ** 9 ** 9 ** 9", return_invalid=True)[0][0] == np.inf
    literal = "9" * 400
    assert m.evaluate(literal)[0] == np.inf
    assert m.evaluate("-" + literal)[0] == -np.inf


def test_division_by_zero_and_log_of_negative_give_nan_inf_with_a_count():
    m = make(shape=(2, 2, 2), depth=0)
    m.add_attribute("a")
    m.add_attribute("b")
    m.set("a", [1.0, -1.0, 0.0, 4.0, np.nan, 2.0, -3.0, 5.0])
    m.set("b", [0.0, 0.0, 0.0, 2.0, 0.0, np.nan, 1.0, 0.5])
    a, b = m["a"], m["b"]
    with np.errstate(all="ignore"):
        values, n_invalid = m.evaluate("a / b", return_invalid=True)
        np.testing.assert_array_equal(values, a / b)
        assert values[0] == np.inf and values[1] == -np.inf and np.isnan(values[2])
        assert n_invalid == 3  # the NaN inputs (rows 4, 5) are missing data, not errors

        values, n_invalid = m.evaluate("log(a)", return_invalid=True)
        np.testing.assert_array_equal(values, np.log(a))
        assert n_invalid == 3  # log(-1), log(0), log(-3)

        values, n_invalid = m.evaluate("sqrt(a) + log10(a - 10)", return_invalid=True)
        np.testing.assert_array_equal(values, np.sqrt(a) + np.log10(a - 10))
        assert n_invalid == 2 + 7

        values, n_invalid = m.evaluate("exp(a * 1000) + 10 ** (a * 400)", return_invalid=True)
        assert n_invalid == 4 + 4 and np.isinf(values[[0, 3, 5, 7]]).all()
        values, n_invalid = m.evaluate("0 ** -1", return_invalid=True)
        assert np.isinf(values).all() and n_invalid == 8  # a constant counts for every block
        assert m.evaluate("a + 1 / 0", return_invalid=True)[1] == 8
        assert m.evaluate("(0 - 8) ** 0.5", return_invalid=True)[1] == 8

    assert m.evaluate("a / where(b == 0, nan, b)", return_invalid=True)[1] == 0
    # without return_invalid: a warning carrying the count, never an exception
    with pytest.warns(ExpressionWarning, match="3 value"):
        values = m.evaluate("a / b")
    assert np.isinf(values[:2]).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.evaluate("a / (b + 100)")  # nothing invalid: no warning
        assert m.calculate("ratio", "a / b") == 3  # calculate returns the count
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(m["ratio"], a / b)


def test_expression_engine_is_usable_without_a_model():
    expr = parse("where(rock == 'ORE', au * 2, 0) > 1")
    assert expr.names == {"rock", "au"} and "Expression(" in repr(expr)
    out = expr.evaluate(
        {
            "au": np.array([1.0, 0.2, 3.0]),
            "rock": CategoryValues(np.array([0, 0, 1]), ("ORE", "WASTE")),
        }
    )
    assert out.values.tolist() == [True, False, False] and out.n_invalid == 0
    assert parse("1 + 2").evaluate({}).values == 3.0
    with pytest.raises(ExpressionError, match="unknown name 'cu'.*available: au"):
        parse("cu + 1").evaluate({"au": 1.0})
    assert parse("a + 1") == parse("a + 1") and hash(parse("a")) == hash(parse("a"))
    seen = []
    parse("a + a * a").evaluate(lambda name: seen.append(name) or np.ones(2))
    assert seen == ["a"]  # each name looked up once


# ---- P7.T5 regression: round() with extreme digits
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("round(au, 400)", [0.2, 1.0, np.nan, 3.0, 1e10]),
        ("round(au, 300)", [0.2, 1.0, np.nan, 3.0, 1e10]),
        ("round(au, -400)", [0.0, 0.0, np.nan, 0.0, 0.0]),
        ("round(au, 2)", [0.2, 1.0, np.nan, 3.0, 1e10]),
    ],
)
def test_round_extreme_digits(source, expected):
    au = np.array([0.2, 1.0, np.nan, 3.0, 1e10])
    out = parse(source).evaluate({"au": au})
    np.testing.assert_array_equal(out.values, expected)
    assert out.n_invalid == 0


def test_round_extreme_scalar():
    assert parse("round(1e300, -400)").evaluate({}).values == 0.0


@pytest.mark.parametrize(
    ("digits", "expected"), [("1e300", 1.5), ("-10**30", 0.0), ("10**400", None), ("nan", None)]
)
def test_round_huge_digits(digits, expected):
    """Huge whole digits behave like +-400; inf / NaN digits are an ExpressionError
    (previously OverflowError escaped the CLI and API error handlers)."""
    expression = parse(f"round(au, {digits})")
    if expected is None:
        with pytest.raises(ExpressionError):
            expression.evaluate({"au": np.array([1.5])})
    else:
        assert expression.evaluate({"au": np.array([1.5])}).values[0] == expected
