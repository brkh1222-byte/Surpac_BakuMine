"""P2.T8 acceptance: Surpac .str text read/write.

The layout is PROVISIONAL (research.md section 5.6): these tests fix the
behaviour of our reader and writer against the documented layout, not against
a file exported by Surpac.
"""

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bakumine.core.io._text import BinaryFileError, FormatWarning
from bakumine.core.io.strings import AxisRecord, StringFile, StringHeader, Strings, StringSegment
from bakumine.core.io.surpac_str import (
    COORD_COLUMNS,
    SurpacFormatError,
    format_surpac_str,
    parse_surpac_str,
    read_surpac_str,
    write_surpac_str,
)

SAMPLE = """\
pit_design, 19-Nov-26, test strings, hand written sample
0, 5300000.000, 500000.000, 100.000, 5300100.000, 500000.000, 100.000
1, 5300000.456, 500000.123, 100.500, crest, bench 1
1, 5300010.000, 500020.000, 100.500, crest
1, 5300020.000, 500000.123, 101.000,
0, 0.000, 0.000, 0.000,
2, 5300100.000, 500100.000, 90.000, toe
2, 5300100.000, 500200.000, 90.000
2, 5300200.000, 500200.000, 90.000, , ramp
2, 5300100.000, 500100.000, 90.000, toe
0, 0.000, 0.000, 0.000,
0, 0.000, 0.000, 0.000, END
"""

# The same content as SAMPLE, as internal (east, north, up).
SAMPLE_SEG1 = np.array(
    [
        [500000.123, 5300000.456, 100.5],
        [500020.0, 5300010.0, 100.5],
        [500000.123, 5300020.0, 101.0],
    ]
)
SAMPLE_SEG2 = np.array(
    [
        [500100.0, 5300100.0, 90.0],
        [500200.0, 5300100.0, 90.0],
        [500200.0, 5300200.0, 90.0],
        [500100.0, 5300100.0, 90.0],
    ]
)


def quiet(text, **kwargs) -> StringFile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FormatWarning)
        return parse_surpac_str(text, **kwargs)


def as_written(points, decimals):
    """What a reader must return for ``points`` written with ``decimals``."""
    return np.array([[float(f"{v:.{decimals}f}") for v in p] for p in np.asarray(points).tolist()])


def random_strings(rng, n_segments=12) -> StringFile:
    words = ["crest", "toe", "bench 12", "ramp north side", "", "x", "a b  c"]
    segments = []
    for _ in range(n_segments):
        n = int(rng.integers(1, 40))
        pts = rng.uniform(-500.0, 500.0, size=(n, 3)) + [500_000.0, 5_300_000.0, 900.0]
        if n >= 4 and rng.random() < 0.5:
            pts[-1] = pts[0]  # closed
        descriptions = []
        for _ in range(n):
            k = int(rng.integers(0, 4))
            descriptions.append([words[int(rng.integers(len(words)))] for _ in range(k)])
        segments.append(StringSegment(int(rng.integers(1, 6)), pts, descriptions))
    header = StringHeader("pit", "19-Nov-26", "design", "memo with, a comma")
    axis = AxisRecord([500000.0, 5300000.0, 0.0], [500000.0, 5300100.0, 0.0])
    return StringFile(segments=segments, header=header, axis=axis)


# ------------------------------------------------------------ hand-written


def test_sample_parses_to_expected_arrays():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the canonical layout gives no warning
        sf = parse_surpac_str(SAMPLE)
    assert sf.warnings == [] and sf.skipped == []
    assert sf.header == StringHeader(
        "pit_design", "19-Nov-26", "test strings", "hand written sample"
    )
    assert [s.number for s in sf.segments] == [1, 2]
    np.testing.assert_array_equal(sf.segments[0].points, SAMPLE_SEG1)
    np.testing.assert_array_equal(sf.segments[1].points, SAMPLE_SEG2)
    assert sf.segments[0].points.dtype == np.float64
    assert sf.n_points == 7 and sf.string_numbers == [1, 2]


def test_first_coordinate_column_is_north():
    """The file's first coordinate column (Y) must land in internal y (north)."""
    assert COORD_COLUMNS == ("y", "x", "z")
    sf = parse_surpac_str(SAMPLE)
    first = sf.segments[0].points[0]
    # record: "1, 5300000.456, 500000.123, 100.500"
    assert first[1] == 5300000.456  # internal y = north = first column
    assert first[0] == 500000.123  # internal x = east = second column
    assert first[2] == 100.5
    # axis record goes through the same swap
    np.testing.assert_array_equal(sf.axis.start, [500000.0, 5300000.0, 100.0])
    np.testing.assert_array_equal(sf.axis.end, [500000.0, 5300100.0, 100.0])


def test_sample_descriptions_and_closed_flags():
    sf = parse_surpac_str(SAMPLE)
    s1, s2 = sf.segments
    assert s1.descriptions == (("crest", "bench 1"), ("crest",), ())
    assert s2.descriptions == (("toe",), (), ("", "ramp"), ("toe",))
    assert s1.description_column(0) == ["crest", "crest", ""]
    assert s2.description_column(1) == ["", "", "ramp", ""]
    assert not s1.is_closed
    assert s2.is_closed


def test_sample_rewrites_identically():
    """The writer's canonical form of the sample is the sample itself, but for
    the trailing comma that one sample line leaves out."""
    text = format_surpac_str(parse_surpac_str(SAMPLE))
    expected = SAMPLE.replace("90.000\n2, 5300200", "90.000,\n2, 5300200")
    assert text == expected


def test_strings_alias_and_from_arrays():
    assert Strings is StringFile
    sf = StringFile.from_arrays([SAMPLE_SEG1, SAMPLE_SEG2], numbers=[3, 7])
    assert [s.number for s in sf] == [3, 7] and len(sf) == 2
    assert sf.segments_of(7)[0].is_closed
    np.testing.assert_array_equal(sf.all_points(), np.vstack([SAMPLE_SEG1, SAMPLE_SEG2]))
    assert not sf.segments[0].points.flags.writeable


def test_closed_tolerance():
    ring = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1e-4]], dtype=float)
    s = StringSegment(1, ring)
    assert not s.is_closed
    assert s.closed(tolerance=1e-3)
    assert not StringSegment(1, [[0, 0, 0], [0, 0, 0]]).is_closed  # too short to be a ring


# -------------------------------------------------------------- round trip


@pytest.mark.parametrize("decimals", [0, 2, 3, 6])
def test_round_trip_lossless_at_written_precision(tmp_path, decimals):
    rng = np.random.default_rng(decimals)
    original = random_strings(rng)
    path = tmp_path / "a.str"
    write_surpac_str(path, original, decimals=decimals)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_surpac_str(path)

    assert back.header == original.header
    assert back.n_segments == original.n_segments
    for a, b in zip(original.segments, back.segments, strict=True):
        assert b.number == a.number
        assert b.descriptions == a.descriptions
        np.testing.assert_array_equal(b.points, as_written(a.points, decimals))
        assert np.abs(b.points - a.points).max() <= 0.5 * 10.0**-decimals + 1e-9
    np.testing.assert_array_equal(back.axis.start, as_written([original.axis.start], decimals)[0])
    # A second pass changes nothing: the file is a fixed point of read -> write.
    assert format_surpac_str(back, decimals=decimals) == path.read_text()
    again = parse_surpac_str(format_surpac_str(back, decimals=decimals))
    assert again.segments == back.segments


def test_round_trip_keeps_open_closed_and_empty_descriptions(tmp_path):
    closed = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 0, 0]], dtype=float)
    opened = closed[:3]
    descriptions = [["", "second only"], ["has  two spaces", "", "third"], [], [""]]
    original = StringFile(
        segments=[
            StringSegment(5, closed, descriptions),
            StringSegment(5, opened),
            StringSegment(9, closed[:1], [["single point"]]),
        ]
    )
    path = tmp_path / "b.str"
    write_surpac_str(path, original)
    back = read_surpac_str(path)
    assert back.segments == original.segments
    assert [s.is_closed for s in back.segments] == [True, False, False]
    assert back.segments[0].descriptions == (
        ("", "second only"),
        ("has  two spaces", "", "third"),
        (),
        (),  # a lone empty field is the same as no field
    )
    assert back.string_numbers == [5, 9]


def test_large_coordinates_keep_precision(tmp_path):
    pts = np.array(
        [
            [500_000.123, 5_300_000.456, 1234.567],
            [500_000.124, 5_300_000.457, 1234.568],
            [999_999.999, 9_999_999.999, -415.001],
        ]
    )
    path = tmp_path / "utm.str"
    write_surpac_str(path, [StringSegment(1, pts)])
    assert "1, 5300000.456, 500000.123, 1234.567," in path.read_text().splitlines()
    back = read_surpac_str(path).segments[0].points
    np.testing.assert_array_equal(back, pts)  # exact: 3 decimals are representable in text
    assert back[0, 0] == 500_000.123 and back[0, 1] == 5_300_000.456
    # millimetre differences survive
    np.testing.assert_allclose(back[1] - back[0], [0.001, 0.001, 0.001], atol=1e-9)

    write_surpac_str(path, [StringSegment(1, pts + 1e-7)], decimals=7)
    np.testing.assert_allclose(read_surpac_str(path).segments[0].points, pts + 1e-7, atol=6e-8)


def test_canonical_layout_and_windows_newlines(tmp_path):
    sf = StringFile(
        segments=[StringSegment(3, [[2.0, 1.0, -0.0001]], [["a b", "c"]])],
        header=StringHeader("loc", "date", "purpose", "memo"),
    )
    assert format_surpac_str(sf, decimals=2) == (
        "loc, date, purpose, memo\n"
        "0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00\n"
        "3, 1.00, 2.00, 0.00, a b, c\n"  # north first; no "-0.00"
        "0, 0.00, 0.00, 0.00,\n"
        "0, 0.00, 0.00, 0.00, END\n"
    )
    path = tmp_path / "w.str"
    write_surpac_str(path, sf, newline="\r\n")
    raw = path.read_bytes()
    assert raw.count(b"\r\n") == 5 and raw.count(b"\n") == 5
    assert read_surpac_str(path).segments == parse_surpac_str(format_surpac_str(sf)).segments


@pytest.mark.parametrize(
    "bad",
    [
        StringFile(segments=[StringSegment(0, [[1.0, 2.0, 3.0]])]),
        StringFile(segments=[StringSegment(-4, [[1.0, 2.0, 3.0]])]),
        StringFile(segments=[StringSegment(1, np.zeros((0, 3)))]),
        StringFile(segments=[StringSegment(1, [[1.0, np.nan, 3.0]])]),
        StringFile(segments=[StringSegment(1, [[1.0, 2.0, 3.0]], [["a,b"]])]),
        StringFile(segments=[StringSegment(1, [[1.0, 2.0, 3.0]], [["a\nb"]])]),
        StringFile(header=StringHeader(location="a,b")),
        StringFile(header=StringHeader(memo="two\nlines")),
    ],
)
def test_writer_is_strict(bad):
    with pytest.raises(SurpacFormatError):
        format_surpac_str(bad)


# ---------------------------------------------------------------- tolerant


def test_tolerant_whitespace_trailing_commas_blank_lines_crlf(tmp_path):
    messy = (
        "pit_design ,19-Nov-26,test strings,   hand written sample   \r\n"
        "\r\n"
        "0,5300000.000,500000.000,100.000,5300100.000,500000.000,100.000,\r\n"
        "   1 ,\t5300000.456 ,500000.123,   100.500,crest ,  bench 1,,,\r\n"
        "1,5300010.000,500020.000,100.500,crest,\r\n"
        "\r\n"
        "   \r\n"
        "1, 5300020.000, 500000.123, 101.000\r\n"
        "0,0,0,0,\r\n"
        "2 5300100.000 500100.000 90.000 toe\r\n"  # no commas at all
        "2, 5300100.000, 500200.000, 90.000,,,\r\n"
        "2, 5300200.000, 500200.000, 90.000, , ramp ,\r\n"
        "2.0, 5300100.000, 500100.000, 90.000, toe\r\n"
        "0, 0.000, 0.000, 0.000\r\n"
        "0, 0.000, 0.000, 0.000, end\r\n"
    )
    path = tmp_path / "messy.str"
    path.write_bytes(messy.encode())
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # all of this is accepted silently
        sf = read_surpac_str(path)
    expected = parse_surpac_str(SAMPLE)
    assert sf.segments == expected.segments
    assert sf.header == expected.header
    assert sf.axis == expected.axis
    assert sf.skipped == []


def test_tolerant_scientific_notation():
    text = (
        "h\n0,0,0,0,0,0,0\n"
        "1, 5.300000456E+06, 5.00000123e5, 1.005E2, sci\n"
        "1, 5.3D+06, 5.0d+05, -1.5E-3\n"
        "0, 0.0E0, 0.0, 0\n"
        "0, 0, 0, 0, END\n"
    )
    sf = parse_surpac_str(text)
    np.testing.assert_allclose(
        sf.segments[0].points,
        [[500000.123, 5300000.456, 100.5], [500000.0, 5300000.0, -0.0015]],
        rtol=0,
        atol=1e-9,
    )
    assert sf.segments[0].descriptions == (("sci",), ())
    assert sf.n_segments == 1 and sf.warnings == []


def test_missing_end_record_warns():
    text = SAMPLE.replace("0, 0.000, 0.000, 0.000, END\n", "")
    with pytest.warns(FormatWarning, match="no END record"):
        sf = parse_surpac_str(text)
    assert sf.segments == parse_surpac_str(SAMPLE).segments
    assert any("END" in w for w in sf.warnings)
    # also when the last segment has no separator either
    text = text.rstrip("\n").rsplit("\n", 1)[0] + "\n"
    with pytest.warns(FormatWarning):
        assert parse_surpac_str(text).segments == parse_surpac_str(SAMPLE).segments
    with pytest.raises(SurpacFormatError, match="END"):
        parse_surpac_str(text, strict=True)


def test_tolerant_structure_deviations():
    # no axis record
    lines = SAMPLE.splitlines()
    no_axis = "\n".join([lines[0]] + lines[2:]) + "\n"
    sf = quiet(no_axis)
    assert sf.axis is None and sf.segments == parse_surpac_str(SAMPLE).segments
    assert any("axis" in w for w in sf.warnings)

    # no header and no axis: the file starts with data
    sf = quiet("\n".join(lines[2:]) + "\n")
    assert sf.segments == parse_surpac_str(SAMPLE).segments
    assert any("header" in w for w in sf.warnings)

    # string number changes without a separator record
    last = "1, 5300020.000, 500000.123, 101.000,\n"
    merged = SAMPLE.replace(last + "0, 0.000, 0.000, 0.000,\n", last)
    sf = quiet(merged)
    assert sf.segments == parse_surpac_str(SAMPLE).segments
    assert any("without a separator" in w for w in sf.warnings)

    # extra header fields are kept in the memo
    tail = "\n0,0,0,0,0,0,0\n0,0,0,0,END\n"
    sf = parse_surpac_str("loc, date, purpose, memo, smooth=yes, extra" + tail)
    assert sf.header.memo == "memo, smooth=yes, extra"
    assert sf.header.location == "loc" and sf.n_segments == 0

    # header with fewer fields
    sf = parse_surpac_str("only a location\n0,0,0,0,0,0,0\n0,0,0,0,END\n")
    assert sf.header == StringHeader("only a location", "", "", "")


def test_unparseable_lines_are_skipped_and_reported():
    text = SAMPLE.replace(
        "2, 5300100.000, 500200.000, 90.000\n",
        "2, 5300100.000, 500200.000, 90.000\nthis is not a record\n2, 1.0, 2.0\n",
    ) + "trailing junk after END\n"
    with pytest.warns(FormatWarning, match="skipped"):
        sf = parse_surpac_str(text)
    assert sf.segments == parse_surpac_str(SAMPLE).segments
    assert [ln for _, ln in sf.skipped] == [
        "this is not a record",
        "2, 1.0, 2.0",
        "trailing junk after END",
    ]
    assert sf.skipped[0][0] == 9  # 1-based line number in the file
    with pytest.raises(SurpacFormatError, match="line 9"):
        parse_surpac_str(text, strict=True)


def test_binary_str_is_rejected(tmp_path):
    path = tmp_path / "binary.str"
    path.write_bytes(b"SURPAC\x00\x00\x01\x02" + bytes(range(256)) * 4)
    with pytest.raises(BinaryFileError, match=r"binary \.str is not supported; re-save as text"):
        read_surpac_str(path)


def test_latin1_and_bom(tmp_path):
    path = tmp_path / "enc.str"
    path.write_bytes("﻿Şahdağ, d, p, m\n0,0,0,0,0,0,0\n1, 2, 1, 3, qərb\n0,0,0,0,END\n".encode())
    sf = read_surpac_str(path)
    assert sf.header.location == "Şahdağ"
    assert sf.segments[0].descriptions == (("qərb",),)
    path.write_bytes("caf\xe9, d, p, m\n0,0,0,0,0,0,0\n0,0,0,0,END\n".encode("latin-1"))
    assert read_surpac_str(path).header.location == "café"


# ---------------------------------------------------------------- property

_description = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs", "Cc", "Zl", "Zp"), blacklist_characters=","
    ),
    max_size=12,
)
_coordinate = st.floats(min_value=-1e7, max_value=1e7, allow_nan=False, allow_infinity=False)


@st.composite
def _segments(draw):
    n = draw(st.integers(min_value=1, max_value=8))
    pts = draw(st.lists(st.tuples(_coordinate, _coordinate, _coordinate), min_size=n, max_size=n))
    if n >= 3 and draw(st.booleans()):
        pts[-1] = pts[0]
    descriptions = draw(
        st.lists(st.lists(_description, max_size=3), min_size=n, max_size=n)
    )
    number = draw(st.integers(min_value=1, max_value=32000))
    return StringSegment(number, pts, descriptions)


@settings(max_examples=150, deadline=None)
@given(
    segments=st.lists(_segments(), max_size=6),
    decimals=st.integers(min_value=0, max_value=8),
)
def test_property_round_trip(segments, decimals):
    original = StringFile(segments=segments, header=StringHeader("h", "d", "p", "m"))
    text = format_surpac_str(original, decimals=decimals)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = parse_surpac_str(text, strict=True)
    assert back.header == original.header
    assert len(back.segments) == len(segments)
    for a, b in zip(segments, back.segments, strict=True):
        assert b.number == a.number
        assert b.descriptions == a.descriptions
        np.testing.assert_array_equal(b.points, as_written(a.points, decimals))
        assert np.abs(b.points - a.points).max() <= 0.5 * 10.0**-decimals * (1 + 1e-6) + 1e-8
        if decimals >= 6 or not a.is_closed:
            continue
        assert b.is_closed  # rounding cannot open a closed ring
    # idempotent: writing what was read gives the same text
    assert format_surpac_str(back, decimals=decimals) == text
