"""Block model CSV interchange (P3.T8b).

The "surpac_like" layout is provisional (no real Surpac export to compare with); these
tests pin down what *we* write and that the reader copes with the likely variants.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.core.blockmodel import AttributeConstraint, BlockModel, BlockModelDefinition
from bakumine.core.io.block_csv import (
    SURPAC_LIKE,
    XYZ,
    BlockCsvError,
    BlockCsvLayout,
    export_block_csv,
    get_layout,
    import_block_csv,
    infer_definition,
    read_block_csv,
    sniff_block_csv,
)

REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "data" / "synthetic" / "default"

DEFINITION = BlockModelDefinition((1000.0, 5000.0, 100.0), (10.0, 20.0, 5.0), (4, 3, 3),
                                  max_depth=(2, 2, 1))  # fmt: skip


def make_model(definition: BlockModelDefinition = DEFINITION, seed: int = 0) -> BlockModel:
    rng = np.random.default_rng(seed)
    model = BlockModel(definition)
    for _ in range(model.max_level):
        can = model.level < model.max_level
        model.split(can & (rng.random(model.n_blocks) < 0.35))
    assert len(np.unique(model.level)) == model.max_level + 1
    n = model.n_blocks
    model.add_attribute("au", unit="g/t")
    model["au"][:] = rng.lognormal(size=n)
    model["au"][rng.random(n) < 0.2] = np.nan
    model.add_attribute("density", dtype=np.float32, background=2.7)
    model["density"][:] = rng.uniform(2.5, 3.0, n)
    model.add_attribute("pass_no", "int")
    model["pass_no"][:] = rng.integers(0, 4, n)
    model.add_attribute("rock", "category", categories=["WASTE", "ORE"])
    model["rock"][:] = rng.integers(-1, 2, n)
    return model


def same_structure(a: BlockModel, b: BlockModel) -> bool:
    return (
        a.definition == b.definition
        and np.array_equal(a.parent, b.parent)
        and np.array_equal(a.level, b.level)
        and np.array_equal(a.sub, b.sub)
    )


def assert_same_values(a: BlockModel, b: BlockModel, names=None) -> None:
    for name in names or a.attribute_names:
        if a.attribute(name).kind == "category":
            assert np.array_equal(a.get(name, decode=True), b.get(name, decode=True)), name
        else:
            # a CSV does not carry the dtype: float32 comes back as float64 holding the
            # same float32 values
            assert np.array_equal(a[name], b[name].astype(a[name].dtype), equal_nan=True), name


# ------------------------------------------------------------------ the layout


def test_default_layout_writes_northing_first(tmp_path):
    model = make_model()
    result = export_block_csv(model, tmp_path / "m.csv")
    assert result.layout == "surpac_like" and result.n_rows == model.n_blocks
    lines = (tmp_path / "m.csv").read_text().splitlines()
    title = f"block model export; layout surpac_like [provisional]; {model.n_blocks} blocks"
    assert lines[0] == title
    assert lines[1].startswith("columns: centroid northing (Y) easting (X) elevation (Z)")
    assert lines[2] == "ycentre,xcentre,zcentre,ysize,xsize,zsize,au,density,pass_no,rock"
    assert len(lines) == 3 + model.n_blocks
    first = lines[3].split(",")
    x, y, z = model.centroids()[0]
    dx, dy, dz = model.sizes()[0]
    # northing (5 000 000-ish in real grids; here 5000+) comes first, then easting
    assert [float(v) for v in first[:6]] == [y, x, z, dy, dx, dz]
    assert float(first[0]) > 5000 > float(first[1])


def test_xyz_layout_is_a_plain_table(tmp_path):
    model = make_model()
    export_block_csv(model, tmp_path / "m.csv", layout="xyz")
    frame = pd.read_csv(tmp_path / "m.csv")
    assert list(frame.columns) == ["x", "y", "z", "dx", "dy", "dz", "au", "density", "pass_no",
                                   "rock"]  # fmt: skip
    assert np.array_equal(frame[["x", "y", "z"]].to_numpy(), model.centroids())
    assert np.array_equal(frame[["dx", "dy", "dz"]].to_numpy(), model.sizes())
    assert get_layout("xyz") is XYZ and get_layout(None) is SURPAC_LIKE
    with pytest.raises(BlockCsvError, match="unknown block CSV layout"):
        get_layout("vulcan")


# ------------------------------------------------------------------ round trip


@pytest.mark.parametrize("layout", ["surpac_like", "xyz"])
def test_round_trip_into_a_new_model(tmp_path, layout):
    model = make_model()
    export_block_csv(model, tmp_path / "m.csv", layout=layout)
    loaded, report = import_block_csv(tmp_path / "m.csv")
    assert report.definition_inferred
    assert loaded.definition == model.definition  # origin, sizes, counts, sub-block depth
    assert same_structure(model, loaded)
    assert np.array_equal(loaded.block_ids(), model.block_ids())
    assert loaded.attribute_names == model.attribute_names
    assert_same_values(model, loaded)  # floats exactly: shortest round-trip text
    assert [loaded.attribute(n).kind for n in loaded.attribute_names] == [
        "float", "float", "int", "category"]  # fmt: skip
    assert report.n_rows == report.n_assigned == model.n_blocks and report.n_unmatched == 0
    assert report.n_split > 0 and report.max_deviation == 0.0
    assert report.created == model.attribute_names and report.updated == []


def test_round_trip_with_attribute_specs_reproduces_codes_and_dtypes(tmp_path):
    model = make_model()
    export_block_csv(model, tmp_path / "m.csv")
    loaded, _ = import_block_csv(tmp_path / "m.csv", kinds=model.attributes)
    for name in model.attribute_names:
        assert loaded.attribute(name).to_dict() == model.attribute(name).to_dict()
        assert loaded[name].dtype == model[name].dtype
        assert np.array_equal(loaded[name], model[name], equal_nan=True)


def test_round_trip_into_an_existing_model(tmp_path):
    model = make_model()
    export_block_csv(model, tmp_path / "m.csv")
    # an unsplit model of the same definition: the import must split it to match
    target = BlockModel(DEFINITION)
    target.add_attribute("rock", "category", categories=["ORE", "WASTE", "DYKE"])
    target.add_attribute("keep", background=7.0)
    result, report = import_block_csv(tmp_path / "m.csv", target)
    assert result is target
    assert same_structure(model, target)
    assert_same_values(model, target)
    assert (target["keep"] == 7.0).all()
    assert target.attribute("rock").categories == ["ORE", "WASTE", "DYKE"]  # existing codes kept
    assert report.updated == ["rock"] and report.created == ["au", "density", "pass_no"]
    assert report.n_split == (model.level > 0).sum() // 2 or report.n_split > 0
    # into a model that already has the structure: nothing is split
    again, report = import_block_csv(tmp_path / "m.csv", make_model())
    assert report.n_split == 0 and same_structure(model, again)
    with pytest.raises(BlockCsvError, match="either a model.*or a definition"):
        import_block_csv(tmp_path / "m.csv", target, definition=DEFINITION)


# ------------------------------------------------------------ hand-written CSV

HAND_DEFINITION = BlockModelDefinition((1000.0, 5000.0, 100.0), (10.0, 20.0, 5.0), (3, 2, 2),
                                       max_depth=1)  # fmt: skip

# Parent blocks are 10 m east (X) x 20 m north (Y) x 5 m high. Surpac order: Y, X, Z.
#   row 1: parent (i=0, j=0, k=0): centre east 1005, north 5010, z 102.5
#   row 2: parent (i=2, j=1, k=1): centre east 1025, north 5030, z 107.5
#   rows 3-4: two of the eight sub-blocks (5 x 10 x 2.5) of parent (i=1, j=0, k=0), whose
#   minimum corner is east 1010, north 5000, z 100:
#     first child  -> centre east 1012.5, north 5005, z 101.25
#     last child   -> centre east 1017.5, north 5015, z 103.75
HAND_CSV = """\
ycentre,xcentre,zcentre,ysize,xsize,zsize,au,rock
5010,1005,102.5,20,10,5,1.5,ORE
5030,1025,107.5,20,10,5,,WASTE
5005,1012.5,101.25,10,5,2.5,0.25,ORE
5015,1017.5,103.75,10,5,2.5,4,
"""


def test_hand_written_csv_in_the_default_layout(tmp_path):
    path = tmp_path / "hand.csv"
    path.write_text(HAND_CSV)
    fmt = sniff_block_csv(path)
    # the FIRST coordinate column is the northing, the second the easting
    assert fmt.centroid == {"y": 0, "x": 1, "z": 2}
    assert fmt.size == {"y": 3, "x": 4, "z": 5}
    table = read_block_csv(path)
    assert table.centroids[0].tolist() == [1005.0, 5010.0, 102.5]  # x = east, y = north, z
    assert table.sizes[0].tolist() == [10.0, 20.0, 5.0]

    model, report = import_block_csv(path, definition=HAND_DEFINITION)
    d = model.definition
    assert model.n_blocks == d.n_parents + 7 and report.n_split == 1
    row = lambda i, j, k: int(d.flat_index((i, j, k)))  # noqa: E731
    r1 = np.flatnonzero(model.parent == row(0, 0, 0))
    r2 = np.flatnonzero(model.parent == row(2, 1, 1))
    children = np.flatnonzero(model.parent == row(1, 0, 0))
    assert len(r1) == 1 and len(r2) == 1 and len(children) == 8
    au, rock = model["au"], model.get("rock", decode=True)
    assert au[r1[0]] == 1.5 and rock[r1[0]] == "ORE"
    assert np.isnan(au[r2[0]]) and rock[r2[0]] == "WASTE"  # empty field = missing, not 0
    assert au[children[0]] == 0.25 and rock[children[0]] == "ORE"
    assert au[children[7]] == 4.0 and rock[children[7]] is None
    assert np.isnan(au[children[1:7]]).all()  # the six sub-blocks without a row
    assert model.centroids()[children[7]].tolist() == [1017.5, 5015.0, 103.75]
    assert model.sizes()[children[0]].tolist() == [5.0, 10.0, 2.5]
    # had Y and X been swapped, (5010, 1005) would be far outside the model
    assert model.locate([[1005.0, 5010.0, 102.5]])[0] == r1[0]
    assert model.locate([[5010.0, 1005.0, 102.5]])[0] == -1
    n_assigned = (~np.isnan(au)).sum() + 1  # row 2 has no grade
    assert n_assigned == report.n_assigned == 4
    # the same rows also define the model when no definition is given, except that the
    # inferred model only spans the blocks present: here the full 3 x 2 x 2 box
    inferred, _ = infer_definition(table.centroids, table.sizes)
    assert inferred == HAND_DEFINITION


# -------------------------------------------------------------- reader variants

HAND_ROWS = [line.split(",") for line in HAND_CSV.strip().splitlines()[1:]]


def check_hand(model: BlockModel, au="au", rock="rock") -> None:
    rows = model.locate([[1005.0, 5010.0, 102.5], [1012.5, 5005.0, 101.25]])
    assert model[au][rows].tolist() == [1.5, 0.25]
    if rock:
        assert model.get(rock, rows, decode=True).tolist() == ["ORE", "ORE"]


def test_reader_without_header_assumes_y_x_z_and_says_so(tmp_path):
    path = tmp_path / "bare.csv"
    path.write_text("\n".join(",".join(r) for r in HAND_ROWS) + "\n")
    model, report = import_block_csv(path, definition=HAND_DEFINITION)
    check_hand(model, "attr1", "attr2")
    text = "\n".join(report.assumptions)
    assert "no column names in the file: assumed columns 1-3 are the centroid in Y, X, Z" in text
    assert "assumed columns 4-6 are the block size along Y, X, Z" in text
    assert "attributes named attr1, attr2" in text
    # the caller can name the columns instead
    names = ["y", "x", "z", "dy", "dx", "dz", "au", "rock"]
    model, report = import_block_csv(path, definition=HAND_DEFINITION, columns=names)
    check_hand(model)
    # ... or say the file is x, y, z: then the same numbers are a different place
    with pytest.raises(BlockCsvError, match="not the parent size"):
        import_block_csv(path, definition=HAND_DEFINITION, layout="xyz")


@pytest.mark.parametrize(
    "header",
    [
        "YCENTRE,XCENTRE,ZCENTRE,YINC,XINC,ZINC,au,rock",
        "Y,X,Z,DY,DX,DZ,au,rock",
        "Centroid_Y,Centroid_X,Centroid_Z,Y Size,X Size,Z Size,au,rock",
        "Northing,Easting,Elevation,ydim,xdim,zdim,au,rock",
        "yc,xc,zc,ylength,xlength,zlength,au,rock",
    ],
)
def test_reader_recognises_column_name_synonyms(tmp_path, header):
    path = tmp_path / "syn.csv"
    path.write_text(header + "\n" + "\n".join(",".join(r) for r in HAND_ROWS) + "\n")
    fmt = sniff_block_csv(path)
    assert fmt.centroid == {"y": 0, "x": 1, "z": 2} and fmt.size == {"y": 3, "x": 4, "z": 5}
    model, _ = import_block_csv(path, definition=HAND_DEFINITION)
    check_hand(model)


def test_reader_column_order_follows_the_names_not_the_position(tmp_path):
    path = tmp_path / "order.csv"
    lines = ["rock;x;y;z;au;dx;dy;dz"]
    lines += [";".join([r[7], r[1], r[0], r[2], r[6], r[4], r[3], r[5]]) for r in HAND_ROWS]
    path.write_text("\n".join(lines) + "\n")
    model, report = import_block_csv(path, definition=HAND_DEFINITION)
    check_hand(model)
    assert report.assumptions[0] == "semicolon-separated"
    assert model.attribute_names == ["rock", "au"]


def test_reader_whitespace_separated_with_free_header_lines(tmp_path):
    path = tmp_path / "ws.txt"
    rows = [[v if v else "-" for v in r] for r in HAND_ROWS]
    text = "Block model report\nexported 2026-01-01, 4 blocks\n\n"
    text += "  y      x      z    ysize xsize zsize au rock\n"
    text += "\n".join("   ".join(r) for r in rows) + "\n"
    path.write_text(text)
    model, report = import_block_csv(path, definition=HAND_DEFINITION, missing=["-"])
    check_hand(model)
    assert np.isnan(model["au"][model.locate([[1025.0, 5030.0, 107.5]])[0]])
    assert report.assumptions[0] == "whitespace-separated"
    assert "skipped 2 header line(s) that are not column names" in report.assumptions
    assert "values read as missing: ['-']" in report.assumptions


def test_reader_extra_columns_and_awkward_names(tmp_path):
    path = tmp_path / "extra.csv"
    lines = ["i,j,k,ycentre,xcentre,zcentre,ysize,xsize,zsize,volume,Au (g/t),rock,2nd"]
    lines += [",".join(["1", "2", "3", *r[:6], "1000", *r[6:], "9"]) for r in HAND_ROWS]
    path.write_text("\n".join(lines) + "\n")
    model, report = import_block_csv(path, definition=HAND_DEFINITION)
    # reserved or invalid names are made valid, and the report says how
    assert model.attribute_names == ["i", "j", "k", "volume_attr", "Au__g_t", "rock", "nd"]
    assert "column 'volume' imported as attribute 'volume_attr'" in report.assumptions
    assert "column 'Au (g/t)' imported as attribute 'Au__g_t'" in report.assumptions
    check_hand(model, "Au__g_t")
    assert model.attribute("i").kind == "int" and model.attribute("Au__g_t").kind == "float"
    # or only some of them
    model, _ = import_block_csv(path, definition=HAND_DEFINITION, attributes=["rock"])
    assert model.attribute_names == ["rock"]
    with pytest.raises(BlockCsvError, match="no attribute column"):
        import_block_csv(path, definition=HAND_DEFINITION, attributes=["cu"])


def test_reader_missing_size_columns_means_parent_blocks(tmp_path):
    path = tmp_path / "nosize.csv"
    path.write_text("y,x,z,au\n5010,1005,102.5,1.5\n5030,1025,107.5,2.5\n5010,1015,107.5,3.5\n")
    model, report = import_block_csv(path, definition=HAND_DEFINITION)
    assert model.is_dense and report.n_split == 0 and report.n_assigned == 3
    assert "no block size columns: every row is taken as a parent block" in report.assumptions
    assert model["au"][model.locate([[1025.0, 5030.0, 107.5]])[0]] == 2.5
    # as a new model the parent size comes from the centroid spacing
    model, report = import_block_csv(path)
    assert model.definition.block_size == (10.0, 20.0, 5.0)
    assert model.definition.origin == (1000.0, 5000.0, 100.0)
    assert model.definition.shape == (3, 2, 2)
    assert any("smallest centroid spacing" in a for a in report.assumptions)
    path.write_text("y,x,z,au\n5010,1005,102.5,1.5\n5030,1025,102.5,2.5\n")
    with pytest.raises(BlockCsvError, match="cannot infer the block size along z"):
        import_block_csv(path)


def test_reader_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("some text\nmore text\n")
    with pytest.raises(BlockCsvError, match="no data found"):
        read_block_csv(path)
    path.write_text("a,b,c,au\n1,2,3,4\n")
    with pytest.raises(BlockCsvError, match="no centroid column recognised for x, y, z"):
        read_block_csv(path)
    path.write_text("y,x,z,dx,au\n1,2,3,4,5\n")
    with pytest.raises(BlockCsvError, match="block size columns found for"):
        read_block_csv(path)
    path.write_text("y,x,z,au\n5010,1005,102.5,1.5\n5030,,107.5,2.5\n")
    with pytest.raises(BlockCsvError, match="missing or non-finite centroid.*data row.*2"):
        read_block_csv(path)
    path.write_text("y,x,z\n5010,1005,102.5,1.5\n")
    with pytest.raises(BlockCsvError, match="3 column names but 4 columns"):
        read_block_csv(path)


# ------------------------------------------------- misaligned and odd-sized rows


def test_odd_sized_and_misaligned_rows_are_a_clear_error(tmp_path):
    path = tmp_path / "odd.csv"
    good = "5010,1005,102.5,20,10,5,1.5,ORE"
    head = "ycentre,xcentre,zcentre,ysize,xsize,zsize,au,rock\n"
    # size not parent / 2**n: a 10 x 3 x 5 block (rows 2 and 3 of the data)
    path.write_text(head + good + "\n5030,1025,107.5,20,3,5,1,ORE\n5030,1015,107.5,6,10,5,1,ORE\n")
    with pytest.raises(BlockCsvError) as error:
        import_block_csv(path, definition=HAND_DEFINITION)
    text = str(error.value)
    assert "2 row(s) have a block size that is not the parent size" in text
    assert "data row 2: centroid Y 5030.0, X 1025.0, Z 107.5; size Y 20.0, X 3.0, Z 5.0" in text
    assert "data row 3" in text and "data row 1" not in text

    # a size the model's depth does not allow: X halved but Y and Z whole
    path.write_text(head + good + "\n5030,1022.5,107.5,20,5,5,1,ORE\n")
    with pytest.raises(BlockCsvError, match="block size the model cannot hold") as error:
        import_block_csv(path, definition=HAND_DEFINITION)
    assert "data row 2" in str(error.value)
    # deeper than the model's max_depth (a quarter block in a max_depth = 1 model)
    path.write_text(head + "5002.5,1001.25,100.625,5,2.5,1.25,1,ORE\n")
    with pytest.raises(BlockCsvError, match="cannot hold"):
        import_block_csv(path, definition=HAND_DEFINITION)

    # right size, wrong place: a parent-sized block shifted 2 m east
    path.write_text(head + good + "\n5030,1027,107.5,20,10,5,1,ORE\n")
    with pytest.raises(BlockCsvError) as error:
        import_block_csv(path, definition=HAND_DEFINITION)
    text = str(error.value)
    assert "1 row(s) have a centroid that is not on the model's grid" in text
    assert "data row 2: centroid Y 5030.0, X 1027.0" in text
    # a sub-block centred where only a parent can be
    path.write_text(head + "5010,1005,102.5,10,5,2.5,1,ORE\n")
    with pytest.raises(BlockCsvError, match="not on the model's grid"):
        import_block_csv(path, definition=HAND_DEFINITION)
    # many offenders: the list is cut, the count is not
    rows = "".join(f"5030,{1000.3 + i},107.5,20,10,5,1,ORE\n" for i in range(35))
    path.write_text(head + rows)
    with pytest.raises(BlockCsvError, match=r"35 row\(s\).*\n(.*\n)*.*35 rows in all"):
        import_block_csv(path, definition=HAND_DEFINITION)


def test_rows_outside_the_model_are_reported_not_assigned(tmp_path):
    path = tmp_path / "out.csv"
    path.write_text(
        "ycentre,xcentre,zcentre,ysize,xsize,zsize,au\n"
        "5010,1005,102.5,20,10,5,1.5\n"
        "5050,1005,102.5,20,10,5,9\n"  # one block north of the model
        "5010,995,102.5,20,10,5,9\n"  # one block west
        "5010,1005,102.5,20,10,5,2.5\n"  # the first block again: the last row wins
    )
    model, report = import_block_csv(path, definition=HAND_DEFINITION)
    assert report.n_rows == 4 and report.n_assigned == 2 and report.n_unmatched == 2
    assert report.unmatched == [(2, "outside the model"), (3, "outside the model")]
    assert report.n_duplicates == 1
    assert model["au"][0] == 2.5 and np.isnan(model["au"][1:]).all()
    assert "unmatched data row 2: outside the model" in report.summary()


def test_row_coarser_than_the_model_fills_the_blocks_it_covers(tmp_path):
    model = BlockModel(HAND_DEFINITION)
    model.split([0, 5])
    path = tmp_path / "coarse.csv"
    path.write_text("y,x,z,dy,dx,dz,au\n5010,1005,102.5,20,10,5,1.5\n5030,1005,102.5,20,10,5,2\n")
    _, report = import_block_csv(path, model)
    assert report.n_coarser == 1 and report.n_split == 0
    assert (model["au"][model.parent == 0] == 1.5).all() and (model.parent == 0).sum() == 8
    assert np.nansum(model["au"]) == 8 * 1.5 + 2.0


# ----------------------------------------------------------- export options


def test_missing_value_token_round_trip(tmp_path):
    model = make_model()
    assert np.isnan(model["au"]).any() and (model["rock"] < 0).any()
    for token, read in (("-99", ["-99"]), ("NULL", ["NULL"]), ("", ())):
        path = tmp_path / f"m{token}.csv"
        export_block_csv(model, path, missing=token)
        body = path.read_text().splitlines()[3:]
        nan_rows = np.flatnonzero(np.isnan(model["au"]))
        assert body[nan_rows[0]].split(",")[6] == token  # never 0, never dropped
        kwargs = {"missing": read} if read else {}
        loaded, _ = import_block_csv(path, **kwargs)
        assert same_structure(model, loaded)
        assert_same_values(model, loaded)
    # without being told, -99 is a number like any other (and a rock code)
    loaded, _ = import_block_csv(tmp_path / "m-99.csv")
    assert (loaded["au"] == -99.0).sum() == np.isnan(model["au"]).sum()
    # numeric tokens match by value, whatever the decimals
    export_block_csv(model, tmp_path / "d.csv", missing="-99.000", decimals=3)
    loaded, _ = import_block_csv(tmp_path / "d.csv", missing=[-99])
    assert np.array_equal(np.isnan(loaded["au"]), np.isnan(model["au"]))


def test_missing_token_that_cannot_be_told_apart_is_refused(tmp_path):
    model = make_model()
    model["au"][np.flatnonzero(~np.isnan(model["au"]))[0]] = -99.0
    with pytest.raises(BlockCsvError, match="'au' contains the value -99"):
        export_block_csv(model, tmp_path / "m.csv", missing="-99")
    with pytest.raises(BlockCsvError, match="category 'ORE', which is also the missing"):
        export_block_csv(model, tmp_path / "m.csv", missing="ORE")
    with pytest.raises(BlockCsvError, match="cannot be used with delimiter"):
        export_block_csv(model, tmp_path / "m.csv", missing="a,b")
    spaced = BlockCsvLayout("spaced", delimiter="whitespace")
    with pytest.raises(BlockCsvError, match="non-empty for whitespace"):
        export_block_csv(model, tmp_path / "m.csv", layout=spaced)
    assert not list(tmp_path.iterdir())  # nothing written, no temporary file left


def test_decimals(tmp_path):
    model = make_model()
    model.attribute("density").decimals = 1  # the attribute's own setting
    export_block_csv(model, tmp_path / "m.csv", decimals={"au": 2}, coord_decimals=3)
    lines = (tmp_path / "m.csv").read_text().splitlines()
    first = lines[3].split(",")
    assert all(len(v.split(".")[1]) == 3 for v in first[:6])
    row = int(np.flatnonzero(~np.isnan(model["au"]))[0])
    fields = lines[3 + row].split(",")
    assert fields[6] == f"{model['au'][row]:.2f}" and fields[7] == f"{model['density'][row]:.1f}"
    loaded, report = import_block_csv(tmp_path / "m.csv")
    assert same_structure(model, loaded)  # structure exactly, values to the written precision
    assert np.allclose(loaded["au"], model["au"], atol=0.005, rtol=0, equal_nan=True)
    assert np.allclose(loaded["density"], model["density"], atol=0.05, rtol=0)
    assert not np.array_equal(loaded["au"], model["au"], equal_nan=True)
    # one number for all float attributes
    export_block_csv(model, tmp_path / "m4.csv", decimals=4)
    fields = (tmp_path / "m4.csv").read_text().splitlines()[3 + row].split(",")
    assert fields[6] == f"{model['au'][row]:.4f}" and fields[8] == str(model["pass_no"][row])


def test_coordinates_written_with_few_decimals_need_a_tolerance(tmp_path):
    definition = BlockModelDefinition((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), (2, 2, 2), max_depth=3)
    model = make_model(definition, seed=3)
    # the smallest block is 0.625 m: written as 0.6 it is 4 % off, its centroid up to 8 %
    export_block_csv(model, tmp_path / "m.csv", coord_decimals=1)
    with pytest.raises(BlockCsvError, match="not the largest size"):
        import_block_csv(tmp_path / "m.csv")
    loaded, report = import_block_csv(tmp_path / "m.csv", tolerance=0.1)
    assert same_structure(model, loaded)  # including the origin (0, 0, 0)
    assert 0.01 < report.max_deviation <= 0.1
    assert any("origin estimated" in a for a in report.assumptions)
    with pytest.raises(BlockCsvError, match="tolerance must be"):
        import_block_csv(tmp_path / "m.csv", tolerance=0.5)


def test_constraint_and_attribute_subsets(tmp_path):
    model = make_model()
    ore = AttributeConstraint('rock == "ORE"')
    n_ore = int((model.get("rock", decode=True) == "ORE").sum())
    result = export_block_csv(model, tmp_path / "ore.csv", constraint=ore,
                              attributes=["pass_no", "au"])  # fmt: skip
    assert result.n_rows == n_ore and result.attributes == ("pass_no", "au")
    lines = (tmp_path / "ore.csv").read_text().splitlines()
    assert lines[2] == "ycentre,xcentre,zcentre,ysize,xsize,zsize,pass_no,au"
    assert len(lines) == 3 + n_ore and f"{n_ore} blocks" in lines[0]
    # a mask as well: ore in the lowest bench only
    low = model.centroids()[:, 2] < 105.0
    result = export_block_csv(model, tmp_path / "low.csv", constraint=ore, where=low)
    assert result.n_rows == int((ore.mask(model) & low).sum()) > 0

    # into a model with the same structure: only the exported blocks change
    target = make_model()
    target["au"][:] = -1.0
    target["pass_no"][:] = -1
    _, report = import_block_csv(tmp_path / "ore.csv", target)
    mask = ore.mask(model)
    assert report.n_assigned == n_ore and report.n_split == 0
    assert np.array_equal(target["au"][mask], model["au"][mask], equal_nan=True)
    assert np.array_equal(target["pass_no"][mask], model["pass_no"][mask])
    assert (target["au"][~mask] == -1.0).all() and (target["pass_no"][~mask] == -1).all()
    with pytest.raises(KeyError):
        export_block_csv(model, tmp_path / "x.csv", attributes=["cu"])
    empty = export_block_csv(model, tmp_path / "none.csv", where=np.zeros(model.n_blocks, bool))
    assert empty.n_rows == 0
    with pytest.raises(BlockCsvError, match="no data found"):
        import_block_csv(tmp_path / "none.csv")


def test_custom_layout(tmp_path):
    model = make_model()
    layout = BlockCsvLayout(
        name="datamine_like",
        axis_order=("x", "y", "z"),
        centroid_names=("XC", "YC", "ZC"),
        size_names=("XINC", "YINC", "ZINC"),
        delimiter="whitespace",
        preamble=("{n_blocks} blocks, attributes {attributes}",),
    )
    export_block_csv(model, tmp_path / "m.txt", layout=layout, missing="-")
    lines = (tmp_path / "m.txt").read_text().splitlines()
    assert lines[0] == f"{model.n_blocks} blocks, attributes au density pass_no rock"
    assert lines[1] == "XC YC ZC XINC YINC ZINC au density pass_no rock"
    loaded, report = import_block_csv(tmp_path / "m.txt", missing=["-"])
    assert same_structure(model, loaded)
    assert_same_values(model, loaded)
    # no sizes, no header at all: positions only, needs the layout to be read back
    bare = BlockCsvLayout("bare", size_names=None, column_header=False)
    dense = BlockModel(HAND_DEFINITION)
    dense.add_attribute("au", background=1.25)
    export_block_csv(dense, tmp_path / "bare.csv", layout=bare)
    assert (tmp_path / "bare.csv").read_text().splitlines()[0] == "5010.0,1005.0,102.5,1.25"
    loaded, _ = import_block_csv(tmp_path / "bare.csv", definition=HAND_DEFINITION, layout=bare)
    assert (loaded["attr1"] == 1.25).all()
    with pytest.raises(BlockCsvError, match="permutation"):
        BlockCsvLayout("bad", axis_order=("x", "x", "z"))
    with pytest.raises(BlockCsvError, match="delimiter"):
        BlockCsvLayout("bad", delimiter="|")


# --------------------------------------------------------------- rotated model


def test_rotated_model_needs_and_uses_a_supplied_definition(tmp_path):
    definition = BlockModelDefinition((500400.0, 5300250.0, 825.0), (10.0, 20.0, 5.0), (4, 3, 3),
                                      angles=(30.0, 10.0, 5.0), max_depth=(2, 2, 1))  # fmt: skip
    model = make_model(definition, seed=5)
    export_block_csv(model, tmp_path / "rot.csv")
    lines = (tmp_path / "rot.csv").read_text().splitlines()
    first = [float(v) for v in lines[3].split(",")[:6]]
    x, y, z = model.centroids()[0]
    assert first[:3] == [y, x, z]  # world coordinates
    assert first[3:] == [20.0, 10.0, 5.0] or first[3:] == list(model.sizes()[0][[1, 0, 2]])
    loaded, report = import_block_csv(tmp_path / "rot.csv", definition=definition)
    assert same_structure(model, loaded) and not report.definition_inferred
    assert_same_values(model, loaded)
    assert report.max_deviation < 1e-6
    with pytest.raises(BlockCsvError, match="is the model rotated"):
        import_block_csv(tmp_path / "rot.csv")
    # written to the millimetre it still lands in the right blocks
    export_block_csv(model, tmp_path / "rot3.csv", coord_decimals=3)
    loaded, _ = import_block_csv(tmp_path / "rot3.csv", definition=definition)
    assert same_structure(model, loaded)
    assert_same_values(model, loaded)


# ------------------------------------------------------------------- chunking


def test_chunk_size_does_not_change_the_output(tmp_path):
    model = make_model()
    export_block_csv(model, tmp_path / "a.csv", chunk_size=7)
    export_block_csv(model, tmp_path / "b.csv", chunk_size=10**6)
    export_block_csv(model, tmp_path / "c.csv", chunk_size=1)
    reference = (tmp_path / "b.csv").read_bytes()
    assert (tmp_path / "a.csv").read_bytes() == reference
    assert (tmp_path / "c.csv").read_bytes() == reference
    models = [import_block_csv(tmp_path / "a.csv", chunk_size=size)[0] for size in (3, 50, 10**6)]
    for loaded in models:
        assert same_structure(model, loaded)
        assert loaded.attribute("rock").categories == models[0].attribute("rock").categories
        for name in model.attribute_names:
            assert np.array_equal(loaded[name], models[0][name], equal_nan=True)
        assert_same_values(model, loaded)


def test_column_kind_is_decided_over_the_whole_file_not_per_chunk(tmp_path):
    path = tmp_path / "mixed.csv"
    rows = ["y,x,z,code,grade,note"]
    # 'code' is numbers for the first chunks and text later; 'grade' whole numbers, then
    # decimals; 'note' empty for the first chunks
    for n in range(12):
        code = str(n) if n < 8 else f"R{n}"
        grade = str(n) if n < 6 else f"{n}.5"
        note = "" if n < 9 else "late"
        rows.append(f"{5010 + 20 * (n % 2)},{1005 + 10 * (n % 3)},{102.5 + 5 * (n // 6)},"
                    f"{code},{grade},{note}")  # fmt: skip
    path.write_text("\n".join(rows) + "\n")
    tables = [read_block_csv(path, chunk_size=size) for size in (2, 5, 100)]
    for table in tables:
        assert table.kinds == {"code": "category", "grade": "float", "note": "category"}
        assert table.categories["code"] == [*map(str, range(8)), "R8", "R9", "R10", "R11"]
        assert table.values["code"].tolist() == list(range(12))
        assert table.values["grade"].tolist() == [0, 1, 2, 3, 4, 5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
        assert table.values["note"].tolist() == [-1] * 9 + [0, 0, 0]
    # the caller can overrule the guess
    table = read_block_csv(path, attributes=["grade"], kinds={"grade": "category"})
    assert table.categories["grade"][:2] == ["0", "1"]
    with pytest.raises(BlockCsvError, match="holds text and cannot be read as float"):
        read_block_csv(path, kinds={"code": "float"})
    with pytest.raises(BlockCsvError, match="not whole numbers"):
        read_block_csv(path, kinds={"grade": "int"})


# ------------------------------------------------------------- the truth grid


def truth_model() -> tuple[BlockModel, pd.DataFrame]:
    frame = pd.read_parquet(SYNTH / "truth_blocks.parquet")
    definition = BlockModelDefinition((500400.0, 5300250.0, 825.0), (10.0, 10.0, 5.0),
                                      (39, 52, 84))  # fmt: skip
    model = BlockModel(definition)
    flat = definition.flat_index(frame[["i", "j", "k"]].to_numpy())
    assert np.array_equal(model.centroids()[flat], frame[["x", "y", "z"]].to_numpy())
    for name in ("au", "au_ore", "density", "ore_fraction"):
        model.add_attribute(name)
        model[name][flat] = frame[name].to_numpy()
    model.add_attribute("rock", "category")
    model.set("rock", frame["rock"].to_numpy(dtype=object), flat)
    return model, frame


def test_truth_grid_exported_and_reimported_gives_the_same_au_and_rock(tmp_path):
    model, frame = truth_model()
    result = export_block_csv(model, tmp_path / "truth.csv", chunk_size=50_000)
    assert result.n_rows == len(frame) == 170352
    loaded, report = import_block_csv(tmp_path / "truth.csv", chunk_size=40_000)
    assert loaded.definition == model.definition and loaded.is_dense
    assert report.n_assigned == len(frame) and report.n_split == 0
    flat = model.definition.flat_index(frame[["i", "j", "k"]].to_numpy())
    assert np.array_equal(loaded["au"][flat], frame["au"].to_numpy(), equal_nan=True)  # exact
    assert np.isnan(frame["au"]).sum() == np.isnan(loaded["au"]).sum()
    assert np.array_equal(loaded.get("rock", decode=True)[flat], frame["rock"].to_numpy(object))
    assert np.array_equal(loaded["au_ore"], model["au_ore"], equal_nan=True)
    assert np.isnan(loaded["au_ore"]).sum() == np.isnan(frame["au_ore"]).sum() > 0
    # and the first data line is northing, easting, elevation of the first block
    line = (tmp_path / "truth.csv").read_text().splitlines()[3].split(",")
    assert [float(v) for v in line[:3]] == [5300255.0, 500405.0, 827.5]
