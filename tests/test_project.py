"""Regression tests for bakumine.core.project (P7.T5 review findings)."""

from __future__ import annotations

import json

import pytest

from bakumine.core.project import PROJECT_FILE, Project, ProjectError


@pytest.fixture
def source(tmp_path):
    file = tmp_path / "src" / "a.csv"
    file.parent.mkdir()
    file.write_text("x,y\n1,2\n")
    return file


def test_object_id_cannot_escape_project(tmp_path, source):
    project = Project.create(tmp_path / "p")
    project.add("string", source)
    with pytest.raises(ProjectError):
        project.add("string", source, object_id="x/../../../escaped")
    assert not (tmp_path / "escaped.csv").exists()
    assert [o.id for o in Project.open(tmp_path / "p").list()] == ["a"]


@pytest.mark.parametrize(
    "path", ["../victim", "strings/../../victim", "/abs/victim", "solids/x", "strings"]
)
def test_registry_path_outside_project_is_rejected(tmp_path, path):
    victim = tmp_path / "victim"
    victim.mkdir()
    folder = tmp_path / "p"
    Project.create(folder)
    doc = json.loads((folder / PROJECT_FILE).read_text())
    doc["objects"] = [{"id": "evil", "kind": "string", "name": "evil", "path": path}]
    (folder / PROJECT_FILE).write_text(json.dumps(doc))
    with pytest.raises(ProjectError):
        Project.open(folder)
    assert victim.exists()


def test_two_handles_do_not_lose_updates(tmp_path, source):
    folder = tmp_path / "p"
    Project.create(folder)
    first, second = Project.open(folder), Project.open(folder)
    first.add("string", source, object_id="a")
    second.add("string", source, object_id="b")
    ids = sorted(o.id for o in Project.open(folder).list())
    assert ids == ["a", "b"]
    paths = {o.path for o in Project.open(folder).list()}
    assert len(paths) == 2  # the second copy did not overwrite the first
    first.remove("b")
    assert [o.id for o in Project.open(folder).list()] == ["a"]


# ----------------------------------------------------------------- kind geology (P9.T12)


@pytest.fixture
def geology_folder(tmp_path):
    """A stand-in for a geology model folder: the project copies it byte for byte."""
    folder = tmp_path / "src" / "demo.geology"
    (folder / "coding").mkdir(parents=True)
    (folder / "geology.json").write_text('{"format": "bakumine.geology", "name": "demo"}\n')
    (folder / "coding" / "model.provenance.json").write_text("{}\n")
    return folder


def test_geology_kind_add_list_remove(tmp_path, geology_folder):
    from bakumine.core.project import KIND_FOLDERS, KINDS

    assert KIND_FOLDERS["geology"] == "geology" and "geology" in KINDS
    folder = tmp_path / "p"
    project = Project.create(folder)
    assert (folder / "geology").is_dir()
    obj = project.add("geology", geology_folder, name="Demo geology")
    assert (obj.kind, obj.name, obj.id, obj.path) == (
        "geology", "Demo geology", "demo-geology", "geology/demo.geology",
    )  # fmt: skip
    copied = project.path_of(obj)
    assert copied == folder / "geology" / "demo.geology"
    for file in ("geology.json", "coding/model.provenance.json"):
        assert (copied / file).read_bytes() == (geology_folder / file).read_bytes()
    reopened = Project.open(folder)
    assert [o.to_dict() for o in reopened.list("geology")] == [obj.to_dict()]
    assert reopened.list("model") == []
    second = reopened.add("geology", geology_folder, name="Demo geology")  # same source again
    assert second.id == "demo-geology-2" and second.path != obj.path
    reopened.remove(second.id)
    assert not reopened.path_of(obj).parent.joinpath(second.path.split("/")[1]).exists()
    assert [o.id for o in Project.open(folder).list()] == ["demo-geology"]


def test_geology_path_must_be_in_its_folder(tmp_path):
    folder = tmp_path / "p"
    Project.create(folder)
    doc = json.loads((folder / PROJECT_FILE).read_text())
    doc["objects"] = [{"id": "g", "kind": "geology", "name": "g", "path": "models/g"}]
    (folder / PROJECT_FILE).write_text(json.dumps(doc))
    with pytest.raises(ProjectError):
        Project.open(folder)


def test_geology_kind_on_the_command_line(tmp_path, geology_folder, capsys):
    from bakumine.cli import main

    folder = tmp_path / "p"
    assert main(["project", "init", str(folder)]) == 0
    code = main(["project", "add", str(folder), "geology", str(geology_folder), "--name", "N"])
    assert code == 0
    capsys.readouterr()
    assert main(["project", "list", str(folder), "--kind", "geology"]) == 0
    out = capsys.readouterr().out
    assert "geology" in out and "geology/demo.geology" in out and " N " in out
