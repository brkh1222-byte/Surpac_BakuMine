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
