import pytest
from fastapi.testclient import TestClient

import bakumine
from bakumine import cli
from bakumine.api import app


def test_import_has_version():
    assert bakumine.__version__


def test_health():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert bakumine.__version__ in capsys.readouterr().out
