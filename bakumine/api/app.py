"""FastAPI app. The UI is a client of the same core functions as the CLI."""

from fastapi import FastAPI

from bakumine import __version__

app = FastAPI(title="BakuMine Block Modeller", version=__version__)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
