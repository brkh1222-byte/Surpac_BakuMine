"""FastAPI app. The UI is a client of the same core functions as the CLI."""

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

from bakumine import __version__
from bakumine.api.routes_scene import router as scene_router

app = FastAPI(title="BakuMine Block Modeller", version=__version__)
# Level 4: float64 coordinate arrays compress to about half; higher levels cost time
# for little gain on localhost.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)
app.include_router(scene_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
