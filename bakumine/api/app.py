"""FastAPI app. The UI is a client of the same core functions as the CLI."""

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

from bakumine import __version__
from bakumine.api.guard import RequestGuard
from bakumine.api.ops_classify import router as classify_router
from bakumine.api.ops_intercepts import router as intercepts_router
from bakumine.api.routes_digitise import router as digitise_router
from bakumine.api.routes_geometry import router as geometry_router
from bakumine.api.routes_ops import router as ops_router
from bakumine.api.routes_report import router as report_router
from bakumine.api.routes_scene import router as scene_router

app = FastAPI(title="BakuMine Block Modeller", version=__version__)
# Level 4: float64 coordinate arrays compress to about half; higher levels cost time
# for little gain on localhost.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)
# Added last, so it runs first: DNS rebinding and cross-site requests (bakumine.api.guard).
app.add_middleware(RequestGuard)
app.include_router(scene_router)
app.include_router(report_router)
app.include_router(digitise_router)
app.include_router(geometry_router)
app.include_router(ops_router)
app.include_router(intercepts_router)
app.include_router(classify_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
