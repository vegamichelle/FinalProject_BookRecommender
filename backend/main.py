"""FastAPI service serving personalized book recommendations.

Endpoints:
    GET  /health     -> liveness + whether the model is loaded
    GET  /metrics    -> offline metrics of the loaded recommender
    GET  /catalog    -> a few catalogue titles (for the frontend dropdown)
    POST /recommend  -> recommend books from a list of favourite titles
    POST /feedback   -> mark a past recommendation helpful / not helpful

Every /recommend call is logged to the DB (for monitoring). When a user_id is
supplied and caching is on, repeat requests are served from the cache table.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.model_loader import load_model
from src import config, db

_MODEL: dict | None = None


def get_model() -> dict:
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        get_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] model not loaded yet: {exc}")
    yield


app = FastAPI(title="Book Recommender API", version="1.0.0", lifespan=lifespan)


class RecommendRequest(BaseModel):
    favorite_titles: list[str] = Field(
        ..., min_length=1, examples=[["The Silent Forest", "The Crimson Machine"]]
    )
    n: int = Field(config.DEFAULT_TOP_N, ge=1, le=50)
    user_id: str | None = Field(None, examples=["user-123"])


class RecommendResponse(BaseModel):
    request_id: str
    recommendations: list[dict]
    source: str
    cache_hit: bool
    latency_ms: float


class FeedbackRequest(BaseModel):
    request_id: str
    helpful: bool


@app.get("/health")
def health() -> dict:
    loaded = True
    try:
        get_model()
    except Exception:  # noqa: BLE001
        loaded = False
    return {"status": "ok", "model_loaded": loaded}


@app.get("/metrics")
def metrics() -> dict:
    return get_model().get("metrics", {})


@app.get("/catalog")
def catalog(n: int = 30) -> dict:
    rec = get_model()["recommender"]
    return {"titles": rec.sample_titles(n)}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        rec = get_model()["recommender"]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"Model unavailable: {exc}"
        ) from exc

    start = time.perf_counter()

    # Serve from cache for repeat users when enabled.
    cache_hit = False
    if config.CACHE_ENABLED and req.user_id:
        cached = db.get_cached(req.user_id)
        if cached:
            recs = cached[: req.n]
            cache_hit = True

    if not cache_hit:
        recs = rec.recommend(req.favorite_titles, n=req.n)
        if config.CACHE_ENABLED and req.user_id:
            db.set_cached(req.user_id, recs)

    source = recs[0]["source"] if recs else "none"
    latency_ms = (time.perf_counter() - start) * 1000.0

    request_id = db.log_request(
        input_titles=req.favorite_titles,
        recommendations=recs,
        latency_ms=latency_ms,
        source="cache" if cache_hit else source,
        cache_hit=cache_hit,
        user_id=req.user_id,
    )
    return RecommendResponse(
        request_id=request_id,
        recommendations=recs,
        source="cache" if cache_hit else source,
        cache_hit=cache_hit,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    ok = db.record_feedback(req.request_id, 1 if req.helpful else 0)
    if not ok:
        raise HTTPException(status_code=404, detail="request_id not found")
    return {"status": "recorded"}
