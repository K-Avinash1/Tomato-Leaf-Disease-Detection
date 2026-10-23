"""Tomato leaf disease API.   Start with:  uvicorn app.main:app --reload   then open http://127.0.0.1:8000/docs"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from app.schemas import Health, PredictResponse
from src.inference.loader import load_engine
from src.inference.pipeline import analyze
from src.knowledge_base.loader import LANGS

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Lang = Literal["en", "hi"]


@asynccontextmanager
async def lifespan(app):
    app.state.engine = load_engine()      # model, kNN bank, configs, knowledge base: loaded once
    yield


app = FastAPI(title="Tomato Leaf Disease Detection and Treatment Recommendation", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=Health)
def health():
    e = app.state.engine
    return {"status": "ok", "model_sha256": e.model_sha256, "classes": e.classes, "languages": list(LANGS)}


@app.get("/api/ui")
def ui_text(lang: Lang = "en"):
    """All app text (messages, section headings) in one language."""
    return app.state.engine.kb.ui(lang)


@app.get("/api/treatment/{disease_id}")
def treatment(disease_id: str, lang: Lang = "en"):
    try:
        return app.state.engine.kb.entry(disease_id, lang)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown disease id '{disease_id}'")


@app.post("/api/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...), lang: Lang = Form("en")):
    """Upload one photo. Uploaded images are processed in memory and never saved."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file larger than 10 MB")
    return analyze(app.state.engine, data, lang)