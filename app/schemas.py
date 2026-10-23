"""Response models for the API (they also drive the /docs page)."""
from typing import Any, Literal, Optional

from pydantic import BaseModel


class QualityInfo(BaseModel):
    passed: bool
    reasons: list[str]
    messages: list[str]
    metrics: Optional[dict[str, float]] = None


class GateInfo(BaseModel):
    status: Literal["reliable", "uncertain", "unsupported"]
    knn_similarity: float
    tau_knn: float
    confidence: float
    tau_conf: float


class Prediction(BaseModel):
    class_id: str
    class_name_en: str
    class_name: str            # in the requested language
    confidence: float
    confidence_text: str       # whole percent, capped at "99%+"


class PredictResponse(BaseModel):
    status: Literal["quality_failed", "unsupported", "uncertain", "reliable"]
    lang: str
    headline: str
    messages: dict[str, str]
    quality: QualityInfo
    gate: Optional[GateInfo] = None
    prediction: Optional[Prediction] = None
    alternatives: list[Prediction] = []
    gradcam_png_base64: Optional[str] = None
    treatment: Optional[dict[str, Any]] = None
    timing_ms: float


class Health(BaseModel):
    status: str
    model_sha256: str
    classes: list[str]
    languages: list[str]