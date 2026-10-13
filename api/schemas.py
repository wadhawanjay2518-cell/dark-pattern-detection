"""
api/schemas.py

Pydantic schemas for request/response serialization across the 4-class
Dark Pattern Detection API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="E-commerce interface text string to classify for potential dark pattern cues.",
        json_schema_extra={"example": "Hurry! Only 2 items left in stock — 48 people are viewing right now!"}
    )
    model_choice: Optional[str] = Field(
        default="model_d",
        description="Target model: 'model_a' (TF-IDF), 'model_b' (DistilBERT), 'model_c' (Linguistic), 'model_d' (Late Fusion).",
        json_schema_extra={"example": "model_d"}
    )


class ObservedCue(BaseModel):
    type: str
    detail: str
    associated_category: str


class PredictResponse(BaseModel):
    success: bool
    text: str
    model_used: str
    predicted_category: str
    predicted_label: int
    confidence: float
    class_probabilities: Dict[str, float]
    observed_cues: List[ObservedCue]
    linguistic_features: Dict[str, float]
    latency_ms: float
    disclaimer: str


class ModelComparisonDetail(BaseModel):
    name: str
    predicted_category: Optional[str] = None
    confidence: Optional[float] = None
    class_probabilities: Optional[Dict[str, float]] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class MultiModelPredictResponse(BaseModel):
    success: bool
    text: str
    models: Dict[str, ModelComparisonDetail]
    observed_cues: List[ObservedCue]
    linguistic_features: Dict[str, float]
    disclaimer: str


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of text strings to classify in batch.",
    )
    model_choice: Optional[str] = "model_d"


class BatchPredictItem(BaseModel):
    text: str
    predicted_category: str
    confidence: float
    class_probabilities: Dict[str, float]


class BatchPredictResponse(BaseModel):
    success: bool
    count: int
    model_used: str
    predictions: List[BatchPredictItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    models_available: Dict[str, bool]
    timestamp: str


class MetadataResponse(BaseModel):
    app_title: str
    model_version: str
    target_classes: List[str]
    device: str
    experiments: Dict[str, Any]
    feature_definitions: Dict[str, str]
