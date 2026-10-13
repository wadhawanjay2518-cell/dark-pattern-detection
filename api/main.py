"""
api/main.py

FastAPI serving application for Dark Pattern Detection:
Semantic–Linguistic Feature Fusion for E-Commerce Text Classification.
Provides endpoints for single inference, multi-model comparison, batch evaluation,
dataset auditing, and research metadata.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    PredictRequest,
    PredictResponse,
    MultiModelPredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictItem,
    HealthResponse,
    MetadataResponse,
)
from src.inference import DarkPatternInferenceService
from src.data_loader import DATA_DICTIONARY, DatasetLoader
from src.preprocessing import TARGET_CLASSES

inference_service: Optional[DarkPatternInferenceService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_service
    models_dir = os.getenv("MODELS_DIR", "outputs/models")
    metadata_path = os.getenv("METADATA_PATH", "outputs/metrics/model_metadata.json")

    inference_service = DarkPatternInferenceService(
        checkpoint_dir=models_dir,
        metadata_path=metadata_path,
    )
    yield


app = FastAPI(
    title="Dark Pattern Detection API",
    description="Semantic–Linguistic Feature Fusion for E-Commerce Text Classification (4 Classes)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    if inference_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference service initializing.")
    return {
        "status": "healthy",
        "model_loaded": inference_service.is_loaded,
        "device": str(inference_service.device),
        "models_available": {
            "model_a_tfidf": inference_service.model_a is not None,
            "model_b_distilbert": inference_service.model_b is not None,
            "model_c_linguistic": inference_service.model_c is not None,
            "model_d_fusion": inference_service.model_d is not None,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/metadata", tags=["Metadata"])
async def get_metadata():
    if inference_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference service not ready.")

    meta_path = "outputs/metrics/model_metadata.json"
    experiments_data = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                experiments_data = meta.get("experiments", {})
        except Exception:
            pass

    return {
        "app_title": "Dark Pattern Detection: Semantic–Linguistic Feature Fusion",
        "model_version": "1.0.0",
        "target_classes": TARGET_CLASSES,
        "device": str(inference_service.device),
        "experiments": experiments_data,
        "feature_definitions": inference_service.extractor.feature_names,
    }


@app.get("/experiments", tags=["Research Benchmarks"])
async def get_experiments():
    """Returns experimental benchmark comparison across Models A, B, C, D."""
    summary_path = "outputs/metrics/experiments_summary.json"
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load experiment metrics: {e}")

    # Fallback to model_metadata.json
    meta_path = "outputs/metrics/model_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("experiments", {})

    raise HTTPException(status_code=404, detail="Experiment metrics not yet generated. Run scripts/train_all.py first.")


@app.get("/audit", tags=["Dataset & Provenance"])
async def get_audit_report():
    """Returns dataset verification, preprocessing audit, and real count breakdown."""
    report_path = "outputs/metrics/preprocessing_report.json"
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading preprocessing report: {e}")

    # If report file doesn't exist yet, generate on the fly
    try:
        loader = DatasetLoader()
        df = loader.load_raw()
        df_target, df_excluded, audit_report = loader.audit_and_clean(df)
        return {**audit_report, "data_dictionary": DATA_DICTIONARY}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not run audit on dataset: {e}")


@app.get("/data-dictionary", tags=["Dataset & Provenance"])
async def get_data_dictionary():
    """Returns the comprehensive data dictionary explaining all columns used."""
    return {
        "dataset_name": "dark-patterns(2).csv",
        "columns": DATA_DICTIONARY,
    }


@app.get("/dataset-sample", tags=["Dataset & Provenance"])
async def get_dataset_sample(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
):
    """Returns paginated and searchable rows from the cleaned 4-class dataset."""
    clean_path = "data/processed/cleaned_4class.csv"
    if not os.path.exists(clean_path):
        clean_path = "data/dark-patterns(2).csv"

    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    try:
        df = pd.read_csv(clean_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}")

    text_col = "clean_text" if "clean_text" in df.columns else "Pattern String"
    cat_col = "clean_category" if "clean_category" in df.columns else "Pattern Category"

    if category and category.strip():
        df = df[df[cat_col].str.lower() == category.strip().lower()]

    if search and search.strip():
        search_lower = search.strip().lower()
        mask = df[text_col].astype(str).str.lower().str.contains(search_lower)
        if "Pattern Type" in df.columns:
            mask = mask | df["Pattern Type"].astype(str).str.lower().str.contains(search_lower)
        df = df[mask]

    total_matches = len(df)
    page_df = df.iloc[offset : offset + limit]

    records = []
    for _, row in page_df.iterrows():
        records.append({
            "text": str(row.get(text_col, "")),
            "category": str(row.get(cat_col, "")),
            "pattern_type": str(row.get("Pattern Type", "N/A")),
            "where_in_website": str(row.get("Where in website?", "N/A")),
            "deceptive": str(row.get("Deceptive?", "N/A")),
            "url": str(row.get("Website Page", "")),
        })

    return {
        "total": total_matches,
        "offset": offset,
        "limit": limit,
        "samples": records,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(payload: PredictRequest):
    """Predict dark pattern category for a single interface text string."""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not ready.")

    try:
        result = inference_service.predict_single(
            text=payload.text,
            model_choice=payload.model_choice or "model_d",
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/all-models", response_model=MultiModelPredictResponse, tags=["Inference"])
async def predict_all_models(payload: PredictRequest):
    """Runs input through all 4 research models simultaneously for comparative analysis."""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not ready.")

    try:
        result = inference_service.predict_all_models(text=payload.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparative inference error: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictResponse, tags=["Inference"])
async def batch_predict(payload: BatchPredictRequest):
    """Classifies multiple text strings in a batch."""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not ready.")

    model_choice = payload.model_choice or "model_d"
    results = []

    for text in payload.texts:
        if not text or not text.strip():
            continue
        try:
            res = inference_service.predict_single(text=text, model_choice=model_choice)
            results.append(BatchPredictItem(
                text=res["text"],
                predicted_category=res["predicted_category"],
                confidence=res["confidence"],
                class_probabilities=res["class_probabilities"],
            ))
        except Exception:
            continue

    return BatchPredictResponse(
        success=True,
        count=len(results),
        model_used=model_choice,
        predictions=results,
    )


# Static frontend mounting
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", tags=["UI"])
    async def serve_ui():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"service": "Dark Pattern Detection API", "status": "online"}
