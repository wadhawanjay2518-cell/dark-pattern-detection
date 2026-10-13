"""
tests/test_api.py

Tests for FastAPI endpoints: /health, /metadata, /experiments, /audit,
/data-dictionary, /dataset-sample, /predict, /predict/all-models, /batch-predict.
Validates input schemas, status codes, and 4-class response structures.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.preprocessing import TARGET_CLASSES


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "device" in data
    assert "models_available" in data


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "target_classes" in data
    assert len(data["target_classes"]) == 4
    assert data["target_classes"] == TARGET_CLASSES


def test_experiments_endpoint(client):
    response = client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) >= 3


def test_audit_endpoint(client):
    response = client.get("/audit")
    assert response.status_code == 200
    data = response.json()
    assert "initial_dataset_verification" in data
    assert "data_cleaning_and_exclusion" in data
    assert "primary_4class_dataset" in data
    assert data["initial_dataset_verification"]["total_raw_rows"] == 1818


def test_data_dictionary_endpoint(client):
    response = client.get("/data-dictionary")
    assert response.status_code == 200
    data = response.json()
    assert "columns" in data
    assert "Pattern String" in data["columns"]
    assert "Pattern Category" in data["columns"]


def test_dataset_sample_endpoint(client):
    response = client.get("/dataset-sample?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) <= 5
    if len(data["samples"]) > 0:
        sample = data["samples"][0]
        assert "text" in sample
        assert "category" in sample
        assert "url" in sample


def test_predict_validation_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in (400, 422)


def test_predict_validation_excessive_text(client):
    too_long = "x" * 2500
    response = client.post("/predict", json={"text": too_long})
    assert response.status_code in (400, 422)


def test_predict_endpoint_valid_cue(client):
    health_resp = client.get("/health").json()
    response = client.post("/predict", json={"text": "Hurry! Only 2 items left in stock!"})

    if health_resp["model_loaded"]:
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["predicted_category"] in TARGET_CLASSES
        assert data["predicted_label"] in (0, 1, 2, 3)
        assert "confidence" in data
        assert "class_probabilities" in data
        assert len(data["class_probabilities"]) == 4
        assert "observed_cues" in data
        assert "linguistic_features" in data
        assert "disclaimer" in data
    else:
        assert response.status_code == 503


def test_predict_all_models_endpoint(client):
    health_resp = client.get("/health").json()
    response = client.post("/predict/all-models", json={"text": "5338 people viewed this in the last hour"})

    if health_resp["model_loaded"]:
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "models" in data
        assert "model_d" in data["models"]
    else:
        assert response.status_code == 503


def test_batch_predict_endpoint(client):
    health_resp = client.get("/health").json()
    payload = {
        "texts": [
            "Hurry! Only 2 left in stock",
            "Faith in Glendale purchased 1 hour ago",
            "FLASH SALE | LIMITED TIME ONLY"
        ],
        "model_choice": "model_d"
    }
    response = client.post("/batch-predict", json=payload)

    if health_resp["model_loaded"]:
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 3
        assert len(data["predictions"]) == 3
        for pred in data["predictions"]:
            assert pred["predicted_category"] in TARGET_CLASSES
    else:
        assert response.status_code == 503
