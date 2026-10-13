"""
tests/test_features.py

Tests for spaCy linguistic and syntactic feature extraction, edge cases,
feature vector dimensions, and StandardScaler fitting.
"""

import pytest
import numpy as np
import pandas as pd
from src.syntactic_features import SyntacticFeatureExtractor, FEATURE_NAMES, FEATURE_DEFINITIONS


@pytest.fixture
def extractor():
    return SyntacticFeatureExtractor()


def test_feature_count_and_documentation():
    assert len(FEATURE_NAMES) == 40, f"Expected 40 features, got {len(FEATURE_NAMES)}"
    for name in FEATURE_NAMES:
        assert name in FEATURE_DEFINITIONS, f"Feature {name} lacks documentation."
        assert len(FEATURE_DEFINITIONS[name]) > 10, f"Definition for {name} too brief."


def test_extract_single_dimension(extractor):
    feats = extractor.extract_single("Hurry! Only 2 items left in stock!")
    assert isinstance(feats, dict)
    assert len(feats) == len(FEATURE_NAMES)
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_empty_and_whitespace_edge_cases(extractor):
    # Empty string
    feats_empty = extractor.extract_single("")
    assert len(feats_empty) == len(FEATURE_NAMES)
    assert all(v == 0.0 for v in feats_empty.values())

    # Whitespace only
    feats_ws = extractor.extract_single("   \t\n  ")
    assert len(feats_ws) == len(FEATURE_NAMES)
    assert all(v == 0.0 for v in feats_ws.values())

    # Punctuation only
    feats_punct = extractor.extract_single("!?!?!?!?")
    assert feats_punct["exclamation_count"] > 0
    assert feats_punct["question_count"] > 0


def test_linguistic_indicator_detection(extractor):
    # Urgency & Scarcity with exclamation and numbers
    cue_text = "Hurry! Only 2 items left in stock — 48 people viewing right now!"
    extracted = extractor.extract_single(cue_text)
    assert extracted["exclamation_count"] >= 1.0
    assert extracted["digit_token_ratio"] > 0.0
    assert extracted["quantity_scarcity_cues"] > 0.0
    assert extracted["social_reference_cues"] > 0.0

    # Temporal constraint
    temporal = extractor.extract_single("Flash sale ends in 5 minutes! Hurry!")
    assert temporal["has_time_expression"] == 1.0
    assert temporal["time_expression_count"] >= 1.0


def test_dataframe_batch_extraction_and_scaling(extractor):
    texts = [
        "Hurry! Limited time only!",
        "1,142 people have added to cart recently.",
        "Only 3 left in stock - order today!",
        "",
        "Regular price $120.00, sale price $59.99"
    ]
    df_feats = extractor.extract_dataframe(texts)
    assert df_feats.shape == (len(texts), len(FEATURE_NAMES))

    # Scaling test
    scaled = extractor.fit_transform(df_feats)
    assert scaled.shape == (len(texts), len(FEATURE_NAMES))
    assert not np.isnan(scaled).any(), "Found NaNs in scaled features!"
    assert not np.isinf(scaled).any(), "Found Infs in scaled features!"
