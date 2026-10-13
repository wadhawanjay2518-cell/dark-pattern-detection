"""
src/inference.py

Production-ready inference pipeline for 4-class Dark Pattern Detection:
  - Serves predictions from Model A, Model B, Model C, or Model D (Proposed Late Fusion)
  - Provides multi-model comparative predictions
  - Extracts and interprets linguistic feature activations
  - Validates and sanitizes text inputs safely
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import joblib
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch

from src.preprocessing import sanitize_text, validate_input_text, TARGET_CLASSES, LABEL_MAP
from src.syntactic_features import SyntacticFeatureExtractor, FEATURE_NAMES, FEATURE_DEFINITIONS
from src.semantic_encoder import SemanticEncoder
from src.fusion_model import DarkPatternFusionModel

DISCLAIMER_TEXT = (
    "This research prototype classifies potential dark-pattern linguistic and semantic cues "
    "into four categories (Social Proof, Scarcity, Urgency, Misdirection). "
    "It is designed for academic analysis and does not constitute formal legal determination."
)


class DarkPatternInferenceService:
    def __init__(
        self,
        checkpoint_dir: str = "outputs/models",
        metadata_path: str = "outputs/metrics/model_metadata.json",
        device: Optional[str] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.metadata_path = metadata_path

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.extractor = SyntacticFeatureExtractor()
        self.encoder = SemanticEncoder(checkpoint="distilbert-base-uncased", max_length=64)

        # Loaded model references
        self.model_d: Optional[DarkPatternFusionModel] = None
        self.model_b: Optional[DarkPatternFusionModel] = None
        self.model_a: Optional[Any] = None  # TF-IDF bundle
        self.model_c: Optional[Any] = None  # Linguistic bundle

        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False

        self._try_load()

    def _try_load(self) -> bool:
        """Attempts to load fitted scaler and available model artifacts."""
        scaler_path = os.path.join(self.checkpoint_dir, "syntax_scaler.joblib")
        if os.path.exists(scaler_path):
            try:
                self.extractor.load_scaler(scaler_path)
            except Exception as e:
                print(f"[INFERENCE] Error loading scaler: {e}")

        # Try load Model D (Fusion)
        model_d_path = os.path.join(self.checkpoint_dir, "fusion_model.pt")
        if not os.path.exists(model_d_path):
            model_d_path = os.path.join(self.checkpoint_dir, "model_d_fusion.pt")
        if os.path.exists(model_d_path):
            try:
                self.model_d, _ = DarkPatternFusionModel.load_checkpoint(model_d_path, device=str(self.device))
                self.is_loaded = True
            except Exception as e:
                print(f"[INFERENCE] Error loading Model D: {e}")

        # Try load Model B (DistilBERT-only)
        model_b_path = os.path.join(self.checkpoint_dir, "model_b_distilbert.pt")
        if os.path.exists(model_b_path):
            try:
                self.model_b, _ = DarkPatternFusionModel.load_checkpoint(model_b_path, device=str(self.device))
            except Exception as e:
                print(f"[INFERENCE] Error loading Model B: {e}")

        # Try load Model A (TF-IDF)
        model_a_path = os.path.join(self.checkpoint_dir, "model_a_tfidf.joblib")
        if os.path.exists(model_a_path):
            try:
                self.model_a = joblib.load(model_a_path)
            except Exception as e:
                print(f"[INFERENCE] Error loading Model A: {e}")

        # Try load Model C (Linguistic)
        model_c_path = os.path.join(self.checkpoint_dir, "model_c_linguistic.joblib")
        if os.path.exists(model_c_path):
            try:
                self.model_c = joblib.load(model_c_path)
            except Exception as e:
                print(f"[INFERENCE] Error loading Model C: {e}")

        # Load metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"[INFERENCE] Error loading metadata: {e}")

        return self.is_loaded or (self.model_a is not None)

    def extract_salient_cues(self, text: str, raw_features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identifies prominent human-interpretable linguistic and surface cues."""
        cues = []
        text_lower = text.lower()

        if raw_features.get("time_expression_count", 0) > 0:
            cues.append({
                "type": "Temporal Constraint / Urgency",
                "detail": "Contains temporal deadlines or countdown intervals (e.g., minutes, hours, ends soon, hurry).",
                "associated_category": "Urgency",
            })
        if raw_features.get("quantity_scarcity_cues", 0) > 0:
            cues.append({
                "type": "Inventory / Stock Scarcity",
                "detail": "Mentions limited stock or availability constraints (e.g., only X left, remaining, claimed).",
                "associated_category": "Scarcity",
            })
        if raw_features.get("social_reference_cues", 0) > 0:
            cues.append({
                "type": "Crowd Activity / Social Proof",
                "detail": "References peer activity, viewing counts, or past purchases (e.g., people, bought, viewers).",
                "associated_category": "Social Proof",
            })
        if raw_features.get("comparative_price_cues", 0) > 0:
            cues.append({
                "type": "Comparative Price / Discount Cue",
                "detail": "Highlights perceived savings, relative discounts, or best-seller endorsements.",
                "associated_category": "Misdirection",
            })
        if raw_features.get("is_imperative", 0) == 1.0:
            cues.append({
                "type": "Imperative Directive",
                "detail": "Grammatically constructed as a direct command (base verb root without overt subject).",
                "associated_category": "Urgency / Misdirection",
            })
        if raw_features.get("uppercase_ratio", 0) > 0.3 and len(text) > 10:
            cues.append({
                "type": "Visual Orthographic Salience",
                "detail": f"High proportion of uppercase characters ({int(raw_features['uppercase_ratio']*100)}%) signaling simulated alarm or urgency.",
                "associated_category": "Urgency",
            })
        if raw_features.get("exclamation_count", 0) > 0:
            cues.append({
                "type": "Punctuation Intensity",
                "detail": f"Contains emphatic punctuation ({int(raw_features['exclamation_count'])} exclamation marks).",
                "associated_category": "Urgency / Scarcity",
            })

        return cues

    def predict_single(
        self,
        text: str,
        model_choice: str = "model_d",
    ) -> Dict[str, Any]:
        """Runs classification using the selected model."""
        is_valid, clean_text = validate_input_text(text)
        if not is_valid:
            raise ValueError(clean_text)

        start_time = time.time()
        raw_features = self.extractor.extract_single(clean_text)
        salient_cues = self.extract_salient_cues(clean_text, raw_features)

        # Scale features
        if self.extractor.scaler is not None:
            feat_df = pd.DataFrame([raw_features])[FEATURE_NAMES]
            scaled_features = self.extractor.transform(feat_df)
        else:
            scaled_features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)

        probs: Dict[str, float] = {}
        predicted_class_idx = 0

        # Model A: TF-IDF
        if model_choice == "model_a":
            if self.model_a is None:
                raise RuntimeError("Model A (TF-IDF) artifact is not loaded.")
            vec = self.model_a["vectorizer"]
            clf = self.model_a["classifier"]
            X = vec.transform([clean_text])
            prob_arr = clf.predict_proba(X)[0]
            predicted_class_idx = int(np.argmax(prob_arr))
            probs = {TARGET_CLASSES[i]: round(float(p), 4) for i, p in enumerate(prob_arr)}

        # Model C: Linguistic-Only
        elif model_choice == "model_c":
            if self.model_c is None:
                raise RuntimeError("Model C (Linguistic) artifact is not loaded.")
            clf = self.model_c["classifier"]
            prob_arr = clf.predict_proba(scaled_features)[0]
            predicted_class_idx = int(np.argmax(prob_arr))
            probs = {TARGET_CLASSES[i]: round(float(p), 4) for i, p in enumerate(prob_arr)}

        # Model B: DistilBERT-Only
        elif model_choice == "model_b":
            if self.model_b is None:
                raise RuntimeError("Model B (DistilBERT) artifact is not loaded.")
            inputs = self.encoder.tokenize_batch([clean_text])
            with torch.no_grad():
                logits, _, _ = self.model_b(
                    input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                    syntax_features=None,
                )
                prob_arr = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_class_idx = int(np.argmax(prob_arr))
            probs = {TARGET_CLASSES[i]: round(float(p), 4) for i, p in enumerate(prob_arr)}

        # Model D: Late Fusion (Default)
        else:
            if self.model_d is None:
                raise RuntimeError("Model D (Late Fusion) artifact is not loaded.")
            inputs = self.encoder.tokenize_batch([clean_text])
            syn_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits, _, _ = self.model_d(
                    input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                    syntax_features=syn_tensor,
                )
                prob_arr = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_class_idx = int(np.argmax(prob_arr))
            probs = {TARGET_CLASSES[i]: round(float(p), 4) for i, p in enumerate(prob_arr)}

        latency_ms = round((time.time() - start_time) * 1000, 2)
        predicted_category = TARGET_CLASSES[predicted_class_idx]
        confidence = probs.get(predicted_category, 0.0)

        return {
            "success": True,
            "text": clean_text,
            "model_used": model_choice,
            "predicted_category": predicted_category,
            "predicted_label": predicted_class_idx,
            "confidence": confidence,
            "class_probabilities": probs,
            "observed_cues": salient_cues,
            "linguistic_features": {k: round(float(v), 4) for k, v in raw_features.items()},
            "latency_ms": latency_ms,
            "disclaimer": DISCLAIMER_TEXT,
        }

    def predict_all_models(self, text: str) -> Dict[str, Any]:
        """Runs inference across all 4 research models simultaneously for comparative inspection."""
        is_valid, clean_text = validate_input_text(text)
        if not is_valid:
            raise ValueError(clean_text)

        model_keys = ["model_a", "model_b", "model_c", "model_d"]
        model_names = {
            "model_a": "Model A: TF-IDF + Logistic Regression",
            "model_b": "Model B: DistilBERT Contextual Transformer",
            "model_c": "Model C: Linguistic-Only Feature Classifier",
            "model_d": "Model D: Proposed Semantic-Linguistic Fusion",
        }

        comparisons = {}
        for k in model_keys:
            try:
                res = self.predict_single(clean_text, model_choice=k)
                comparisons[k] = {
                    "name": model_names[k],
                    "predicted_category": res["predicted_category"],
                    "confidence": res["confidence"],
                    "class_probabilities": res["class_probabilities"],
                    "latency_ms": res["latency_ms"],
                }
            except Exception as e:
                comparisons[k] = {
                    "name": model_names[k],
                    "error": str(e),
                }

        # Common linguistic cues
        raw_features = self.extractor.extract_single(clean_text)
        salient_cues = self.extract_salient_cues(clean_text, raw_features)

        return {
            "success": True,
            "text": clean_text,
            "models": comparisons,
            "observed_cues": salient_cues,
            "linguistic_features": {k: round(float(v), 4) for k, v in raw_features.items()},
            "disclaimer": DISCLAIMER_TEXT,
        }
