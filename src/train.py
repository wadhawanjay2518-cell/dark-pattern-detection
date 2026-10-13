"""
src/train.py

Training workflows for the 4 comparative research models:
  - Model A (Baseline): TF-IDF + Multinomial Logistic Regression
  - Model B (Transformer-Only): Fine-tuned DistilBERT Contextual Classifier
  - Model C (Linguistic-Only): Normalized spaCy Linguistic Features + Classifier
  - Model D (Proposed Late Fusion): Contextual DistilBERT + Linguistic Feature Fusion
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import joblib
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import TARGET_CLASSES
from src.syntactic_features import SyntacticFeatureExtractor, FEATURE_NAMES
from src.semantic_encoder import SemanticEncoder, DarkPatternDataset
from src.fusion_model import DarkPatternFusionModel
from src.evaluate import compute_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model_a_tfidf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Optional[str] = "outputs/models/model_a_tfidf.joblib",
) -> Tuple[Any, Dict[str, Any], np.ndarray, np.ndarray]:
    """Model A: TF-IDF vectorization + Logistic Regression (4-class)."""
    print("[Model A] Training TF-IDF + Logistic Regression baseline...")
    start_time = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    train_time = round(time.time() - start_time, 2)
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)

    metrics = compute_metrics(y_test, test_preds, test_probs, class_names=TARGET_CLASSES)
    metrics["training_time_sec"] = train_time
    metrics["model_name"] = "Model A: TF-IDF + Logistic Regression"
    metrics["model_code"] = "model_a"

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump({"vectorizer": vectorizer, "classifier": clf, "classes": TARGET_CLASSES}, save_path)
        print(f"[Model A] Saved artifact to {save_path}")

    print(f"[Model A] Macro-F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f} in {train_time}s")
    return clf, metrics, test_preds, test_probs


def train_model_c_linguistic(
    X_train_syntax: np.ndarray,
    X_val_syntax: np.ndarray,
    X_test_syntax: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Optional[str] = "outputs/models/model_c_linguistic.joblib",
) -> Tuple[Any, Dict[str, Any], np.ndarray, np.ndarray]:
    """Model C: Linguistic-Feature-Only Classifier (4-class)."""
    print("[Model C] Training Linguistic-Feature-Only Classifier...")
    start_time = time.time()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train_syntax, y_train)

    train_time = round(time.time() - start_time, 2)
    test_preds = clf.predict(X_test_syntax)
    test_probs = clf.predict_proba(X_test_syntax)

    metrics = compute_metrics(y_test, test_preds, test_probs, class_names=TARGET_CLASSES)
    metrics["training_time_sec"] = train_time
    metrics["model_name"] = "Model C: Linguistic-Only Classifier"
    metrics["model_code"] = "model_c"

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump({"classifier": clf, "classes": TARGET_CLASSES, "feature_names": FEATURE_NAMES}, save_path)
        print(f"[Model C] Saved artifact to {save_path}")

    print(f"[Model C] Macro-F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f} in {train_time}s")
    return clf, metrics, test_preds, test_probs


def train_neural_model(
    model: DarkPatternFusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 4,
    transformer_lr: float = 2e-5,
    syntax_lr: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 2,
    device: Optional[torch.device] = None,
    model_name: str = "Neural Model",
    model_code: str = "neural",
) -> Tuple[DarkPatternFusionModel, Dict[str, Any], np.ndarray, np.ndarray]:
    """Trains DistilBERT (Model B) or Fusion (Model D) with validation monitoring."""
    if device is None:
        device = get_device()
    model.to(device)

    # Optimizer parameters with differential learning rates
    optimizer_grouped_parameters = []
    if not model.freeze_transformer:
        optimizer_grouped_parameters.append({
            "params": [p for p in model.transformer.parameters() if p.requires_grad],
            "lr": transformer_lr,
            "weight_decay": weight_decay,
        })
    other_params = []
    if model.syntax_projection is not None:
        other_params.extend(list(model.syntax_projection.parameters()))
    other_params.extend(list(model.classifier.parameters()))
    optimizer_grouped_parameters.append({
        "params": other_params,
        "lr": syntax_lr,
        "weight_decay": weight_decay,
    })

    optimizer = AdamW(optimizer_grouped_parameters)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    no_improve_epochs = 0
    training_start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            syntax_features = batch["syntax_features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _, _ = model(input_ids, attention_mask, syntax_features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Validation phase
        model.eval()
        val_preds, val_probs, val_trues = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                syntax_features = batch["syntax_features"].to(device)
                labels = batch["label"].to(device)

                logits, _, _ = model(input_ids, attention_mask, syntax_features)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                val_preds.extend(preds.tolist())
                val_probs.extend(probs.tolist())
                val_trues.extend(labels.cpu().numpy().tolist())

        val_loss /= max(1, len(val_loader))
        val_metrics = compute_metrics(np.array(val_trues), np.array(val_preds), np.array(val_probs), class_names=TARGET_CLASSES)
        print(f"[{model_code.upper()}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"[{model_code.upper()}] Early stopping triggered after epoch {epoch+1}")
                break

    train_time = round(time.time() - training_start, 2)

    # Load best checkpoint for testing
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_probs, test_trues = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            syntax_features = batch["syntax_features"].to(device)
            labels = batch["label"].to(device)

            logits, _, _ = model(input_ids, attention_mask, syntax_features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            test_preds.extend(preds.tolist())
            test_probs.extend(probs.tolist())
            test_trues.extend(labels.cpu().numpy().tolist())

    y_true = np.array(test_trues)
    y_pred = np.array(test_preds)
    y_prob = np.array(test_probs)

    test_metrics = compute_metrics(y_true, y_pred, y_prob, class_names=TARGET_CLASSES)
    test_metrics["training_time_sec"] = train_time
    test_metrics["best_val_macro_f1"] = round(float(best_val_f1), 4)
    test_metrics["model_name"] = model_name
    test_metrics["model_code"] = model_code

    print(f"[{model_code.upper()}] Test Macro-F1: {test_metrics['macro_f1']:.4f} | Accuracy: {test_metrics['accuracy']:.4f} in {train_time}s")
    return model, test_metrics, y_pred, y_prob
