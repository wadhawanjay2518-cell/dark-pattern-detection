#!/usr/bin/env python3
"""
scripts/train_all.py

Executes the complete controlled 4-class research experiment:
  - Model A: Baseline TF-IDF + Multinomial Logistic Regression
  - Model B: Transformer-Only (DistilBERT contextual fine-tuning)
  - Model C: Linguistic-Feature-Only (spaCy POS/syntax/cues classifier)
  - Model D: Proposed Semantic-Linguistic Feature Fusion Model

Generates:
  - Model artifacts in outputs/models/
  - Metrics & classification reports in outputs/metrics/
  - Publication figures (confusion matrices, comparison bar chart) in outputs/figures/
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import numpy as np
import pandas as pd
import torch

from src.preprocessing import TARGET_CLASSES, LABEL_MAP
from src.data_loader import prepare_dataset_pipeline
from src.syntactic_features import SyntacticFeatureExtractor, FEATURE_NAMES, FEATURE_DEFINITIONS
from src.semantic_encoder import SemanticEncoder
from src.fusion_model import DarkPatternFusionModel
from src.train import (
    train_model_a_tfidf,
    train_model_c_linguistic,
    train_neural_model,
    get_device,
)
from src.evaluate import (
    compute_metrics,
    perform_error_analysis,
    plot_confusion_matrix,
    plot_comparison_bar,
)


def run_all_experiments(
    raw_path: str = "data/dark-patterns(2).csv",
    output_dir: str = "outputs",
    batch_size: int = 32,
    epochs: int = 4,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"[SETUP] Experiment hardware device: {device}")

    metrics_dir = os.path.join(output_dir, "metrics")
    figures_dir = os.path.join(output_dir, "figures")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Step 1: Prepare and audit dataset
    print("\n=======================================================")
    print("STEP 1: Dataset Partitioning & 4-Class Audit")
    print("=======================================================")
    train_df, val_df, test_df, audit_data = prepare_dataset_pipeline(
        raw_path=raw_path,
        processed_dir="data/processed",
        output_metrics_dir=metrics_dir,
        seed=seed,
    )

    # Step 2: Linguistic Parsing & Feature Scaling (spaCy)
    print("\n=======================================================")
    print("STEP 2: Linguistic Parsing & Feature Extraction (spaCy)")
    print("=======================================================")
    extractor = SyntacticFeatureExtractor(spacy_model="en_core_web_sm")

    print("[SYNTAX] Extracting training linguistic features...")
    df_feat_train = extractor.extract_dataframe(train_df["clean_text"].tolist())
    print("[SYNTAX] Extracting validation linguistic features...")
    df_feat_val = extractor.extract_dataframe(val_df["clean_text"].tolist())
    print("[SYNTAX] Extracting test linguistic features...")
    df_feat_test = extractor.extract_dataframe(test_df["clean_text"].tolist())

    # Fit scaler strictly on train set
    X_train_syntax = extractor.fit_transform(df_feat_train)
    X_val_syntax = extractor.transform(df_feat_val)
    X_test_syntax = extractor.transform(df_feat_test)

    # Save fitted scaler
    scaler_path = os.path.join(models_dir, "syntax_scaler.joblib")
    extractor.save_scaler(scaler_path)

    # Step 3: Semantic Encoding & DataLoaders
    print("\n=======================================================")
    print("STEP 3: Transformer Tokenization & DataLoaders")
    print("=======================================================")
    encoder = SemanticEncoder(checkpoint="distilbert-base-uncased", max_length=64)

    train_ds = encoder.create_dataset(train_df["clean_text"].tolist(), X_train_syntax, train_df["label"].values)
    val_ds = encoder.create_dataset(val_df["clean_text"].tolist(), X_val_syntax, val_df["label"].values)
    test_ds = encoder.create_dataset(test_df["clean_text"].tolist(), X_test_syntax, test_df["label"].values)

    train_loader = encoder.get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = encoder.get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = encoder.get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}

    # Model A: Baseline TF-IDF + Logistic Regression
    print("\n=======================================================")
    print("EXPERIMENT A: TF-IDF + Logistic Regression Baseline")
    print("=======================================================")
    _, metrics_a, preds_a, probs_a = train_model_a_tfidf(
        train_df, val_df, test_df,
        save_path=os.path.join(models_dir, "model_a_tfidf.joblib")
    )
    results["Model A: TF-IDF + LogReg"] = metrics_a

    # Model C: Linguistic-Only Classifier
    print("\n=======================================================")
    print("EXPERIMENT C: Linguistic-Only Features Classifier")
    print("=======================================================")
    _, metrics_c, preds_c, probs_c = train_model_c_linguistic(
        X_train_syntax, X_val_syntax, X_test_syntax,
        train_df, val_df, test_df,
        save_path=os.path.join(models_dir, "model_c_linguistic.joblib")
    )
    results["Model C: Linguistic-Only"] = metrics_c

    # Model B: Transformer-Only (DistilBERT)
    print("\n=======================================================")
    print("EXPERIMENT B: DistilBERT Contextual Transformer-Only")
    print("=======================================================")
    model_b = DarkPatternFusionModel(
        checkpoint="distilbert-base-uncased",
        num_classes=4,
        ablation_syntax=True,
    )
    model_b, metrics_b, preds_b, probs_b = train_neural_model(
        model_b, train_loader, val_loader, test_loader,
        epochs=epochs, device=device,
        model_name="Model B: DistilBERT-Only",
        model_code="model_b"
    )
    model_b.save_checkpoint(
        os.path.join(models_dir, "model_b_distilbert.pt"),
        metadata={
            "model_type": "distilbert_only",
            "classes": TARGET_CLASSES,
            "test_macro_f1": metrics_b["macro_f1"],
            "test_accuracy": metrics_b["accuracy"],
        }
    )
    results["Model B: DistilBERT-Only"] = metrics_b

    # Model D: Proposed Semantic-Linguistic Fusion Model
    print("\n=======================================================")
    print("EXPERIMENT D: Proposed Semantic-Linguistic Feature Fusion")
    print("=======================================================")
    model_d = DarkPatternFusionModel(
        checkpoint="distilbert-base-uncased",
        syntax_dim=X_train_syntax.shape[1],
        syntax_proj_dim=64,
        hidden_dim=128,
        num_classes=4,
        ablation_syntax=False,
    )
    model_d, metrics_d, preds_d, probs_d = train_neural_model(
        model_d, train_loader, val_loader, test_loader,
        epochs=epochs, device=device,
        model_name="Model D: Semantic-Linguistic Fusion",
        model_code="model_d"
    )

    # Save Model D checkpoints (both named and default fusion_model.pt)
    model_d_path = os.path.join(models_dir, "model_d_fusion.pt")
    fusion_model_path = os.path.join(models_dir, "fusion_model.pt")
    metadata_d = {
        "model_type": "semantic_linguistic_fusion",
        "classes": TARGET_CLASSES,
        "feature_names": FEATURE_NAMES,
        "test_macro_f1": metrics_d["macro_f1"],
        "test_accuracy": metrics_d["accuracy"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    model_d.save_checkpoint(model_d_path, metadata=metadata_d)
    model_d.save_checkpoint(fusion_model_path, metadata=metadata_d)
    results["Model D: Late Fusion (Proposed)"] = metrics_d

    # Step 4: Error Analysis on Proposed Model D
    print("\n=======================================================")
    print("STEP 4: Qualitative Error Analysis on Proposed Model D")
    print("=======================================================")
    error_analysis = perform_error_analysis(
        texts=test_df["clean_text"].tolist(),
        y_true=test_df["label"].values,
        y_pred=preds_d,
        y_prob=probs_d,
        syntax_features_df=df_feat_test,
        class_names=TARGET_CLASSES,
    )
    with open(os.path.join(metrics_dir, "error_analysis_model_d.json"), "w") as f:
        json.dump(error_analysis, f, indent=2)

    # Save summary of all experiments
    summary_path = os.path.join(metrics_dir, "experiments_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[RESULTS] Experiments summary saved to {summary_path}")

    # Step 5: Generate Figures
    print("\n=======================================================")
    print("STEP 5: Generating Publication Figures")
    print("=======================================================")
    # Confusion Matrix for Model D
    cm_d_path = os.path.join(figures_dir, "confusion_matrix_model_d.png")
    plot_confusion_matrix(
        metrics_d["confusion_matrix"],
        TARGET_CLASSES,
        cm_d_path,
        title="Model D (Semantic-Linguistic Fusion) Test Confusion Matrix",
    )

    # Comparison Bar Chart
    comp_path = os.path.join(figures_dir, "model_comparison.png")
    plot_comparison_bar(results, comp_path)

    # Save global metadata for serving
    metadata_payload = {
        "app_title": "Dark Pattern Detection: Semantic–Linguistic Feature Fusion",
        "model_version": "1.0.0",
        "target_classes": TARGET_CLASSES,
        "label_map": LABEL_MAP,
        "feature_definitions": FEATURE_DEFINITIONS,
        "device": str(device),
        "experiments": results,
    }
    with open(os.path.join(metrics_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata_payload, f, indent=2)

    print("\n[COMPLETE] All 4 models trained and evaluated successfully!")
    return results


if __name__ == "__main__":
    run_all_experiments()
