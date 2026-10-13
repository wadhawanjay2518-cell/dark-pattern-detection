"""
src/evaluate.py

Evaluation metrics calculation, multi-class error analysis, and figure generation
for the 4-class Dark Pattern Detection research task:
  1. Social Proof
  2. Scarcity
  3. Urgency
  4. Misdirection
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import TARGET_CLASSES, LABEL_MAP


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculates all standardized 4-class research evaluation metrics."""
    if class_names is None:
        class_names = TARGET_CLASSES

    num_classes = len(class_names)
    acc = float(accuracy_score(y_true, y_pred))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()

    # Multi-class ROC-AUC (One-vs-Rest)
    roc_auc = None
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
        try:
            roc_auc = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4)
        except Exception:
            roc_auc = None

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": round(float(p_per_class[i]), 4),
            "recall": round(float(r_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4),
            "support": int(support[i]),
        }

    return {
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "accuracy": round(acc, 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "roc_auc_ovr_macro": roc_auc,
        "confusion_matrix": cm,
        "per_class": per_class_metrics,
        "classes": class_names,
    }


def perform_error_analysis(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    syntax_features_df: Optional[pd.DataFrame] = None,
    class_names: Optional[List[str]] = None,
    n_examples_per_pair: int = 5,
) -> Dict[str, Any]:
    """Identifies and dissects misclassified examples across the 4 categories."""
    if class_names is None:
        class_names = TARGET_CLASSES

    error_mask = (y_true != y_pred)
    total_errors = int(error_mask.sum())
    total_samples = len(y_true)

    error_records = []
    for idx in np.where(error_mask)[0]:
        true_lbl = int(y_true[idx])
        pred_lbl = int(y_pred[idx])
        conf = float(np.max(y_prob[idx])) if y_prob is not None else 0.0

        item: Dict[str, Any] = {
            "index": int(idx),
            "text": str(texts[idx]),
            "true_class": class_names[true_lbl] if true_lbl < len(class_names) else str(true_lbl),
            "predicted_class": class_names[pred_lbl] if pred_lbl < len(class_names) else str(pred_lbl),
            "confidence": round(conf, 4),
        }
        if syntax_features_df is not None:
            row = syntax_features_df.iloc[idx].to_dict()
            item["salient_linguistic_cues"] = {
                k: round(float(v), 3)
                for k, v in row.items()
                if k in [
                    "is_imperative",
                    "exclamation_count",
                    "time_expression_count",
                    "quantity_scarcity_cues",
                    "comparative_price_cues",
                    "social_reference_cues",
                    "digit_token_ratio",
                    "uppercase_ratio",
                ] and float(v) > 0.0
            }
        error_records.append(item)

    # Group sample errors by pair
    sample_discrepancies = error_records[:25]

    return {
        "total_test_samples": total_samples,
        "total_errors": total_errors,
        "overall_error_rate": round(float(total_errors / max(1, total_samples)), 4),
        "sample_discrepancies": sample_discrepancies,
    }


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: str,
    title: str = "4-Class Confusion Matrix",
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Purples)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Category",
        xlabel="Predicted Category",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=11, fontweight="bold",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_comparison_bar(results_dict: Dict[str, Dict[str, Any]], save_path: str) -> None:
    """Generates comparison bar chart of Macro-F1 and Accuracy across all 4 research models."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    models = list(results_dict.keys())
    macro_f1s = [results_dict[m]["macro_f1"] for m in models]
    accuracies = [results_dict[m]["accuracy"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width/2, macro_f1s, width, label="Macro F1", color="#6366F1")
    rects2 = ax.bar(x + width/2, accuracies, width, label="Accuracy", color="#A5B4FC", edgecolor="#4F46E5")

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title("Model Architecture Comparison: Baselines vs Transformer vs Fusion", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
