"""
src/data_loader.py

Robust dataset-loading, validation, and stratification pipeline for
the 4-class Dark Pattern Detection research task:
  - Detects column names dynamically without rigid assumptions
  - Handles encodings, missing values, and whitespace
  - Preserves original raw CSV without modification
  - Filters to primary 4 classes (Social Proof, Scarcity, Urgency, Misdirection)
  - Excludes Obstruction, Sneaking, Forced Action without relabeling
  - Generates comprehensive statistical audit & preprocessing report
  - Produces stratified train/val/test splits with identical partitions
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.preprocessing import TARGET_CLASSES, EXCLUDED_CLASSES, LABEL_MAP, REVERSE_LABEL_MAP, sanitize_text

logger = logging.getLogger(__name__)

# Data dictionary defining every column used by the application
DATA_DICTIONARY: Dict[str, Dict[str, str]] = {
    "Pattern String": {
        "description": "The exact interface text string extracted from the e-commerce webpage representing the potential dark pattern cue.",
        "type": "string (text)",
        "role": "Primary Model Input Feature",
        "handling": "Sanitized (HTML entities decoded, whitespace normalized). Rows with null/empty text are excluded from model training.",
    },
    "Pattern Category": {
        "description": "The high-level dark pattern category according to e-commerce dark pattern taxonomies (e.g. Mathur et al., 2019).",
        "type": "categorical string",
        "role": "Classification Target Label",
        "handling": "Primary experiment focuses on 4 classes: Social Proof, Scarcity, Urgency, Misdirection. Obstruction, Sneaking, and Forced Action are excluded without relabeling.",
    },
    "Pattern Type": {
        "description": "Specific sub-category or UI pattern mechanism (e.g., Activity Notification, Countdown Timer, Confirmshaming, Low-stock message).",
        "type": "categorical string",
        "role": "Metadata & Provenance",
        "handling": "Preserved for error analysis and qualitative breakdown.",
    },
    "Comment": {
        "description": "Annotator observations detailing the visual or interactive presentation of the element (e.g., 'Periodic popup', 'Hardcoded in JS').",
        "type": "string (text)",
        "role": "Metadata & Provenance",
        "handling": "Preserved for qualitative inspection.",
    },
    "Where in website?": {
        "description": "Location within the e-commerce purchase funnel where the cue was identified (Product Page, Cart Page, Home Page, Checkout Process, Other).",
        "type": "categorical string",
        "role": "Metadata & Provenance",
        "handling": "Preserved for contextual reporting.",
    },
    "Deceptive?": {
        "description": "Annotator determination of whether the pattern was deceptive, non-deceptive, or context-dependent ('Yes', 'No', 'Depends').",
        "type": "categorical string",
        "role": "Metadata & Provenance",
        "handling": "Preserved for exploratory analysis.",
    },
    "Website Page": {
        "description": "Full URL source provenance indicating the e-commerce website where the pattern instance was observed and collected.",
        "type": "string (URL)",
        "role": "Provenance / Origin URL",
        "handling": "Preserved for verification and data audit.",
    },
}


class DatasetLoader:
    def __init__(
        self,
        raw_path: str = "data/dark-patterns(2).csv",
        processed_dir: str = "data/processed",
        seed: int = 42,
    ):
        self.raw_path = raw_path
        self.processed_dir = processed_dir
        self.seed = seed
        os.makedirs(self.processed_dir, exist_ok=True)

    def resolve_raw_path(self) -> str:
        """Finds the dataset file if path variation exists."""
        candidates = [
            self.raw_path,
            "data/dark-patterns(2).csv",
            "data/dark-patterns.csv",
            "data/raw/dataset.tsv",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Dataset not found at '{self.raw_path}'. "
            "Please ensure 'dark-patterns(2).csv' is placed at 'data/dark-patterns(2).csv'."
        )

    def load_raw(self) -> pd.DataFrame:
        """Loads raw CSV safely handling encoding fallbacks without altering the original file."""
        actual_path = self.resolve_raw_path()
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        df: Optional[pd.DataFrame] = None

        for enc in encodings:
            try:
                # Use comma first, tab fallback if TSV
                sep = "\t" if actual_path.endswith(".tsv") else ","
                df = pd.read_csv(actual_path, sep=sep, encoding=enc)
                logger.info(f"Loaded dataset from {actual_path} with encoding {enc}")
                break
            except Exception as e:
                continue

        if df is None:
            raise RuntimeError(f"Failed to read dataset from {actual_path} with supported encodings.")

        return df

    @staticmethod
    def detect_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
        """Dynamically detects the text column, category column, and URL column."""
        cols = df.columns.tolist()

        # Detect text column
        text_col = None
        for candidate in ["Pattern String", "pattern_string", "text", "PatternString", "string"]:
            for col in cols:
                if col.strip().lower() == candidate.lower():
                    text_col = col
                    break
            if text_col:
                break

        # Detect category column
        cat_col = None
        for candidate in ["Pattern Category", "pattern_category", "category", "label", "Category"]:
            for col in cols:
                if col.strip().lower() == candidate.lower():
                    cat_col = col
                    break
            if cat_col:
                break

        # Detect URL / provenance column
        url_col = None
        for candidate in ["Website Page", "website_page", "url", "WebsitePage", "page_url", "source_url"]:
            for col in cols:
                if col.strip().lower() == candidate.lower():
                    url_col = col
                    break
            if url_col:
                break

        if not text_col:
            raise ValueError(f"Could not identify text column in dataset columns: {cols}")
        if not cat_col:
            raise ValueError(f"Could not identify category column in dataset columns: {cols}")

        return text_col, cat_col, url_col

    def audit_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Performs statistical audit, handles missing text, excludes non-target categories,
        and outputs clean 4-class dataset and audit report.
        """
        raw_rows = len(df)
        text_col, cat_col, url_col = self.detect_columns(df)

        # Audit initial distribution
        raw_category_counts = df[cat_col].value_counts(dropna=False).to_dict()
        raw_missing = df.isnull().sum().to_dict()

        # Check for usable text
        has_text_mask = df[text_col].notnull() & (df[text_col].astype(str).str.strip() != "")
        missing_text_count = int((~has_text_mask).sum())
        missing_text_by_category = df[~has_text_mask][cat_col].value_counts(dropna=False).to_dict()

        df_with_text = df[has_text_mask].copy()

        # Sanitize text
        df_with_text["clean_text"] = df_with_text[text_col].apply(sanitize_text)

        # Check text duplicates
        exact_text_duplicates = int(df_with_text["clean_text"].duplicated().sum())
        unique_texts = int(df_with_text["clean_text"].nunique())

        # Category normalization and filtering
        df_with_text["clean_category"] = df_with_text[cat_col].astype(str).str.strip()

        # Excluded categories (without relabeling)
        excluded_mask = df_with_text["clean_category"].isin(EXCLUDED_CLASSES)
        df_excluded = df_with_text[excluded_mask].copy()
        excluded_counts = df_excluded["clean_category"].value_counts().to_dict()

        # Primary 4 classes
        target_mask = df_with_text["clean_category"].isin(TARGET_CLASSES)
        df_target = df_with_text[target_mask].copy()

        # Check for any unexpected/invalid labels
        invalid_mask = (~target_mask) & (~excluded_mask)
        invalid_records_count = int(invalid_mask.sum())
        invalid_categories = df_with_text[invalid_mask]["clean_category"].value_counts().to_dict() if invalid_records_count > 0 else {}

        # Encode 4-class numeric labels
        df_target["label"] = df_target["clean_category"].map(REVERSE_LABEL_MAP)

        target_distribution = df_target["clean_category"].value_counts().to_dict()

        # Text length statistics on 4-class dataset
        text_lengths = df_target["clean_text"].str.len()
        word_counts = df_target["clean_text"].apply(lambda t: len(t.split()))

        audit_report: Dict[str, Any] = {
            "initial_dataset_verification": {
                "total_raw_rows": raw_rows,
                "detected_columns": df.columns.tolist(),
                "text_column": text_col,
                "category_column": cat_col,
                "url_provenance_column": url_col,
                "raw_category_distribution": {str(k): int(v) for k, v in raw_category_counts.items()},
                "raw_missing_values_per_column": {str(k): int(v) for k, v in raw_missing.items()},
            },
            "data_cleaning_and_exclusion": {
                "missing_text_rows_excluded": missing_text_count,
                "missing_text_by_category": {str(k): int(v) for k, v in missing_text_by_category.items()},
                "rows_with_usable_text": int(len(df_with_text)),
                "exact_text_duplicates": exact_text_duplicates,
                "unique_text_count": unique_texts,
                "excluded_categories_count": int(len(df_excluded)),
                "excluded_categories_breakdown": {str(k): int(v) for k, v in excluded_counts.items()},
                "invalid_labels_count": invalid_records_count,
                "invalid_labels_breakdown": {str(k): int(v) for k, v in invalid_categories.items()},
            },
            "primary_4class_dataset": {
                "eligible_sample_count": int(len(df_target)),
                "classes": TARGET_CLASSES,
                "category_distribution": {str(k): int(v) for k, v in target_distribution.items()},
                "class_proportions": {
                    str(k): round(float(v) / len(df_target), 4) for k, v in target_distribution.items()
                },
                "text_statistics": {
                    "min_chars": int(text_lengths.min()),
                    "max_chars": int(text_lengths.max()),
                    "mean_chars": round(float(text_lengths.mean()), 2),
                    "median_chars": float(text_lengths.median()),
                    "min_words": int(word_counts.min()),
                    "max_words": int(word_counts.max()),
                    "mean_words": round(float(word_counts.mean()), 2),
                },
            },
        }

        return df_target, df_excluded, audit_report

    def create_stratified_splits(
        self,
        df_target: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Creates stratified train/val/test splits using fixed seed for fair comparison.
        Splits preserve class proportions across all three sets.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

        # First split: train vs (val + test)
        val_test_ratio = val_ratio + test_ratio
        split1 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=self.seed)
        train_idx, val_test_idx = next(split1.split(df_target, df_target["label"]))

        train_df = df_target.iloc[train_idx].reset_index(drop=True)
        val_test_df = df_target.iloc[val_test_idx].reset_index(drop=True)

        # Second split: val vs test (equal split of the remaining 30%)
        relative_test_ratio = test_ratio / val_test_ratio
        split2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=self.seed)
        val_idx, test_idx = next(split2.split(val_test_df, val_test_df["label"]))

        val_df = val_test_df.iloc[val_idx].reset_index(drop=True)
        test_df = val_test_df.iloc[test_idx].reset_index(drop=True)

        split_metadata = {
            "random_seed": self.seed,
            "train_size": len(train_df),
            "train_class_counts": {LABEL_MAP[k]: int(v) for k, v in train_df["label"].value_counts().items()},
            "val_size": len(val_df),
            "val_class_counts": {LABEL_MAP[k]: int(v) for k, v in val_df["label"].value_counts().items()},
            "test_size": len(test_df),
            "test_class_counts": {LABEL_MAP[k]: int(v) for k, v in test_df["label"].value_counts().items()},
        }

        return train_df, val_df, test_df, split_metadata

    def save_processed(
        self,
        df_target: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Saves clean dataset and splits without modifying raw files."""
        clean_path = os.path.join(self.processed_dir, "cleaned_4class.csv")
        train_path = os.path.join(self.processed_dir, "train.csv")
        val_path = os.path.join(self.processed_dir, "val.csv")
        test_path = os.path.join(self.processed_dir, "test.csv")

        df_target.to_csv(clean_path, index=False)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved processed datasets to {self.processed_dir}")


def prepare_dataset_pipeline(
    raw_path: str = "data/dark-patterns(2).csv",
    processed_dir: str = "data/processed",
    output_metrics_dir: str = "outputs/metrics",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Runs the full data preparation and audit pipeline."""
    loader = DatasetLoader(raw_path=raw_path, processed_dir=processed_dir, seed=seed)
    raw_df = loader.load_raw()

    df_target, df_excluded, audit_report = loader.audit_and_clean(raw_df)
    train_df, val_df, test_df, split_metadata = loader.create_stratified_splits(df_target)

    loader.save_processed(df_target, train_df, val_df, test_df)

    full_report = {
        **audit_report,
        "split_metadata": split_metadata,
        "data_dictionary": DATA_DICTIONARY,
    }

    os.makedirs(output_metrics_dir, exist_ok=True)
    report_path = os.path.join(output_metrics_dir, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"[DATA] Preprocessing completed. Eligible 4-class samples: {len(df_target)}")
    print(f"[DATA] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"[DATA] Full audit report saved to {report_path}")

    return train_df, val_df, test_df, full_report


if __name__ == "__main__":
    prepare_dataset_pipeline()
