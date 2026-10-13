"""
tests/test_data.py

Tests for dataset loading, dynamic column verification, 4-class filtering,
non-target exclusion, and stratified partitioning for the 4-class research corpus.
"""

import os
import pytest
import pandas as pd
from src.data_loader import DatasetLoader
from src.preprocessing import TARGET_CLASSES, EXCLUDED_CLASSES, LABEL_MAP


@pytest.fixture
def loader():
    return DatasetLoader(raw_path="data/dark-patterns(2).csv", processed_dir="data/processed")


def test_raw_dataset_exists(loader):
    resolved = loader.resolve_raw_path()
    assert os.path.exists(resolved), f"Raw dataset must exist at {resolved}"


def test_dataset_column_detection(loader):
    df = loader.load_raw()
    text_col, cat_col, url_col = loader.detect_columns(df)
    assert text_col == "Pattern String"
    assert cat_col == "Pattern Category"
    assert url_col == "Website Page"
    assert len(df) == 1818, f"Expected 1,818 raw rows, got {len(df)}"


def test_missing_text_and_exclusion(loader):
    df = loader.load_raw()
    df_target, df_excluded, audit = loader.audit_and_clean(df)

    # 306 missing text rows safely excluded
    assert audit["data_cleaning_and_exclusion"]["missing_text_rows_excluded"] == 306
    assert audit["data_cleaning_and_exclusion"]["rows_with_usable_text"] == 1512

    # Non-target classes excluded without relabeling
    assert len(df_excluded) == 46
    excluded_cats = set(df_excluded["clean_category"].unique())
    assert excluded_cats.issubset(set(EXCLUDED_CLASSES))

    # Eligible 4-class dataset
    assert len(df_target) == 1466
    target_cats = set(df_target["clean_category"].unique())
    assert target_cats == set(TARGET_CLASSES)


def test_four_class_labels(loader):
    df = loader.load_raw()
    df_target, _, _ = loader.audit_and_clean(df)
    unique_labels = set(df_target["label"].unique())
    assert unique_labels == {0, 1, 2, 3}, f"Labels must be 0, 1, 2, 3, got {unique_labels}"
    assert len(df_target) == 1466


def test_stratified_splits_distribution(loader):
    df = loader.load_raw()
    df_target, _, _ = loader.audit_and_clean(df)
    train_df, val_df, test_df, meta = loader.create_stratified_splits(df_target)

    # Size checks
    assert len(train_df) == 1026
    assert len(val_df) == 220
    assert len(test_df) == 220
    assert len(train_df) + len(val_df) + len(test_df) == len(df_target)

    # All 4 classes represented in each split
    assert set(train_df["label"].unique()) == {0, 1, 2, 3}
    assert set(val_df["label"].unique()) == {0, 1, 2, 3}
    assert set(test_df["label"].unique()) == {0, 1, 2, 3}


def test_audit_report_structure(loader):
    df = loader.load_raw()
    _, _, report = loader.audit_and_clean(df)
    assert "initial_dataset_verification" in report
    assert "data_cleaning_and_exclusion" in report
    assert "primary_4class_dataset" in report
    assert report["initial_dataset_verification"]["total_raw_rows"] == 1818
