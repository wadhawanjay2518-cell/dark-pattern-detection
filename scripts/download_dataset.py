#!/usr/bin/env python3
"""
scripts/download_dataset.py

Reproducible script to retrieve the e-commerce dark-pattern dataset from:
https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv

Preserves the raw dataset in data/raw/ without modification.
"""

import os
import sys
import urllib.request
import pandas as pd

RAW_DATASET_URL = (
    "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv"
)
DEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dataset.tsv")


def download_dataset(url: str = RAW_DATASET_URL, dest: str = DEST_PATH) -> str:
    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        print(f"[INFO] Raw dataset already exists at {dest} ({os.path.getsize(dest)} bytes).")
    else:
        print(f"[INFO] Downloading dataset from {url} to {dest}...")
        urllib.request.urlretrieve(url, dest)
        print(f"[INFO] Download completed. File size: {os.path.getsize(dest)} bytes.")

    # Audit verification
    df = pd.read_csv(dest, sep="\t")
    print(f"[AUDIT] Total records: {len(df)}")
    print(f"[AUDIT] Columns: {df.columns.tolist()}")
    print(f"[AUDIT] Class distribution:\n{df['label'].value_counts().to_dict()}")
    print(f"[AUDIT] Pattern categories:\n{df['Pattern Category'].value_counts().to_dict()}")
    print(f"[AUDIT] Unique page IDs: {df['page_id'].nunique()}")

    return dest


if __name__ == "__main__":
    download_dataset()
