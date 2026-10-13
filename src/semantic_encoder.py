"""
src/semantic_encoder.py

Handles transformer tokenization, dataset wrapping, and DataLoader creation.
Supports offline HuggingFace cache loading with graceful fallback.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class DarkPatternDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        syntax_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 64,
    ):
        self.texts = [str(t) if t is not None else "" for t in texts]
        self.syntax_features = torch.tensor(syntax_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "syntax_features": self.syntax_features[idx],
        }
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                self.texts[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            item["input_ids"] = encoding["input_ids"].squeeze(0)
            item["attention_mask"] = encoding["attention_mask"].squeeze(0)

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


class SemanticEncoder:
    def __init__(self, checkpoint: str = "distilbert-base-uncased", max_length: int = 64):
        self.checkpoint = checkpoint
        self.max_length = max_length

        # Try offline first to avoid network hangs in sandboxed or isolated environments
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            except Exception as e:
                raise RuntimeError(f"Could not load tokenizer for '{checkpoint}': {e}")

    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        clean_texts = [str(t) if t is not None else "" for t in texts]
        return self.tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def create_dataset(
        self,
        texts: List[str],
        syntax_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> DarkPatternDataset:
        return DarkPatternDataset(
            texts=texts,
            syntax_features=syntax_features,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def get_dataloader(
        self,
        dataset: DarkPatternDataset,
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
