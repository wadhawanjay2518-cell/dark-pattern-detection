"""
src/fusion_model.py

PyTorch implementation of the 4-class Dark Pattern Detection architecture:
Late feature fusion combining contextual semantic embeddings from DistilBERT
with projected linguistic/syntactic features from spaCy parsing.
Supports Model B (DistilBERT-only ablation) and Model D (Semantic-Linguistic Fusion).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from typing import Dict, Any, Optional, Tuple

from src.syntactic_features import FEATURE_NAMES


class DarkPatternFusionModel(nn.Module):
    def __init__(
        self,
        checkpoint: str = "distilbert-base-uncased",
        syntax_dim: int = len(FEATURE_NAMES),
        syntax_proj_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        ablation_syntax: bool = False,
        zero_syntax: bool = False,
        freeze_transformer: bool = False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.syntax_dim = syntax_dim
        self.syntax_proj_dim = syntax_proj_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.ablation_syntax = ablation_syntax
        self.zero_syntax = zero_syntax
        self.freeze_transformer = freeze_transformer

        # Branch B: Semantic Contextual Transformer Encoder
        try:
            self.transformer: PreTrainedModel = AutoModel.from_pretrained(checkpoint, local_files_only=True)
        except Exception:
            self.transformer: PreTrainedModel = AutoModel.from_pretrained(checkpoint)

        self.semantic_dim = self.transformer.config.hidden_size  # 768 for DistilBERT

        if self.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Branch A: Linguistic/Syntactic Projection Layer
        if not self.ablation_syntax:
            self.syntax_projection = nn.Sequential(
                nn.Linear(syntax_dim, syntax_proj_dim),
                nn.LayerNorm(syntax_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            fused_dim = self.semantic_dim + syntax_proj_dim
        else:
            self.syntax_projection = None
            fused_dim = self.semantic_dim

        # Fusion Classification Head (Predicts 4 classes)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        syntax_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Returns:
            logits: [batch_size, num_classes] (4 classes)
            semantic_vec: [batch_size, semantic_dim] (768)
            syntax_proj: [batch_size, syntax_proj_dim] (64) or None
        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Extract [CLS] token representation (index 0)
        semantic_vec = transformer_outputs.last_hidden_state[:, 0, :]

        syntax_proj = None
        if not self.ablation_syntax:
            if self.zero_syntax or syntax_features is None:
                syn_in = torch.zeros(
                    (semantic_vec.size(0), self.syntax_dim),
                    device=semantic_vec.device,
                    dtype=torch.float32,
                )
            else:
                syn_in = syntax_features

            syntax_proj = self.syntax_projection(syn_in)
            fused_vec = torch.cat([semantic_vec, syntax_proj], dim=-1)
        else:
            fused_vec = semantic_vec

        logits = self.classifier(fused_vec)
        return logits, semantic_vec, syntax_proj

    def get_config(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "syntax_dim": self.syntax_dim,
            "syntax_proj_dim": self.syntax_proj_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "ablation_syntax": self.ablation_syntax,
            "zero_syntax": self.zero_syntax,
            "freeze_transformer": self.freeze_transformer,
        }

    def save_checkpoint(self, save_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
            "metadata": metadata or {},
        }
        torch.save(payload, save_path)
        print(f"[MODEL] Checkpoint saved successfully to {save_path}")

    @classmethod
    def load_checkpoint(cls, load_path: str, device: str = "cpu") -> Tuple["DarkPatternFusionModel", Dict[str, Any]]:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found at {load_path}")
        payload = torch.load(load_path, map_location=device, weights_only=False)
        config = payload["config"]
        metadata = payload.get("metadata", {})

        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        print(f"[MODEL] Loaded model from {load_path} on device {device}")
        return model, metadata
