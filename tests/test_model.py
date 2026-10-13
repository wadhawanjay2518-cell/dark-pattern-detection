"""
tests/test_model.py

Tests for DarkPatternFusionModel architecture, late fusion layer,
forward pass dimensions for 4 classes, gradient flow, and ablations.
"""

import os
import pytest
import torch
from src.fusion_model import DarkPatternFusionModel
from src.syntactic_features import FEATURE_NAMES


@pytest.fixture
def sample_batch():
    b = 4
    seq_len = 16
    input_ids = torch.randint(0, 1000, (b, seq_len))
    attention_mask = torch.ones((b, seq_len), dtype=torch.long)
    syntax = torch.randn((b, len(FEATURE_NAMES)), dtype=torch.float32)
    return input_ids, attention_mask, syntax


def test_fusion_forward_pass_shapes(sample_batch):
    input_ids, attention_mask, syntax = sample_batch
    model = DarkPatternFusionModel(syntax_dim=len(FEATURE_NAMES), syntax_proj_dim=64, hidden_dim=128, num_classes=4)

    logits, sem_vec, syn_proj = model(input_ids, attention_mask, syntax)
    assert logits.shape == (4, 4), f"Expected (4, 4) logits, got {logits.shape}"
    assert sem_vec.shape == (4, 768)
    assert syn_proj.shape == (4, 64)


def test_syntax_ablation_mode(sample_batch):
    input_ids, attention_mask, _ = sample_batch
    model = DarkPatternFusionModel(num_classes=4, ablation_syntax=True)

    logits, sem_vec, syn_proj = model(input_ids, attention_mask)
    assert logits.shape == (4, 4)
    assert sem_vec.shape == (4, 768)
    assert syn_proj is None


def test_zero_syntax_diagnostic_mode(sample_batch):
    input_ids, attention_mask, syntax = sample_batch
    model = DarkPatternFusionModel(syntax_dim=len(FEATURE_NAMES), num_classes=4, zero_syntax=True)

    logits, sem_vec, syn_proj = model(input_ids, attention_mask, syntax)
    assert logits.shape == (4, 4)
    assert syn_proj is not None


def test_gradient_flow(sample_batch):
    input_ids, attention_mask, syntax = sample_batch
    model = DarkPatternFusionModel(syntax_dim=len(FEATURE_NAMES), num_classes=4)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits, _, _ = model(input_ids, attention_mask, syntax)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # Check gradients on semantic branch
    assert model.transformer.transformer.layer[0].attention.q_lin.weight.grad is not None
    # Check gradients on syntactic branch
    assert model.syntax_projection[0].weight.grad is not None
    # Check gradients on classifier head
    assert model.classifier[-1].weight.grad is not None


def test_save_and_load_checkpoint(sample_batch, tmp_path):
    input_ids, attention_mask, syntax = sample_batch
    model = DarkPatternFusionModel(syntax_dim=len(FEATURE_NAMES), num_classes=4)
    model.eval()
    save_path = os.path.join(tmp_path, "test_model.pt")

    model.save_checkpoint(save_path, metadata={"version": "test", "num_classes": 4})
    assert os.path.exists(save_path)

    loaded_model, meta = DarkPatternFusionModel.load_checkpoint(save_path, device="cpu")
    assert meta["version"] == "test"

    with torch.no_grad():
        logits_orig, _, _ = model(input_ids, attention_mask, syntax)
        logits_loaded, _, _ = loaded_model(input_ids, attention_mask, syntax)
        assert torch.allclose(logits_orig, logits_loaded, atol=1e-5)
