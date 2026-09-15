# Created: 2026-09-15
# Purpose: NNStompEQ 모델 계약 테스트 — 입출력 shape 및 파라미터 수

import pytest
import torch

from nnstomps.training.eq_model import NNStompEQ


def test_output_shape_is_batch_by_fir_len():
    model = NNStompEQ(cond_dim=4, fir_len=256)
    out = model(torch.randn(8, 4))
    assert out.shape == (8, 256)


def test_single_sample_unbatched_input():
    model = NNStompEQ(cond_dim=3, fir_len=128)
    out = model(torch.randn(1, 3))
    assert out.shape == (1, 128)


def test_parameter_count_matches_the_recorded_training_summary():
    """config [128, 256, 256] with cond_dim=4 and fir_len=256 is the shape whose
    parameter count is recorded in models/eq_training_summary.json."""
    model = NNStompEQ(cond_dim=4, fir_len=256, hidden_dims=[128, 256, 256])
    n = sum(p.numel() for p in model.parameters())
    assert n == 165_248


def test_custom_hidden_dims_are_respected():
    model = NNStompEQ(cond_dim=4, fir_len=64, hidden_dims=[16, 16])
    out = model(torch.randn(2, 4))
    assert out.shape == (2, 64)


def test_default_hidden_dims_are_used_when_omitted():
    a = NNStompEQ(cond_dim=4, fir_len=64)
    b = NNStompEQ(cond_dim=4, fir_len=64, hidden_dims=[128, 256, 256])
    assert sum(p.numel() for p in a.parameters()) == sum(
        p.numel() for p in b.parameters()
    )


def test_frequency_response_shape_and_units():
    n_fft = 4096
    model = NNStompEQ(cond_dim=4, fir_len=256)
    mag_db, phase = model.predict_frequency_response(torch.randn(2, 4), n_fft=n_fft)

    assert mag_db.shape == (2, n_fft // 2 + 1)
    assert phase.shape == (2, n_fft // 2 + 1)
    assert torch.isfinite(mag_db).all()
    assert torch.isfinite(phase).all()
    assert phase.abs().max() <= torch.pi + 1e-5


def test_gradients_flow_to_all_parameters():
    model = NNStompEQ(cond_dim=4, fir_len=64, hidden_dims=[16, 16])
    model(torch.randn(4, 4)).sum().backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"