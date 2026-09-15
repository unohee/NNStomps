# Created: 2026-09-15
# Purpose: pytest 공용 픽스처 — src 경로 및 합성 EQ 체크포인트

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nnstomps.training.eq_dataset import encoded_dim, param_spec_for  # noqa: E402
from nnstomps.training.eq_model import NNStompEQ  # noqa: E402


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    """A tiny Neural EQ checkpoint.

    Built from scratch rather than loaded from models/, so the suite does not
    depend on gitignored training artifacts.
    """
    def _make(plugin_name="channel_eq", fir_len=64, hidden_dims=(16, 16), with_spec=True):
        torch.manual_seed(0)
        spec = param_spec_for(plugin_name)
        cond_dim = encoded_dim(spec)

        model = NNStompEQ(cond_dim, fir_len, list(hidden_dims))
        config = {
            "cond_dim": cond_dim,
            "fir_len": fir_len,
            "hidden_dims": list(hidden_dims),
            "plugin_name": plugin_name,
        }
        if with_spec:
            config["param_spec"] = spec

        path = tmp_path / f"{plugin_name}_{fir_len}.pt"
        torch.save(
            {"model_state": model.state_dict(), "config": config,
             "epoch": 0, "val_mag_error": 0.1},
            str(path),
        )
        return str(path)

    return _make