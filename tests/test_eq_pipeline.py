# Created: 2026-09-15
# Purpose: EQ 파이프라인 가드 테스트 — export 계약, 학습 설정 검증

import json

import pytest
import torch

from nnstomps.training.eq_dataset import EQ_PLUGIN_CONFIGS
from nnstomps.training.export_eq import _build_controls_metadata, export_eq_to_onnx
from nnstomps.training.train_eq import EQTrainConfig, train_eq

onnx = pytest.importorskip("onnx", reason="onnx is a training extra")


def _checkpoint(tmp_path, plugin_name="channel_eq", fir_len=64, hidden=(16, 16)):
    from nnstomps.training.eq_dataset import encoded_dim, param_spec_for
    from nnstomps.training.eq_model import NNStompEQ

    spec = param_spec_for(plugin_name)
    model = NNStompEQ(encoded_dim(spec), fir_len, list(hidden))
    path = tmp_path / "best_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "cond_dim": encoded_dim(spec), "fir_len": fir_len,
            "hidden_dims": list(hidden), "plugin_name": plugin_name,
            "param_spec": spec,
        },
        "epoch": 0, "val_mag_error": 0.25,
    }, str(path))
    return str(path)


def test_export_metadata_declares_the_natural_unit_contract(tmp_path):
    """Consumers must be told the domain; the engine normalizes natural units."""
    ckpt = _checkpoint(tmp_path)
    meta = export_eq_to_onnx(ckpt, str(tmp_path / "m.onnx"))

    assert meta["controls_domain"] == "natural"
    assert meta["param_spec"]["params"] == EQ_PLUGIN_CONFIGS["channel_eq"]["params"]
    assert len(meta["controls"]) == 4


def test_export_rejects_an_unknown_plugin_name(tmp_path):
    """Silently emitting controls: [] would render a UI with no sliders — and
    would publish whatever name the checkpoint carries."""
    ckpt = _checkpoint(tmp_path)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    blob["config"]["plugin_name"] = "commercial_product_name"
    torch.save(blob, ckpt)

    with pytest.raises(KeyError, match="not in EQ_PLUGIN_CONFIGS"):
        export_eq_to_onnx(ckpt, str(tmp_path / "m.onnx"))


def test_controls_cover_every_declared_parameter():
    for plugin_name, cfg in EQ_PLUGIN_CONFIGS.items():
        controls = _build_controls_metadata(plugin_name)
        assert [c["name"] for c in controls] == cfg["params"], plugin_name


def test_continuous_controls_report_natural_bounds():
    controls = {c["name"]: c for c in _build_controls_metadata("channel_eq")}
    assert controls["hf_gain"]["type"] == "continuous"
    assert (controls["hf_gain"]["min"], controls["hf_gain"]["max"]) == (-12, 12)


def test_categorical_controls_list_their_options():
    controls = {c["name"]: c for c in _build_controls_metadata("passive_eq")}
    assert controls["ch1loenable"]["type"] == "categorical"
    assert controls["ch1loenable"]["options"] == ["OUT", "BOOST", "CUT"]


def test_metadata_round_trips_as_json(tmp_path):
    ckpt = _checkpoint(tmp_path)
    out = tmp_path / "meta.json"
    export_eq_to_onnx(ckpt, str(tmp_path / "m.onnx"), metadata_path=str(out))

    reloaded = json.loads(out.read_text())
    assert reloaded["controls_domain"] == "natural"


def test_training_rejects_non_positive_epochs(tmp_path):
    """A zero-epoch run used to save model_state: None and report success."""
    cfg = EQTrainConfig(plugin_name="channel_eq", plugin_dir=str(tmp_path), epochs=0)
    with pytest.raises(ValueError, match="epochs must be positive"):
        train_eq(cfg)


def test_training_rejects_fir_len_that_contradicts_the_contract(tmp_path):
    cfg = EQTrainConfig(plugin_name="channel_eq", plugin_dir=str(tmp_path), fir_len=512)
    with pytest.raises(ValueError, match="does not match the plugin contract"):
        train_eq(cfg)