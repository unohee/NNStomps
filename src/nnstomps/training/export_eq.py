# Created: 2026-04-01
# Purpose: Neural EQ 모델 → ONNX export + 메타데이터

import json
import logging
from pathlib import Path

import torch

from .eq_model import NNStompEQ
from .eq_dataset import EQ_PLUGIN_CONFIGS, param_spec_for

logger = logging.getLogger(__name__)


def export_eq_to_onnx(
    model_path: str,
    output_path: str,
    metadata_path: str | None = None,
) -> dict:
    """Export a Neural EQ model to ONNX plus UI metadata.

    Args:
        model_path: best_model.pt path
        output_path: .onnx output path
        metadata_path: metadata JSON path (slider spec for the web UI)
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]

    plugin_name = cfg.get("plugin_name", "")
    # Fail loudly on an unknown plugin. A silent fallback here would emit
    # `controls: []` (the web UI renders zero sliders) and would write whatever
    # name the checkpoint carries straight into the published metadata.
    if plugin_name not in EQ_PLUGIN_CONFIGS:
        raise KeyError(
            f"checkpoint reports plugin_name={plugin_name!r}, which is not in "
            f"EQ_PLUGIN_CONFIGS {sorted(EQ_PLUGIN_CONFIGS)}; refusing to export"
        )

    model = NNStompEQ(cfg["cond_dim"], cfg["fir_len"], cfg.get("hidden_dims"))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, cfg["cond_dim"])

    torch.onnx.export(
        model, dummy,
        output_path,
        input_names=["parameters"],
        output_names=["fir_coefficients"],
        opset_version=17,
        dynamo=False,
    )

    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f"ONNX: {output_path} ({size_kb:.1f} KB)")

    spec = cfg.get("param_spec") or param_spec_for(plugin_name)
    metadata = {
        "plugin_name": plugin_name,
        "cond_dim": cfg["cond_dim"],
        "fir_len": cfg["fir_len"],
        "val_mag_error": ckpt.get("val_mag_error"),
        # Controls are reported in natural units (dB / seconds). The engine
        # normalizes before feeding the model — see NeuralEQEngine.set_params.
        "controls_domain": "natural",
        "controls": _build_controls_metadata(plugin_name),
        "param_spec": spec,
    }

    if metadata_path:
        Path(metadata_path).write_text(json.dumps(metadata, indent=2))
        logger.info(f"Metadata: {metadata_path}")

    return metadata


def _build_controls_metadata(plugin_name: str) -> list[dict]:
    """Build the slider spec for the web UI, in natural units."""
    cfg = EQ_PLUGIN_CONFIGS[plugin_name]
    param_names = cfg["params"]
    ranges = cfg.get("ranges", {})
    categorical = cfg.get("categorical", {})

    controls = []
    for p in param_names:
        cats = categorical.get(p)
        if cats:
            controls.append({
                "name": p,
                "type": "categorical",
                "options": cats,
                "default": cats[0],
            })
        else:
            lo, hi = ranges.get(p, (0, 1))
            controls.append({
                "name": p,
                "type": "continuous",
                "min": lo,
                "max": hi,
                "default": (lo + hi) / 2,
                "step": 0.5,
            })

    if not controls:
        raise ValueError(f"{plugin_name}: built an empty control set — check the config")

    return controls