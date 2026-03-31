# Created: 2026-03-25
# Purpose: PyTorch GRU → RTNeural JSON 변환
# Dependencies: torch, json
# Test Status: 미완료
#
# RTNeural JSON 포맷:
#   GRU weights: W_ih (input), W_hh (recurrent), b_ih, b_hh
#   PyTorch GRU gate 순서: [reset, update, new] (각 hidden_size 행)
#   RTNeural도 동일한 순서 사용

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nnstomps.training.model import NNStompGRU, NNStompGRU2

logger = logging.getLogger(__name__)


def export_gru_to_rtneural(
    model_path: str,
    output_path: str,
    model_class: str = "NNStompGRU",
) -> dict:
    """PyTorch GRU 모델 → RTNeural JSON

    Args:
        model_path: best_model.pt 경로
        output_path: 출력 JSON 경로
        model_class: "NNStompGRU" 또는 "NNStompGRU2"

    Returns: {"output_path", "input_size", "output_size", "n_layers", "n_params"}
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state"]
    config = ckpt["config"]

    cond_dim = config["cond_dim"]
    hidden_size = config["hidden_size"]
    input_size = 1 + cond_dim

    layers = []
    total_params = 0

    if model_class == "NNStompGRU":
        # GRU 레이어
        gru_layer = _export_gru_layer(state, "gru", input_size, hidden_size)
        layers.append(gru_layer)
        total_params += gru_layer["n_params"]

        # Dense 레이어
        dense_layer = _export_dense_layer(state, "dense", hidden_size, 1, "tanh")
        layers.append(dense_layer)
        total_params += dense_layer["n_params"]

    elif model_class == "NNStompGRU2":
        hidden1 = config.get("hidden1", hidden_size)
        hidden2 = config.get("hidden2", hidden_size // 2)

        # GRU 1
        gru1 = _export_gru_layer(state, "gru1", input_size, hidden1)
        layers.append(gru1)
        total_params += gru1["n_params"]

        # GRU 2
        gru2 = _export_gru_layer(state, "gru2", hidden1, hidden2)
        layers.append(gru2)
        total_params += gru2["n_params"]

        # Dense
        dense = _export_dense_layer(state, "dense", hidden2, 1, "tanh")
        layers.append(dense)
        total_params += dense["n_params"]

    else:
        raise ValueError(f"지원하지 않는 모델: {model_class}")

    # RTNeural JSON 구성
    rtneural_json = {
        "model_class": model_class,
        "input_size": input_size,
        "output_size": 1,
        "cond_dim": cond_dim,
        "hidden_size": hidden_size,
        "n_params": total_params,
        "layers": [_clean_layer(l) for l in layers],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(rtneural_json, indent=2))

    logger.info(f"RTNeural JSON 내보내기: {output_path} "
                f"(input={input_size}, params={total_params:,})")

    return {
        "output_path": output_path,
        "input_size": input_size,
        "output_size": 1,
        "n_layers": len(layers),
        "n_params": total_params,
    }


def _export_gru_layer(
    state: dict, prefix: str, input_size: int, hidden_size: int
) -> dict:
    """PyTorch GRU state → RTNeural GRU 레이어 딕셔너리"""
    # PyTorch weight 키: {prefix}.weight_ih_l0, {prefix}.weight_hh_l0,
    #                     {prefix}.bias_ih_l0, {prefix}.bias_hh_l0
    W_ih = state[f"{prefix}.weight_ih_l0"].numpy()  # (3*H, input_size)
    W_hh = state[f"{prefix}.weight_hh_l0"].numpy()  # (3*H, H)
    b_ih = state[f"{prefix}.bias_ih_l0"].numpy()     # (3*H,)
    b_hh = state[f"{prefix}.bias_hh_l0"].numpy()     # (3*H,)

    n_params = W_ih.size + W_hh.size + b_ih.size + b_hh.size

    return {
        "type": "gru",
        "shape": [input_size, hidden_size],
        "weights": {
            "W": W_ih.flatten().tolist(),
            "U": W_hh.flatten().tolist(),
            "b_W": b_ih.tolist(),
            "b_U": b_hh.tolist(),
        },
        "n_params": n_params,
    }


def _export_dense_layer(
    state: dict, prefix: str, in_features: int, out_features: int,
    activation: str = "tanh",
) -> dict:
    """PyTorch Linear state → RTNeural Dense 레이어 딕셔너리"""
    W = state[f"{prefix}.weight"].numpy()  # (out, in)
    b = state[f"{prefix}.bias"].numpy()    # (out,)

    n_params = W.size + b.size

    return {
        "type": "dense",
        "shape": [in_features, out_features],
        "weights": {
            "W": W.flatten().tolist(),
            "b": b.tolist(),
        },
        "activation": activation,
        "n_params": n_params,
    }


def _clean_layer(layer: dict) -> dict:
    """n_params 등 내부 필드 제거 (RTNeural JSON 정규화)"""
    return {k: v for k, v in layer.items() if k != "n_params"}


def verify_export(
    model_path: str,
    rtneural_json_path: str,
    test_length: int = 1000,
    cond_dim: int | None = None,
    atol: float = 1e-5,
) -> dict:
    """PyTorch 모델과 RTNeural JSON의 출력 일치 검증

    PyTorch 모델로 순전파한 결과와, RTNeural JSON에서 수동으로
    GRU를 재현한 결과를 비교합니다.

    Returns: {"max_diff", "mean_diff", "passed"}
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]

    if cond_dim is None:
        cond_dim = config["cond_dim"]

    model = NNStompGRU(cond_dim, config["hidden_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 테스트 입력 생성
    rng = np.random.RandomState(42)
    test_input = rng.randn(1, test_length, 1).astype(np.float32) * 0.5
    test_cond = rng.rand(1, cond_dim).astype(np.float32)

    x = torch.from_numpy(test_input)
    c = torch.from_numpy(test_cond)

    with torch.no_grad():
        pytorch_out, _ = model(x, c)
    pytorch_out = pytorch_out.numpy()

    # RTNeural JSON에서 수동 GRU 재현
    rt_json = json.loads(Path(rtneural_json_path).read_text())
    rt_out = _manual_gru_forward(rt_json, test_input[0], test_cond[0])

    diff = np.abs(pytorch_out[0] - rt_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    passed = max_diff < atol

    logger.info(f"검증: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
                f"passed={passed} (atol={atol})")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "passed": passed,
        "atol": atol,
    }


def _manual_gru_forward(
    rt_json: dict, x: np.ndarray, cond: np.ndarray
) -> np.ndarray:
    """RTNeural JSON 가중치로 수동 GRU 순전파 (검증용)

    Args:
        rt_json: RTNeural JSON
        x: (seq_len, 1) 입력
        cond: (cond_dim,) 조건 벡터

    Returns: (seq_len, 1) 출력
    """
    layers = rt_json["layers"]
    seq_len = len(x)

    # GRU 레이어 파싱
    gru_l = layers[0]
    assert gru_l["type"] == "gru"
    input_size, hidden_size = gru_l["shape"]

    W = np.array(gru_l["weights"]["W"]).reshape(3 * hidden_size, input_size)
    U = np.array(gru_l["weights"]["U"]).reshape(3 * hidden_size, hidden_size)
    b_W = np.array(gru_l["weights"]["b_W"])
    b_U = np.array(gru_l["weights"]["b_U"])

    # Dense 레이어 파싱
    dense_l = layers[1]
    assert dense_l["type"] == "dense"
    W_d = np.array(dense_l["weights"]["W"]).reshape(
        dense_l["shape"][1], dense_l["shape"][0]
    )
    b_d = np.array(dense_l["weights"]["b"])
    activation = dense_l.get("activation", "tanh")

    # GRU 순전파
    h = np.zeros(hidden_size, dtype=np.float32)
    outputs = []

    for t in range(seq_len):
        # 입력: [x(t), cond]
        inp = np.concatenate([x[t], cond]).astype(np.float32)

        # gates: W*inp + b_W, U*h + b_U
        w_gates = W @ inp + b_W
        u_gates = U @ h + b_U

        H = hidden_size
        # reset gate
        r = _sigmoid(w_gates[:H] + u_gates[:H])
        # update gate
        z = _sigmoid(w_gates[H:2*H] + u_gates[H:2*H])
        # new gate
        n = np.tanh(w_gates[2*H:3*H] + r * u_gates[2*H:3*H])

        h = (1 - z) * n + z * h

        # Dense + activation
        out = W_d @ h + b_d
        if activation == "tanh":
            out = np.tanh(out)

        outputs.append(out)

    return np.array(outputs, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
