#!/usr/bin/env python3
# Created: 2026-03-26
# Purpose: NNStomps Gradio 데모 — GRU 디스토션 모델 실시간 시연
# Usage: python scripts/demo.py

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnstomps.training.model import NNStompGRU

# ============================================================================
# 모델 레지스트리
# ============================================================================

MODELS = {
    # v2 모델 (대규모 데이터 + pre-emphasis + 올바른 파라미터)
    "Blackstar (Drive A/B)": {
        "path": "models/blackstar_v2/best_model.pt",
        "cond_dim": 2,
        "controls": {
            "Drive A": {"idx": 0, "min": 0, "max": 100, "default": 50},
            "Drive B": {"idx": 1, "min": 0, "max": 100, "default": 0},
        },
    },
    # 다른 모델은 data/ 디렉토리에 프로파일 추가 후 여기에 등록
}

# 기본 샘플
DEFAULT_SAMPLE = "/Volumes/Library SSD/Audio Asset/rhythm-lab.com_amen_vol.1/WAV/cw_amen01_175.wav"

# 모델 캐시
_model_cache: dict[str, NNStompGRU] = {}


def load_model_cached(name: str) -> NNStompGRU | None:
    if name in _model_cache:
        return _model_cache[name]

    cfg = MODELS.get(name)
    if not cfg or not Path(cfg["path"]).exists():
        return None

    ckpt = torch.load(cfg["path"], map_location="cpu", weights_only=True)
    model = NNStompGRU(ckpt["config"]["cond_dim"], ckpt["config"]["hidden_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _model_cache[name] = model
    return model


def process_audio(
    audio_input,
    model_name: str,
    param1: float,
    param2: float,
    mix: float,
    input_gain_db: float,
):
    """Gradio 처리 함수"""
    if audio_input is None:
        return None

    # Gradio audio: (sr, np.ndarray)
    sr, data = audio_input

    # float32 변환
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # 스테레오 → 모노
    if data.ndim == 2:
        mono = data.mean(axis=1) if data.shape[1] <= 2 else data.mean(axis=0)
    else:
        mono = data

    # 입력 게인
    gain = 10 ** (input_gain_db / 20.0)
    mono = mono * gain

    # 모델 로드
    model = load_model_cached(model_name)
    if model is None:
        return (sr, mono)

    cfg = MODELS[model_name]
    controls = cfg["controls"]

    # 조건 벡터 구성
    cond = [0.0] * cfg["cond_dim"]
    ctrl_list = list(controls.values())

    if len(ctrl_list) >= 1:
        c = ctrl_list[0]
        cond[c["idx"]] = (param1 - c["min"]) / (c["max"] - c["min"])
    if len(ctrl_list) >= 2:
        c = ctrl_list[1]
        cond[c["idx"]] = (param2 - c["min"]) / (c["max"] - c["min"])

    # GRU 추론 (청크 처리)
    chunk_size = 8192
    output = np.zeros_like(mono)
    hidden = None

    with torch.no_grad():
        cond_t = torch.tensor([cond], dtype=torch.float32)
        for start in range(0, len(mono), chunk_size):
            end = min(start + chunk_size, len(mono))
            chunk = mono[start:end]
            x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(-1)
            pred, hidden = model(x, cond_t, hidden)
            output[start:end] = pred[0, :, 0].numpy()

    # Dry/Wet 믹스
    wet = mono * (1 - mix) + output * mix

    # 클리핑 방지
    peak = np.max(np.abs(wet))
    if peak > 0.99:
        wet = wet * (0.99 / peak)

    return (sr, wet.astype(np.float32))


def update_controls(model_name: str):
    """모델 선택 시 컨트롤 라벨/범위 업데이트"""
    cfg = MODELS.get(model_name, {})
    controls = cfg.get("controls", {})
    ctrl_list = list(controls.items())

    if len(ctrl_list) >= 1:
        name1, c1 = ctrl_list[0]
        p1_update = gr.update(
            label=name1, minimum=c1["min"], maximum=c1["max"],
            value=c1["default"], visible=True,
        )
    else:
        p1_update = gr.update(visible=False)

    if len(ctrl_list) >= 2:
        name2, c2 = ctrl_list[1]
        p2_update = gr.update(
            label=name2, minimum=c2["min"], maximum=c2["max"],
            value=c2["default"], visible=True,
        )
    else:
        p2_update = gr.update(visible=False, value=0)

    return p1_update, p2_update


def build_ui():
    with gr.Blocks(
        title="NNStomps — Neural Drive",
        theme=gr.themes.Soft(primary_hue="orange"),
    ) as demo:
        gr.Markdown(
            "# NNStomps — Neural Drive\n"
            "GRU 신경망 기반 새추레이션/디스토션. "
            "플러그인을 프로파일링한 데이터로 학습된 모델입니다."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_sel = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                    label="Model",
                )

                param1 = gr.Slider(
                    minimum=0, maximum=100, value=50, step=1,
                    label="Pentode",
                )
                param2 = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1,
                    label="Triode",
                )

                input_gain = gr.Slider(
                    minimum=-12, maximum=12, value=0, step=0.5,
                    label="Input Gain (dB)",
                )
                mix_slider = gr.Slider(
                    minimum=0, maximum=1.0, value=1.0, step=0.05,
                    label="Dry/Wet Mix",
                )

                process_btn = gr.Button("Process", variant="primary", size="lg")

            with gr.Column(scale=2):
                audio_in = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    value=DEFAULT_SAMPLE if Path(DEFAULT_SAMPLE).exists() else None,
                )
                audio_out = gr.Audio(label="Output Audio", type="numpy")

        # 모델 변경 시 컨트롤 업데이트
        model_sel.change(
            fn=update_controls,
            inputs=[model_sel],
            outputs=[param1, param2],
        )

        # 처리 버튼
        process_btn.click(
            fn=process_audio,
            inputs=[audio_in, model_sel, param1, param2, mix_slider, input_gain],
            outputs=[audio_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=7870, share=False)
