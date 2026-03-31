# Created: 2026-03-25
# Purpose: 모델 평가 — A/B 비교, ESR 계산, 오디오 렌더링
# Dependencies: torch, soundfile, numpy

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from nnstomps.training.model import NNStompGRU

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> tuple[NNStompGRU, dict]:
    """저장된 모델 로드

    Returns: (model, config)
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    model = NNStompGRU(config["cond_dim"], config["hidden_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


def compute_esr(pred: np.ndarray, target: np.ndarray) -> float:
    """Error-to-Signal Ratio (낮을수록 좋음)"""
    error = pred - target
    return float(np.sum(error ** 2) / (np.sum(target ** 2) + 1e-10))


def process_audio(
    model: NNStompGRU,
    audio: np.ndarray,
    cond: list[float],
    chunk_size: int = 8192,
) -> np.ndarray:
    """모델로 오디오 처리 (긴 파일도 청크 단위로)

    Args:
        model: 학습된 GRU 모델
        audio: (samples,) float32 모노 오디오
        cond: 조건 벡터
        chunk_size: 처리 청크 크기

    Returns: (samples,) float32 처리된 오디오
    """
    model.eval()
    cond_t = torch.tensor([cond], dtype=torch.float32)

    n = len(audio)
    output = np.zeros(n, dtype=np.float32)
    hidden = None

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = audio[start:end]

            x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(-1)  # (1, len, 1)
            pred, hidden = model(x, cond_t, hidden)

            output[start:end] = pred[0, :, 0].numpy()

    return output


def evaluate_pair(
    model_path: str,
    input_wav: str,
    target_wav: str,
    cond: list[float],
    output_wav: str | None = None,
    channel: int = 0,
) -> dict:
    """단일 input/output 쌍에 대한 모델 평가

    Returns: {"esr", "pred_peak", "target_peak", "output_wav"}
    """
    model, config = load_model(model_path)

    # 오디오 로드
    in_audio, sr = sf.read(input_wav, dtype="float32", always_2d=True)
    target_audio, _ = sf.read(target_wav, dtype="float32", always_2d=True)

    in_mono = in_audio[:, channel]
    target_mono = target_audio[:, min(channel, target_audio.shape[1] - 1)]

    # 모델 추론
    pred = process_audio(model, in_mono, cond)

    # ESR 계산
    esr = compute_esr(pred, target_mono)

    result = {
        "esr": round(esr, 6),
        "pred_peak": round(float(np.max(np.abs(pred))), 4),
        "target_peak": round(float(np.max(np.abs(target_mono))), 4),
        "sample_rate": sr,
        "samples": len(in_mono),
    }

    # 결과 저장
    if output_wav:
        sf.write(output_wav, pred, sr, subtype="FLOAT")
        result["output_wav"] = output_wav

    return result


def evaluate_manifest(
    model_path: str,
    manifest_path: str,
    cond_map: dict[int, list[float]],
    output_dir: str | None = None,
    max_pairs: int | None = None,
) -> dict:
    """manifest의 모든 쌍을 평가

    Returns: {"mean_esr", "results": [...]}
    """
    manifest = json.loads(Path(manifest_path).read_text())
    base = Path(manifest_path).parent

    out_dir = None
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, pair in enumerate(manifest["pairs"]):
        if max_pairs and i >= max_pairs:
            break

        cond = cond_map.get(pair["setting_idx"], [])
        out_wav = None
        if out_dir:
            out_wav = str(out_dir / f"pred_{pair['setting_idx']:03d}_{pair['signal']}.wav")

        r = evaluate_pair(
            model_path,
            str(base / pair["input"]),
            str(base / pair["output"]),
            cond,
            out_wav,
        )
        r["setting_idx"] = pair["setting_idx"]
        r["signal"] = pair["signal"]
        results.append(r)

        logger.info(f"[{i+1}/{len(manifest['pairs'])}] "
                    f"{pair['signal']} set={pair['setting_idx']} ESR={r['esr']:.5f}")

    esr_values = [r["esr"] for r in results]
    summary = {
        "mean_esr": round(float(np.mean(esr_values)), 6) if esr_values else 0,
        "min_esr": round(float(np.min(esr_values)), 6) if esr_values else 0,
        "max_esr": round(float(np.max(esr_values)), 6) if esr_values else 0,
        "n_pairs": len(results),
        "results": results,
    }

    logger.info(f"평가 완료: mean_esr={summary['mean_esr']:.5f} "
                f"({summary['n_pairs']} 쌍)")

    return summary
