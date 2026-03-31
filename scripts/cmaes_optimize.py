#!/usr/bin/env python3
# Created: 2026-03-26
# Purpose: CMA-ES로 GRU 디스토션 모델 하이퍼파라미터 최적화
#          목적함수: 하모닉 재현 정확도 (스펙트럼 ESR)
# Usage: python scripts/cmaes_optimize.py [--plugin blackstar] [--generations 30] [--popsize 8]

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cma
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnstomps.training.model import NNStompGRU
from nnstomps.training.losses import ESRLoss, MultiScaleSTFTLoss, DCLoss
from nnstomps.training.dataset import create_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ============================================================================
# 플러그인별 설정 (train_all.py와 동일)
# ============================================================================
PLUGIN_CONFIGS = {
    "blackstar": {
        "cond_dim": 2,
        "cond_map": {
            0: [0.00, 0.00], 1: [0.25, 0.00], 2: [0.50, 0.00],
            3: [0.75, 0.00], 4: [1.00, 0.00], 5: [0.00, 0.00],
            6: [0.00, 0.25], 7: [0.00, 0.50], 8: [0.00, 0.75], 9: [0.00, 1.00],
        },
    },
    # 다른 플러그인은 여기에 추가
}


def harmonic_esr(pred: np.ndarray, target: np.ndarray, sr: int = 44100) -> float:
    """하모닉 영역(>500Hz)에서의 스펙트럼 ESR — CMA-ES 목적함수의 핵심

    기본 주파수를 제외한 하모닉 영역의 재현 정확도를 측정.
    일반 ESR과 달리, 하모닉이 부족하면 높은 값을 반환.
    """
    N = min(8192, len(pred))
    window = np.hanning(N).astype(np.float32)

    pred_spec = np.abs(np.fft.rfft(pred[:N] * window))
    target_spec = np.abs(np.fft.rfft(target[:N] * window))
    freqs = np.fft.rfftfreq(N, 1 / sr)

    # 500Hz 이상 (하모닉 영역)만 사용
    mask = freqs >= 500
    pred_h = pred_spec[mask]
    target_h = target_spec[mask]

    # 로그 스펙트럼 ESR (하모닉 레벨 차이에 민감)
    pred_log = np.log(pred_h + 1e-10)
    target_log = np.log(target_h + 1e-10)

    error = np.sum((pred_log - target_log) ** 2)
    signal = np.sum(target_log ** 2) + 1e-10

    return float(error / signal)


def pre_emphasis(audio: np.ndarray, coeff: float = 0.95) -> np.ndarray:
    """Pre-emphasis 필터: y[n] = x[n] - coeff * x[n-1]"""
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


class PreEmphasisLoss(nn.Module):
    """Pre-emphasis 적용 후 ESR — 고주파 하모닉 강조"""

    def __init__(self, coeff: float = 0.95):
        super().__init__()
        self.coeff = coeff

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # (batch, seq, 1) → pre-emphasis 적용
        p = pred.squeeze(-1)
        t = target.squeeze(-1)

        p_emph = p[:, 1:] - self.coeff * p[:, :-1]
        t_emph = t[:, 1:] - self.coeff * t[:, :-1]

        error = torch.sum((p_emph - t_emph) ** 2)
        signal = torch.sum(t_emph ** 2) + 1e-8
        return error / signal


def train_short(
    plugin: str,
    hidden_size: int,
    w_esr: float,
    w_stft: float,
    w_dc: float,
    w_preemph: float,
    preemph_coeff: float,
    lr: float,
    epochs: int = 30,
    device: str = "mps",
) -> dict:
    """짧은 학습 + 하모닉 ESR 평가 (CMA-ES 1회 호출용)"""
    pcfg = PLUGIN_CONFIGS[plugin]
    manifest = f"training_data/{plugin}/manifest.json"

    hidden_size = int(hidden_size)
    if hidden_size < 8:
        hidden_size = 8
    if hidden_size > 128:
        hidden_size = 128

    model = NNStompGRU(pcfg["cond_dim"], hidden_size).to(device)

    # 손실 함수 조합
    esr_fn = ESRLoss().to(device)
    stft_fn = MultiScaleSTFTLoss().to(device)
    dc_fn = DCLoss().to(device)
    preemph_fn = PreEmphasisLoss(preemph_coeff).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds, val_ds = create_split(
        manifest, seq_len=4096,
        cond_map=pcfg["cond_map"],
        val_ratio=0.1, seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)

            pred, _ = model(x, cond)

            loss = (w_esr * esr_fn(pred, target)
                    + w_stft * stft_fn(pred, target)
                    + w_dc * dc_fn(pred, target)
                    + w_preemph * preemph_fn(pred, target))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # 검증
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                target = batch["target"].to(device)
                cond = batch["cond"].to(device)
                pred, _ = model(x, cond)
                val_losses.append(esr_fn(pred, target).item())

        val_esr = np.mean(val_losses) if val_losses else 999
        if val_esr < best_val:
            best_val = val_esr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 하모닉 ESR 평가: 1kHz 사인파로 측정
    model.load_state_dict(best_state)
    model.eval()

    sr = 44100
    t = np.arange(int(sr * 2), dtype=np.float32) / sr
    test_sine = 0.5 * np.sin(2 * np.pi * 1000 * t)

    # 모델 추론
    with torch.no_grad():
        x = torch.from_numpy(test_sine).unsqueeze(0).unsqueeze(-1).to(device)
        # 중간 세팅 조건
        mid_cond = list(pcfg["cond_map"].values())[len(pcfg["cond_map"]) // 2]
        c = torch.tensor([mid_cond], dtype=torch.float32).to(device)
        pred_audio, _ = model(x, c)
        pred_np = pred_audio[0, :, 0].cpu().numpy()

    # 원본 플러그인 출력 로드
    ref_path = f"audio_demos/ref_{plugin}_*.wav"
    import glob
    ref_files = glob.glob(ref_path)
    if ref_files:
        ref_audio, _ = sf.read(ref_files[0], dtype="float32")
        if ref_audio.ndim == 2:
            ref_audio = ref_audio[:, 0]
        ref_audio = ref_audio[:len(pred_np)]
    else:
        ref_audio = test_sine[:len(pred_np)]

    h_esr = harmonic_esr(pred_np, ref_audio, sr)

    return {
        "val_esr": best_val,
        "harmonic_esr": h_esr,
        "hidden_size": hidden_size,
        "model_state": best_state,
        "config": {"cond_dim": pcfg["cond_dim"], "hidden_size": hidden_size},
    }


def objective(x: np.ndarray, plugin: str, epochs: int, device: str) -> float:
    """CMA-ES 목적함수

    탐색 벡터 x (7차원):
        [0] hidden_size (8~128)
        [1] w_esr (0~2)
        [2] w_stft (0~2)
        [3] w_dc (0~0.5)
        [4] w_preemph (0~2)
        [5] preemph_coeff (0.8~0.99)
        [6] lr (1e-5 ~ 1e-2, log scale)
    """
    hidden_size = int(np.clip(x[0], 8, 128))
    w_esr = float(np.clip(x[1], 0.01, 2.0))
    w_stft = float(np.clip(x[2], 0.01, 2.0))
    w_dc = float(np.clip(x[3], 0.0, 0.5))
    w_preemph = float(np.clip(x[4], 0.0, 2.0))
    preemph_coeff = float(np.clip(x[5], 0.8, 0.99))
    lr = float(10 ** np.clip(x[6], -4, -2))  # log scale

    try:
        result = train_short(
            plugin, hidden_size,
            w_esr, w_stft, w_dc, w_preemph, preemph_coeff,
            lr, epochs, device,
        )
        # 목적함수: 하모닉 ESR (최소화)
        # val_esr도 고려 (가중 합산)
        score = 0.7 * result["harmonic_esr"] + 0.3 * result["val_esr"]

        logger.info(
            f"  h={hidden_size:3d} w=[{w_esr:.2f},{w_stft:.2f},{w_dc:.2f},{w_preemph:.2f}] "
            f"pe={preemph_coeff:.2f} lr={lr:.1e} "
            f"→ h_esr={result['harmonic_esr']:.4f} v_esr={result['val_esr']:.4f} "
            f"score={score:.4f}"
        )
        return score

    except Exception as e:
        logger.warning(f"  ERROR: {e}")
        return 999.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", "-p", default="blackstar")
    parser.add_argument("--generations", "-g", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--epochs", "-e", type=int, default=30,
                        help="각 후보의 학습 에폭 수")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"CMA-ES 최적화: {args.plugin}, {args.generations}세대, popsize={args.popsize}")
    logger.info(f"각 후보 {args.epochs}에폭 학습, device={device}")

    # 초기값: 기존 학습에서 사용한 값 기반
    x0 = [
        40,     # hidden_size
        0.3,    # w_esr (기존 0.7에서 낮춤)
        1.0,    # w_stft (기존 0.25에서 대폭 올림)
        0.05,   # w_dc
        0.5,    # w_preemph (새로 추가)
        0.95,   # preemph_coeff
        -3.3,   # lr = 10^-3.3 ≈ 5e-4
    ]

    sigma0 = 0.5  # 초기 탐색 폭

    # CMA-ES 실행
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "maxiter": args.generations,
        "popsize": args.popsize,
        "seed": 42,
        "verbose": -1,  # 내부 로그 억제
    })

    best_score = float("inf")
    best_x = None
    best_result = None
    history = []

    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        logger.info(f"\n--- 세대 {gen} ---")
        for i, x in enumerate(solutions):
            logger.info(f"[{gen}/{args.generations}] 후보 {i+1}/{len(solutions)}")
            f = objective(x, args.plugin, args.epochs, device)
            fitnesses.append(f)

            if f < best_score:
                best_score = f
                best_x = x.copy()

        es.tell(solutions, fitnesses)

        gen_best = min(fitnesses)
        logger.info(f"세대 {gen} 완료: best={gen_best:.4f}, global_best={best_score:.4f}")
        history.append({
            "generation": gen,
            "best_fitness": gen_best,
            "global_best": best_score,
        })
        gen += 1

    # 최적 파라미터로 최종 학습 (더 긴 에폭)
    logger.info(f"\n{'='*60}")
    logger.info(f"최적 파라미터 발견! 최종 학습 (100에폭)...")
    logger.info(f"{'='*60}")

    final_hidden = int(np.clip(best_x[0], 8, 128))
    final_w_esr = float(np.clip(best_x[1], 0.01, 2.0))
    final_w_stft = float(np.clip(best_x[2], 0.01, 2.0))
    final_w_dc = float(np.clip(best_x[3], 0.0, 0.5))
    final_w_preemph = float(np.clip(best_x[4], 0.0, 2.0))
    final_preemph = float(np.clip(best_x[5], 0.8, 0.99))
    final_lr = float(10 ** np.clip(best_x[6], -4, -2))

    logger.info(f"  hidden={final_hidden}, w_esr={final_w_esr:.3f}, "
                f"w_stft={final_w_stft:.3f}, w_dc={final_w_dc:.3f}, "
                f"w_preemph={final_w_preemph:.3f}, preemph={final_preemph:.3f}, "
                f"lr={final_lr:.1e}")

    final = train_short(
        args.plugin, final_hidden,
        final_w_esr, final_w_stft, final_w_dc,
        final_w_preemph, final_preemph, final_lr,
        epochs=100, device=device,
    )

    # 저장
    out_dir = Path(f"models/{args.plugin}_cmaes")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": final["model_state"],
        "config": final["config"],
        "epoch": 100,
        "val_esr": final["val_esr"],
        "harmonic_esr": final["harmonic_esr"],
        "cmaes_params": {
            "hidden_size": final_hidden,
            "w_esr": final_w_esr, "w_stft": final_w_stft,
            "w_dc": final_w_dc, "w_preemph": final_w_preemph,
            "preemph_coeff": final_preemph, "lr": final_lr,
        },
    }, str(out_dir / "best_model.pt"))

    # 이력 저장
    (out_dir / "cmaes_history.json").write_text(json.dumps(history, indent=2))

    # 오디오 렌더링
    pcfg = PLUGIN_CONFIGS[args.plugin]
    model = NNStompGRU(pcfg["cond_dim"], final_hidden)
    model.load_state_dict(final["model_state"])
    model.eval()

    sr = 44100
    t_arr = np.arange(int(sr * 3), dtype=np.float32) / sr
    test_sine = 0.5 * np.sin(2 * np.pi * 1000 * t_arr)

    mid_cond = list(pcfg["cond_map"].values())[len(pcfg["cond_map"]) // 2]
    with torch.no_grad():
        x_t = torch.from_numpy(test_sine).unsqueeze(0).unsqueeze(-1)
        c_t = torch.tensor([mid_cond], dtype=torch.float32)
        pred, _ = model(x_t, c_t)
        pred_np = pred[0, :, 0].numpy()

    sf.write(str(out_dir / f"wet_{args.plugin}_cmaes.wav"), pred_np, sr, subtype="FLOAT")

    logger.info(f"\n최종 결과:")
    logger.info(f"  val_esr:     {final['val_esr']:.5f}")
    logger.info(f"  harmonic_esr: {final['harmonic_esr']:.5f}")
    logger.info(f"  모델: {out_dir / 'best_model.pt'}")
    logger.info(f"  오디오: {out_dir / f'wet_{args.plugin}_cmaes.wav'}")


if __name__ == "__main__":
    main()
