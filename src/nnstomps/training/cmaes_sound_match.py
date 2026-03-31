#!/usr/bin/env python3
# Created: 2026-03-26
# Purpose: CMA-ES render-in-the-loop 사운드 매칭
#          원본 플러그인 출력을 앵커(타겟)로, GRU 모델 출력을 반복적으로 최적화
#
# NeuralSound cmaes_optim.py 패턴 차용:
#   타겟(원본 플러그인) → 스펙트럼 추출 → CMA-ES 루프:
#     모델 파라미터 후보 → GRU 렌더링 → 스펙트럼 추출 → loss → 반복
#
# 최적화 대상: GRU hidden state 초기값 + 조건 벡터 미세조정
# 손실: 하모닉 스펙트럼 매칭 (로그 magnitude + 위상 무시)

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

_SR = 44100
_N_FFT = 8192


@dataclass
class SoundMatchResult:
    """사운드 매칭 결과"""
    cond_vector: list[float]        # 최적화된 조건 벡터
    hidden_init: np.ndarray         # 최적화된 hidden state 초기값
    loss: float                     # 최종 스펙트럼 손실
    harmonic_esr: float             # 하모닉 ESR
    n_evals: int                    # 평가 횟수
    elapsed_sec: float              # 소요 시간
    generations: int                # 세대 수


def extract_harmonic_spectrum(
    audio: np.ndarray,
    sr: int = _SR,
    n_fft: int = _N_FFT,
    min_freq: float = 100.0,
) -> np.ndarray:
    """오디오에서 로그 magnitude 스펙트럼 추출 (하모닉 영역)

    Returns: (n_bins,) 로그 magnitude — min_freq 이상만
    """
    if audio.ndim == 2:
        audio = audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]

    # 안정 구간 사용
    skip = min(int(0.2 * sr), len(audio) // 4)
    seg = audio[skip:skip + n_fft]
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))

    window = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # min_freq 이상만
    mask = freqs >= min_freq
    log_spec = np.log(spec[mask] + 1e-10)

    return log_spec.astype(np.float32)


def spectral_loss(
    target_spec: np.ndarray,
    candidate_spec: np.ndarray,
) -> float:
    """로그 스펙트럼 간 MSE"""
    ml = min(len(target_spec), len(candidate_spec))
    return float(np.mean((target_spec[:ml] - candidate_spec[:ml]) ** 2))


def harmonic_peaks_loss(
    target: np.ndarray,
    candidate: np.ndarray,
    fundamental_freq: float,
    sr: int = _SR,
    n_fft: int = _N_FFT,
    n_harmonics: int = 8,
) -> float:
    """하모닉 피크별 dB 차이 (가장 직접적인 하모닉 매칭)

    각 하모닉(H1~H8)의 피크 레벨을 비교.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    def _get_peaks(audio):
        if audio.ndim == 2:
            audio = audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]
        skip = min(int(0.2 * sr), len(audio) // 4)
        seg = audio[skip:skip + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        window = np.hanning(n_fft).astype(np.float32)
        spec = np.abs(np.fft.rfft(seg * window))
        spec_db = 20 * np.log10(spec + 1e-10)

        peaks = []
        for h in range(1, n_harmonics + 1):
            f = h * fundamental_freq
            bin_idx = int(round(f * n_fft / sr))
            if bin_idx >= len(spec_db):
                break
            search = max(3, bin_idx // 20)
            peak = spec_db[max(0, bin_idx - search):min(len(spec_db), bin_idx + search)].max()
            peaks.append(peak)
        return np.array(peaks)

    t_peaks = _get_peaks(target)
    c_peaks = _get_peaks(candidate)

    ml = min(len(t_peaks), len(c_peaks))
    if ml == 0:
        return 100.0

    # dB 차이의 가중 MSE — 높은 하모닉에 더 가중 (H2~H5가 핵심)
    weights = np.array([1.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0])[:ml]
    diff = (t_peaks[:ml] - c_peaks[:ml]) ** 2
    return float(np.sum(weights * diff) / np.sum(weights))


def render_gru(
    model,
    audio_input: np.ndarray,
    cond: list[float],
    hidden_init: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> np.ndarray:
    """GRU 모델로 오디오 렌더링

    Args:
        model: NNStompGRU 모델
        audio_input: (samples,) float32 입력
        cond: 조건 벡터
        hidden_init: (1, 1, hidden_size) 초기 hidden state

    Returns: (samples,) float32 출력
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(audio_input).unsqueeze(0).unsqueeze(-1).to(device)
        c = torch.tensor([cond], dtype=torch.float32).to(device)

        h0 = None
        if hidden_init is not None:
            h0 = torch.from_numpy(hidden_init).to(device)

        pred, _ = model(x, c, h0)
        return pred[0, :, 0].cpu().numpy()


def sound_match(
    model,
    target_audio: np.ndarray,
    input_audio: np.ndarray,
    init_cond: list[float],
    fundamental_freq: float = 1000.0,
    n_generations: int = 40,
    population_size: int = 8,
    sigma: float = 0.1,
    optimize_hidden: bool = True,
    optimize_cond: bool = True,
    early_stop_patience: int = 10,
    early_stop_tol: float = 1e-4,
    device: str = "cpu",
) -> SoundMatchResult:
    """CMA-ES render-in-the-loop 사운드 매칭

    원본 플러그인 출력(target)에 가장 가까운 GRU 모델 출력을 찾기 위해
    조건 벡터와 hidden state 초기값을 CMA-ES로 최적화.

    Args:
        model: 학습된 NNStompGRU
        target_audio: 원본 플러그인 출력 (앵커)
        input_audio: 입력 신호 (dry)
        init_cond: 초기 조건 벡터
        fundamental_freq: 테스트 신호 기본 주파수 (하모닉 피크 계산용)
        n_generations: CMA-ES 세대 수
        population_size: 세대당 후보 수
        sigma: 초기 탐색 범위
        optimize_hidden: hidden state도 최적화할지
        optimize_cond: 조건 벡터도 최적화할지

    Returns: SoundMatchResult
    """
    from cmaes import CMA

    hidden_size = model.hidden_size
    cond_dim = model.cond_dim

    # 타겟 스펙트럼 추출
    target_spec = extract_harmonic_spectrum(target_audio)

    # 최적화 벡터 구성: [cond..., hidden...]
    x0_parts = []
    dim_labels = []

    if optimize_cond:
        x0_parts.extend(init_cond)
        dim_labels.extend([f"cond_{i}" for i in range(cond_dim)])

    if optimize_hidden:
        # hidden state 초기값 (0으로 시작)
        x0_parts.extend([0.0] * hidden_size)
        dim_labels.extend([f"h_{i}" for i in range(hidden_size)])

    x0 = np.array(x0_parts, dtype=np.float64)
    n_dims = len(x0)

    # 바운드: cond [0,1], hidden [-1,1]
    bounds = []
    for label in dim_labels:
        if label.startswith("cond"):
            bounds.append([0.0, 1.0])
        else:
            bounds.append([-0.5, 0.5])  # hidden state는 작은 범위
    bounds = np.array(bounds)

    optimizer = CMA(
        mean=x0,
        sigma=sigma,
        bounds=bounds,
        population_size=population_size,
    )

    best_loss = float("inf")
    best_x = x0.copy()
    n_evals = 0
    no_improve = 0
    start = time.time()
    gen = 0

    for gen in range(n_generations):
        solutions = []

        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            n_evals += 1

            # 벡터 분해
            idx = 0
            if optimize_cond:
                cond = x[idx:idx + cond_dim].tolist()
                idx += cond_dim
            else:
                cond = init_cond

            if optimize_hidden:
                h_vec = x[idx:idx + hidden_size]
                h_init = h_vec.reshape(1, 1, hidden_size).astype(np.float32)
                idx += hidden_size
            else:
                h_init = None

            # GRU 렌더링
            pred = render_gru(model, input_audio, cond, h_init, device)

            # 복합 손실: 스펙트럼 매칭 + 하모닉 피크 매칭
            pred_spec = extract_harmonic_spectrum(pred)
            s_loss = spectral_loss(target_spec, pred_spec)
            h_loss = harmonic_peaks_loss(target_audio, pred, fundamental_freq)

            loss = 0.5 * s_loss + 0.5 * h_loss

            solutions.append((x, loss))

            if loss < best_loss - early_stop_tol:
                best_loss = loss
                best_x = x.copy()
                no_improve = 0

        optimizer.tell(solutions)

        gen_best = min(l for _, l in solutions)
        if gen % 5 == 0 or gen == n_generations - 1:
            logger.info(f"  [gen {gen:3d}] best={gen_best:.4f} global={best_loss:.4f}")

        no_improve += 1
        if no_improve >= early_stop_patience:
            logger.info(f"  Early stop @ gen {gen} (patience={early_stop_patience})")
            break

    elapsed = time.time() - start

    # 최적 결과 분해
    idx = 0
    if optimize_cond:
        final_cond = best_x[idx:idx + cond_dim].tolist()
        idx += cond_dim
    else:
        final_cond = init_cond

    if optimize_hidden:
        final_hidden = best_x[idx:idx + hidden_size].reshape(1, 1, hidden_size).astype(np.float32)
        idx += hidden_size
    else:
        final_hidden = None

    # 최종 렌더링으로 하모닉 ESR 계산
    final_pred = render_gru(model, input_audio, final_cond, final_hidden, device)
    final_h_esr = harmonic_peaks_loss(target_audio, final_pred, fundamental_freq)

    logger.info(f"  완료: loss={best_loss:.4f} h_esr={final_h_esr:.4f} "
                f"evals={n_evals} ({elapsed:.1f}s)")

    return SoundMatchResult(
        cond_vector=final_cond,
        hidden_init=final_hidden,
        loss=best_loss,
        harmonic_esr=final_h_esr,
        n_evals=n_evals,
        elapsed_sec=elapsed,
        generations=gen + 1,
    )


def iterate_sound(
    model,
    plugin_path: str,
    plugin_params: dict,
    init_cond: list[float],
    test_freq: float = 1000.0,
    duration_sec: float = 3.0,
    n_iterations: int = 3,
    cmaes_generations: int = 30,
    cmaes_popsize: int = 8,
    output_dir: str = "audio_demos",
    plugin_name: str = "plugin",
    device: str = "cpu",
) -> list[dict]:
    """반복 사운드 매칭: 원본 프리셋을 앵커로, 점진적으로 모델 출력 개선

    1회차: 기본 조건 벡터로 렌더링 → CMA-ES로 매칭
    2회차: 1회차 결과를 초기값으로 → 더 정밀한 CMA-ES
    3회차: 2회차 결과 + hidden state 최적화

    각 반복마다 WAV 저장 → 청감 비교 가능

    Args:
        model: 학습된 NNStompGRU
        plugin_path: 원본 플러그인 VST3 경로
        plugin_params: 플러그인 파라미터 (앵커 세팅)
        init_cond: 초기 조건 벡터
        test_freq: 테스트 사인파 주파수
        duration_sec: 렌더링 길이
        n_iterations: 반복 횟수
        output_dir: WAV 출력 디렉토리

    Returns: 반복별 결과 리스트
    """
    import os
    from pathlib import Path
    from nnstomps.core.vst3_wrapper import VST3PluginWrapper

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sr = _SR

    # 입력 신호 생성
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    input_mono = 0.5 * np.sin(2 * np.pi * test_freq * t)
    input_stereo = np.stack([input_mono, input_mono])

    # 원본 플러그인 렌더링 (앵커)
    wrapper = VST3PluginWrapper(plugin_path)
    wrapper.load()
    wrapper.set_parameters(plugin_params)
    ref_output = wrapper.process(input_stereo, sr)
    ref_mono = ref_output[0]

    # 앵커 저장
    param_label = "_".join(f"{k}={v}" for k, v in plugin_params.items())
    sf.write(str(out / f"anchor_{plugin_name}_{param_label}.wav"), ref_mono, sr, subtype="FLOAT")
    sf.write(str(out / f"dry_sine_{int(test_freq)}hz.wav"), input_mono, sr, subtype="FLOAT")

    logger.info(f"앵커: {plugin_name} {param_label}")
    logger.info(f"입력: {test_freq}Hz sine, {duration_sec}s")

    results = []
    current_cond = init_cond
    current_hidden = None

    for it in range(n_iterations):
        logger.info(f"\n--- 반복 {it + 1}/{n_iterations} ---")

        # 반복별 전략
        if it == 0:
            # 1회차: 조건 벡터만 최적화, 넓은 탐색
            opt_hidden = False
            opt_cond = True
            sigma = 0.15
            gens = cmaes_generations
        elif it == 1:
            # 2회차: 조건 + hidden, 좁은 탐색
            opt_hidden = True
            opt_cond = True
            sigma = 0.08
            gens = cmaes_generations
        else:
            # 3회차+: 정밀 조정
            opt_hidden = True
            opt_cond = True
            sigma = 0.05
            gens = cmaes_generations // 2

        result = sound_match(
            model=model,
            target_audio=ref_mono,
            input_audio=input_mono,
            init_cond=current_cond,
            fundamental_freq=test_freq,
            n_generations=gens,
            population_size=cmaes_popsize,
            sigma=sigma,
            optimize_hidden=opt_hidden,
            optimize_cond=opt_cond,
            device=device,
        )

        # 최적화된 파라미터로 렌더링
        pred = render_gru(model, input_mono, result.cond_vector, result.hidden_init, device)
        wav_path = out / f"iter{it+1}_{plugin_name}_{param_label}.wav"
        sf.write(str(wav_path), pred, sr, subtype="FLOAT")

        # 하모닉 분석
        logger.info(f"  저장: {wav_path}")
        _print_harmonics(ref_mono, pred, test_freq, sr)

        results.append({
            "iteration": it + 1,
            "loss": result.loss,
            "harmonic_esr": result.harmonic_esr,
            "cond": result.cond_vector,
            "n_evals": result.n_evals,
            "elapsed": result.elapsed_sec,
            "wav": str(wav_path),
        })

        # 다음 반복의 초기값 업데이트
        current_cond = result.cond_vector
        current_hidden = result.hidden_init

    return results


def _print_harmonics(ref: np.ndarray, pred: np.ndarray, freq: float, sr: int):
    """하모닉 피크 비교 출력"""
    N = _N_FFT
    window = np.hanning(N).astype(np.float32)
    skip = int(0.2 * sr)

    def _peaks(audio):
        seg = audio[skip:skip + N]
        if len(seg) < N:
            seg = np.pad(seg, (0, N - len(seg)))
        spec = np.abs(np.fft.rfft(seg * window))
        spec_db = 20 * np.log10(spec + 1e-10)
        peaks = []
        for h in range(1, 8):
            bin_idx = int(round(h * freq * N / sr))
            if bin_idx >= len(spec_db):
                break
            search = max(3, bin_idx // 20)
            peak = spec_db[max(0, bin_idx - search):min(len(spec_db), bin_idx + search)].max()
            peaks.append(peak)
        return peaks

    r_peaks = _peaks(ref)
    p_peaks = _peaks(pred)

    logger.info(f"  하모닉 비교:")
    for h, (r, p) in enumerate(zip(r_peaks, p_peaks), 1):
        diff = p - r
        bar = "+" * max(0, int(diff / 2)) + "-" * max(0, int(-diff / 2))
        logger.info(f"    H{h}: ref={r:+6.1f}dB  model={p:+6.1f}dB  diff={diff:+5.1f}dB {bar}")
