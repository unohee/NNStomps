# Created: 2026-03-25
# Purpose: 플러그인 분석 엔진 — PluginDoctor 스타일 측정
#
# 측정 항목:
# 1. Linear: impulse response → frequency response (magnitude + phase)
# 2. Harmonic: THD, THD+N, IMD
# 3. Sweep: THD vs frequency, 2D spectrogram (앨리어싱 감지)
# 4. Dynamics: ramp (I/O 곡선), attack/release
# 5. Oscilloscope: waveshaper 곡선
# 6. Performance: 처리 시간 측정

import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from nnstomps.core.test_signal import (
    generate_impulse, generate_sine, generate_two_tone,
    generate_white_noise, generate_sweep,
    generate_dynamics_ramp, generate_dynamics_attack_release,
    to_mid_side, from_mid_side,
)
from nnstomps.core.vst3_wrapper import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class LinearResult:
    """주파수 응답 측정 결과"""
    frequencies: list[float]      # Hz
    magnitude_db: list[float]     # dB
    phase_deg: list[float]        # degrees
    sample_rate: int
    fft_size: int
    method: str                   # "impulse" or "noise"


@dataclass
class HarmonicResult:
    """하모닉 왜곡 측정 결과"""
    thd_percent: float
    thd_plus_n_percent: float
    fundamental_freq: float
    fundamental_db: float
    harmonics: list[dict]         # [{freq, db, order}]
    imd_percent: Optional[float] = None
    method: str = "thd"


@dataclass
class SweepResult:
    """스윕 분석 결과"""
    frequencies: list[float]
    thd_per_freq: list[float]     # THD vs frequency
    gain_per_freq: list[float]    # dB gain vs frequency
    spectrogram: Optional[np.ndarray] = None  # 2D (time, freq) for aliasing
    time_axis: Optional[list[float]] = None
    freq_axis: Optional[list[float]] = None


@dataclass
class DynamicsResult:
    """다이내믹스 측정 결과"""
    input_levels_db: list[float]
    output_levels_db: list[float]
    gain_reduction_db: list[float]
    method: str = "ramp"          # "ramp" or "attack_release"
    attack_release_audio: Optional[np.ndarray] = None


@dataclass
class OscilloscopeResult:
    """오실로스코프/웨이브셰이퍼 결과"""
    input_signal: np.ndarray
    output_signal: np.ndarray
    # waveshaper: input→output 매핑
    waveshaper_input: list[float]
    waveshaper_output: list[float]


@dataclass
class WaveshaperV2Result:
    """다중 진폭 waveshaper 추출 결과 (v2)"""
    input_values: np.ndarray       # (n_points,) 균등 분포 [-1, +1]
    output_values: np.ndarray      # (n_points,) 매핑된 출력
    n_points: int                  # 256 기본
    levels_db: list[float]         # 측정에 사용된 진폭 레벨
    input_coverage: float          # 커버리지 비율 (0~1)
    is_symmetric: bool             # 홀수 하모닉 대칭 여부
    raw_pairs: Optional[list[tuple[np.ndarray, np.ndarray]]] = None  # 레벨별 raw (in, out)


@dataclass
class PerformanceResult:
    """성능 측정 결과"""
    buffer_sizes: list[int]
    process_times_ms: list[float]
    samples_per_second: list[float]
    realtime_ratio: list[float]


def _load_plugin(plugin_path: str, params: Optional[dict] = None) -> VST3PluginWrapper:
    wrapper = VST3PluginWrapper(plugin_path)
    wrapper.load()
    if params:
        wrapper.set_parameters(params)
    return wrapper


# =============================================================================
# 1. Linear Analysis
# =============================================================================


def measure_linear(
    plugin_path: str,
    params: Optional[dict] = None,
    sample_rate: int = 44100,
    fft_size: int = 16384,
    method: str = "impulse",
    level_db: float = 0.0,
) -> LinearResult:
    """주파수 응답 측정 (magnitude + phase)

    method: "impulse" (delta) or "noise" (white noise averaged)
    """
    wrapper = _load_plugin(plugin_path, params)

    if method == "impulse":
        test = generate_impulse(sample_rate, duration_sec=fft_size / sample_rate + 0.1,
                                level_db=level_db)
    else:
        test = generate_white_noise(sample_rate, duration_sec=2.0, level_db=level_db)

    output = wrapper.process(test, sample_rate)

    # 모노 변환
    mono = output[0] if output.ndim == 2 else output

    # FFT
    window = np.hanning(fft_size).astype(np.float32)
    frame = mono[:fft_size] * window
    spectrum = np.fft.rfft(frame)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum, deg=True)

    # dB 변환
    mag_db = 20 * np.log10(magnitude + 1e-10)

    return LinearResult(
        frequencies=freqs.tolist(),
        magnitude_db=mag_db.tolist(),
        phase_deg=phase.tolist(),
        sample_rate=sample_rate,
        fft_size=fft_size,
        method=method,
    )


# =============================================================================
# 2. Harmonic Analysis (THD, IMD)
# =============================================================================


def measure_thd(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 1000.0,
    level_db: float = -6.0,
    sample_rate: int = 44100,
    fft_size: int = 16384,
) -> HarmonicResult:
    """THD + THD+N 측정"""
    wrapper = _load_plugin(plugin_path, params)

    duration = fft_size / sample_rate + 0.5
    test = generate_sine(frequency, sample_rate, duration, level_db)
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    # 안정 구간 사용 (처음 0.1초 제외)
    skip = int(0.1 * sample_rate)
    frame = mono[skip:skip + fft_size]
    if len(frame) < fft_size:
        frame = np.pad(frame, (0, fft_size - len(frame)))

    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # 기본 주파수 피크
    fund_bin = int(round(frequency * fft_size / sample_rate))
    search_range = max(3, fund_bin // 20)
    fund_region = spectrum[max(0, fund_bin - search_range):fund_bin + search_range]
    fund_peak = np.max(fund_region)
    fund_db = 20 * np.log10(fund_peak + 1e-10)

    # 하모닉 피크 찾기
    harmonics = []
    harmonic_energy = 0.0
    max_harmonic = min(16, int(sample_rate / 2 / frequency))

    for h in range(2, max_harmonic + 1):
        h_bin = int(round(h * frequency * fft_size / sample_rate))
        if h_bin >= len(spectrum):
            break
        sr = max(3, h_bin // 50)
        region = spectrum[max(0, h_bin - sr):min(len(spectrum), h_bin + sr)]
        if len(region) == 0:
            continue
        h_peak = np.max(region)
        h_db = 20 * np.log10(h_peak + 1e-10)
        harmonic_energy += h_peak**2
        harmonics.append({"freq": round(h * frequency, 1), "db": round(h_db, 2), "order": h})

    # THD = sqrt(sum(harmonics^2)) / fundamental
    thd = np.sqrt(harmonic_energy) / (fund_peak + 1e-10) * 100

    # THD+N = sqrt(sum(everything_except_fundamental^2)) / fundamental
    total_energy = np.sum(spectrum**2)
    fund_energy = fund_peak**2
    thd_n = np.sqrt(max(0, total_energy - fund_energy)) / (fund_peak + 1e-10) * 100

    return HarmonicResult(
        thd_percent=round(thd, 4),
        thd_plus_n_percent=round(thd_n, 4),
        fundamental_freq=frequency,
        fundamental_db=round(fund_db, 2),
        harmonics=harmonics,
        method="thd",
    )


def measure_imd(
    plugin_path: str,
    params: Optional[dict] = None,
    freq_low: float = 60.0,
    freq_high: float = 7000.0,
    sample_rate: int = 44100,
    fft_size: int = 16384,
) -> HarmonicResult:
    """IMD (상호변조 왜곡) 측정"""
    wrapper = _load_plugin(plugin_path, params)

    duration = fft_size / sample_rate + 0.5
    test = generate_two_tone(freq_low, freq_high, sample_rate, duration)
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    skip = int(0.1 * sample_rate)
    frame = mono[skip:skip + fft_size]
    if len(frame) < fft_size:
        frame = np.pad(frame, (0, fft_size - len(frame)))

    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # 7kHz 피크
    high_bin = int(round(freq_high * fft_size / sample_rate))
    sr = max(3, high_bin // 50)
    high_peak = np.max(spectrum[max(0, high_bin - sr):high_bin + sr])

    # IMD 사이드밴드: 7000 ± N*60 Hz
    imd_energy = 0.0
    harmonics = []
    for n in range(1, 11):
        for sign in [-1, 1]:
            sb_freq = freq_high + sign * n * freq_low
            if sb_freq <= 0 or sb_freq >= sample_rate / 2:
                continue
            sb_bin = int(round(sb_freq * fft_size / sample_rate))
            if sb_bin >= len(spectrum):
                continue
            sr2 = max(2, sb_bin // 100)
            region = spectrum[max(0, sb_bin - sr2):min(len(spectrum), sb_bin + sr2)]
            if len(region) == 0:
                continue
            sb_peak = np.max(region)
            imd_energy += sb_peak**2
            sb_db = 20 * np.log10(sb_peak + 1e-10)
            harmonics.append({"freq": round(sb_freq, 1), "db": round(sb_db, 2), "order": f"±{n}"})

    imd = np.sqrt(imd_energy) / (high_peak + 1e-10) * 100

    return HarmonicResult(
        thd_percent=0.0,
        thd_plus_n_percent=0.0,
        fundamental_freq=freq_high,
        fundamental_db=round(20 * np.log10(high_peak + 1e-10), 2),
        harmonics=harmonics,
        imd_percent=round(imd, 4),
        method="imd",
    )


# =============================================================================
# 3. Sweep Analysis
# =============================================================================


def measure_sweep(
    plugin_path: str,
    params: Optional[dict] = None,
    freq_start: float = 20.0,
    freq_end: float = 20000.0,
    sample_rate: int = 44100,
    duration_sec: float = 6.0,
    level_db: float = -6.0,
    fft_size: int = 4096,
    hop_size: int = 1024,
) -> SweepResult:
    """주파수 스윕 → THD vs freq + 2D spectrogram"""
    wrapper = _load_plugin(plugin_path, params)

    test = generate_sweep(freq_start, freq_end, sample_rate, duration_sec, level_db)
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    n = len(mono)

    # STFT → 2D spectrogram
    window = np.hanning(fft_size).astype(np.float32)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    n_frames = (n - fft_size) // hop_size + 1

    spectrogram = np.zeros((n_frames, len(freqs)), dtype=np.float32)
    time_axis = []

    for i in range(n_frames):
        start = i * hop_size
        frame = mono[start:start + fft_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectrogram[i] = 20 * np.log10(spectrum + 1e-10)
        time_axis.append(start / sample_rate)

    # 스윕 시 각 시점의 기본 주파수 계산
    sweep_freqs = []
    thd_per_freq = []
    gain_per_freq = []

    for i in range(n_frames):
        t = time_axis[i]
        # 현재 스윕 주파수 (exponential)
        ratio = t / duration_sec
        current_freq = freq_start * (freq_end / freq_start) ** ratio

        if current_freq > sample_rate / 4:  # Nyquist/2 이상은 THD 의미 없음
            break

        sweep_freqs.append(round(current_freq, 1))

        # 기본 주파수 피크
        fund_bin = int(round(current_freq * fft_size / sample_rate))
        if fund_bin >= len(freqs) or fund_bin < 1:
            thd_per_freq.append(0.0)
            gain_per_freq.append(0.0)
            continue

        spec = 10 ** (spectrogram[i] / 20)  # linear
        sr = max(2, fund_bin // 20)
        fund_peak = np.max(spec[max(0, fund_bin - sr):min(len(spec), fund_bin + sr)])

        # 하모닉 에너지
        h_energy = 0.0
        for h in range(2, 8):
            h_bin = int(round(h * current_freq * fft_size / sample_rate))
            if h_bin >= len(spec):
                break
            sr2 = max(2, h_bin // 30)
            h_energy += np.max(spec[max(0, h_bin - sr2):min(len(spec), h_bin + sr2)])**2

        thd = np.sqrt(h_energy) / (fund_peak + 1e-10) * 100
        thd_per_freq.append(round(thd, 4))
        gain_per_freq.append(round(20 * np.log10(fund_peak + 1e-10), 2))

    return SweepResult(
        frequencies=sweep_freqs,
        thd_per_freq=thd_per_freq,
        gain_per_freq=gain_per_freq,
        spectrogram=spectrogram,
        time_axis=time_axis,
        freq_axis=freqs.tolist(),
    )


# =============================================================================
# 4. Dynamics
# =============================================================================


def measure_dynamics_ramp(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    level_start_db: float = -80.0,
    level_end_db: float = 0.0,
    step_db: float = 1.0,
) -> DynamicsResult:
    """입력 레벨별 출력 레벨 측정 (컴프레서 I/O 곡선)"""
    wrapper = _load_plugin(plugin_path, params)

    test, levels = generate_dynamics_ramp(
        frequency, sample_rate, level_start_db, level_end_db, step_db,
    )
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    step_samples = int(0.5 * sample_rate)

    output_levels = []
    for i in range(len(levels)):
        start = i * step_samples
        end = start + step_samples
        segment = mono[start:end]
        peak = np.max(np.abs(segment))
        out_db = 20 * np.log10(peak + 1e-10)
        output_levels.append(round(out_db, 2))

    gain_reduction = [round(o - i, 2) for i, o in zip(levels, output_levels)]

    return DynamicsResult(
        input_levels_db=levels,
        output_levels_db=output_levels,
        gain_reduction_db=gain_reduction,
        method="ramp",
    )


def measure_dynamics_ar(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    level_below_db: float = -30.0,
    level_above_db: float = 0.0,
) -> DynamicsResult:
    """Attack/Release 응답 측정"""
    wrapper = _load_plugin(plugin_path, params)

    test = generate_dynamics_attack_release(
        frequency, sample_rate, level_below_db, level_above_db,
    )
    output = wrapper.process(test, sample_rate)

    # RMS envelope 추출
    hop = 256
    mono = output[0] if output.ndim == 2 else output
    rms_env = []
    for i in range(0, len(mono) - hop, hop):
        rms = np.sqrt(np.mean(mono[i:i + hop]**2))
        rms_env.append(round(20 * np.log10(rms + 1e-10), 2))

    return DynamicsResult(
        input_levels_db=[level_below_db, level_above_db, level_below_db],
        output_levels_db=rms_env,
        gain_reduction_db=[],
        method="attack_release",
        attack_release_audio=output,
    )


# =============================================================================
# 5. Oscilloscope / Waveshaper
# =============================================================================


def measure_waveshaper(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 100.0,
    level_db: float = 0.0,
    sample_rate: int = 44100,
) -> OscilloscopeResult:
    """입력→출력 웨이브셰이퍼 곡선 추출"""
    wrapper = _load_plugin(plugin_path, params)

    test = generate_sine(frequency, sample_rate, 0.5, level_db)
    output = wrapper.process(test, sample_rate)

    in_mono = test[0]
    out_mono = output[0] if output.ndim == 2 else output

    # 안정 구간 1주기 추출
    period = int(sample_rate / frequency)
    skip = int(0.1 * sample_rate)
    in_cycle = in_mono[skip:skip + period]
    out_cycle = out_mono[skip:skip + period]

    # 정렬: 입력값 기준 정렬 → waveshaper 곡선
    sort_idx = np.argsort(in_cycle)
    ws_input = in_cycle[sort_idx].tolist()
    ws_output = out_cycle[sort_idx].tolist()

    return OscilloscopeResult(
        input_signal=in_cycle,
        output_signal=out_cycle,
        waveshaper_input=ws_input,
        waveshaper_output=ws_output,
    )


def measure_waveshaper_v2(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 100.0,
    levels_db: Optional[list[float]] = None,
    sample_rate: int = 44100,
    n_points: int = 256,
    n_cycles_avg: int = 3,
    preroll_sec: float = 0.2,
) -> WaveshaperV2Result:
    """다중 진폭 레벨 waveshaper 곡선 추출 (v2)

    기존 measure_waveshaper의 문제 해결:
    - 다중 진폭 레벨로 [-1, +1] 전체 범위 커버
    - 복수 주기 평균으로 노이즈 감소
    - 레이턴시 보상 (silence 프리롤)

    Args:
        plugin_path: VST3 플러그인 경로
        params: 플러그인 파라미터
        frequency: 테스트 사인파 주파수 (Hz)
        levels_db: 측정 진폭 레벨 리스트 (dBFS)
        sample_rate: 샘플레이트
        n_points: 출력 waveshaper 해상도
        n_cycles_avg: 평균할 주기 수
        preroll_sec: 레이턴시 보상용 프리롤 (초)

    Returns: WaveshaperV2Result
    """
    if levels_db is None:
        levels_db = [-24.0, -18.0, -12.0, -6.0, -3.0, -1.0, 0.0]

    wrapper = _load_plugin(plugin_path, params)
    period = int(sample_rate / frequency)

    # 모든 레벨에서 수집한 raw input/output 쌍
    all_in = []
    all_out = []
    raw_pairs = []

    for level in levels_db:
        # 프리롤(silence) + 테스트 신호 생성
        preroll_samples = int(preroll_sec * sample_rate)
        test_duration = preroll_sec + (n_cycles_avg + 2) / frequency + 0.1
        test = generate_sine(frequency, sample_rate, test_duration, level)

        # 프리롤 구간을 silence로 교체 (레이턴시 보상)
        test[:, :preroll_samples] = 0.0

        output = wrapper.process(test, sample_rate)
        wrapper.reset()

        in_mono = test[0]
        out_mono = output[0] if output.ndim == 2 else output

        # 프리롤 이후 안정 구간에서 n_cycles_avg 주기 추출
        # 프리롤 + 1주기 스킵 (과도 응답 회피)
        skip = preroll_samples + period
        extract_len = period * n_cycles_avg

        if skip + extract_len > len(in_mono):
            # 신호가 짧으면 가능한 만큼 사용
            extract_len = len(in_mono) - skip
            if extract_len < period:
                logger.warning(f"level={level}dB: 추출 구간 부족, 스킵")
                continue

        in_seg = in_mono[skip:skip + extract_len]
        out_seg = out_mono[skip:skip + extract_len]

        # 주기 단위로 잘라서 평균
        n_full_cycles = len(in_seg) // period
        if n_full_cycles == 0:
            continue

        in_cycles = in_seg[:n_full_cycles * period].reshape(n_full_cycles, period)
        out_cycles = out_seg[:n_full_cycles * period].reshape(n_full_cycles, period)

        in_avg = in_cycles.mean(axis=0)
        out_avg = out_cycles.mean(axis=0)

        raw_pairs.append((in_avg.copy(), out_avg.copy()))
        all_in.append(in_avg)
        all_out.append(out_avg)

    if not all_in:
        raise RuntimeError("waveshaper 추출 실패: 유효한 데이터 없음")

    # 모든 레벨의 데이터를 합쳐서 입력값 기준 정렬
    combined_in = np.concatenate(all_in)
    combined_out = np.concatenate(all_out)

    sort_idx = np.argsort(combined_in)
    sorted_in = combined_in[sort_idx]
    sorted_out = combined_out[sort_idx]

    # 균등 분포 n_points로 리샘플링 (-1 ~ +1)
    x_uniform = np.linspace(-1.0, 1.0, n_points, dtype=np.float32)

    # 입력 범위 내에서만 보간 (범위 밖은 외삽하지 않음)
    in_min, in_max = float(sorted_in[0]), float(sorted_in[-1])
    y_uniform = np.interp(x_uniform, sorted_in, sorted_out).astype(np.float32)

    # 커버리지 계산: 입력이 [-1, +1] 중 얼마를 커버하는지
    coverage = (in_max - in_min) / 2.0  # 전체 범위 2.0 기준

    # 대칭성 검증: f(-x) ≈ -f(x) 이면 홀수 하모닉 대칭
    # 중앙(0) 기준 좌우 비교
    half = n_points // 2
    left = y_uniform[:half]           # 음의 입력 구간
    right = y_uniform[half:][::-1]    # 양의 입력 구간 (뒤집기)
    if len(left) == len(right) and np.max(np.abs(left)) > 1e-6:
        asymmetry = np.mean(np.abs(left + right)) / (np.mean(np.abs(left)) + 1e-10)
        is_symmetric = asymmetry < 0.15  # 15% 이하 차이면 대칭
    else:
        is_symmetric = False

    logger.info(f"waveshaper v2: {n_points}pt, "
                f"coverage={coverage:.1%}, "
                f"range=[{in_min:.3f}, {in_max:.3f}], "
                f"symmetric={is_symmetric}")

    return WaveshaperV2Result(
        input_values=x_uniform,
        output_values=y_uniform,
        n_points=n_points,
        levels_db=levels_db,
        input_coverage=round(coverage, 4),
        is_symmetric=is_symmetric,
        raw_pairs=raw_pairs,
    )


# =============================================================================
# 6. Performance
# =============================================================================


def measure_performance(
    plugin_path: str,
    params: Optional[dict] = None,
    sample_rate: int = 44100,
    buffer_sizes: Optional[list[int]] = None,
    n_iterations: int = 100,
) -> PerformanceResult:
    """프로세싱 콜백 시간 측정"""
    wrapper = _load_plugin(plugin_path, params)

    if buffer_sizes is None:
        buffer_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    process_times = []
    sps_list = []
    rt_ratios = []

    for bs in buffer_sizes:
        test = np.random.randn(2, bs).astype(np.float32) * 0.1
        times = []

        for _ in range(n_iterations):
            t0 = time.perf_counter()
            wrapper.process(test, sample_rate)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        avg_ms = np.median(times)
        process_times.append(round(avg_ms, 4))

        # samples per second
        sps = bs / (avg_ms / 1000) if avg_ms > 0 else 0
        sps_list.append(round(sps))

        # realtime ratio
        rt = sps / sample_rate if sample_rate > 0 else 0
        rt_ratios.append(round(rt, 2))

    return PerformanceResult(
        buffer_sizes=buffer_sizes,
        process_times_ms=process_times,
        samples_per_second=sps_list,
        realtime_ratio=rt_ratios,
    )


# =============================================================================
# 통합: 2 플러그인 비교
# =============================================================================


def compare_linear(
    plugin_path_1: str,
    plugin_path_2: str,
    params_1: Optional[dict] = None,
    params_2: Optional[dict] = None,
    sample_rate: int = 44100,
) -> dict:
    """2 플러그인 주파수 응답 비교 (차이)"""
    r1 = measure_linear(plugin_path_1, params_1, sample_rate)
    r2 = measure_linear(plugin_path_2, params_2, sample_rate)

    diff_db = [round(a - b, 4) for a, b in zip(r1.magnitude_db, r2.magnitude_db)]
    diff_phase = [round(a - b, 4) for a, b in zip(r1.phase_deg, r2.phase_deg)]

    return {
        "plugin_1": asdict(r1),
        "plugin_2": asdict(r2),
        "diff_magnitude_db": diff_db,
        "diff_phase_deg": diff_phase,
    }


# =============================================================================
# 7. CLAP 임베딩 프로파일링
# =============================================================================


def measure_clap_profile(
    plugin_path: str,
    param_sweeps: dict[str, list],
    base_params: Optional[dict] = None,
    sample_rate: int = 44100,
    duration_sec: float = 2.0,
    test_frequency: float = 1000.0,
    test_level_db: float = -6.0,
) -> dict:
    """파라미터 스윕별 CLAP 임베딩 생성 — 새추레이션 "지문"

    Args:
        plugin_path: VST3 경로
        param_sweeps: {"drive": [0, 25, 50, 75, 100], "style": ["Soft", "Hard"]}
        base_params: 기본 파라미터 {"mix": 100.0, ...}

    Returns: {
        "embeddings": [(param_values, embedding_512d), ...],
        "labels": ["drive=0", "drive=25", ...],
        "embeddings_npy": np.ndarray (N, 512),
    }
    """
    try:
        import laion_clap
    except ImportError:
        raise ImportError("CLAP 필요: pip install laion-clap")

    import soundfile as sf
    import tempfile
    import os

    wrapper = _load_plugin(plugin_path, base_params)

    # 테스트 신호
    n_samples = int(duration_sec * sample_rate)
    test = generate_sine(test_frequency, sample_rate, duration_sec, test_level_db)

    # 모든 파라미터 조합 생성
    import itertools
    param_names = list(param_sweeps.keys())
    param_values_list = list(param_sweeps.values())
    combinations = list(itertools.product(*param_values_list))

    # 플러그인 한 번만 로딩, 파라미터만 교체
    from pedalboard import load_plugin as pb_load
    plugin = pb_load(plugin_path)
    if base_params:
        for k, v in base_params.items():
            try:
                setattr(plugin, k, v)
            except Exception:
                pass

    tmpdir = tempfile.mkdtemp()
    wav_paths = []
    labels = []
    param_records = []

    for combo in combinations:
        # 파라미터 적용 (같은 인스턴스 재사용)
        param_dict = dict(zip(param_names, combo))
        for k, v in param_dict.items():
            try:
                setattr(plugin, k, v)
            except Exception:
                try:
                    setattr(plugin, k.replace(' ', '_'), v)
                except Exception:
                    pass

        output = plugin.process(test, sample_rate)

        # WAV 저장
        label = ", ".join(f"{k}={v}" for k, v in param_dict.items())
        labels.append(label)
        param_records.append(param_dict)

        wav_path = os.path.join(tmpdir, f"{len(wav_paths):04d}.wav")
        if output.ndim == 2:
            sf.write(wav_path, output.T, sample_rate, subtype='FLOAT')
        else:
            sf.write(wav_path, output, sample_rate, subtype='FLOAT')
        wav_paths.append(wav_path)

    # CLAP 인코딩
    logger.info(f"CLAP 인코딩: {len(wav_paths)}개 설정")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    batch_size = 32
    all_emb = []
    for i in range(0, len(wav_paths), batch_size):
        batch = wav_paths[i:i + batch_size]
        emb = model.get_audio_embedding_from_filelist(batch, use_tensor=False)
        all_emb.append(emb)

    embeddings = np.concatenate(all_emb, axis=0)

    # 정리
    for p in wav_paths:
        os.unlink(p)
    os.rmdir(tmpdir)

    return {
        "labels": labels,
        "params": param_records,
        "embeddings_npy": embeddings,
        "n_settings": len(combinations),
        "embedding_dim": embeddings.shape[1],
    }
