# Created: 2026-03-26
# Purpose: 하이브리드 디스토션 엔진 — Waveshaper LUT (정적) + GRU (동적)
#
# 아키텍처:
#   입력 → [정적] Waveshaper LUT (다중 레벨 보간)
#        → [동적] GRU residual (attack/release, 주파수 의존적 변화)
#        → 출력
#
# Waveshaper LUT: audioman doctor로 캡처한 실제 전달함수
#   - 256포인트 × N 레벨로 세밀하게 샘플링
#   - 입력 레벨에 따라 LUT 간 보간 (비선형 게인 특성 정확 재현)
#
# GRU residual: LUT가 재현 못하는 동적 특성만 학습
#   - attack/release 특성
#   - 주파수 의존적 새추레이션
#   - 인터샘플 피킹/오버슈트

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class WaveshaperLUT:
    """다중 레벨 Waveshaper Lookup Table

    여러 입력 레벨에서 캡처한 waveshaper 곡선을 저장하고,
    입력 진폭에 따라 적절한 곡선을 선형 보간하여 적용.

    IR처럼 각 이펙터의 정적 비선형 반응을 세밀하게 재현.
    """

    def __init__(self):
        self.curves: list[np.ndarray] = []     # [(n_points,), ...] 레벨별 곡선
        self.levels_amp: list[float] = []       # 각 곡선의 입력 진폭 (선형)
        self.n_points: int = 256
        self.x_table: np.ndarray = np.linspace(-1.0, 1.0, 256, dtype=np.float32)

    @classmethod
    def from_multi_level_capture(
        cls,
        plugin_path: str,
        params: dict,
        levels_db: list[float] | None = None,
        frequency: float = 100.0,
        sample_rate: int = 44100,
        n_points: int = 256,
        n_cycles: int = 5,
    ) -> "WaveshaperLUT":
        """플러그인에서 다중 레벨 waveshaper를 직접 캡처

        각 레벨에서 독립적으로 waveshaper를 측정하여
        레벨 의존적 비선형성을 정확히 캡처.
        """
        from nnstomps.core.vst3_wrapper import VST3PluginWrapper
        from nnstomps.core.test_signal import generate_sine

        if levels_db is None:
            # 세밀한 레벨 샘플링 — IR 수준의 정밀도
            levels_db = [-40, -36, -30, -24, -18, -12, -9, -6, -3, -1, 0]

        lut = cls()
        lut.n_points = n_points
        lut.x_table = np.linspace(-1.0, 1.0, n_points, dtype=np.float32)

        wrapper = VST3PluginWrapper(plugin_path)
        wrapper.load()
        wrapper.set_parameters(params)

        period = int(sample_rate / frequency)
        preroll = int(0.2 * sample_rate)

        for level_db in levels_db:
            amp = 10 ** (level_db / 20.0)
            wrapper.reset()

            # 프리롤 + 측정 구간
            duration = 0.2 + (n_cycles + 2) / frequency
            sine = generate_sine(frequency, sample_rate, duration, level_db)

            # 프리롤을 silence로
            sine[:, :preroll] = 0.0
            output = wrapper.process(sine, sample_rate)

            in_mono = sine[0]
            out_mono = output[0] if output.ndim == 2 else output

            # 안정 구간에서 복수 주기 평균
            stable_start = preroll + period
            cycles_in = []
            cycles_out = []
            for c in range(n_cycles):
                s = stable_start + c * period
                e = s + period
                if e > len(in_mono):
                    break
                cycles_in.append(in_mono[s:e])
                cycles_out.append(out_mono[s:e])

            if not cycles_in:
                continue

            avg_in = np.mean(cycles_in, axis=0)
            avg_out = np.mean(cycles_out, axis=0)

            # 입력 기준 정렬 → 균등 리샘플링
            sort_idx = np.argsort(avg_in)
            sorted_in = avg_in[sort_idx]
            sorted_out = avg_out[sort_idx]

            # 이 레벨의 입력 범위 내에서 보간
            curve = np.interp(
                lut.x_table * amp,  # 이 레벨의 실제 입력 범위
                sorted_in, sorted_out,
            ).astype(np.float32)

            lut.curves.append(curve)
            lut.levels_amp.append(amp)

        lut.levels_amp = np.array(lut.levels_amp, dtype=np.float32)
        logger.info(f"WaveshaperLUT: {len(lut.curves)} 레벨, {n_points} 포인트")
        return lut

    @classmethod
    def from_npy(cls, curves_path: str, levels_db: list[float]) -> "WaveshaperLUT":
        """저장된 waveshaper_curves_v2.npy에서 로드"""
        lut = cls()
        curves = np.load(curves_path)
        lut.n_points = curves.shape[1]
        lut.x_table = np.linspace(-1.0, 1.0, lut.n_points, dtype=np.float32)
        lut.curves = [curves[i] for i in range(len(curves))]
        lut.levels_amp = np.array([10 ** (db / 20.0) for db in levels_db], dtype=np.float32)
        return lut

    def process_sample(self, x: float) -> float:
        """단일 샘플 처리 — 입력 진폭에 따라 LUT 보간"""
        abs_x = abs(x)

        if len(self.curves) == 1:
            return float(np.interp(x, self.x_table, self.curves[0]))

        # 입력 진폭에 가장 가까운 두 레벨 찾기
        idx = np.searchsorted(self.levels_amp, abs_x)
        idx = np.clip(idx, 1, len(self.levels_amp) - 1)

        lo = idx - 1
        hi = idx
        amp_lo = self.levels_amp[lo]
        amp_hi = self.levels_amp[hi]

        # 레벨 간 보간 비율
        if amp_hi - amp_lo > 1e-10:
            t = (abs_x - amp_lo) / (amp_hi - amp_lo)
        else:
            t = 0.0
        t = np.clip(t, 0.0, 1.0)

        # 각 레벨의 LUT에서 보간
        y_lo = np.interp(x, self.x_table, self.curves[lo])
        y_hi = np.interp(x, self.x_table, self.curves[hi])

        return float((1 - t) * y_lo + t * y_hi)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """배열 처리 (벡터화)"""
        abs_audio = np.abs(audio)

        if len(self.curves) == 1:
            return np.interp(audio, self.x_table, self.curves[0]).astype(np.float32)

        # 각 샘플의 진폭에 가장 가까운 레벨 인덱스
        indices = np.searchsorted(self.levels_amp, abs_audio)
        indices = np.clip(indices, 1, len(self.levels_amp) - 1)

        output = np.zeros_like(audio)

        # 레벨 쌍별로 배치 처리
        for hi_idx in range(1, len(self.levels_amp)):
            lo_idx = hi_idx - 1
            mask = indices == hi_idx

            if not np.any(mask):
                continue

            amp_lo = self.levels_amp[lo_idx]
            amp_hi = self.levels_amp[hi_idx]

            t = np.clip((abs_audio[mask] - amp_lo) / (amp_hi - amp_lo + 1e-10), 0, 1)

            y_lo = np.interp(audio[mask], self.x_table, self.curves[lo_idx])
            y_hi = np.interp(audio[mask], self.x_table, self.curves[hi_idx])

            output[mask] = (1 - t) * y_lo + t * y_hi

        return output.astype(np.float32)

    def save(self, path: str):
        """LUT를 npz로 저장"""
        np.savez(path,
                 curves=np.stack(self.curves),
                 levels_amp=self.levels_amp,
                 x_table=self.x_table)

    @classmethod
    def load(cls, path: str) -> "WaveshaperLUT":
        """npz에서 로드"""
        data = np.load(path)
        lut = cls()
        curves = data["curves"]
        lut.curves = [curves[i] for i in range(len(curves))]
        lut.levels_amp = data["levels_amp"]
        lut.x_table = data["x_table"]
        lut.n_points = len(lut.x_table)
        return lut


class HybridDrive:
    """하이브리드 디스토션 엔진: Waveshaper LUT + GRU Residual

    정적 비선형: WaveshaperLUT (다중 레벨 전달함수, IR 정밀도)
    동적 잔차:  GRU (attack/release, 주파수 의존적 변화)

    처리 흐름:
        input → WaveshaperLUT → static_output
        input → GRU(input, cond) → dynamic_residual
        output = static_output + residual_mix * dynamic_residual
    """

    def __init__(
        self,
        lut: WaveshaperLUT,
        gru_model=None,
        cond: list[float] | None = None,
        residual_mix: float = 0.3,
    ):
        self.lut = lut
        self.gru = gru_model
        self.cond = cond or []
        self.residual_mix = residual_mix
        self._hidden = None

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        mix: float = 1.0,
    ) -> np.ndarray:
        """하이브리드 처리

        Args:
            audio: (samples,) float32 모노
            mix: dry/wet (0=dry, 1=full wet)
        """
        dry = audio.copy()

        # 정적: LUT
        static = self.lut.process(audio)

        # 동적: GRU residual
        if self.gru is not None and self.cond:
            import torch
            self.gru.eval()
            with torch.no_grad():
                x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(-1)
                c = torch.tensor([self.cond], dtype=torch.float32)
                pred, self._hidden = self.gru(x, c, self._hidden)
                dynamic = pred[0, :, 0].numpy()

            # residual = GRU 출력 - LUT 출력 (동적 차이만)
            wet = static + self.residual_mix * (dynamic - static)
        else:
            wet = static

        return dry * (1 - mix) + wet * mix

    def reset(self):
        """hidden state 리셋"""
        self._hidden = None
