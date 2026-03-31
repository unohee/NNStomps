# Created: 2026-03-21
# Purpose: 오디오 분석 함수 (스펙트럼, silence 감지 등)

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FrameMetrics:
    """프레임 단위 분석 결과"""
    rms: list[float] = field(default_factory=list)
    peak: list[float] = field(default_factory=list)
    spectral_centroid: list[float] = field(default_factory=list)
    spectral_entropy: list[float] = field(default_factory=list)
    zero_crossing_rate: list[float] = field(default_factory=list)


@dataclass
class SilenceRegion:
    start_sample: int
    end_sample: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "duration_sec": round(self.duration_sec, 6),
        }


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """(channels, samples) → mono 1D"""
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def compute_frame_metrics(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> FrameMetrics:
    """프레임 단위 오디오 메트릭 계산"""
    mono = _to_mono(audio)
    n_samples = len(mono)
    metrics = FrameMetrics()

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start : start + frame_size]

        # RMS
        rms = float(np.sqrt(np.mean(frame**2)))
        metrics.rms.append(rms)

        # Peak
        metrics.peak.append(float(np.max(np.abs(frame))))

        # Zero crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size))
        metrics.zero_crossing_rate.append(zcr)

        # Spectrum
        window = np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(frame * window))
        spectrum_sum = np.sum(spectrum)

        # Spectral centroid
        if spectrum_sum > 1e-10:
            centroid = float(np.sum(freqs * spectrum) / spectrum_sum)
        else:
            centroid = 0.0
        metrics.spectral_centroid.append(centroid)

        # Spectral entropy
        if spectrum_sum > 1e-10:
            prob = spectrum / spectrum_sum
            prob = prob[prob > 1e-10]
            entropy = float(-np.sum(prob * np.log2(prob)))
        else:
            entropy = 0.0
        metrics.spectral_entropy.append(entropy)

    return metrics


def compute_summary(metrics: FrameMetrics) -> dict:
    """프레임 메트릭의 요약 통계 (mean/min/max/std)"""
    summary = {}
    for name in ["rms", "peak", "spectral_centroid", "spectral_entropy", "zero_crossing_rate"]:
        values = np.array(getattr(metrics, name))
        if len(values) == 0:
            summary[name] = {"mean": 0, "min": 0, "max": 0, "std": 0}
        else:
            summary[name] = {
                "mean": round(float(np.mean(values)), 6),
                "min": round(float(np.min(values)), 6),
                "max": round(float(np.max(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
    return summary


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_duration_sec: float = 0.1,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> list[SilenceRegion]:
    """RMS 임계값 이하 구간을 silence로 감지"""
    mono = _to_mono(audio)
    threshold_linear = 10 ** (threshold_db / 20.0)
    n_samples = len(mono)

    regions = []
    in_silence = False
    silence_start = 0

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start : start + frame_size]
        rms = np.sqrt(np.mean(frame**2))

        if rms < threshold_linear:
            if not in_silence:
                silence_start = start
                in_silence = True
        else:
            if in_silence:
                duration = (start - silence_start) / sample_rate
                if duration >= min_duration_sec:
                    regions.append(SilenceRegion(
                        start_sample=silence_start,
                        end_sample=start,
                        duration_sec=duration,
                    ))
                in_silence = False

    # 파일 끝까지 silence인 경우
    if in_silence:
        duration = (n_samples - silence_start) / sample_rate
        if duration >= min_duration_sec:
            regions.append(SilenceRegion(
                start_sample=silence_start,
                end_sample=n_samples,
                duration_sec=duration,
            ))

    return regions
