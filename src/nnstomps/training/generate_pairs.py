# Created: 2026-03-25
# Purpose: 학습 데이터 생성 — 테스트 신호 × 플러그인 세팅 → input/output 쌍
# Dependencies: numpy, soundfile, pedalboard (nnstomps.core)

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from nnstomps.core.test_signal import (
    generate_sine,
    generate_sweep,
    generate_white_noise,
)
from nnstomps.core.vst3_wrapper import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class TrainingPairConfig:
    """학습 데이터 생성 설정"""

    # 사인파: 주파수 × 레벨
    sine_frequencies: list[float] = field(
        default_factory=lambda: [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    )
    sine_levels_db: list[float] = field(
        default_factory=lambda: [-24.0, -12.0, -6.0, -3.0, 0.0]
    )
    sine_duration_sec: float = 5.0

    # 스윕
    sweep_levels_db: list[float] = field(
        default_factory=lambda: [-12.0, -6.0, 0.0]
    )
    sweep_duration_sec: float = 6.0

    # 화이트노이즈
    noise_levels_db: list[float] = field(
        default_factory=lambda: [-12.0, -6.0, 0.0]
    )
    noise_duration_sec: float = 5.0

    # 음악 클립 경로 (있으면 사용)
    music_clip_dir: Optional[str] = None

    sample_rate: int = 44100


def _generate_test_signals(config: TrainingPairConfig) -> list[tuple[str, np.ndarray]]:
    """테스트 신호 세트 생성. 반환: [(이름, audio (2, samples)), ...]"""
    signals = []
    sr = config.sample_rate

    # 사인파
    for freq in config.sine_frequencies:
        for level in config.sine_levels_db:
            name = f"sine_{int(freq)}hz_{int(level)}db"
            sig = generate_sine(freq, sr, config.sine_duration_sec, level)
            signals.append((name, sig))

    # 스윕
    for level in config.sweep_levels_db:
        name = f"sweep_20_20k_{int(level)}db"
        sig = generate_sweep(20.0, 20000.0, sr, config.sweep_duration_sec, level)
        signals.append((name, sig))

    # 화이트노이즈
    for level in config.noise_levels_db:
        name = f"noise_{int(level)}db"
        sig = generate_white_noise(sr, config.noise_duration_sec, level)
        signals.append((name, sig))

    # 음악 클립
    if config.music_clip_dir:
        clip_dir = Path(config.music_clip_dir)
        if clip_dir.exists():
            for wav in sorted(clip_dir.glob("*.wav")):
                data, clip_sr = sf.read(str(wav), dtype="float32", always_2d=True)
                audio = data.T  # (ch, samples)
                if clip_sr != sr:
                    logger.warning(f"음악 클립 {wav.name}: sr={clip_sr} != {sr}, 스킵")
                    continue
                signals.append((f"music_{wav.stem}", audio))

    logger.info(f"테스트 신호 {len(signals)}개 생성")
    return signals


def generate_training_pairs(
    plugin_path: str,
    param_sets: list[dict],
    output_dir: str,
    config: Optional[TrainingPairConfig] = None,
) -> dict:
    """단일 플러그인의 모든 세팅별 input/output 쌍 생성

    Args:
        plugin_path: VST3 플러그인 경로
        param_sets: 파라미터 세팅 리스트 [{"drive": 50, ...}, ...]
        output_dir: 출력 디렉토리
        config: 생성 설정

    Returns: {"n_pairs", "n_settings", "output_dir", "manifest"}
    """
    if config is None:
        config = TrainingPairConfig()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    signals = _generate_test_signals(config)
    wrapper = VST3PluginWrapper(plugin_path)
    wrapper.load()

    sr = config.sample_rate
    manifest = {"plugin": plugin_path, "sample_rate": sr, "pairs": []}
    total_pairs = 0

    for set_idx, params in enumerate(param_sets):
        set_dir = out / f"setting_{set_idx:03d}"
        set_dir.mkdir(exist_ok=True)

        wrapper.set_parameters(params)
        logger.info(f"세팅 {set_idx}/{len(param_sets)}: {params}")

        for sig_name, input_audio in signals:
            wrapper.reset()
            output_audio = wrapper.process(input_audio, sr)

            # 저장
            in_path = set_dir / f"input_{sig_name}.wav"
            out_path = set_dir / f"output_{sig_name}.wav"

            sf.write(str(in_path), input_audio.T, sr, subtype="FLOAT")
            sf.write(str(out_path), output_audio.T, sr, subtype="FLOAT")

            manifest["pairs"].append({
                "setting_idx": set_idx,
                "params": params,
                "signal": sig_name,
                "input": str(in_path.relative_to(out)),
                "output": str(out_path.relative_to(out)),
                "samples": input_audio.shape[1],
            })
            total_pairs += 1

    # manifest 저장
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    logger.info(f"학습 데이터 생성 완료: {total_pairs} 쌍, {len(param_sets)} 세팅")

    return {
        "n_pairs": total_pairs,
        "n_settings": len(param_sets),
        "n_signals": len(signals),
        "output_dir": str(out),
        "manifest": str(manifest_path),
    }


def generate_all_plugins(
    plugin_configs: dict[str, dict],
    output_base: str,
    config: Optional[TrainingPairConfig] = None,
) -> dict:
    """모든 플러그인에 대해 학습 데이터 생성

    Args:
        plugin_configs: {
            "blackstar": {
                "path": "example_plugin.vst3",
                "param_sets": [{"pentode": 0}, {"pentode": 25}, ...]
            },
            ...
        }
        output_base: 기본 출력 디렉토리

    Returns: 플러그인별 결과 딕셔너리
    """
    results = {}
    base = Path(output_base)

    for plugin_name, pcfg in plugin_configs.items():
        logger.info(f"=== {plugin_name} ===")
        result = generate_training_pairs(
            plugin_path=pcfg["path"],
            param_sets=pcfg["param_sets"],
            output_dir=str(base / plugin_name),
            config=config,
        )
        results[plugin_name] = result

    return results
