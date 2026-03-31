# Created: 2026-03-21
# Purpose: pedalboard 기반 VST3 플러그인 래퍼

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from nnstomps.core.parameter import ParameterInfo

logger = logging.getLogger(__name__)


class VST3PluginWrapper:
    """pedalboard load_plugin 기반 VST3 래퍼"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._plugin = None
        self._parameters: Optional[list[ParameterInfo]] = None

    @property
    def name(self) -> str:
        return self._path.stem

    @property
    def is_loaded(self) -> bool:
        return self._plugin is not None

    def load(self) -> None:
        if self._plugin is not None:
            return
        import os
        import sys
        from pedalboard import load_plugin
        logger.debug(f"VST3 로드: {self._path}")
        # iZotope 플러그인 로드 시 objc 런타임 로그가 stdout에 출력되는 문제 억제
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            self._plugin = load_plugin(str(self._path))
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stdout)
            os.close(old_stderr)

    def get_parameters(self) -> list[ParameterInfo]:
        """플러그인 파라미터 목록 추출"""
        if self._parameters is not None:
            return self._parameters

        self.load()
        params = []

        for attr_name, param in self._plugin.parameters.items():
            # 파라미터 범위 추출
            try:
                rng = param.range
                min_val, max_val, step = rng
            except Exception:
                min_val = max_val = step = None

            # 현재값 읽기
            try:
                current = getattr(self._plugin, attr_name.replace(" ", "_"), None)
            except Exception:
                current = None

            # 타입 추론
            if isinstance(current, bool):
                param_type = "bool"
            elif isinstance(current, str):
                param_type = "enum"
            else:
                param_type = "float"

            info = ParameterInfo(
                name=attr_name,
                label=attr_name.replace("_", " ").title(),
                min_value=float(min_val) if min_val is not None else None,
                max_value=float(max_val) if max_val is not None else None,
                default_value=None,
                step_size=float(step) if step is not None else None,
                current_value=current if not isinstance(current, (int, float)) else float(current),
                type=param_type,
            )
            params.append(info)

        self._parameters = params
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """파라미터 설정 (이름 → 값 딕셔너리)"""
        self.load()
        for name, value in params.items():
            # 언더스코어/공백 양쪽 지원
            attr_name = name.replace(" ", "_")
            try:
                setattr(self._plugin, attr_name, value)
                logger.debug(f"파라미터 설정: {attr_name} = {value}")
            except AttributeError:
                # 공백 포함 이름 시도
                space_name = name.replace("_", " ")
                try:
                    setattr(self._plugin, space_name, value)
                except Exception as e:
                    logger.warning(f"파라미터 설정 실패: {name} = {value}: {e}")

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """오디오 처리. audio shape: (channels, samples), float32"""
        self.load()

        # shape 검증/변환
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        # float32 보장
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        return self._plugin.process(audio, sample_rate)

    def reset(self) -> None:
        """플러그인 상태 리셋"""
        if self._plugin is not None:
            try:
                self._plugin.reset()
            except Exception:
                pass
