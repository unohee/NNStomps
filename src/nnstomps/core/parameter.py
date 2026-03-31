# Created: 2026-03-21
# Purpose: 플러그인 파라미터 정보 데이터클래스

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParameterInfo:
    """플러그인 파라미터 메타데이터"""
    name: str
    label: str  # 사람이 읽는 이름 (공백 포함)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[float] = None
    step_size: Optional[float] = None
    current_value: Any = None
    type: str = "float"  # "float", "bool", "enum"
    enum_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "current_value": self.current_value,
        }
        if self.type == "float":
            d["min"] = self.min_value
            d["max"] = self.max_value
            d["default"] = self.default_value
        elif self.type == "enum":
            d["values"] = self.enum_values
        return d


@dataclass
class PluginMeta:
    """플러그인 메타데이터"""
    name: str           # 전체 이름 ("RX 10 Spectral De-noise")
    short_name: str     # CLI용 ("spectral-de-noise")
    path: str           # 플러그인 파일 경로
    format: str         # "vst3" | "au"
    vendor: str = ""
    version: str = ""
    aliases: list[str] = field(default_factory=list)
    param_count: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "short_name": self.short_name,
            "path": self.path,
            "format": self.format,
            "vendor": self.vendor,
            "aliases": self.aliases,
            "param_count": self.param_count,
        }
