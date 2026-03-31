# Created: 2026-03-25
# Purpose: Neural Drive — 플러그인 프로파일 기반 새추레이션 재합성
# License: MIT
#
# 플러그인 프로파일 데이터(THD, waveshaper, CLAP)를 사용하여
# 텍스트 또는 참조 오디오로 새추레이션 스타일을 선택하고 적용.

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NeuralDrive:
    """프로파일 기반 새추레이션 엔진

    Usage:
        drive = NeuralDrive.from_data_dir("data/")
        drive.search("warm tape saturation")  # → 가장 가까운 세팅
        output = drive.process(audio, sr, style_idx=5)  # waveshaper 적용
    """

    def __init__(self):
        self.clap_embeddings: Optional[np.ndarray] = None  # (N, 512)
        self.labels: list[str] = []
        self.params: list[dict] = []
        self.waveshaper_curves: list[np.ndarray] = []  # 각 세팅의 I/O 곡선
        self.thd_data: list[dict] = []  # THD, odd%, harmonics
        self.plugin_names: list[str] = []
        self._clap_model = None

    @classmethod
    def from_data_dir(cls, data_dir: str) -> "NeuralDrive":
        """stompbox 데이터 디렉토리에서 전체 프로파일 로드"""
        nd = cls()
        data = Path(data_dir)

        all_emb = []
        all_labels = []
        all_params = []
        all_ws = []
        all_thd = []
        all_plugins = []

        for plugin_dir in sorted(data.iterdir()):
            if not plugin_dir.is_dir():
                continue

            plugin_name = plugin_dir.name

            # CLAP 임베딩 로드
            clap_files = sorted(plugin_dir.glob("*_clap.npy"))
            for cf in clap_files:
                emb = np.load(str(cf))
                label_file = cf.with_name(cf.stem + "_labels.json")
                if label_file.exists():
                    meta = json.loads(label_file.read_text())
                    labels = meta.get("labels", [f"{plugin_name}_{i}" for i in range(len(emb))])
                    params = meta.get("params", [{}] * len(emb))
                else:
                    labels = [f"{plugin_name}_{i}" for i in range(len(emb))]
                    params = [{}] * len(emb)

                all_emb.append(emb)
                all_labels.extend([f"[{plugin_name}] {l}" for l in labels])
                all_params.extend(params)
                all_plugins.extend([plugin_name] * len(emb))

            # Profile (THD + waveshaper) 로드
            profile_file = plugin_dir / "profile.json"
            if profile_file.exists():
                profile = json.loads(profile_file.read_text())
                for m in profile.get("measurements", []):
                    all_thd.append({
                        "plugin": plugin_name,
                        "label": f"{m['sweep']}={m['value']}",
                        "thd_percent": m["thd_percent"],
                        "odd_percent": m["odd_percent"],
                        "harmonics": m.get("harmonics", {}),
                    })

            # Waveshaper curves 로드
            ws_file = plugin_dir / "waveshaper_curves.npy"
            if ws_file.exists():
                ws = np.load(str(ws_file))
                for i in range(len(ws)):
                    all_ws.append(ws[i])

        if all_emb:
            nd.clap_embeddings = np.concatenate(all_emb, axis=0)
        nd.labels = all_labels
        nd.params = all_params
        nd.waveshaper_curves = all_ws
        nd.thd_data = all_thd
        nd.plugin_names = all_plugins

        logger.info(f"NeuralDrive 로드: {len(nd.labels)} 세팅, "
                     f"{len(nd.waveshaper_curves)} waveshaper curves, "
                     f"{len(nd.thd_data)} THD measurements")

        return nd

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """텍스트 → CLAP 검색 → 가장 가까운 새추레이션 세팅

        Returns: [{"index", "label", "similarity", "plugin"}, ...]
        """
        if self.clap_embeddings is None:
            raise RuntimeError("CLAP 임베딩이 로드되지 않음")

        if self._clap_model is None:
            import laion_clap
            self._clap_model = laion_clap.CLAP_Module(enable_fusion=False)
            self._clap_model.load_ckpt()

        text_emb = self._clap_model.get_text_embedding([query], use_tensor=False)[0]

        from numpy.linalg import norm
        sims = np.dot(self.clap_embeddings, text_emb) / (
            norm(self.clap_embeddings, axis=1) * norm(text_emb) + 1e-10
        )

        top_idx = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in top_idx:
            results.append({
                "index": int(idx),
                "label": self.labels[idx],
                "similarity": round(float(sims[idx]), 4),
                "plugin": self.plugin_names[idx] if idx < len(self.plugin_names) else "",
            })

        return results

    def get_waveshaper(self, index: int) -> Optional[np.ndarray]:
        """인덱스에 해당하는 waveshaper I/O 곡선"""
        if index < len(self.waveshaper_curves):
            return self.waveshaper_curves[index]
        return None

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        style_idx: int = 0,
        mix: float = 1.0,
    ) -> np.ndarray:
        """waveshaper 곡선으로 오디오에 새추레이션 적용

        Args:
            audio: (channels, samples) float32
            style_idx: 세팅 인덱스 (search 결과의 index)
            mix: dry/wet 비율 (0=dry, 1=wet)

        Returns: (channels, samples) float32
        """
        ws = self.get_waveshaper(style_idx)
        if ws is None:
            logger.warning(f"Waveshaper curve 없음 (idx={style_idx}), 바이패스")
            return audio

        # waveshaper: 입력 -1~+1 → 출력 매핑
        # ws는 균등 분포된 입력에 대한 출력값 (64 또는 256 포인트)
        n_points = len(ws)
        x_table = np.linspace(-1.0, 1.0, n_points, dtype=np.float32)

        # 보간 적용
        dry = audio.copy()
        wet = np.interp(audio, x_table, ws).astype(np.float32)

        return dry * (1.0 - mix) + wet * mix

    def process_by_text(
        self,
        audio: np.ndarray,
        sample_rate: int,
        description: str,
        mix: float = 1.0,
    ) -> tuple[np.ndarray, dict]:
        """텍스트 설명 → 가장 가까운 새추레이션 스타일 적용

        Returns: (processed_audio, matched_info)
        """
        results = self.search(description, top_k=1)
        if not results:
            return audio, {"error": "매칭 없음"}

        best = results[0]
        output = self.process(audio, sample_rate, style_idx=best["index"], mix=mix)

        return output, best

    def summary(self) -> dict:
        """로드된 데이터 요약"""
        plugins = {}
        for name in self.plugin_names:
            plugins[name] = plugins.get(name, 0) + 1

        return {
            "total_settings": len(self.labels),
            "total_waveshapers": len(self.waveshaper_curves),
            "total_thd_measurements": len(self.thd_data),
            "embedding_dim": self.clap_embeddings.shape[1] if self.clap_embeddings is not None else 0,
            "plugins": plugins,
        }
