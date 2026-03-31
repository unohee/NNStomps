# Created: 2026-03-25
# Purpose: PyTorch Dataset — input/output 오디오 쌍 로드 + 시퀀스 크롭
# Dependencies: torch, soundfile

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AudioPairDataset(Dataset):
    """학습 데이터셋 — input/output 오디오 쌍에서 랜덤 시퀀스 크롭

    manifest.json에서 쌍 목록을 읽고, 초기화 시 전체 오디오를 메모리에
    프리로드합니다. __getitem__에서는 메모리에서 랜덤 크롭만 수행.

    Args:
        manifest_path: generate_pairs가 생성한 manifest.json
        seq_len: 크롭 시퀀스 길이 (샘플)
        cond_map: 세팅 인덱스 → 조건 벡터 매핑. None이면 빈 벡터 사용
        channel: 사용할 채널 (0=Left, None=mono 변환)
        preload: True이면 초기화 시 전체 오디오를 메모리에 로드
    """

    def __init__(
        self,
        manifest_path: str,
        seq_len: int = 4096,
        cond_map: dict[int, list[float]] | None = None,
        channel: int = 0,
        preload: bool = True,
    ):
        self.seq_len = seq_len
        self.cond_map = cond_map or {}
        self.channel = channel

        manifest = json.loads(Path(manifest_path).read_text())
        self.base_dir = Path(manifest_path).parent
        self.sample_rate = manifest.get("sample_rate", 44100)

        # 유효한 쌍만 필터 (seq_len보다 긴 파일만)
        self.pairs = []
        for p in manifest["pairs"]:
            in_path = self.base_dir / p["input"]
            out_path = self.base_dir / p["output"]
            n_samples = p.get("samples", 0)

            if n_samples < seq_len:
                continue
            if not in_path.exists() or not out_path.exists():
                logger.warning(f"파일 없음: {in_path} or {out_path}")
                continue

            self.pairs.append({
                "input": str(in_path),
                "output": str(out_path),
                "setting_idx": p["setting_idx"],
                "samples": n_samples,
                "signal": p.get("signal", ""),
            })

        # 메모리 프리로드
        self._preloaded = preload
        self._inputs: list[np.ndarray] = []
        self._outputs: list[np.ndarray] = []

        if preload and self.pairs:
            logger.info(f"AudioPairDataset: {len(self.pairs)} 쌍 메모리 프리로드 중...")
            for pair in self.pairs:
                in_audio, _ = sf.read(pair["input"], dtype="float32", always_2d=True)
                out_audio, _ = sf.read(pair["output"], dtype="float32", always_2d=True)
                self._inputs.append(in_audio[:, channel])
                self._outputs.append(out_audio[:, min(channel, out_audio.shape[1] - 1)])

            total_mb = sum(a.nbytes for a in self._inputs + self._outputs) / 1e6
            logger.info(f"AudioPairDataset: {len(self.pairs)} 쌍, seq_len={seq_len}, "
                        f"메모리 {total_mb:.0f}MB")
        else:
            logger.info(f"AudioPairDataset: {len(self.pairs)} 쌍, seq_len={seq_len} (디스크 모드)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        if self._preloaded:
            in_ch = self._inputs[idx]
            out_ch = self._outputs[idx]
        else:
            in_audio, _ = sf.read(pair["input"], dtype="float32", always_2d=True)
            out_audio, _ = sf.read(pair["output"], dtype="float32", always_2d=True)
            in_ch = in_audio[:, self.channel]
            out_ch = out_audio[:, min(self.channel, out_audio.shape[1] - 1)]

        # 랜덤 크롭
        max_start = len(in_ch) - self.seq_len
        start = np.random.randint(0, max(1, max_start))
        in_crop = in_ch[start:start + self.seq_len]
        out_crop = out_ch[start:start + self.seq_len]

        # 조건 벡터
        cond = self.cond_map.get(pair["setting_idx"], [])

        return {
            "input": torch.from_numpy(in_crop).unsqueeze(-1),    # (seq_len, 1)
            "target": torch.from_numpy(out_crop).unsqueeze(-1),  # (seq_len, 1)
            "cond": torch.tensor(cond, dtype=torch.float32),
            "setting_idx": pair["setting_idx"],
        }


def create_split(
    manifest_path: str,
    seq_len: int = 4096,
    cond_map: dict[int, list[float]] | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[AudioPairDataset, AudioPairDataset]:
    """학습/검증 분할 (파일 단위)

    Returns: (train_dataset, val_dataset)
    """
    manifest = json.loads(Path(manifest_path).read_text())
    pairs = manifest["pairs"]

    # 세팅별로 그룹화 후, 세팅 단위로 분할
    from collections import defaultdict
    by_setting = defaultdict(list)
    for p in pairs:
        by_setting[p["setting_idx"]].append(p)

    settings = sorted(by_setting.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(settings)

    n_val = max(1, int(len(settings) * val_ratio))
    val_settings = set(settings[:n_val])

    train_pairs = []
    val_pairs = []
    for s in settings:
        if s in val_settings:
            val_pairs.extend(by_setting[s])
        else:
            train_pairs.extend(by_setting[s])

    # 임시 manifest 생성
    base_dir = Path(manifest_path).parent
    sr = manifest.get("sample_rate", 44100)

    def _write_temp(pairs_list, suffix):
        temp = {
            "plugin": manifest.get("plugin", ""),
            "sample_rate": sr,
            "pairs": pairs_list,
        }
        path = base_dir / f"manifest_{suffix}.json"
        path.write_text(json.dumps(temp, indent=2, default=str))
        return str(path)

    train_manifest = _write_temp(train_pairs, "train")
    val_manifest = _write_temp(val_pairs, "val")

    train_ds = AudioPairDataset(train_manifest, seq_len, cond_map)
    val_ds = AudioPairDataset(val_manifest, seq_len, cond_map)

    logger.info(f"분할: train={len(train_ds)}, val={len(val_ds)} "
                f"(val_settings={val_settings})")

    return train_ds, val_ds
