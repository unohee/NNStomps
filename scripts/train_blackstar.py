#!/usr/bin/env python3
# Created: 2026-03-26
# Purpose: Blackstar GRU 모델 학습
# Usage: python scripts/train_blackstar.py

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

from nnstomps.training.train import TrainConfig, train

# Blackstar 조건 벡터 설계
# 세팅 10개: drive_a 0/25/50/75/100 + drive_b 0/25/50/75/100
# 조건 벡터 (cond_dim=2): [drive_a_normalized, drive_b_normalized]
COND_MAP = {
    0:  [0.00, 0.00],  # drive_a=0
    1:  [0.25, 0.00],  # drive_a=25
    2:  [0.50, 0.00],  # drive_a=50
    3:  [0.75, 0.00],  # drive_a=75
    4:  [1.00, 0.00],  # drive_a=100
    5:  [0.00, 0.00],  # drive_b=0
    6:  [0.00, 0.25],  # drive_b=25
    7:  [0.00, 0.50],  # drive_b=50
    8:  [0.00, 0.75],  # drive_b=75
    9:  [0.00, 1.00],  # drive_b=100
}


def main():
    import torch

    # 디바이스 선택
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logging.getLogger(__name__).info(f"디바이스: {device}")

    config = TrainConfig(
        cond_dim=2,
        hidden_size=40,

        epochs=100,
        batch_size=32,
        lr=5e-4,
        weight_decay=1e-6,

        seq_len_start=2048,
        seq_len_final=8192,
        curriculum_epoch=20,
        tbptt_len=2048,

        w_esr=0.7,
        w_stft=0.25,
        w_dc=0.05,

        val_ratio=0.1,
        save_every=10,
        device=device,
        seed=42,

        manifest_path="training_data/blackstar/manifest.json",
        cond_map=COND_MAP,
        output_dir="models/blackstar/",
    )

    result = train(config)

    print(f"\n{'='*50}")
    print(f"  Blackstar 학습 완료")
    print(f"{'='*50}")
    print(f"  Best ESR: {result['best_esr']:.5f} @ epoch {result['best_epoch']}")
    print(f"  모델: {result['model_path']}")


if __name__ == "__main__":
    main()
