#!/usr/bin/env python3
# Created: 2026-03-26
# Purpose: 전체 플러그인 GRU 모델 학습 + 평가 + RTNeural 내보내기 자동 파이프라인
# Usage: python scripts/train_all.py [--plugin NAME] [--epochs N] [--skip-trained]

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch
from nnstomps.training.train import TrainConfig, train
from nnstomps.training.export import export_gru_to_rtneural, verify_export
from nnstomps.training.evaluate import evaluate_manifest

# ============================================================================
# 플러그인별 조건 벡터 매핑
# ============================================================================

PLUGIN_CONFIGS = {
    "blackstar": {
        "cond_dim": 2,
        # [drive_a_norm, drive_b_norm]
        "cond_map": {
            0: [0.00, 0.00],  # drive_a=0
            1: [0.25, 0.00],  # drive_a=25
            2: [0.50, 0.00],  # drive_a=50
            3: [0.75, 0.00],  # drive_a=75
            4: [1.00, 0.00],  # drive_a=100
            5: [0.00, 0.00],  # drive_b=0
            6: [0.00, 0.25],  # drive_b=25
            7: [0.00, 0.50],  # drive_b=50
            8: [0.00, 0.75],  # drive_b=75
            9: [0.00, 1.00],  # drive_b=100
        },
    },
    # 다른 플러그인은 여기에 추가
}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_plugin(name: str, epochs: int, device: str) -> dict:
    """단일 플러그인 학습"""
    pcfg = PLUGIN_CONFIGS[name]
    manifest = f"training_data/{name}/manifest.json"

    if not Path(manifest).exists():
        return {"error": f"학습 데이터 없음: {manifest}"}

    config = TrainConfig(
        cond_dim=pcfg["cond_dim"],
        hidden_size=40,
        epochs=epochs,
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
        save_every=25,
        device=device,
        seed=42,
        manifest_path=manifest,
        cond_map=pcfg["cond_map"],
        output_dir=f"models/{name}/",
    )

    return train(config)


def export_plugin(name: str) -> dict:
    """RTNeural JSON 내보내기"""
    model_path = f"models/{name}/best_model.pt"
    json_path = f"models/{name}/{name}_rtneural.json"

    if not Path(model_path).exists():
        return {"error": f"모델 없음: {model_path}"}

    result = export_gru_to_rtneural(model_path, json_path)
    verify = verify_export(model_path, json_path, test_length=1000)

    return {**result, "verify": verify}


def eval_plugin(name: str) -> dict:
    """모델 평가"""
    pcfg = PLUGIN_CONFIGS[name]
    model_path = f"models/{name}/best_model.pt"
    manifest = f"training_data/{name}/manifest.json"

    if not Path(model_path).exists():
        return {"error": f"모델 없음: {model_path}"}

    return evaluate_manifest(
        model_path, manifest,
        cond_map=pcfg["cond_map"],
        max_pairs=30,  # 전체 평가는 느리므로 30쌍만
    )


def main():
    parser = argparse.ArgumentParser(description="NNStomps 전체 학습 파이프라인")
    parser.add_argument("--plugin", "-p", help="특정 플러그인만 (예: blackstar)")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--skip-trained", action="store_true",
                        help="이미 best_model.pt가 있으면 건너뜀")
    parser.add_argument("--eval-only", action="store_true",
                        help="학습 건너뛰고 평가만")
    parser.add_argument("--export-only", action="store_true",
                        help="학습 건너뛰고 내보내기만")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"디바이스: {device}")

    plugins = [args.plugin] if args.plugin else list(PLUGIN_CONFIGS.keys())
    summary = {}

    for name in plugins:
        logger.info(f"\n{'='*60}")
        logger.info(f"  {name.upper()}")
        logger.info(f"{'='*60}")

        t0 = time.time()

        # 학습
        if not args.eval_only and not args.export_only:
            if args.skip_trained and Path(f"models/{name}/best_model.pt").exists():
                logger.info(f"  [SKIP] 이미 학습됨")
                train_result = {"skipped": True}
            else:
                logger.info(f"  [TRAIN] {args.epochs} epochs...")
                train_result = train_plugin(name, args.epochs, device)
        else:
            train_result = {"skipped": True}

        # 내보내기
        if not args.eval_only:
            logger.info(f"  [EXPORT] RTNeural JSON...")
            export_result = export_plugin(name)
        else:
            export_result = {"skipped": True}

        # 평가
        logger.info(f"  [EVAL] 모델 평가 (30쌍)...")
        eval_result = eval_plugin(name)

        dt = time.time() - t0

        summary[name] = {
            "train": train_result if isinstance(train_result, dict) and "best_esr" in train_result
                     else train_result,
            "export": export_result,
            "eval_mean_esr": eval_result.get("mean_esr", None),
            "time_sec": round(dt, 1),
        }

    # 최종 요약
    logger.info(f"\n{'='*60}")
    logger.info(f"  최종 요약")
    logger.info(f"{'='*60}")

    for name, s in summary.items():
        train_esr = s["train"].get("best_esr", "N/A") if isinstance(s["train"], dict) else "skipped"
        eval_esr = s.get("eval_mean_esr", "N/A")
        export_ok = "error" not in s["export"] if isinstance(s["export"], dict) else False
        verify_ok = s["export"].get("verify", {}).get("passed", False) if isinstance(s["export"], dict) else False

        logger.info(
            f"  {name:12s} | train_esr={train_esr!s:>8s} | "
            f"eval_esr={eval_esr!s:>8s} | "
            f"export={'OK' if export_ok else 'FAIL':>4s} | "
            f"verify={'OK' if verify_ok else 'FAIL':>4s} | "
            f"{s['time_sec']:.0f}s"
        )

    # 요약 JSON 저장
    Path("models").mkdir(exist_ok=True)
    summary_path = Path("models/training_summary.json")

    # 직렬화 가능한 형태로 변환
    serializable = {}
    for name, s in summary.items():
        entry = {"time_sec": s["time_sec"], "eval_mean_esr": s.get("eval_mean_esr")}
        if isinstance(s["train"], dict):
            entry["best_esr"] = s["train"].get("best_esr")
            entry["best_epoch"] = s["train"].get("best_epoch")
        if isinstance(s["export"], dict):
            entry["export_ok"] = "error" not in s["export"]
            entry["verify_ok"] = s["export"].get("verify", {}).get("passed", False)
        serializable[name] = entry

    summary_path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"\n요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
