# Created: 2026-03-25
# Purpose: GRU 디스토션 모델 학습 파이프라인
# Dependencies: torch
# Test Status: 미완료

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from nnstomps.training.model import NNStompGRU
from nnstomps.training.losses import NNStompLoss, ESRLoss, MultiScaleSTFTLoss, DCLoss
from nnstomps.training.dataset import AudioPairDataset, create_split

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """학습 설정"""
    # 모델
    cond_dim: int = 2
    hidden_size: int = 40

    # 학습
    epochs: int = 100
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-6

    # 시퀀스
    seq_len_start: int = 2048     # curriculum: 초기 시퀀스 길이
    seq_len_final: int = 8192     # curriculum: 최종 시퀀스 길이
    curriculum_epoch: int = 20    # 이 에폭까지 seq_len 점진 증가

    # TBPTT (Truncated Backpropagation Through Time)
    tbptt_len: int = 2048         # hidden state 전달 단위

    # 손실 가중치
    w_esr: float = 0.7
    w_stft: float = 0.25
    w_dc: float = 0.05
    w_preemph: float = 0.0        # pre-emphasis ESR 가중치
    preemph_coeff: float = 0.95   # pre-emphasis 계수

    # 기타
    val_ratio: float = 0.1
    save_every: int = 10
    device: str = "cpu"           # "cpu", "cuda", "mps"
    seed: int = 42

    # 데이터
    manifest_path: str = ""
    cond_map: dict[int, list[float]] = field(default_factory=dict)
    output_dir: str = "models/"


def _get_seq_len(epoch: int, config: TrainConfig) -> int:
    """Curriculum learning: 시퀀스 길이 점진 증가"""
    if epoch >= config.curriculum_epoch:
        return config.seq_len_final
    ratio = epoch / config.curriculum_epoch
    return int(config.seq_len_start + ratio * (config.seq_len_final - config.seq_len_start))


def _validate(
    model: NNStompGRU,
    val_loader: DataLoader,
    criterion: NNStompLoss,
    device: torch.device,
) -> dict:
    """검증 루프"""
    model.eval()
    total_loss = 0.0
    total_esr = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)

            pred, _ = model(x, cond)
            loss, detail = criterion(pred, target)

            total_loss += detail["total"]
            total_esr += detail["esr"]
            n_batches += 1

    if n_batches == 0:
        return {"val_loss": 0, "val_esr": 0}

    return {
        "val_loss": total_loss / n_batches,
        "val_esr": total_esr / n_batches,
    }


def train(config: TrainConfig) -> dict:
    """학습 메인 루프

    Args:
        config: 학습 설정

    Returns: {"best_esr", "best_epoch", "model_path", "history"}
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모델
    model = NNStompGRU(config.cond_dim, config.hidden_size).to(device)
    logger.info(f"모델: cond_dim={config.cond_dim}, hidden={config.hidden_size}, "
                f"params={sum(p.numel() for p in model.parameters()):,}")

    # 손실
    criterion = NNStompLoss(config.w_esr, config.w_stft, config.w_dc).to(device)

    # Pre-emphasis 손실 (하모닉 강제)
    preemph_fn = None
    if config.w_preemph > 0:
        from nnstomps.training.losses import PreEmphasisLoss
        preemph_fn = PreEmphasisLoss(config.preemph_coeff).to(device)
        logger.info(f"Pre-emphasis 손실: w={config.w_preemph}, coeff={config.preemph_coeff}")

    # Optimizer + Scheduler
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 데이터
    train_ds, val_ds = create_split(
        config.manifest_path,
        seq_len=config.seq_len_final,  # 최대 길이로 로드
        cond_map=config.cond_map,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    best_esr = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(config.epochs):
        t0 = time.time()
        model.train()

        # Curriculum: 현재 에폭의 시퀀스 길이
        cur_seq_len = _get_seq_len(epoch, config)
        train_ds.seq_len = cur_seq_len
        val_ds.seq_len = min(cur_seq_len, config.seq_len_final)

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        epoch_loss = 0.0
        epoch_esr = 0.0
        n_batches = 0

        scaler = torch.amp.GradScaler("cpu") if device.type == "cpu" else None
        amp_dtype = torch.float16 if device.type == "mps" else torch.float32

        for batch in train_loader:
            x = batch["input"].to(device)       # (B, seq, 1)
            target = batch["target"].to(device)  # (B, seq, 1)
            cond = batch["cond"].to(device)      # (B, cond_dim)

            # TBPTT: 시퀀스를 청크로 분할하여 hidden state 전달
            hidden = None
            chunk_size = min(config.tbptt_len, cur_seq_len)
            n_chunks = cur_seq_len // chunk_size

            batch_loss = torch.tensor(0.0, device=device)

            for c in range(n_chunks):
                s = c * chunk_size
                e = s + chunk_size

                x_chunk = x[:, s:e, :]
                t_chunk = target[:, s:e, :]

                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pred, hidden = model(x_chunk, cond, hidden)
                    loss, detail = criterion(pred.float(), t_chunk)
                    if preemph_fn is not None:
                        loss = loss + config.w_preemph * preemph_fn(pred.float(), t_chunk)

                batch_loss = batch_loss + loss

                # hidden state detach (TBPTT 핵심)
                hidden = hidden.detach()

            # 역전파
            optimizer.zero_grad()
            batch_loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += batch_loss.item() / n_chunks
            epoch_esr += detail["esr"]
            n_batches += 1

        scheduler.step()

        # 검증
        val_metrics = _validate(model, val_loader, criterion, device)
        dt = time.time() - t0

        avg_loss = epoch_loss / max(1, n_batches)
        avg_esr = epoch_esr / max(1, n_batches)

        record = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "train_esr": round(avg_esr, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_esr": round(val_metrics["val_esr"], 6),
            "seq_len": cur_seq_len,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(dt, 1),
        }
        history.append(record)

        # 로그
        logger.info(
            f"[{epoch:3d}/{config.epochs}] "
            f"loss={avg_loss:.5f} esr={avg_esr:.5f} "
            f"val_esr={val_metrics['val_esr']:.5f} "
            f"seq={cur_seq_len} lr={record['lr']:.2e} ({dt:.1f}s)"
        )

        # 최적 모델 저장
        if val_metrics["val_esr"] < best_esr:
            best_esr = val_metrics["val_esr"]
            best_epoch = epoch
            best_path = out_dir / "best_model.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "cond_dim": config.cond_dim,
                    "hidden_size": config.hidden_size,
                },
                "epoch": epoch,
                "val_esr": best_esr,
            }, str(best_path))

        # 주기적 체크포인트
        if (epoch + 1) % config.save_every == 0:
            ckpt_path = out_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
            }, str(ckpt_path))

    # 학습 이력 저장
    history_path = out_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    logger.info(f"학습 완료: best_esr={best_esr:.5f} @ epoch {best_epoch}")

    return {
        "best_esr": best_esr,
        "best_epoch": best_epoch,
        "model_path": str(out_dir / "best_model.pt"),
        "history": history,
    }
