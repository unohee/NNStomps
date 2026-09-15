# Created: 2026-04-01
# Purpose: Neural EQ 학습 루프

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .eq_model import NNStompEQ
from .eq_losses import EQLoss
from .eq_dataset import EQProfileDataset, EQ_PLUGIN_CONFIGS, param_spec_for

logger = logging.getLogger(__name__)


@dataclass
class EQTrainConfig:
    plugin_name: str = ""
    plugin_dir: str = ""
    fir_len: int = 256
    hidden_dims: list[int] = field(default_factory=lambda: [128, 256, 256])

    epochs: int = 5000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    augment_factor: int = 10

    w_mag: float = 1.0
    w_phase: float = 0.1
    w_reg: float = 0.01
    n_fft: int = 32768

    val_ratio: float = 0.15
    save_every: int = 500
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "models/"


def _build_checkpoint(model, config: EQTrainConfig, cond_dim: int, best_state, best_epoch, best_val_mag):
    return {
        "model_state": best_state,
        "config": {
            "cond_dim": cond_dim,
            "fir_len": config.fir_len,
            "hidden_dims": config.hidden_dims,
            "plugin_name": config.plugin_name,
            # Carries the parameter contract so the realtime engine can build
            # model inputs without importing the plugin table.
            "param_spec": param_spec_for(config.plugin_name),
        },
        "epoch": best_epoch,
        "val_mag_error": best_val_mag,
    }


def train_eq(config: EQTrainConfig) -> dict:
    """Train a Neural EQ model.

    Returns:
        dict with best_mag_error, best_epoch, model_path
    """
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.epochs}")

    torch.manual_seed(config.seed)

    expected_fir_len = EQ_PLUGIN_CONFIGS[config.plugin_name]["fir_len"]
    if config.fir_len != expected_fir_len:
        raise ValueError(
            f"{config.plugin_name}: fir_len {config.fir_len} does not match the "
            f"plugin contract ({expected_fir_len})"
        )

    # Split base row indices first, then augment only the training half.
    # Augmenting before splitting would let interpolated samples sit between two
    # training rows while their endpoints land in validation — the validation
    # error would no longer measure held-out performance.
    n_base = EQProfileDataset.base_count(config.plugin_dir)
    if n_base < 2:
        raise ValueError(f"{config.plugin_name}: need >= 2 rows, got {n_base}")

    indices = list(range(n_base))
    random.Random(config.seed).shuffle(indices)
    n_val = max(1, int(n_base * config.val_ratio))
    val_indices, train_indices = indices[:n_val], indices[n_val:]

    train_ds = EQProfileDataset(
        config.plugin_dir, config.plugin_name,
        augment_factor=config.augment_factor, base_indices=train_indices,
        seed=config.seed,
    )
    val_ds = EQProfileDataset(
        config.plugin_dir, config.plugin_name,
        augment_factor=1, base_indices=val_indices,
    )
    cond_dim = train_ds.cond_dim
    if val_ds.cond_dim != cond_dim:
        raise ValueError(f"cond_dim mismatch: {cond_dim} vs {val_ds.cond_dim}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    logger.info(f"Dataset: {n_base} base rows -> {len(train_ds)} train, "
                f"{len(val_ds)} held-out val")
    logger.info(f"cond_dim={cond_dim}, fir_len={config.fir_len}")

    model = NNStompEQ(
        cond_dim=cond_dim,
        fir_len=config.fir_len,
        hidden_dims=config.hidden_dims,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters ({n_params * 4 / 1024:.1f} KB)")

    criterion = EQLoss(
        w_mag=config.w_mag, w_phase=config.w_phase,
        w_reg=config.w_reg, n_fft=config.n_fft,
    ).to(config.device)

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mag = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            params = batch["params"].to(config.device)
            mag_target = batch["mag_db"].to(config.device)
            phase_target = batch["phase_deg"].to(config.device)

            fir_pred = model(params)
            losses = criterion(fir_pred, mag_target, phase_target)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(losses["mag"].item())

        scheduler.step()

        model.eval()
        val_mags = []
        with torch.no_grad():
            for batch in val_loader:
                params = batch["params"].to(config.device)
                mag_target = batch["mag_db"].to(config.device)
                phase_target = batch["phase_deg"].to(config.device)

                fir_pred = model(params)
                losses = criterion(fir_pred, mag_target, phase_target)
                val_mags.append(losses["mag"].item())

        val_mag = sum(val_mags) / len(val_mags) if val_mags else float("inf")

        if val_mag < best_val_mag:
            best_val_mag = val_mag
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 200 == 0 or epoch == 0:
            train_mag = sum(train_losses) / len(train_losses)
            logger.info(f"  [{epoch+1:5d}/{config.epochs}] "
                        f"train_mag={train_mag:.4f} dB, val_mag={val_mag:.4f} dB, "
                        f"best={best_val_mag:.4f} dB @ {best_epoch+1}")

        if (epoch + 1) % config.save_every == 0:
            torch.save(
                _build_checkpoint(model, config, cond_dim, best_state, best_epoch, best_val_mag),
                str(out_dir / "best_model.pt"),
            )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint (no validation improvement)")

    model_path = out_dir / "best_model.pt"
    torch.save(
        _build_checkpoint(model, config, cond_dim, best_state, best_epoch, best_val_mag),
        str(model_path),
    )

    logger.info(f"Best: mag_error={best_val_mag:.4f} dB @ epoch {best_epoch+1}")
    logger.info(f"Saved: {model_path}")

    return {
        "best_mag_error": best_val_mag,
        "best_epoch": best_epoch,
        "model_path": str(model_path),
        "n_params": n_params,
        "cond_dim": cond_dim,
    }