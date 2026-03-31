# Created: 2026-03-25
# Purpose: 학습 손실 함수 — ESR + Multi-scale STFT + DC offset
# Dependencies: torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESRLoss(nn.Module):
    """Error-to-Signal Ratio 손실

    L = sum((pred - target)^2) / sum(target^2 + eps)

    시간 영역에서 가장 기본적이고 안정적인 오디오 손실.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        signal_energy = torch.sum(target ** 2) + self.eps
        return torch.sum(error ** 2) / signal_energy


class DCLoss(nn.Module):
    """DC 오프셋 페널티

    L = mean(pred)^2

    모델 출력의 DC 오프셋을 억제.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(pred) ** 2


class PreEmphasisLoss(nn.Module):
    """Pre-emphasis 적용 후 ESR — 고주파 하모닉 강조

    y[n] = x[n] - coeff * x[n-1] 필터를 적용한 뒤 ESR 계산.
    coeff가 높을수록 (0.99 근처) 고주파(하모닉)에 대한 가중치가 커짐.
    """

    def __init__(self, coeff: float = 0.95):
        super().__init__()
        self.coeff = coeff

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.squeeze(-1)
        t = target.squeeze(-1)

        p_emph = p[:, 1:] - self.coeff * p[:, :-1]
        t_emph = t[:, 1:] - self.coeff * t[:, :-1]

        error = torch.sum((p_emph - t_emph) ** 2)
        signal = torch.sum(t_emph ** 2) + 1e-8
        return error / signal


class STFTLoss(nn.Module):
    """단일 스케일 STFT 손실 = Spectral Convergence + Log Magnitude

    Args:
        fft_size: FFT 크기
        hop_size: hop 크기
        win_size: 윈도우 크기
    """

    def __init__(self, fft_size: int = 1024, hop_size: int = 256, win_size: int = 1024):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.register_buffer(
            "window", torch.hann_window(win_size)
        )

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq) → magnitude spectrogram
        # 1D → 2D
        if x.dim() == 3:
            x = x.squeeze(-1)  # (batch, seq, 1) → (batch, seq)

        spec = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=self.window,
            return_complex=True,
        )
        return torch.abs(spec)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mag = self._stft(pred)
        target_mag = self._stft(target)

        # Spectral Convergence: ||target_mag - pred_mag||_F / ||target_mag||_F
        sc = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )

        # Log Magnitude: L1(log(target) - log(pred))
        log_mag = F.l1_loss(
            torch.log(pred_mag + 1e-8),
            torch.log(target_mag + 1e-8),
        )

        return sc + log_mag


class MultiScaleSTFTLoss(nn.Module):
    """멀티 스케일 STFT 손실

    여러 FFT 크기에서 STFT 손실을 계산하여 평균.
    주파수 해상도와 시간 해상도를 모두 커버.
    """

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_sizes: list[int] | None = None,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048, 4096]
        if hop_sizes is None:
            hop_sizes = [s // 4 for s in fft_sizes]
        if win_sizes is None:
            win_sizes = fft_sizes

        self.losses = nn.ModuleList([
            STFTLoss(fft, hop, win)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 시퀀스 길이보다 큰 FFT 크기는 건너뜀
        seq_len = pred.shape[1] if pred.dim() == 3 else pred.shape[-1]
        total = torch.tensor(0.0, device=pred.device)
        n_used = 0
        for loss_fn in self.losses:
            if loss_fn.fft_size <= seq_len:
                total = total + loss_fn(pred, target)
                n_used += 1
        if n_used == 0:
            return total
        return total / n_used


class NNStompLoss(nn.Module):
    """NNStomps 통합 손실 함수

    L = w_esr * ESR + w_stft * MultiSTFT + w_dc * DC

    기본 가중치: 0.7 * ESR + 0.25 * STFT + 0.05 * DC
    """

    def __init__(
        self,
        w_esr: float = 0.7,
        w_stft: float = 0.25,
        w_dc: float = 0.05,
        stft_fft_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.w_esr = w_esr
        self.w_stft = w_stft
        self.w_dc = w_dc

        self.esr = ESRLoss()
        self.stft = MultiScaleSTFTLoss(fft_sizes=stft_fft_sizes)
        self.dc = DCLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """손실 계산

        Args:
            pred: (batch, seq_len, 1) 모델 출력
            target: (batch, seq_len, 1) 정답

        Returns: (total_loss, {"esr": ..., "stft": ..., "dc": ...})
        """
        l_esr = self.esr(pred, target)
        l_stft = self.stft(pred, target)
        l_dc = self.dc(pred, target)

        total = self.w_esr * l_esr + self.w_stft * l_stft + self.w_dc * l_dc

        detail = {
            "esr": l_esr.item(),
            "stft": l_stft.item(),
            "dc": l_dc.item(),
            "total": total.item(),
        }
        return total, detail
