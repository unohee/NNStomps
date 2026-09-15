# Created: 2026-04-01
# Purpose: EQ 학습용 손실 함수 — 주파수 도메인 기반

import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_response(
    fir_pred: torch.Tensor, n_fft: int, n_target: int
) -> torch.Tensor:
    """Complex frequency response of `fir_pred`, resampled to `n_target` bins.

    Resampling happens on the real and imaginary parts, not on magnitude and
    angle. Interpolating the angle would corrupt every bin whose neighbours
    straddle the +-180 degree branch cut — a pair like (+179, -179) interpolates
    to 0, which reads as a spurious phase reversal and injects a false gradient.

    Args:
        fir_pred: (batch, fir_len) FIR coefficients
        n_fft: FFT size used to evaluate the response
        n_target: number of frequency bins to resample to

    Returns:
        (batch, n_target) complex response
    """
    padded = torch.zeros(
        fir_pred.shape[0], n_fft, device=fir_pred.device, dtype=fir_pred.dtype
    )
    n_fir = fir_pred.shape[1]
    if n_fir > n_fft:
        raise ValueError(f"fir_len {n_fir} exceeds n_fft {n_fft}")
    padded[:, :n_fir] = fir_pred

    H = torch.fft.rfft(padded)
    if H.shape[1] == n_target:
        return H

    def _interp(x: torch.Tensor) -> torch.Tensor:
        # align_corners=True maps output bin i to source coordinate
        # i * (n_src - 1) / (n_dst - 1). rfft bins partition DC..Nyquist evenly,
        # so that is the mapping that keeps a given bin on the same frequency.
        # With align_corners=False the axis is scaled by n_src/n_dst instead and
        # shifted by half a bin, which puts every resampled bin slightly off.
        return F.interpolate(
            x.unsqueeze(1), size=n_target, mode="linear", align_corners=True
        ).squeeze(1)

    return torch.complex(_interp(H.real), _interp(H.imag))


class MagnitudeLoss(nn.Module):
    """Frequency-response magnitude loss (L1 on a dB scale).

    dB weighting makes every band contribute comparably, which is closer to how
    the ear judges an EQ curve than a linear-scale error would be.
    """

    def __init__(self, n_fft: int = 32768):
        super().__init__()
        self.n_fft = n_fft

    def forward(self, fir_pred: torch.Tensor, mag_db_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fir_pred: (batch, fir_len) predicted FIR coefficients
            mag_db_target: (batch, n_freq) target magnitude response in dB
        """
        H = resample_response(fir_pred, self.n_fft, mag_db_target.shape[1])
        mag_db_pred = 20 * torch.log10(torch.abs(H) + 1e-10)
        return torch.mean(torch.abs(mag_db_pred - mag_db_target))


class PhaseLoss(nn.Module):
    """Phase-response loss, in degrees."""

    def __init__(self, n_fft: int = 32768):
        super().__init__()
        self.n_fft = n_fft

    def forward(self, fir_pred: torch.Tensor, phase_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fir_pred: (batch, fir_len) predicted FIR coefficients
            phase_target: (batch, n_freq) target phase in degrees
        """
        H = resample_response(fir_pred, self.n_fft, phase_target.shape[1])
        phase_pred = torch.rad2deg(torch.angle(H))

        # Shortest angular distance, wrapped into (-180, 180].
        diff = phase_pred - phase_target
        diff = (diff + 180) % 360 - 180

        return torch.mean(torch.abs(diff))


class FIRRegularization(nn.Module):
    """Penalise late-tap energy to encourage a compact impulse response."""

    def forward(self, fir_pred: torch.Tensor) -> torch.Tensor:
        fir_len = fir_pred.shape[1]
        weight = torch.linspace(0, 1, fir_len, device=fir_pred.device) ** 2
        return torch.mean(fir_pred ** 2 * weight)


class EQLoss(nn.Module):
    """Combined EQ loss.

    L = w_mag * MagnitudeLoss + w_phase * PhaseLoss + w_reg * FIRRegularization
    """

    def __init__(
        self,
        w_mag: float = 1.0,
        w_phase: float = 0.1,
        w_reg: float = 0.01,
        n_fft: int = 32768,
    ):
        super().__init__()
        self.w_mag = w_mag
        self.w_phase = w_phase
        self.w_reg = w_reg
        self.mag_loss = MagnitudeLoss(n_fft)
        self.phase_loss = PhaseLoss(n_fft)
        self.fir_reg = FIRRegularization()

    def forward(
        self,
        fir_pred: torch.Tensor,
        mag_db_target: torch.Tensor,
        phase_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_mag = self.mag_loss(fir_pred, mag_db_target)
        l_phase = self.phase_loss(fir_pred, phase_target)
        l_reg = self.fir_reg(fir_pred)

        total = self.w_mag * l_mag + self.w_phase * l_phase + self.w_reg * l_reg

        return {
            "total": total,
            "mag": l_mag,
            "phase": l_phase,
            "reg": l_reg,
        }
