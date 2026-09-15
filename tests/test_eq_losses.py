# Created: 2026-09-15
# Purpose: EQ 손실 함수 회귀 테스트 — dB 스케일, 위상 branch cut, 해상도 리샘플링

import numpy as np
import pytest
import torch

from nnstomps.training.eq_losses import (
    EQLoss,
    FIRRegularization,
    MagnitudeLoss,
    PhaseLoss,
    resample_response,
)

N_FFT = 2048


def _random_fir(batch=2, fir_len=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, fir_len, generator=g)


def _response(fir, n_fft):
    """Full-resolution response — n_fft//2+1 bins, so no resampling is involved."""
    padded = torch.zeros(fir.shape[0], n_fft)
    padded[:, :fir.shape[1]] = fir
    return torch.fft.rfft(padded)


def test_magnitude_loss_is_zero_for_identical_curves():
    fir = _random_fir()
    H = _response(fir, N_FFT)
    mag_db = 20 * torch.log10(torch.abs(H) + 1e-10)

    loss = MagnitudeLoss(n_fft=N_FFT)(fir, mag_db)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_magnitude_loss_grows_with_a_constant_offset():
    """A uniform dB offset must cost exactly that many dB."""
    fir = _random_fir()
    H = _response(fir, N_FFT)
    mag_db = 20 * torch.log10(torch.abs(H) + 1e-10)

    for offset in (1.0, 3.0, 6.0):
        loss = MagnitudeLoss(n_fft=N_FFT)(fir, mag_db - offset)
        assert loss.item() == pytest.approx(offset, abs=1e-3)


def test_phase_loss_is_zero_for_the_matching_response():
    fir = _random_fir()
    H = _response(fir, N_FFT)
    phase_deg = torch.rad2deg(torch.angle(H))

    loss = PhaseLoss(n_fft=N_FFT)(fir, phase_deg)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_phase_loss_respects_wrapping():
    """A small perturbation crossing +-180 must not read as a huge error."""
    fir = _random_fir()
    H = _response(fir, N_FFT)
    phase_deg = torch.rad2deg(torch.angle(H))

    shifted = phase_deg + 5.0
    shifted = (shifted + 180) % 360 - 180

    loss = PhaseLoss(n_fft=N_FFT)(fir, shifted)
    assert loss.item() == pytest.approx(5.0, abs=1e-3)


def test_resample_response_matches_analytic_pure_delay():
    """A single-tap delay has a known closed-form response at any resolution.

    Resampling must interpolate the complex response, not magnitude and angle
    separately. Interpolating wrapped angles turns a pair like (+179, -179) into
    0 degrees — a false reversal that injects a gradient pointing the wrong way.
    """
    n_fft, n_target, delay = 512, 1024, 8
    fir = torch.zeros(1, 64)
    fir[0, delay] = 1.0

    got = resample_response(fir, n_fft, n_target)[0].numpy()

    # rfft bins partition DC..Nyquist evenly, so bin i sits at frequency
    # i/(n_target-1) * fs/2, giving exp(-j*pi*i*delay/(n_target-1)) for a
    # pure delay of `delay` samples.
    idx = np.arange(n_target)
    expected = np.exp(-1j * np.pi * idx * delay / (n_target - 1))

    # Phase is the invariant that matters. Magnitude is checked separately
    # against its exact bound: interpolating linearly between two unit-modulus
    # points traces a chord, so |H| dips by 1 - cos(step/2) at the midpoint of
    # each interval. That is inherent to linear resampling, not an error.
    phase_err = np.abs(np.angle(got) - np.angle(expected))
    phase_err = np.minimum(phase_err, 2 * np.pi - phase_err)

    step = np.pi * delay / (n_fft // 2)
    chord_sag = 1.0 - np.cos(step / 2)
    mag = np.abs(got)

    assert phase_err.max() < 1e-3, f"phase error {phase_err.max():.6f}"
    assert mag.max() <= 1.0 + 1e-6
    assert mag.min() >= 1.0 - chord_sag - 1e-6


def test_angle_interpolation_would_be_worse_than_complex():
    """Regression guard: documents why the implementation resamples complex."""
    n_fft, n_target, delay = 512, 1024, 8
    fir = torch.zeros(1, 64)
    fir[0, delay] = 1.0

    padded = torch.zeros(1, n_fft)
    padded[:, :64] = fir
    H = torch.fft.rfft(padded)

    def resample(x):
        return torch.nn.functional.interpolate(
            x[None, None], size=n_target, mode="linear", align_corners=True
        )[0, 0]

    ang = torch.angle(H)[0]
    mag = torch.abs(H)[0]
    naive = torch.polar(resample(mag), resample(ang))

    idx = np.arange(n_target)
    expected = np.exp(-1j * np.pi * idx * delay / (n_target - 1))

    err_complex = np.max(np.abs(resample_response(fir, n_fft, n_target)[0].numpy() - expected))
    err_naive = np.max(np.abs(naive.numpy() - expected))

    assert err_naive > 10 * err_complex, (
        f"expected angle interpolation to be clearly worse: "
        f"complex={err_complex:.5f}, naive={err_naive:.5f}"
    )


def test_fir_regularization_prefers_compact_filters():
    early = torch.zeros(1, 64)
    early[0, 0] = 1.0
    late = torch.zeros(1, 64)
    late[0, -1] = 1.0

    reg = FIRRegularization()
    assert reg(early).item() < reg(late).item()


def test_combined_loss_is_finite_and_differentiable():
    fir = _random_fir(seed=3).requires_grad_(True)
    H = _response(fir.detach(), N_FFT)
    mag_db = 20 * torch.log10(torch.abs(H) + 1e-10)
    phase_deg = torch.rad2deg(torch.angle(H))

    out = EQLoss(n_fft=N_FFT)(fir, mag_db, phase_deg)
    out["total"].backward()

    assert torch.isfinite(out["total"])
    assert fir.grad is not None and torch.isfinite(fir.grad).all()


def test_fir_len_larger_than_fft_is_rejected():
    with pytest.raises(ValueError, match="exceeds n_fft"):
        resample_response(torch.zeros(1, 128), n_fft=64, n_target=33)