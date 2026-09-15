# Created: 2026-04-01
# Purpose: Neural EQ 실시간 엔진 — MLP → FIR → overlap-add convolution

import numpy as np
import torch


class NeuralEQEngine:
    """Realtime Neural EQ engine.

    An MLP predicts FIR coefficients from EQ parameters; audio is processed by
    overlap-add FFT convolution.

    Parameters are supplied in natural units (dB, seconds) and normalized
    internally using the `param_spec` stored in the checkpoint. That keeps the
    encoding in exactly one place: the same `encode_params` used to build the
    training set. Callers cannot forget to normalize, and a caller that *does*
    pre-normalize the way the old API required now produces a different filter
    with no error — which is why the natural-unit contract is enforced here
    rather than documented and hoped for.

    Args:
        model_path: best_model.pt path
        block_size: audio block size in samples
        crossfade_blocks: blocks over which a parameter change is blended in.
            0 replaces the filter instantly, which clicks on large jumps.
    """

    def __init__(self, model_path: str, block_size: int = 256, crossfade_blocks: int = 8):
        from nnstomps.training.eq_dataset import encode_param_vector
        from nnstomps.training.eq_model import NNStompEQ

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]

        if "param_spec" not in cfg:
            raise ValueError(
                f"{model_path}: checkpoint has no param_spec, so the parameter "
                f"contract used at training time is unknown. Re-export it or "
                f"inject the spec before using the realtime engine."
            )

        self._encode = encode_param_vector
        self.param_spec = cfg["param_spec"]
        self.plugin_name = cfg.get("plugin_name", "")
        self.cond_dim = cfg["cond_dim"]
        self.fir_len = cfg["fir_len"]
        self.block_size = block_size
        self.crossfade_blocks = max(0, int(crossfade_blocks))

        self.model = NNStompEQ(cfg["cond_dim"], cfg["fir_len"], cfg.get("hidden_dims"))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # FFT size must cover the linear convolution of one block with the FIR.
        self.fft_len = 1
        while self.fft_len < block_size + self.fir_len:
            self.fft_len *= 2

        self.reset()

    def reset(self) -> None:
        """Clear all stream state.

        Call this before reusing the engine on a new stream. Without it the tail
        of the previous stream (up to fir_len - 1 samples) would be convolved
        into the head of the next one.
        """
        self._current_fir = np.zeros(self.fir_len, dtype=np.float32)
        self._current_fir[0] = 1.0  # identity (pass-through) filter
        self._current_H = np.fft.rfft(self._current_fir, n=self.fft_len)

        self._overlap = np.zeros(self.fft_len, dtype=np.float32)

        self._prev_H = None
        self._overlap_prev = np.zeros(self.fft_len, dtype=np.float32)
        self._xfade_remaining = 0

    def set_params(self, values: dict) -> None:
        """Update EQ parameters, in natural units, keyed by parameter name.

        Args:
            values: e.g. {"lf_gain": -3.0, "hf_gain": 6.0}; names come from
                self.param_spec["params"]. Missing keys raise ValueError rather
                than defaulting.
        """
        vec = self._encode(self.param_spec, values)

        with torch.no_grad():
            fir = self.model(torch.from_numpy(vec)[None, :])[0].numpy().astype(np.float32)

        if self.crossfade_blocks > 0:
            self._prev_H = self._current_H
            self._overlap_prev = self._overlap.copy()
            self._xfade_remaining = self.crossfade_blocks

        self._current_fir = fir
        self._current_H = np.fft.rfft(fir, n=self.fft_len)

    def _convolve_block(self, audio: np.ndarray, H: np.ndarray, overlap: np.ndarray):
        """One overlap-add step. Returns (output block, next overlap state)."""
        X = np.fft.rfft(audio, n=self.fft_len)
        y = np.fft.irfft(X * H, n=self.fft_len).astype(np.float32)
        y += overlap

        out = y[:self.block_size].copy()
        next_overlap = np.zeros(self.fft_len, dtype=np.float32)
        next_overlap[:self.fft_len - self.block_size] = y[self.block_size:]
        return out, next_overlap

    def process_block(self, audio: np.ndarray) -> np.ndarray:
        """Process one block.

        Args:
            audio: (block_size,) float32 mono input

        Returns:
            (block_size,) float32 mono output
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or audio.shape[0] != self.block_size:
            raise ValueError(
                f"expected a mono block of {self.block_size} samples, got shape "
                f"{audio.shape}. The overlap-add state is sized for one block "
                f"length; other lengths silently alias and desynchronise the "
                f"stream instead of raising."
            )

        out, self._overlap = self._convolve_block(audio, self._current_H, self._overlap)

        if self._xfade_remaining > 0:
            # Blending the two filter outputs sample-by-sample is equivalent to
            # blending the coefficients (convolution is linear), but doing it on
            # the outputs lets the ramp be per-sample instead of per-block.
            old, self._overlap_prev = self._convolve_block(
                audio, self._prev_H, self._overlap_prev
            )
            done = self.crossfade_blocks - self._xfade_remaining
            pos = (done + np.arange(self.block_size, dtype=np.float32) / self.block_size)
            a = pos / self.crossfade_blocks
            a = (0.5 - 0.5 * np.cos(np.pi * a)).astype(np.float32)  # raised cosine
            out = (1.0 - a) * old + a * out
            self._xfade_remaining -= 1

        return out

    def get_frequency_response(self, n_fft: int = 32768) -> tuple[np.ndarray, np.ndarray]:
        """Frequency response of the current FIR, for display.

        Returns:
            freqs: (n_fft//2+1,) Hz
            mag_db: (n_fft//2+1,) dB
        """
        H = np.fft.rfft(self._current_fir, n=n_fft)
        mag_db = 20 * np.log10(np.abs(H) + 1e-10)
        freqs = np.fft.rfftfreq(n_fft, 1 / 44100)
        return freqs, mag_db