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

    A parameter change is crossfaded rather than switched. The fade is a blend
    of two convolution streams, and `set_params` may arrive again while one is
    still running (a slider drag does exactly that). To keep the state coherent
    across re-entry, the in-flight fade is *collapsed* first: the two streams
    are folded into a single filter and a single overlap state, so there is
    never a filter paired with another filter's tail.

    Args:
        model_path: best_model.pt path
        block_size: audio block size in samples
        crossfade_blocks: blocks over which a parameter change is blended in.
            0 replaces the filter instantly, which clicks on large jumps.
        sample_rate: only used to label the frequency axis of
            get_frequency_response(); it does not affect processing.
    """

    def __init__(
        self,
        model_path: str,
        block_size: int = 256,
        crossfade_blocks: int = 8,
        sample_rate: float = 44100.0,
    ):
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
        self.sample_rate = float(sample_rate)

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
        identity = np.zeros(self.fir_len, dtype=np.float32)
        identity[0] = 1.0

        # The stream that is currently sounding. During a fade this is the
        # incoming filter; `_prev_fir` is the one being faded out.
        self._current_fir = identity
        self._current_H = np.fft.rfft(identity, n=self.fft_len)
        self._overlap = np.zeros(self.fft_len, dtype=np.float32)

        self._prev_fir = None
        self._overlap_prev = np.zeros(self.fft_len, dtype=np.float32)
        self._fade_done = 0
        self._xfade_remaining = 0

    def _raised_cosine(self, t: np.ndarray | float):
        """Equal-gain fade curve on [0, 1]."""
        return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))

    def _collapse(self) -> None:
        """Fold an in-flight fade into one filter and one overlap state.

        The output during a fade is (1-a)*stream_prev + a*stream_cur, and at a
        frozen `a` that equals convolving with the blended filter. Collapsing at
        the alpha already reached therefore reproduces the sounding signal
        exactly, leaving a single coherent stream to fade out of.

        Runs before every new fade. When no fade is in flight it is a no-op, and
        when one just started (`_fade_done == 0`) alpha is 0, so the outgoing
        filter is correctly the previous one — no samples of the new filter have
        been heard yet.
        """
        if self._prev_fir is None or self._xfade_remaining <= 0:
            self._prev_fir = None
            self._xfade_remaining = 0
            return

        a = self._raised_cosine(self._fade_done / self.crossfade_blocks)
        self._current_fir = (
            (1.0 - a) * self._prev_fir + a * self._current_fir
        ).astype(np.float32)
        self._current_H = np.fft.rfft(self._current_fir, n=self.fft_len)
        self._overlap = (
            (1.0 - a) * self._overlap_prev + a * self._overlap
        ).astype(np.float32)

        self._prev_fir = None
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
            self._collapse()
            self._prev_fir = self._current_fir
            self._overlap_prev = self._overlap.copy()
            self._fade_done = 0
            self._xfade_remaining = self.crossfade_blocks
        else:
            self._prev_fir = None
            self._xfade_remaining = 0

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

        if self._prev_fir is not None and self._xfade_remaining > 0:
            out_prev, self._overlap_prev = self._convolve_block(
                audio, np.fft.rfft(self._prev_fir, n=self.fft_len), self._overlap_prev
            )
            out_cur, self._overlap = self._convolve_block(
                audio, self._current_H, self._overlap
            )

            done = self._fade_done
            pos = (done + np.arange(self.block_size, dtype=np.float32) / self.block_size)
            a = self._raised_cosine(pos / self.crossfade_blocks).astype(np.float32)

            out = (1.0 - a) * out_prev + a * out_cur
            self._fade_done += 1
            self._xfade_remaining -= 1
            return out

        out, self._overlap = self._convolve_block(audio, self._current_H, self._overlap)
        return out

    def get_frequency_response(self, n_fft: int = 32768) -> tuple[np.ndarray, np.ndarray]:
        """Frequency response of the current FIR, for display.

        Returns:
            freqs: (n_fft//2+1,) Hz
            mag_db: (n_fft//2+1,) dB
        """
        H = np.fft.rfft(self._current_fir, n=n_fft)
        mag_db = 20 * np.log10(np.abs(H) + 1e-10)
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)
        return freqs, mag_db
