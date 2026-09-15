# Created: 2026-09-15
# Purpose: NeuralEQEngine 회귀 테스트 — overlap-add 등가성, 크로스페이드, 상태 초기화, 입력 검증

import numpy as np
import pytest
import torch

from nnstomps.core.neural_eq import NeuralEQEngine

BLOCK = 64


@pytest.fixture
def engine(synthetic_checkpoint):
    return NeuralEQEngine(synthetic_checkpoint(), block_size=BLOCK, crossfade_blocks=0)


def _natural_params():
    return {"lf_gain": 3.0, "lmf_gain": -4.0, "hmf_gain": 0.0, "hf_gain": 6.0}


def test_block_processing_matches_full_convolution(engine):
    """Streaming output must equal convolving the whole signal at once.

    This is the core invariant of the overlap-add path: every block boundary is
    a place where a state bug hides, so the signal is several blocks long.
    """
    rng = np.random.default_rng(0)
    n_blocks = 5
    x = rng.standard_normal(BLOCK * n_blocks).astype(np.float32)

    fir = engine._current_fir.copy()
    engine.reset()

    streamed = np.concatenate([
        engine.process_block(x[i * BLOCK:(i + 1) * BLOCK]) for i in range(n_blocks)
    ])
    reference = np.convolve(x, fir)[:len(x)]

    assert np.max(np.abs(streamed - reference)) < 1e-4


def test_overlap_add_matches_full_convolution_with_trained_filter(engine):
    """Same invariant, but with a non-trivial predicted filter."""
    engine.set_params(_natural_params())
    fir = engine._current_fir.copy()
    assert not np.allclose(fir, [1.0] + [0.0] * (len(fir) - 1)), "expected a real filter"

    rng = np.random.default_rng(1)
    x = rng.standard_normal(BLOCK * 4).astype(np.float32)

    # No reset() here: it would restore the identity filter while `fir` holds
    # the predicted one. The engine is freshly constructed, so state is clean.
    streamed = np.concatenate([
        engine.process_block(x[i * BLOCK:(i + 1) * BLOCK]) for i in range(4)
    ])
    reference = np.convolve(x, fir)[:len(x)]

    assert np.max(np.abs(streamed - reference)) < 1e-4


def test_wrong_block_length_raises(engine):
    """A short or long block must raise, not silently desynchronise the stream."""
    with pytest.raises(ValueError, match="expected a mono block"):
        engine.process_block(np.zeros(BLOCK + 5, dtype=np.float32))
    with pytest.raises(ValueError, match="expected a mono block"):
        engine.process_block(np.zeros(BLOCK - 5, dtype=np.float32))
    with pytest.raises(ValueError, match="expected a mono block"):
        engine.process_block(np.zeros((BLOCK, 2), dtype=np.float32))


def test_missing_parameter_raises(engine):
    """Missing keys must not be silently defaulted to a plausible value."""
    with pytest.raises(ValueError, match="missing"):
        engine.set_params({"lf_gain": 0.0})


def test_reset_clears_stream_state(engine):
    """After reset, a silent block must produce silence.

    Without clearing, the tail of the previous stream (up to fir_len - 1
    samples) leaks into the head of the next one.
    """
    engine.set_params(_natural_params())

    rng = np.random.default_rng(2)
    engine.process_block(rng.standard_normal(BLOCK).astype(np.float32))

    engine.reset()
    out = engine.process_block(np.zeros(BLOCK, dtype=np.float32))

    assert np.allclose(out, 0.0)


def test_crossfade_reduces_discontinuity(synthetic_checkpoint):
    """A parameter change must not produce a step discontinuity in the output.

    A 100 Hz sine has small natural sample-to-sample deltas, so any click at the
    change point stands out clearly in max|y[n] - y[n-1]|.
    """
    sr = 44100.0
    freq = 100.0
    n_blocks = 24
    change_at = 8

    t = np.arange(BLOCK * n_blocks) / sr
    x = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def run(crossfade_blocks):
        eng = NeuralEQEngine(
            synthetic_checkpoint(), block_size=BLOCK, crossfade_blocks=crossfade_blocks
        )
        eng.set_params({"lf_gain": 0.0, "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0})
        out = []
        for b in range(n_blocks):
            if b == change_at:
                eng.set_params({"lf_gain": 12.0, "lmf_gain": 12.0,
                                "hmf_gain": 12.0, "hf_gain": 12.0})
            out.append(eng.process_block(x[b * BLOCK:(b + 1) * BLOCK]))
        return np.concatenate(out)

    jump_instant = np.max(np.abs(np.diff(run(0))))
    jump_faded = np.max(np.abs(np.diff(run(8))))

    assert jump_faded < jump_instant, (
        f"crossfade did not smooth the transition: instant={jump_instant:.6f}, "
        f"faded={jump_faded:.6f}"
    )


def test_engine_requires_param_spec(synthetic_checkpoint):
    """A checkpoint without a spec must be rejected, not guessed at."""
    path = synthetic_checkpoint(with_spec=False)
    with pytest.raises(ValueError, match="param_spec"):
        NeuralEQEngine(path, block_size=BLOCK)


def test_natural_unit_params_are_normalized(synthetic_checkpoint):
    """Callers pass natural units; the engine normalizes against the spec.

    Guards the contract bug where metadata advertised natural-unit ranges while
    the model expected [0, 1] inputs — the same setting produced two entirely
    different filters with no error.

    The FIR is compared against the model run on the encoded vector. Only
    asserting that two settings differ would also pass if the engine fed raw
    values straight in, which is exactly the bug this guards against.
    """
    from nnstomps.training.eq_dataset import EQ_PLUGIN_CONFIGS, encode_param_vector

    spec = EQ_PLUGIN_CONFIGS["channel_eq"]
    params = {"lf_gain": 3.0, "lmf_gain": -4.0, "hmf_gain": 0.0, "hf_gain": 6.0}

    eng = NeuralEQEngine(synthetic_checkpoint(), block_size=BLOCK, crossfade_blocks=0)
    eng.set_params(params)

    encoded = torch.from_numpy(encode_param_vector(eng.param_spec, params))[None, :]
    with torch.no_grad():
        expected = eng.model(encoded)[0].numpy()

    assert np.allclose(eng._current_fir, expected, atol=1e-6), (
        "engine output does not match the model run on the encoded vector — "
        "it is either skipping normalization or encoding differently"
    )


def test_out_of_range_parameter_raises(synthetic_checkpoint):
    """A value outside the profiled range must raise, not extrapolate.

    `lf_gain=100` would otherwise encode to 4.67 — far outside the training
    distribution — with no signal to the caller.
    """
    eng = NeuralEQEngine(synthetic_checkpoint(), block_size=BLOCK, crossfade_blocks=0)
    with pytest.raises(ValueError, match="outside the profiled range"):
        eng.set_params({"lf_gain": 100.0, "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0})


def test_set_params_mid_crossfade_does_not_click(synthetic_checkpoint):
    """Calling set_params again while a fade is running must not click.

    A no-op A→B→A inside the fade window used to produce a step as large as an
    instant switch: the outgoing branch was re-armed with a filter that had not
    sounded yet, while its overlap state still belonged to the sounding one.
    """
    sr = 44100.0
    freq = 100.0
    n_blocks = 40
    t = np.arange(BLOCK * n_blocks) / sr
    x = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    flat = {"lf_gain": 0.0, "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0}
    boost = {"lf_gain": 12.0, "lmf_gain": 12.0, "hmf_gain": 12.0, "hf_gain": 12.0}

    def run(calls):
        eng = NeuralEQEngine(synthetic_checkpoint(), block_size=BLOCK, crossfade_blocks=8)
        eng.set_params(flat)
        out = []
        for b in range(n_blocks):
            for at, params in calls:
                if b == at:
                    eng.set_params(params)
            out.append(eng.process_block(x[b * BLOCK:(b + 1) * BLOCK]))
        return np.concatenate(out)

    y_single = run([(8, boost)])
    y_reentrant = run([(8, boost), (9, flat)])     # second call inside the fade
    y_baseline = run([])

    # The failure mode is a step discontinuity. Compare the maximum
    # sample-to-sample jump: the re-entrant case used to land at ~100% of an
    # instant switch, ~50x the baseline. After the collapse fix it should stay
    # on the order of a single faded change.
    j_single = np.max(np.abs(np.diff(y_single)))
    j_reentrant = np.max(np.abs(np.diff(y_reentrant)))
    j_baseline = np.max(np.abs(np.diff(y_baseline)))

    assert j_reentrant < 10 * j_single + 1e-6, (
        f"re-entrant set_params jumped {j_reentrant:.5f} vs {j_single:.5f} for a "
        f"single change — the fade state was not collapsed"
    )
    assert j_reentrant < 30 * j_baseline + 1e-6