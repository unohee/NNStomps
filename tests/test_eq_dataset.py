# Created: 2026-09-15
# Purpose: EQ 데이터셋 인코딩·분할 회귀 테스트 — 조용한 키 누락과 증강 누수 방지

import json
from pathlib import Path

import numpy as np
import pytest

from nnstomps.training.eq_dataset import (
    EQ_PLUGIN_CONFIGS,
    EQProfileDataset,
    encode_param_vector,
    encoded_dim,
    param_spec_for,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _spec():
    return param_spec_for("channel_eq")


def test_continuous_encoding_spans_zero_to_one():
    spec = _spec()
    lo = {"lf_gain": -12.0, "lmf_gain": -12.0, "hmf_gain": -12.0, "hf_gain": -12.0}
    hi = {"lf_gain": 12.0, "lmf_gain": 12.0, "hmf_gain": 12.0, "hf_gain": 12.0}

    assert np.allclose(encode_param_vector(spec, lo), 0.0)
    assert np.allclose(encode_param_vector(spec, hi), 1.0)


def test_categorical_encoding_is_one_hot():
    spec = param_spec_for("passive_eq")
    values = {
        "ch1loenable": "BOOST", "ch1logain": 0.0,
        "ch1lomidenable": "CUT", "ch1lomidgain": 0.0,
        "ch1himidenable": "OUT", "ch1himidgain": 0.0,
        "ch1hienable": "OUT", "ch1higain": 0.0,
    }
    vec = encode_param_vector(spec, values)

    assert vec.shape[0] == encoded_dim(spec)
    # 4 continuous + 4 categorical x 3 options
    assert vec.shape[0] == 16

    # Walk the layout to find each categorical block's column offset, rather
    # than assuming the block starts at index 0.
    offset = 0
    layout = {}
    for p in spec["params"]:
        cats = spec.get("categorical", {}).get(p)
        layout[p] = (offset, cats)
        offset += len(cats) if cats else 1

    for name, expected in [("ch1loenable", "BOOST"), ("ch1lomidenable", "CUT")]:
        start, cats = layout[name]
        block = vec[start:start + len(cats)]
        assert block[list(cats).index(expected)] == 1.0
        assert block.sum() == 1.0, f"{name}: not a one-hot block"


def test_missing_continuous_key_raises():
    """A missing key must raise rather than encode as 0.0.

    0.0 min-max scales to the bottom of the range for a (0, 10) parameter and to
    the centre for a symmetric one — both look like valid settings, so a silent
    default corrupts training without any signal.
    """
    with pytest.raises(ValueError, match="missing"):
        encode_param_vector(_spec(), {"lf_gain": 0.0})


def test_unknown_categorical_value_raises():
    spec = param_spec_for("passive_eq")
    values = {
        "ch1loenable": "BOOSTED",  # not in the allowed set
        "ch1logain": 0.0, "ch1lomidenable": "OUT", "ch1lomidgain": 0.0,
        "ch1himidenable": "OUT", "ch1himidgain": 0.0,
        "ch1hienable": "OUT", "ch1higain": 0.0,
    }
    with pytest.raises(ValueError, match="not in allowed"):
        encode_param_vector(spec, values)


def test_non_finite_value_raises():
    values = {"lf_gain": float("nan"), "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0}
    with pytest.raises(ValueError, match="not finite"):
        encode_param_vector(_spec(), values)


def test_all_config_params_exist_in_the_data():
    """Guards the drift the reviewer flagged: config names vs profile data keys.

    EQ_PLUGIN_CONFIGS is maintained by hand while settings_dense.json is written
    by the profiling script. If one side is renamed the other silently encodes
    zeros — so the key sets are checked here against the real data.
    """
    missing = []
    for name, cfg in EQ_PLUGIN_CONFIGS.items():
        data_dir = DATA_ROOT / name
        if not data_dir.is_dir():
            pytest.skip(f"{name}: profile data not present")

        settings = json.loads((data_dir / "settings_dense.json").read_text())
        present = set()
        for row in settings:
            present |= set(row.keys())
        missing += [f"{name}:{p}" for p in cfg["params"] if p not in present]

    assert not missing, f"config params absent from profile data: {missing}"


def test_unexpected_plugin_dir_is_rejected():
    with pytest.raises(KeyError):
        param_spec_for("not_a_plugin")


def test_row_count_mismatch_raises(tmp_path):
    """Truncating to the shorter array would silently misalign rows."""
    d = tmp_path / "channel_eq"
    d.mkdir()
    (d / "settings_dense.json").write_text(json.dumps([
        {"lf_gain": 0.0, "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0}
        for _ in range(4)
    ]))
    np.save(d / "freq_response_dense.npy", np.zeros((4, 16), dtype=np.float32))
    np.save(d / "phase_response_dense.npy", np.zeros((3, 16), dtype=np.float32))

    with pytest.raises(ValueError, match="row count mismatch"):
        EQProfileDataset(str(d), "channel_eq")


def test_base_indices_select_an_exact_subset(tmp_path):
    """Base indices must address pre-augmentation rows."""
    d = tmp_path / "channel_eq"
    d.mkdir()
    (d / "settings_dense.json").write_text(json.dumps([
        {"lf_gain": float(i), "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0}
        for i in range(-12, 13)
    ]))
    n = 25
    np.save(d / "freq_response_dense.npy", np.zeros((n, 16), dtype=np.float32))
    np.save(d / "phase_response_dense.npy", np.zeros((n, 16), dtype=np.float32))

    assert EQProfileDataset.base_count(str(d)) == n

    ds = EQProfileDataset(str(d), "channel_eq", base_indices=[0, 5, 7])
    assert len(ds) == 3
    # row 0 encodes lf_gain=-12 -> 0.0; row 24 would be 1.0
    assert ds.params[0][0] == pytest.approx(0.0)

    with pytest.raises(IndexError):
        EQProfileDataset(str(d), "channel_eq", base_indices=[n])


def test_augmentation_interpolates_phase_along_the_short_arc(tmp_path):
    """Phase blending must hug the +-180 branch cut, not cross it.

    Two rows at +179 and -179 degrees are the same physical phase modulo 360.
    Linear blending would put the midpoint at 0 degrees — a target no real
    response has — poisoning a few percent of augmented rows. The blend must
    stay near +-180, i.e. on the short arc.
    """
    d = tmp_path / "channel_eq"
    d.mkdir()
    (d / "settings_dense.json").write_text(json.dumps([
        {"lf_gain": 0.0, "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0},
        {"lf_gain": 12.0, "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0},
    ]))
    np.save(d / "freq_response_dense.npy", np.zeros((2, 8), dtype=np.float32))
    phases = np.zeros((2, 8), dtype=np.float32)
    phases[0, 0] = 179.0
    phases[1, 0] = -179.0
    np.save(d / "phase_response_dense.npy", phases)

    ds = EQProfileDataset(str(d), "channel_eq", augment_factor=4)

    # Every row's bin 0 must sit near +-180 — never near 0.
    for col in ds.phase_deg[:, 0]:
        assert abs(col) > 170.0, f"augmented phase {col:.1f} crosses the branch cut"


def test_augmentation_stays_within_the_given_subset(tmp_path):
    """Augmented rows must interpolate only between rows handed to the split.

    When augmentation ran before the train/val split, interpolated samples could
    sit between two training rows while their endpoints landed in validation —
    the reported validation error then was not held out.
    """
    d = tmp_path / "channel_eq"
    d.mkdir()
    (d / "settings_dense.json").write_text(json.dumps([
        {"lf_gain": float(i), "lmf_gain": 0.0, "hmf_gain": 0.0, "hf_gain": 0.0}
        for i in range(-12, 13)
    ]))
    n = 25
    np.save(d / "freq_response_dense.npy", np.zeros((n, 16), dtype=np.float32))
    np.save(d / "phase_response_dense.npy", np.zeros((n, 16), dtype=np.float32))

    train_ds = EQProfileDataset(str(d), "channel_eq", augment_factor=4,
                                base_indices=[0, 1, 2])
    assert len(train_ds) == 3 * 4

    # every augmented row must be a convex combination of the chosen rows, i.e.
    # its first encoded coordinate stays inside [min, max] of those rows.
    # Rows 0..2 are lf_gain = -12..-10 -> 0/24 .. 2/24; a leaked row would allow
    # values up to 24/24.
    col = train_ds.params[:, 0]
    assert col.min() >= -1e-6
    assert col.max() <= 2 / 24 + 1e-6