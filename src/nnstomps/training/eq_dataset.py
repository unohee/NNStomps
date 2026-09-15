# Created: 2026-04-01
# Purpose: EQ 프로파일 데이터셋 — 파라미터 인코딩 + 주파수 응답 타깃

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Per-plugin parameter definition + normalization ranges.
# Keys are pseudonyms; see the project public policy before adding entries.
EQ_PLUGIN_CONFIGS = {
    "bronze_eq": {
        "params": ["lf_boost", "lf_atten", "hf_boost", "hf_atten"],
        "ranges": {"lf_boost": (0, 10), "lf_atten": (0, 10),
                   "hf_boost": (0, 10), "hf_atten": (0, 10)},
        "fir_len": 256,
    },
    "midrange_eq": {
        "params": ["lm_peak", "mid_dip", "hm_peak"],
        "ranges": {"lm_peak": (0, 10), "mid_dip": (0, 10), "hm_peak": (0, 10)},
        "fir_len": 256,
    },
    "channel_eq": {
        "params": ["lf_gain", "lmf_gain", "hmf_gain", "hf_gain"],
        "ranges": {"lf_gain": (-12, 12), "lmf_gain": (-12, 12),
                   "hmf_gain": (-12, 12), "hf_gain": (-12, 12)},
        "fir_len": 256,
    },
    "passive_eq": {
        "params": [
            "ch1loenable", "ch1logain",
            "ch1lomidenable", "ch1lomidgain",
            "ch1himidenable", "ch1himidgain",
            "ch1hienable", "ch1higain",
        ],
        "ranges": {
            "ch1logain": (0, 20), "ch1lomidgain": (0, 20),
            "ch1himidgain": (0, 20), "ch1higain": (0, 20),
        },
        "categorical": {
            "ch1loenable": ["OUT", "BOOST", "CUT"],
            "ch1lomidenable": ["OUT", "BOOST", "CUT"],
            "ch1himidenable": ["OUT", "BOOST", "CUT"],
            "ch1hienable": ["OUT", "BOOST", "CUT"],
        },
        "fir_len": 1024,
    },
    "precision_eq": {
        "params": ["low_gain_1", "low_mid_gain_1", "high_mid_gain_1", "high_gain_1"],
        "ranges": {"low_gain_1": (-20, 20), "low_mid_gain_1": (-20, 20),
                   "high_mid_gain_1": (-20, 20), "high_gain_1": (-20, 20)},
        "fir_len": 1024,
    },
}


def param_spec_for(plugin_name: str) -> dict:
    """Build the portable parameter contract for a plugin.

    The spec travels with the checkpoint so the realtime engine can build model
    inputs without importing this table. Both sides call encode_params(), which
    is the single definition of the encoding.
    """
    if plugin_name not in EQ_PLUGIN_CONFIGS:
        raise KeyError(
            f"unknown plugin {plugin_name!r}; known: {sorted(EQ_PLUGIN_CONFIGS)}"
        )
    cfg = EQ_PLUGIN_CONFIGS[plugin_name]
    return {
        "params": list(cfg["params"]),
        "ranges": {k: list(v) for k, v in cfg.get("ranges", {}).items()},
        "categorical": {k: list(v) for k, v in cfg.get("categorical", {}).items()},
    }


def encoded_dim(spec: dict) -> int:
    """Width of the vector encode_params() produces for this spec."""
    dim = 0
    for p in spec["params"]:
        cats = spec.get("categorical", {}).get(p)
        dim += len(cats) if cats else 1
    return dim


def encode_params(rows: list[dict], spec: dict) -> np.ndarray:
    """Encode raw parameter dicts into model input vectors.

    Continuous parameters are min-max scaled to [0, 1]; categorical parameters
    become one-hot blocks. This is the single source of truth for the encoding —
    the training dataset and the realtime engine both call it, so the two cannot
    drift apart.

    Raises rather than substituting defaults. A silent `dict.get(key, 0)` would
    encode a missing key as 0.0, which min-max maps to the bottom of the range
    (or the centre, for symmetric ranges) — a plausible-looking value that
    quietly corrupts training.

    Args:
        rows: list of raw parameter dicts
        spec: contract from param_spec_for()

    Returns:
        (len(rows), encoded_dim(spec)) float32 array
    """
    params = spec["params"]
    ranges = spec.get("ranges", {})
    categorical = spec.get("categorical", {})

    out = np.zeros((len(rows), encoded_dim(spec)), dtype=np.float32)

    for i, row in enumerate(rows):
        col = 0
        for p in params:
            cats = categorical.get(p)
            if cats:
                if p not in row:
                    raise ValueError(f"row {i}: categorical key {p!r} missing")
                val = row[p]
                if val not in cats:
                    raise ValueError(
                        f"row {i}: {p!r}={val!r} not in allowed values {cats}"
                    )
                out[i, col + cats.index(val)] = 1.0
                col += len(cats)
            else:
                if p not in row:
                    raise ValueError(f"row {i}: continuous key {p!r} missing")
                if p not in ranges:
                    raise ValueError(f"{p!r} has no range in spec")
                lo, hi = ranges[p]
                if hi <= lo:
                    raise ValueError(f"{p!r}: invalid range ({lo}, {hi})")
                val = float(row[p])
                if not np.isfinite(val):
                    raise ValueError(f"row {i}: {p!r}={val!r} is not finite")
                out[i, col] = (val - lo) / (hi - lo)
                col += 1

    return out


def encode_param_vector(spec: dict, values: dict) -> np.ndarray:
    """Encode a single parameter dict — convenience wrapper for realtime use."""
    return encode_params([values], spec)[0]


class EQProfileDataset(Dataset):
    """EQ profile dataset yielding (parameter vector, frequency response) pairs.

    Rows are index-aligned across settings_dense.json / freq_response_dense.npy /
    phase_response_dense.npy, so a mismatch is an error rather than something to
    truncate past.

    Args:
        plugin_dir: data/{plugin}/ path
        plugin_name: key in EQ_PLUGIN_CONFIGS
        augment_factor: 1 disables; >1 adds linearly interpolated samples
        base_indices: restrict to these base rows (used to hold out a validation
            split *before* augmentation, so augmented samples cannot straddle
            the split)
        seed: RNG seed for augmentation
    """

    def __init__(
        self,
        plugin_dir: str,
        plugin_name: str,
        augment_factor: int = 1,
        base_indices: list[int] | None = None,
        seed: int = 42,
    ):
        pdir = Path(plugin_dir)
        cfg = EQ_PLUGIN_CONFIGS[plugin_name]
        self.plugin_name = plugin_name
        self.fir_len = cfg["fir_len"]
        self.spec = param_spec_for(plugin_name)

        settings = json.loads((pdir / "settings_dense.json").read_text())
        mag_db = np.load(str(pdir / "freq_response_dense.npy"))
        phase_deg = np.load(str(pdir / "phase_response_dense.npy"))

        n_base = self.base_count(plugin_dir)
        if not (len(settings) == mag_db.shape[0] == phase_deg.shape[0]):
            raise ValueError(
                f"{plugin_name}: row count mismatch — settings={len(settings)}, "
                f"mag={mag_db.shape[0]}, phase={phase_deg.shape[0]}"
            )

        if base_indices is None:
            indices = np.arange(n_base)
        else:
            indices = np.asarray(base_indices, dtype=int)
            if indices.size and (indices.min() < 0 or indices.max() >= n_base):
                raise IndexError(
                    f"{plugin_name}: base_indices out of range [0, {n_base})"
                )

        all_params = encode_params(settings[:n_base], self.spec)
        self.cond_dim = all_params.shape[1]

        self.params = all_params[indices]
        self.mag_db = mag_db[:n_base][indices].astype(np.float32)
        self.phase_deg = phase_deg[:n_base][indices].astype(np.float32)

        if augment_factor > 1:
            self._augment(augment_factor, seed)

        logger.info(f"EQProfileDataset: {plugin_name}, {len(self)} samples, "
                    f"cond_dim={self.cond_dim}, fir_len={self.fir_len}")

    @staticmethod
    def base_count(plugin_dir: str) -> int:
        """Number of rows before augmentation — needed to split indices first."""
        return len(json.loads((Path(plugin_dir) / "settings_dense.json").read_text()))

    def _augment(self, factor: int, seed: int):
        """Add linearly interpolated samples between existing rows.

        Interpolation is applied to already-encoded parameters in [0, 1] space,
        which is linear in the raw parameter values because the encoding is
        affine. Categorical one-hot blocks blend into fractional values; those
        are only ever training inputs, never decoded back to a parameter value.
        """
        n = len(self.params)
        if n < 2:
            logger.warning("%s: augmentation skipped (need >= 2 rows)", self.plugin_name)
            return

        rng = np.random.default_rng(seed)
        n_new = n * (factor - 1)
        i = rng.integers(0, n, size=n_new)
        j = rng.integers(0, n, size=n_new)
        alpha = rng.uniform(0.1, 0.9, size=(n_new, 1)).astype(np.float32)

        blend = lambda a, b: a * alpha + b * (1.0 - alpha)  # noqa: E731

        self.params = np.concatenate([self.params, blend(self.params[i], self.params[j])])
        self.mag_db = np.concatenate([self.mag_db, blend(self.mag_db[i], self.mag_db[j])])
        self.phase_deg = np.concatenate([
            self.phase_deg,
            blend(self.phase_deg[i], self.phase_deg[j]),
        ])

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx: int) -> dict:
        return {
            "params": torch.from_numpy(self.params[idx]),
            "mag_db": torch.from_numpy(self.mag_db[idx]),
            "phase_deg": torch.from_numpy(self.phase_deg[idx]),
        }
