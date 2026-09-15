# Changelog

## [Unreleased] — 2026-09-15

### Added — Neural EQ
- `src/nnstomps/core/neural_eq.py` — `NeuralEQEngine`: MLP-generated FIR coefficients
  with overlap-add FFT convolution, per-sample crossfade on parameter changes, and a
  `reset()` for stream reuse
- `src/nnstomps/training/eq_model.py` — `NNStompEQ`: conditional MLP, parameters → FIR
- `src/nnstomps/training/eq_dataset.py` — `EQProfileDataset` plus `encode_params()`, the
  single definition of the parameter encoding shared by training and the realtime engine
- `src/nnstomps/training/eq_losses.py` — magnitude, phase, and FIR-compactness losses
- `src/nnstomps/training/train_eq.py` — EQ training loop
- `src/nnstomps/training/export_eq.py` — ONNX export plus web UI metadata

### Added — tests
- `tests/` goes from empty to 42 tests. Covers overlap-add equivalence against a direct
  convolution, crossfade smoothing, stream reset, input validation, the parameter
  encoding contract, loss behaviour, and the export/training guards.

### Changed
- Parameter encoding has one definition. `encode_params()` raises on a missing key, an
  unknown categorical value, or a non-finite value — the previous `dict.get(key, 0)`
  encoded a missing key into a plausible-looking value and trained on it silently.
- `NeuralEQEngine` takes parameters in natural units and normalizes internally, using a
  `param_spec` stored in the checkpoint. The previous contract asked callers for
  normalized inputs while the exported metadata advertised natural-unit ranges; the same
  setting produced two different filters with no error.
- Validation split is taken before augmentation, so interpolated samples can no longer
  straddle the train/validation boundary.
- Frequency-response resampling interpolates the complex response with
  `align_corners=True`. Interpolating wrapped angles corrupted bins near the ±180 branch
  cut, and the bin-to-frequency mapping was half a bin off.
- EQ plugin identifiers use pseudonyms, matching the model directory names.

### Fixed
- `export_eq` raises on a `plugin_name` absent from `EQ_PLUGIN_CONFIGS` instead of
  emitting an empty control set, which rendered a UI with no sliders and wrote the
  unknown name into the published metadata.
- `process_block` rejects block lengths other than `block_size`; other lengths aliased
  and desynchronised the stream silently.
- `train_eq` rejects `epochs <= 0` instead of saving a checkpoint with
  `model_state: None` and reporting success.

### Removed
- `fir_target` and `_freq_response_to_fir()` — unused, and the window placement delayed
  the response by `fir_len // 2` samples. Removing them also removed the `scipy`
  dependency.

### Infrastructure
- Local-only scripts moved to `scripts/local/`, excluded by directory. The previous
  `.gitignore` listed files by name, which needed a manual edit for every new script and
  put the plugin names in the committed file.
- `onnx` declared in the `training` extra (required by `export_eq`); `auraloss` dropped
  (never imported).
- Repository history rewritten, and `gh-pages` rebuilt as a single commit, to remove
  commercial plugin names that had been published.

## [0.1.0] — 2026-03-26

### Added — GRU training pipeline
- `src/nnstomps/training/` — `model.py` (NNStompGRU 5,441 params, NNStompGRU2 9,141),
  `losses.py` (ESR, MultiScaleSTFT, PreEmphasisLoss, DCLoss, NNStompLoss), `dataset.py`
  (memory preload), `train.py` (TBPTT, curriculum, AMP, CosineAnnealing), `export.py`
  (PyTorch → RTNeural JSON with verification), `evaluate.py` (A/B comparison, ESR),
  `generate_pairs.py`, `cmaes_sound_match.py`
- `src/nnstomps/core/test_signal_v2.py` — AD envelope test signals: impulse tones,
  multi-tone bursts, velocity sweeps, chord bursts, and the 214-signal full suite
- `src/nnstomps/core/hybrid_drive.py` — `WaveshaperLUT` (multi-level transfer functions)
  and `HybridDrive` (static LUT + dynamic GRU residual)
- Scripts: `demo.py` (Gradio UI on :7870), `realtime.py`, `train_blackstar.py`,
  `train_all.py`, `cmaes_optimize.py`
- `models/blackstar/` — example model (dual-drive tube saturation)

### Added — initial engine
- `NeuralDrive` (from_data_dir, search, process, process_by_text)
- CLI (search, process, info)
- Plugin profiling data: CLAP embeddings, waveshaper curves

### Changed
- `plugin_analysis.py` — fixed an import (`vst3` → `vst3_wrapper`); added
  `WaveshaperV2Result` and `measure_waveshaper_v2()`
- `pyproject.toml` — added the `training` optional-dependency group

### Fixed
- Waveshaper capture covered only 10% of the input range; v2 reaches 99.9%

### Discovered
- **Pre-emphasis loss**: ESR + STFT alone cannot learn harmonics.
  `PreEmphasisLoss(coeff=0.99)` is what forces it
- **Parameter names matter**: a mismatch between the plugin's real parameter names and
  the names in code fails quietly
- **CMA-ES optimum**: w_esr=0.48, w_stft=1.11, w_preemph=1.30
