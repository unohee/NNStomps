# NNStomps — Neural Drive

GRU neural network based distortion/saturation modeling. Profile any audio plugin's nonlinear characteristics, train a neural network clone, and run it in real-time.

## What it does

```
Plugin profiling (audioman doctor)
  → 214 test signals × drive settings = input/output pairs
  → Conditional GRU training (drive value controls timbre)
  → Real-time audio processing (5.8ms latency)
  → JUCE + RTNeural VST3/AU build (planned)
```

A second engine models **equalizers** rather than saturation — see [Neural EQ](#neural-eq).

## Architecture

```
                  ┌─────────────────────────────┐
                  │  NNStomps Hybrid Engine      │
                  ├─────────────────────────────┤
Input Audio ──→   │ [Static]  Waveshaper LUT    │ ──→ Output Audio
                  │           (multi-level lerp) │
Condition   ──→   │ [Dynamic] GRU Residual      │
(drive, tone)     │           (attack/release)   │
                  └─────────────────────────────┘
```

**Two engines:**
- **GRU standalone**: Conditional GRU learns the full nonlinear transfer (current main approach)
- **Hybrid**: Waveshaper LUT (static) + GRU residual (dynamic) — experimental

## What is in this repository

This is the code, plus one worked example. Profiled data and trained models for every
other plugin are **not** tracked:

| Path | Tracked | Why |
|------|---------|-----|
| `src/`, `scripts/`, `tests/` | yes | the code |
| `data/blackstar/` | yes | example profile, plugin name anonymized |
| `data/<other>/` | no | a profile captures one specific commercial plugin |
| `models/` | no | trained artifacts; reproduce by training |
| `training_data/` | no | rendered audio pairs (large `.wav`) |
| `scripts/local/` | no | profiling scripts that embed plugin paths and names |

So the quick-start commands below need a model you have trained yourself. Nothing here
ships a ready-to-use one.

## Example Model

The `blackstar` profile under `data/blackstar/` is a dual-drive tube saturation with 2D
conditioning (drive_a, drive_b). Its plugin name and path are anonymized.

## Quick Start

Train first — the quick-start scripts load from `models/`:

### Gradio Demo

```bash
python scripts/demo.py
# → http://localhost:7870
```

### Realtime Audio (sounddevice)

```bash
# Blackstar drive_a=70
python scripts/realtime.py --model blackstar --input 4 --output 8 --p1 70

# Keyboard: b=bypass, +/-=mix, p1=80, q=quit
```

### Python API

```python
from nnstomps.training.evaluate import load_model, process_audio

# Requires a model trained locally — models/ is not tracked.
model, config = load_model("models/blackstar/best_model.pt")
output = process_audio(model, input_audio, cond=[0.8, 0.0])  # drive_a=80
```

## Training Pipeline

### 1. Data Generation

```bash
# 214 test signals × N settings = input/output pairs
python scripts/local/generate_massive_data.py
```

The signal set (`core/test_signal_v2.py`):
- Sine waves: 11 frequencies × 8 levels = 88
- Sweeps / noise / IMD = 13
- AD impulse tones: 7 frequencies × 4 levels × 3 decays = 84
- Velocity sweeps, chords, dynamics, glides, pulses, triangle/sawtooth

### 2. Model Training

```bash
python scripts/train_blackstar.py            # single plugin
python scripts/train_all.py --epochs 100     # train + export + eval
python scripts/train_all.py --skip-trained
```

**GRU model**: `Input(1 + cond_dim) → GRU(hidden=40) → Dense(1) → Tanh`
- 5,441 parameters (21KB)
- RTNeural compatible (for the VST3 build)
- Loss: ESR + Multi-STFT + Pre-emphasis

### 3. CMA-ES Hyperparameter Optimization

```bash
python scripts/cmaes_optimize.py --plugin blackstar --generations 20 --popsize 6
```

CMA-ES loss weight optimization revealed that **pre-emphasis (coeff=0.99, w=1.30) is the key to harmonic reproduction**.

### 4. RTNeural Export

```bash
python scripts/train_all.py --export-only
# → models/{plugin}/{plugin}_rtneural.json
```

## Neural EQ

A separate engine for equalizers. An MLP maps EQ parameters to FIR coefficients, which
are convolved with the audio through overlap-add FFT convolution. Rather than learning a
waveform, it learns the frequency response.

```
parameters → MLP → FIR coefficients → overlap-add convolution → audio
```

Training fits the predicted response to a profiled magnitude and phase curve:

```bash
python scripts/local/train_eq_all.py
```

Running it:

```python
from nnstomps.core.neural_eq import NeuralEQEngine

engine = NeuralEQEngine("models/channel_eq/best_model.pt", block_size=256, crossfade_blocks=8)
engine.set_params({"lf_gain": 3.0, "lmf_gain": 0.0, "hmf_gain": -2.0, "hf_gain": 6.0})

output = engine.process_block(block)   # exactly block_size samples
```

Parameters are given in **natural units** (dB), keyed by name. The engine normalizes them
for the model using the `param_spec` stored in the checkpoint, so the encoding used at
training time and at run time cannot drift apart. A checkpoint without a spec is rejected
rather than guessed at.

`set_params` crossfades between the old and new filter over `crossfade_blocks` blocks;
`crossfade_blocks=0` switches instantly, which clicks on a large jump.

Profiled EQ plugins are trained under the same pseudonyms as their model directories
(`bronze_eq`, `midrange_eq`, `channel_eq`, `passive_eq`, `precision_eq`).

## Key Findings

### Pre-emphasis Loss
Standard ESR + STFT loss alone cannot learn harmonics (overtones). Applying a **pre-emphasis filter** (y[n] = x[n] - 0.99·x[n-1]) to the ESR forces high-frequency harmonic learning:

| Loss Config | H2 diff | H3 diff | H5 diff |
|-------------|---------|---------|---------|
| ESR 0.7, STFT 0.25 | -22 dB | -35 dB | -71 dB |
| ESR 0.48, STFT 1.11, **PreEmph 1.30** | **+0.6 dB** | **-1.4 dB** | **-2.8 dB** |

### Waveshaper v2
Discovered that the original waveshaper capture only covered 10% of the input range → fixed with `measure_waveshaper_v2()`: multi-amplitude levels + multi-cycle averaging + 256-point resampling.

## Project Structure

```
NNStomps/
├── src/nnstomps/
│   ├── core/
│   │   ├── neural_drive.py      # CLAP search + waveshaper interpolation engine
│   │   ├── hybrid_drive.py      # Waveshaper LUT + GRU hybrid
│   │   ├── neural_eq.py         # MLP → FIR realtime EQ engine
│   │   ├── plugin_analysis.py   # Plugin analysis (THD, waveshaper v2)
│   │   ├── test_signal.py       # Basic test signals (sine, sweep)
│   │   ├── test_signal_v2.py    # AD envelope signals (impulse tones, velocity)
│   │   ├── vst3_wrapper.py      # pedalboard VST3 wrapper
│   │   ├── audio_file.py        # Audio I/O
│   │   ├── analysis.py          # Frame metrics (RMS, spectral)
│   │   └── parameter.py         # Parameter dataclass
│   ├── training/
│   │   ├── model.py             # NNStompGRU, NNStompGRU2
│   │   ├── losses.py            # ESR, MultiSTFT, PreEmphasis, DC
│   │   ├── dataset.py           # AudioPairDataset (memory preload)
│   │   ├── train.py             # Training loop (TBPTT, AMP, curriculum)
│   │   ├── export.py            # PyTorch → RTNeural JSON
│   │   ├── evaluate.py          # A/B comparison, ESR calculation
│   │   ├── generate_pairs.py    # Input/output pair generation
│   │   ├── cmaes_sound_match.py # CMA-ES render-in-the-loop matching
│   │   ├── eq_model.py          # NNStompEQ — parameters → FIR
│   │   ├── eq_dataset.py        # EQ profile dataset + parameter encoding
│   │   ├── eq_losses.py         # Magnitude / phase / compactness losses
│   │   ├── train_eq.py          # EQ training loop
│   │   └── export_eq.py         # EQ → ONNX + web UI metadata
│   └── cli/app.py               # CLI (search, process, info)
├── scripts/                     # public — runnable from a clone
│   ├── demo.py                  # Gradio UI (localhost:7870)
│   ├── realtime.py              # Real-time audio processing
│   ├── train_blackstar.py       # Blackstar training script (example)
│   ├── train_all.py             # Full pipeline: train + eval + export
│   ├── cmaes_optimize.py        # CMA-ES hyperparameter optimization
│   └── local/                   # NOT tracked — profiling scripts with plugin paths
├── tests/                       # pytest
│   ├── test_neural_eq.py        # overlap-add equivalence, crossfade, state reset
│   ├── test_eq_dataset.py       # parameter encoding, split-then-augment
│   ├── test_eq_losses.py        # dB scale, phase wrapping, resampling
│   ├── test_eq_model.py         # shape and parameter-count contract
│   └── test_eq_pipeline.py      # export and training guards
├── models/                      # NOT tracked
├── data/                        # example profile only
└── training_data/               # NOT tracked
```

## Data Format

Each plugin directory (`data/{plugin}/`) contains:
- `*_clap.npy` — (N, 512) CLAP audio embeddings
- `*_clap_labels.json` — parameter labels
- `profile.json` — THD%, odd/even ratio, harmonic spectrum, waveshaper I/O
- `waveshaper_curves.npy` — (N, 64) v1 transfer functions
- `waveshaper_curves_v2.npy` — (N, 256) v2 transfer functions (multi-level)

EQ profiles add:
- `settings_dense.json` — parameter sets, one row per measurement
- `freq_response_dense.npy`, `phase_response_dense.npy` — responses, row-aligned with
  `settings_dense.json`

## Requirements

- Python 3.12+
- numpy, soundfile, pedalboard
- torch >= 2.0 (training)

```bash
pip install -e ".[training]"   # torch, torchaudio, onnx
pytest                          # requires the training extra
```

Optional: `laion-clap` (text search), `sounddevice` (realtime), `gradio` (demo).

## Next Steps

1. **Hybrid engine** — Integrate Waveshaper LUT + GRU residual
2. **JUCE + RTNeural VST3/AU build** — RTNeural JSON export already done
3. **Preset system** — CLAP-based auto-tagging + preset manager

## License

MIT