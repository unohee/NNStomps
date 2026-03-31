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

## Example Model

An example model (`blackstar`) is included — a dual-drive tube saturation profile with 2D conditioning (drive_a, drive_b).

## Quick Start

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

model, config = load_model("models/blackstar/best_model.pt")
output = process_audio(model, input_audio, cond=[0.8, 0.0])  # drive_a=80
```

## Training Pipeline

### 1. Data Generation

```bash
# 214 test signals × N settings = input/output pairs
python scripts/generate_massive_data.py
```

Test signal set:
- Sine waves: 11 frequencies × 8 levels = 88
- Sweeps / noise / IMD = 13
- AD impulse tones: 7 frequencies × 4 levels × 3 decays = 84
- Velocity sweeps, chords, dynamics, glides, pulses, triangle/sawtooth

### 2. Model Training

```bash
# Single plugin
python scripts/train_blackstar.py

# All plugins (train + export + eval)
python scripts/train_all.py --epochs 100

# Skip already trained models
python scripts/train_all.py --skip-trained
```

**GRU model**: `Input(1 + cond_dim) → GRU(hidden=40) → Dense(1) → Tanh`
- 5,441 parameters (21KB)
- RTNeural compatible (for VST3 build)
- Loss: ESR + Multi-STFT + Pre-emphasis (forces harmonic learning)

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
│   │   └── presets.py           # CLAP-based preset generation (planned)
│   └── cli/app.py               # CLI (search, process, info)
├── scripts/
│   ├── demo.py                  # Gradio UI (localhost:7870)
│   ├── realtime.py              # Real-time audio processing
│   ├── train_blackstar.py       # Blackstar training script (example)
│   ├── train_all.py             # Full pipeline: train + eval + export
│   └── cmaes_optimize.py        # CMA-ES hyperparameter optimization
├── models/                      # Trained models (.pt, .json)
├── data/                        # Plugin profile data
├── training_data/               # Input/output audio pairs
└── audio_demos/                 # Rendered comparison audio
```

## Data Format

Each plugin directory (`data/{plugin}/`) contains:
- `*_clap.npy` — (N, 512) CLAP audio embeddings
- `*_clap_labels.json` — Parameter labels
- `profile.json` — THD%, odd/even ratio, harmonic spectrum, waveshaper I/O
- `waveshaper_curves.npy` — (N, 64) v1 transfer functions
- `waveshaper_curves_v2.npy` — (N, 256) v2 transfer functions (multi-level)

## Requirements

- Python 3.12+
- numpy, soundfile, pedalboard
- torch >= 2.0 (training)
- Optional: laion-clap (text search), sounddevice (realtime), gradio (demo)

```bash
pip install -e ".[training]"  # torch, torchaudio, auraloss
```

## Next Steps

1. **Hybrid engine** — Integrate Waveshaper LUT + GRU residual
2. **JUCE + RTNeural VST3/AU build** — RTNeural JSON export already done
3. **Preset system** — CLAP-based auto-tagging + preset manager

## License

MIT
