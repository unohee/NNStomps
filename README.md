# NNStomps — Neural Drive

GRU 신경망 기반 디스토션/새추레이션 모델링. 오디오 플러그인의 비선형 특성을 프로파일링하고, 신경망으로 학습하여 실시간 오디오 처리.

## What it does

```
플러그인 프로파일링 (audioman doctor)
  → 214종 테스트 신호 × drive 세팅 = input/output 쌍 생성
  → 조건부 GRU 학습 (drive 값으로 음색 제어)
  → 실시간 오디오 처리 (5.8ms 레이턴시)
  → JUCE + RTNeural VST3/AU 빌드 (예정)
```

## Architecture

```
                  ┌─────────────────────────────┐
                  │  NNStomps Hybrid Engine      │
                  ├─────────────────────────────┤
Input Audio ──→   │ [Static]  Waveshaper LUT    │ ──→ Output Audio
                  │           (다중 레벨 보간)    │
Condition   ──→   │ [Dynamic] GRU Residual      │
(drive, tone)     │           (attack/release)   │
                  └─────────────────────────────┘
```

**두 가지 엔진:**
- **GRU 단독**: 조건부 GRU가 전체 비선형 특성을 학습 (현재 주력)
- **하이브리드**: Waveshaper LUT (정적) + GRU residual (동적) — 실험 중

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

# 키보드: b=bypass, +/-=mix, p1=80, q=quit
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
# 214종 테스트 신호 × N 세팅 = input/output 쌍 생성
python scripts/generate_massive_data.py
```

테스트 신호 세트:
- 사인파 11주파수 × 8레벨 = 88개
- 스윕/노이즈/IMD = 13개
- AD 임펄스 톤 7주파수 × 4레벨 × 3decay = 84개
- 벨로시티 스윕, 코드, 다이나믹, 글라이드, 펄스, 삼각/톱니파

### 2. Model Training

```bash
# 단일 플러그인
python scripts/train_blackstar.py

# 전체 플러그인 (학습 + export + eval)
python scripts/train_all.py --epochs 100

# 이미 학습된 모델 건너뛰기
python scripts/train_all.py --skip-trained
```

**GRU 모델**: `Input(1 + cond_dim) → GRU(hidden=40) → Dense(1) → Tanh`
- 5,441 파라미터 (21KB)
- RTNeural 호환 (VST3 빌드용)
- 손실: ESR + Multi-STFT + Pre-emphasis (하모닉 강제)

### 3. CMA-ES Hyperparameter Optimization

```bash
python scripts/cmaes_optimize.py --plugin blackstar --generations 20 --popsize 6
```

CMA-ES로 손실 가중치 최적화 → **pre-emphasis(coeff=0.99, w=1.30)가 하모닉 재현의 핵심**임을 발견.

### 4. RTNeural Export

```bash
python scripts/train_all.py --export-only
# → models/{plugin}/{plugin}_rtneural.json
```

## Key Findings

### Pre-emphasis Loss
기존 ESR + STFT 손실만으로는 하모닉(배음)을 학습하지 못함. **Pre-emphasis 필터**(y[n] = x[n] - 0.99·x[n-1])를 적용한 ESR이 고주파 하모닉 학습을 강제:

| Loss Config | H2 diff | H3 diff | H5 diff |
|-------------|---------|---------|---------|
| ESR 0.7, STFT 0.25 | -22 dB | -35 dB | -71 dB |
| ESR 0.48, STFT 1.11, **PreEmph 1.30** | **+0.6 dB** | **-1.4 dB** | **-2.8 dB** |

### Waveshaper v2
기존 waveshaper 캡처가 입력 범위의 10%만 커버하는 버그 발견 → `measure_waveshaper_v2()`로 다중 진폭 레벨 + 복수 주기 평균 + 256포인트 리샘플링으로 해결.

## Project Structure

```
NNStomps/
├── src/nnstomps/
│   ├── core/
│   │   ├── neural_drive.py      # CLAP 검색 + waveshaper 보간 엔진
│   │   ├── hybrid_drive.py      # Waveshaper LUT + GRU 하이브리드
│   │   ├── plugin_analysis.py   # 플러그인 분석 (THD, waveshaper v2)
│   │   ├── test_signal.py       # 기본 테스트 신호 (사인, 스윕)
│   │   ├── test_signal_v2.py    # AD 엔벨로프 기반 신호 (임펄스 톤, 벨로시티)
│   │   ├── vst3_wrapper.py      # pedalboard VST3 래퍼
│   │   ├── audio_file.py        # 오디오 I/O
│   │   ├── analysis.py          # 프레임 메트릭 (RMS, spectral)
│   │   └── parameter.py         # 파라미터 dataclass
│   ├── training/
│   │   ├── model.py             # NNStompGRU, NNStompGRU2
│   │   ├── losses.py            # ESR, MultiSTFT, PreEmphasis, DC
│   │   ├── dataset.py           # AudioPairDataset (메모리 프리로드)
│   │   ├── train.py             # 학습 루프 (TBPTT, AMP, curriculum)
│   │   ├── export.py            # PyTorch → RTNeural JSON
│   │   ├── evaluate.py          # A/B 비교, ESR 계산
│   │   ├── generate_pairs.py    # input/output 쌍 생성
│   │   ├── cmaes_sound_match.py # CMA-ES render-in-the-loop 매칭
│   │   └── presets.py           # CLAP 기반 프리셋 생성 (예정)
│   └── cli/app.py               # CLI (search, process, info)
├── scripts/
│   ├── demo.py                  # Gradio UI (localhost:7870)
│   ├── realtime.py              # 실시간 오디오 처리
│   ├── train_blackstar.py       # Blackstar 학습 스크립트 (예제)
│   ├── train_all.py             # 전체 플러그인 학습 + eval + export
│   └── cmaes_optimize.py        # CMA-ES 하이퍼파라미터 최적화
├── models/                      # 학습된 모델 (.pt, .json)
├── data/                        # 플러그인 프로파일 데이터
├── training_data/               # input/output 오디오 쌍
└── audio_demos/                 # 렌더링된 비교 오디오
```

## Data Format

Each plugin directory (`data/{plugin}/`) contains:
- `*_clap.npy` — (N, 512) CLAP audio embeddings
- `*_clap_labels.json` — parameter labels
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

1. **하이브리드 엔진 완성** — Waveshaper LUT + GRU residual 통합
2. **JUCE + RTNeural VST3/AU 빌드** — RTNeural JSON 이미 내보내기 완료
3. **프리셋 시스템** — CLAP 기반 자동 태깅 + 프리셋 매니저

## License

MIT
