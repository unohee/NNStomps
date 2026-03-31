# Changelog

## [Unreleased] — 2026-03-26

### Added — GRU 학습 파이프라인
- `src/nnstomps/training/` 모듈 전체 신규
  - `model.py` — NNStompGRU (5,441 params), NNStompGRU2 (9,141 params)
  - `losses.py` — ESR, MultiScaleSTFT, PreEmphasisLoss, DCLoss, NNStompLoss
  - `dataset.py` — AudioPairDataset (메모리 프리로드 지원)
  - `train.py` — 학습 루프 (TBPTT, curriculum, AMP, CosineAnnealing)
  - `export.py` — PyTorch → RTNeural JSON 변환 + 검증
  - `evaluate.py` — A/B 비교, ESR 계산, 오디오 렌더링
  - `generate_pairs.py` — input/output 쌍 생성 파이프라인
  - `cmaes_sound_match.py` — CMA-ES render-in-the-loop 사운드 매칭

### Added — 테스트 신호 v2
- `src/nnstomps/core/test_signal_v2.py` — AD 엔벨로프 기반 테스트 신호
  - `generate_impulse_tone()` — 볼륨별 비선형 반응 캡처
  - `generate_multi_tone_burst()` — 주파수별 새추레이션 측정
  - `generate_velocity_sweep()` — pp→ff 레벨 의존적 특성
  - `generate_chord_burst()` — IMD 상호변조 캡처
  - `generate_full_test_suite()` — 214종 전체 세트

### Added — 하이브리드 엔진
- `src/nnstomps/core/hybrid_drive.py` — Waveshaper LUT + GRU Residual
  - `WaveshaperLUT` — 다중 레벨 전달함수, 입력 진폭별 보간
  - `HybridDrive` — 정적 LUT + 동적 GRU 결합

### Added — 스크립트
- `scripts/demo.py` — Gradio UI (localhost:7870)
- `scripts/realtime.py` — sounddevice 실시간 오디오 처리
- `scripts/train_blackstar.py` — Blackstar 학습 (예제)
- `scripts/train_all.py` — 전체 플러그인 학습 + export + eval
- `scripts/cmaes_optimize.py` — CMA-ES 하이퍼파라미터 최적화
- `scripts/generate_massive_data.py` — 대규모 학습 데이터 생성 (214종 × 12세팅)
- `scripts/reprofile_waveshapers.py` — waveshaper v2 재프로파일링

### Added — 학습된 모델
- `models/blackstar/` — Blackstar 예제 모델 (dual-drive tube saturation)

### Changed
- `plugin_analysis.py` — import 오류 수정 (`vst3` → `vst3_wrapper`)
- `plugin_analysis.py` — `WaveshaperV2Result` + `measure_waveshaper_v2()` 추가
- `pyproject.toml` — `[project.optional-dependencies].training` 추가

### Fixed
- waveshaper 캡처 범위 버그: 입력의 10%만 캡처 → v2로 99.9% 커버
- audioman doctor `--mode waveshaper` v2 기본 적용 (audioman 코드베이스 수정)

### Discovered
- **Pre-emphasis Loss**: ESR+STFT만으로는 하모닉 학습 불가. `PreEmphasisLoss(coeff=0.99)`가 핵심
- **파라미터 이름 검증 중요**: 플러그인 실제 파라미터명과 코드 내 이름 불일치 주의
- **CMA-ES 최적 가중치**: w_esr=0.48, w_stft=1.11, w_preemph=1.30

## [0.1.0] — 2026-03-25

### Added
- 초기 NNStomps — CLAP 검색 + waveshaper 보간 엔진
- 플러그인 프로파일링 데이터 (CLAP 임베딩, waveshaper 곡선)
- `NeuralDrive` 클래스 (from_data_dir, search, process, process_by_text)
- CLI (search, process, info)
