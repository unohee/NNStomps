# NNStomps — Neural Drive Plugin

## 프로젝트 개요
오디오 새추레이션/디스토션 플러그인의 비선형 특성을 GRU 신경망으로 학습하여
실시간 오디오 처리가 가능한 Neural Drive 엔진.

**최종 목표**: JUCE + RTNeural로 VST3/AU 플러그인 빌드

## 아키텍처
```
audioman doctor → 플러그인 프로파일링 (THD, waveshaper v2, CLAP)
       ↓
data/{plugin}/ → profile.json + clap.npy + waveshaper_curves_v2.npy
       ↓
generate_massive_data.py → 214종 테스트 신호 × N세팅 = input/output 쌍
       ↓
train.py → 조건부 GRU 학습 (ESR + STFT + PreEmphasis 손실)
       ↓
export.py → RTNeural JSON (JUCE 빌드용)
       ↓
realtime.py / demo.py → 실시간 오디오 처리 / Gradio UI
```

## 핵심 엔진

### GRU 디스토션 모델
- `Input(1 + cond_dim) → GRU(hidden=40) → Dense(1) → Tanh`
- 5,441 파라미터 (21KB), RTNeural 호환
- 조건 벡터로 drive/gain 실시간 제어
- sample-by-sample 처리 (5.8ms 레이턴시 @ buffer=256)

### 하이브리드 엔진 (실험 중)
- Waveshaper LUT: 다중 레벨 전달함수 (정적 비선형)
- GRU Residual: attack/release 동적 보정
- `hybrid_drive.py`

## 학습 데이터 생성
`test_signal_v2.py`의 테스트 신호 세트:
- AD 임펄스 톤: 엔벨로프 × 사인파 → 볼륨별 비선형 반응
- 벨로시티 스윕: pp→ff 레벨 의존적 특성
- 다중 톤 버스트: 주파수별 새추레이션 차이
- 코드: 상호변조 왜곡 (IMD)
- 정상 상태 사인파/스윕/노이즈: 하모닉 정밀 측정

## 손실 함수 (핵심 발견)
**Pre-emphasis Loss가 하모닉 재현의 핵심:**
```
L = 0.48×ESR + 1.11×MultiSTFT + 0.05×DC + 1.30×PreEmphasis(coeff=0.99)
```
- ESR만으로는 기본 주파수에 지배되어 하모닉 학습 불가
- Pre-emphasis(미분 필터)가 고주파 하모닉 학습을 강제
- CMA-ES로 최적 가중치 탐색하여 발견

## 예제 모델
- `blackstar`: 2축 drive (drive_a, drive_b) 진공관 새추레이션 프로파일

## 공개 정책
- 상용 플러그인 이름은 pseudonym으로 대체 (예: blackstar)
- 커밋 히스토리에 상용 플러그인 이름 노출 금지
- data/, models/ 디렉토리는 예제(blackstar)만 트래킹

## 데이터 포맷
- `*_clap.npy`: (N, 512) CLAP 오디오 임베딩
- `*_clap_labels.json`: 파라미터 라벨 + 값
- `profile.json`: THD%, odd/even ratio, 하모닉 분포, waveshaper I/O
- `waveshaper_curves.npy`: (N, 64) v1 전달함수 (범위 제한 버그)
- `waveshaper_curves_v2.npy`: (N, 256) v2 전달함수 (다중 레벨, 99.9% 커버리지)
- `lut_*.npz`: 다중 레벨 Waveshaper LUT

## 학습 설정 (검증됨)
```python
TrainConfig(
    hidden_size=40, batch_size=128,
    lr=3.9e-3,
    w_esr=0.48, w_stft=1.11, w_dc=0.05,
    w_preemph=1.30, preemph_coeff=0.99,
    seq_len_final=8192, epochs=100,
    device='mps',  # Apple Silicon GPU
)
```

## 개발 규칙
- 프로파일링은 audioman doctor로 수행 (`--mode waveshaper` = v2 자동 사용)
- waveshaper v2: -1~+1 범위, 256포인트 균등 샘플링, 다중 진폭 레벨
- 학습 데이터는 메모리 프리로드 (dataset.py `preload=True`)
- AMP (float16) + batch=128로 MPS에서 에폭당 ~20초
- CLAP 모델: laion-clap (선택적 의존성)

## 다음 단계
1. 하이브리드 엔진 완성 (LUT + GRU residual)
2. JUCE + RTNeural VST3/AU 빌드
3. 프리셋 시스템 (CLAP 기반 자동 태깅)
