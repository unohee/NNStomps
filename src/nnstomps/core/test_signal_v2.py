# Created: 2026-03-26
# Purpose: IR 스타일 테스트 신호 생성 — impulse + tone + envelope + pitch
#
# 기존 test_signal.py의 정상 상태 사인파 대신,
# ADSR 엔벨로프가 있는 신호로 한 번에:
#   - 볼륨별 비선형 반응 (엔벨로프가 전 레벨 스윕)
#   - attack/release 동적 특성
#   - 주파수별 새추레이션 차이
#   - 고조파 상호작용
# 을 캡처합니다.

import numpy as np


def ad_envelope(
    n_samples: int,
    attack_samples: int,
    decay_samples: int,
    peak: float = 1.0,
    sustain: float = 0.0,
) -> np.ndarray:
    """Attack-Decay 엔벨로프 (선형)

    0 → peak (attack) → sustain (decay) → sustain (hold)
    """
    env = np.zeros(n_samples, dtype=np.float32)

    # Attack: 0 → peak
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, peak, attack_samples, dtype=np.float32)

    # Decay: peak → sustain
    decay_end = attack_samples + decay_samples
    if decay_samples > 0 and decay_end <= n_samples:
        env[attack_samples:decay_end] = np.linspace(peak, sustain, decay_samples, dtype=np.float32)

    # Sustain (나머지)
    if decay_end < n_samples:
        env[decay_end:] = sustain

    return env


def generate_impulse_tone(
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    duration_sec: float = 1.0,
    peak_db: float = 0.0,
    attack_ms: float = 5.0,
    decay_ms: float = 200.0,
    sustain_db: float = -60.0,
) -> np.ndarray:
    """임펄스 톤: AD 엔벨로프 × 사인파

    짧은 attack으로 피크까지 올라가고, decay로 내려오면서
    볼륨 0dB → sustain_db까지 전 레벨의 비선형 반응을 캡처.

    Returns: (2, samples) float32 스테레오
    """
    n = int(sample_rate * duration_sec)
    peak_amp = 10 ** (peak_db / 20.0)
    sustain_amp = 10 ** (sustain_db / 20.0)

    attack_samples = int(attack_ms / 1000.0 * sample_rate)
    decay_samples = int(decay_ms / 1000.0 * sample_rate)

    env = ad_envelope(n, attack_samples, decay_samples, peak_amp, sustain_amp)

    t = np.arange(n, dtype=np.float32) / sample_rate
    sine = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    mono = env * sine
    return np.stack([mono, mono])


def generate_multi_tone_burst(
    frequencies: list[float] | None = None,
    sample_rate: int = 44100,
    note_duration_sec: float = 0.5,
    gap_sec: float = 0.1,
    peak_db: float = 0.0,
    attack_ms: float = 5.0,
    decay_ms: float = 150.0,
    sustain_db: float = -60.0,
) -> np.ndarray:
    """다중 톤 버스트: 여러 주파수를 순차적으로 재생

    각 노트는 AD 엔벨로프가 적용된 임펄스 톤.
    주파수별 새추레이션 차이 + 볼륨별 반응을 한 번에 캡처.

    기본 주파수: 100, 200, 500, 1k, 2k, 5k, 10kHz

    Returns: (2, samples) float32 스테레오
    """
    if frequencies is None:
        frequencies = [100, 200, 500, 1000, 2000, 5000, 10000]

    segments = []
    gap_samples = int(gap_sec * sample_rate)

    for freq in frequencies:
        tone = generate_impulse_tone(
            freq, sample_rate, note_duration_sec,
            peak_db, attack_ms, decay_ms, sustain_db,
        )
        segments.append(tone)

        # 노트 사이 silence
        if gap_samples > 0:
            silence = np.zeros((2, gap_samples), dtype=np.float32)
            segments.append(silence)

    return np.concatenate(segments, axis=1)


def generate_velocity_sweep(
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    note_duration_sec: float = 0.3,
    gap_sec: float = 0.1,
    velocities_db: list[float] | None = None,
    attack_ms: float = 5.0,
    decay_ms: float = 100.0,
) -> np.ndarray:
    """벨로시티 스윕: 같은 음을 다양한 레벨로 순차 재생

    pp → ff까지 레벨을 올리면서 비선형 반응의 레벨 의존성 캡처.

    Returns: (2, samples) float32 스테레오
    """
    if velocities_db is None:
        velocities_db = [-36, -30, -24, -18, -12, -9, -6, -3, -1, 0]

    segments = []
    gap_samples = int(gap_sec * sample_rate)

    for vel_db in velocities_db:
        tone = generate_impulse_tone(
            frequency, sample_rate, note_duration_sec,
            vel_db, attack_ms, decay_ms, sustain_db=-60.0,
        )
        segments.append(tone)
        if gap_samples > 0:
            segments.append(np.zeros((2, gap_samples), dtype=np.float32))

    return np.concatenate(segments, axis=1)


def generate_chord_burst(
    sample_rate: int = 44100,
    duration_sec: float = 1.0,
    peak_db: float = -3.0,
    attack_ms: float = 10.0,
    decay_ms: float = 300.0,
) -> np.ndarray:
    """코드 버스트: 여러 음이 동시에 울리는 경우의 상호변조 캡처

    C major triad (261.6, 329.6, 392.0 Hz) — IMD 특성 측정용.

    Returns: (2, samples) float32 스테레오
    """
    freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
    n = int(sample_rate * duration_sec)
    peak_amp = 10 ** (peak_db / 20.0)

    attack_samples = int(attack_ms / 1000.0 * sample_rate)
    decay_samples = int(decay_ms / 1000.0 * sample_rate)
    env = ad_envelope(n, attack_samples, decay_samples, peak_amp, 0.0)

    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = np.zeros(n, dtype=np.float32)
    for f in freqs:
        mono += np.sin(2 * np.pi * f * t).astype(np.float32)
    mono = mono / len(freqs)  # 정규화

    mono = env * mono
    return np.stack([mono, mono])


def generate_full_test_suite(
    sample_rate: int = 44100,
) -> list[tuple[str, np.ndarray]]:
    """전체 테스트 신호 세트 — IR 수준의 이펙터 프로파일링용

    Returns: [(이름, audio), ...]
    """
    signals = []

    # 1. 다중 톤 버스트 × 3 레벨
    for peak_db in [-6, -3, 0]:
        name = f"multi_tone_{peak_db}db"
        sig = generate_multi_tone_burst(
            peak_db=peak_db, attack_ms=5, decay_ms=200,
            sample_rate=sample_rate,
        )
        signals.append((name, sig))

    # 2. 벨로시티 스윕 × 주요 주파수
    for freq in [200, 1000, 5000]:
        name = f"velocity_{int(freq)}hz"
        sig = generate_velocity_sweep(
            frequency=freq, sample_rate=sample_rate,
            attack_ms=5, decay_ms=150,
        )
        signals.append((name, sig))

    # 3. 코드 버스트 (IMD)
    for peak_db in [-6, -3, 0]:
        name = f"chord_{peak_db}db"
        sig = generate_chord_burst(
            peak_db=peak_db, attack_ms=10, decay_ms=400,
            sample_rate=sample_rate,
        )
        signals.append((name, sig))

    # 4. 긴 sustain 톤 (정상 상태 하모닉 측정용)
    for freq in [500, 1000, 2000]:
        for peak_db in [-6, 0]:
            name = f"sustain_{int(freq)}hz_{peak_db}db"
            sig = generate_impulse_tone(
                freq, sample_rate, duration_sec=3.0,
                peak_db=peak_db, attack_ms=10, decay_ms=50,
                sustain_db=peak_db,  # sustain = peak (정상 상태)
            )
            signals.append((name, sig))

    # 5. 극짧은 트랜지언트 (클릭/어택 반응)
    for peak_db in [-6, 0]:
        name = f"transient_{peak_db}db"
        sig = generate_impulse_tone(
            1000, sample_rate, duration_sec=0.5,
            peak_db=peak_db, attack_ms=0.5, decay_ms=20,
            sustain_db=-60,
        )
        signals.append((name, sig))

    return signals
