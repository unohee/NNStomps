# Created: 2026-03-21
# Purpose: 오디오 파일 I/O 추상화

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioStats:
    """오디오 파일 통계"""
    duration: float  # 초
    sample_rate: int
    channels: int
    frames: int
    peak: float
    rms: float
    format: str


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """오디오 파일 읽기. 반환: (audio shape (channels, samples), sample_rate)"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile: (samples, channels) → (channels, samples) for pedalboard
    audio = data.T
    logger.debug(f"읽기: {path.name} ({audio.shape[0]}ch, {sr}Hz, {audio.shape[1]} samples)")
    return audio, sr


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_24",
) -> None:
    """오디오 파일 쓰기. audio shape: (channels, samples)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # (channels, samples) → (samples, channels) for soundfile
    if audio.ndim == 1:
        data = audio
    else:
        data = audio.T

    sf.write(str(path), data, sample_rate, subtype=subtype)
    logger.debug(f"쓰기: {path.name} ({sample_rate}Hz, {subtype})")


def get_file_info(path: str | Path) -> dict:
    """파일 메타데이터만 빠르게 읽기 (오디오 로드 없이)"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    info = sf.info(str(path))
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }


def stream_process(
    input_path: str | Path,
    output_path: str | Path,
    process_fn,
    chunk_seconds: float = 10.0,
    subtype: str = "PCM_24",
) -> dict:
    """대용량 파일을 청크 단위로 스트리밍 처리

    Args:
        process_fn: (audio_chunk: ndarray, sr: int) → ndarray 처리 함수
        chunk_seconds: 청크 크기 (초)

    Returns: {"frames_processed", "duration", "chunks"}
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(input_path))
    chunk_frames = int(chunk_seconds * info.samplerate)
    total_frames = info.frames
    frames_done = 0
    chunks = 0

    with sf.SoundFile(str(input_path), 'r') as infile:
        with sf.SoundFile(
            str(output_path), 'w',
            samplerate=info.samplerate,
            channels=info.channels,
            subtype=subtype,
        ) as outfile:
            while frames_done < total_frames:
                n_read = min(chunk_frames, total_frames - frames_done)
                data = infile.read(n_read, dtype="float32", always_2d=True)
                if len(data) == 0:
                    break

                # (samples, channels) → (channels, samples) 변환
                chunk = data.T
                processed = process_fn(chunk, info.samplerate)

                # (channels, samples) → (samples, channels) 저장
                if processed.ndim == 1:
                    outfile.write(processed)
                else:
                    outfile.write(processed.T)

                frames_done += len(data)
                chunks += 1

    logger.debug(f"스트리밍 처리: {chunks} chunks, {frames_done} frames")
    return {
        "frames_processed": frames_done,
        "duration": frames_done / info.samplerate,
        "chunks": chunks,
        "sample_rate": info.samplerate,
    }


def get_audio_stats(audio: np.ndarray, sample_rate: int) -> AudioStats:
    """오디오 데이터 통계 계산"""
    if audio.ndim == 1:
        channels = 1
        samples = len(audio)
    else:
        channels, samples = audio.shape

    return AudioStats(
        duration=samples / sample_rate,
        sample_rate=sample_rate,
        channels=channels,
        frames=samples,
        peak=float(np.max(np.abs(audio))),
        rms=float(np.sqrt(np.mean(audio**2))),
        format="float32",
    )
