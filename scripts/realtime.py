#!/usr/bin/env python3
# Created: 2026-03-26
# Purpose: NNStomps 실시간 오디오 처리 — 오디오 인터페이스 입력 → GRU → 출력
# Usage: python scripts/realtime.py [--model blackstar] [--drive 60]
#        python scripts/realtime.py --list-devices
#        python scripts/realtime.py --input 4 --output 4 --model blackstar --p1 50

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnstomps.training.model import NNStompGRU

# 모델 설정
PRESETS = {
    "blackstar": {
        "path": "models/blackstar/best_model.pt",
        "cond_dim": 2,
        "p1_name": "Drive A", "p1_range": (0, 100),
        "p2_name": "Drive B",  "p2_range": (0, 100),
    },
    # 다른 모델은 프로파일 추가 후 여기에 등록
}


class RealtimeEngine:
    def __init__(self, model: NNStompGRU, cond_dim: int):
        self.model = model
        self.model.eval()
        self.cond_dim = cond_dim
        self.hidden = None
        self.cond = [0.0] * cond_dim
        self.mix = 1.0
        self.input_gain = 1.0
        self.bypass = False
        self._lock = threading.Lock()

    def set_cond(self, idx: int, value: float):
        with self._lock:
            if idx < len(self.cond):
                self.cond[idx] = np.clip(value, 0.0, 1.0)

    def set_mix(self, mix: float):
        self.mix = np.clip(mix, 0.0, 1.0)

    def set_input_gain_db(self, db: float):
        self.input_gain = 10 ** (db / 20.0)

    def toggle_bypass(self):
        self.bypass = not self.bypass
        return self.bypass

    def reset(self):
        self.hidden = None

    def process_block(self, indata: np.ndarray) -> np.ndarray:
        """오디오 블록 처리 (콜백에서 호출)

        indata: (frames, channels) float32
        returns: (frames,) float32 모노
        """
        # 모노 변환 + 입력 게인
        if indata.ndim == 2:
            mono = indata[:, 0] * self.input_gain
        else:
            mono = indata * self.input_gain

        if self.bypass:
            return mono

        with self._lock:
            cond = self.cond.copy()

        with torch.no_grad():
            x = torch.from_numpy(mono).unsqueeze(0).unsqueeze(-1)
            c = torch.tensor([cond], dtype=torch.float32)
            pred, self.hidden = self.model(x, c, self.hidden)
            wet = pred[0, :, 0].numpy()

        return mono * (1 - self.mix) + wet * self.mix


def main():
    parser = argparse.ArgumentParser(
        description="NNStomps 실시간 오디오 — 오디오 인터페이스를 통한 실시간 디스토션"
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--model", "-m", default="blackstar", choices=list(PRESETS.keys()))
    parser.add_argument("--input", "-i", type=int, default=None, help="입력 디바이스 ID")
    parser.add_argument("--output", "-o", type=int, default=None, help="출력 디바이스 ID")
    parser.add_argument("--buffer", "-b", type=int, default=256, help="버퍼 크기")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--p1", type=float, default=None, help="파라미터 1 값")
    parser.add_argument("--p2", type=float, default=None, help="파라미터 2 값")
    parser.add_argument("--mix", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=0.0, help="입력 게인 (dB)")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    preset = PRESETS[args.model]
    model_path = preset["path"]

    if not Path(model_path).exists():
        print(f"모델 없음: {model_path}")
        sys.exit(1)

    # 모델 로드
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model = NNStompGRU(ckpt["config"]["cond_dim"], ckpt["config"]["hidden_size"])
    model.load_state_dict(ckpt["model_state"])

    engine = RealtimeEngine(model, preset["cond_dim"])
    engine.set_mix(args.mix)
    engine.set_input_gain_db(args.gain)

    # 조건 벡터 설정
    p1_min, p1_max = preset["p1_range"]
    if args.p1 is not None:
        engine.set_cond(0, (args.p1 - p1_min) / (p1_max - p1_min))
    else:
        engine.set_cond(0, 0.5)

    if preset["cond_dim"] >= 2 and args.p2 is not None:
        p2_min, p2_max = preset["p2_range"]
        engine.set_cond(1, (args.p2 - p2_min) / (p2_max - p2_min))

    # 오디오 콜백
    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"[!] {status}")
        try:
            processed = engine.process_block(indata)
            # 모노 → 스테레오 출력
            outdata[:, 0] = processed
            if outdata.shape[1] > 1:
                outdata[:, 1] = processed
        except Exception as e:
            outdata[:] = indata
            print(f"[E] {e}")

    # 스트림 시작
    in_dev = args.input
    out_dev = args.output

    print(f"{'='*50}")
    print(f"  NNStomps Realtime — {args.model}")
    print(f"{'='*50}")
    print(f"  모델: {model_path}")
    print(f"  {preset['p1_name']}: {args.p1 or 'default'}")
    if preset["cond_dim"] >= 2:
        print(f"  {preset['p2_name']}: {args.p2 or 'default'}")
    print(f"  Mix: {args.mix}, Gain: {args.gain}dB")
    print(f"  Buffer: {args.buffer}, SR: {args.sr}")
    print(f"  Input: {in_dev or 'default'}, Output: {out_dev or 'default'}")
    print(f"{'='*50}")
    print()
    print("  키보드 명령:")
    print("    b = bypass 토글")
    print("    r = hidden state 리셋")
    print("    +/- = mix 조절 (0.1 단위)")
    print("    q = 종료")
    print()

    try:
        with sd.Stream(
            samplerate=args.sr,
            blocksize=args.buffer,
            device=(in_dev, out_dev),
            channels=1,
            dtype="float32",
            callback=callback,
            latency="low",
        ):
            print("  실시간 처리 중... (q로 종료)")
            while True:
                cmd = input().strip().lower()
                if cmd == "q":
                    break
                elif cmd == "b":
                    bp = engine.toggle_bypass()
                    print(f"  Bypass: {'ON' if bp else 'OFF'}")
                elif cmd == "r":
                    engine.reset()
                    print("  Hidden state 리셋")
                elif cmd == "+":
                    engine.set_mix(min(1.0, engine.mix + 0.1))
                    print(f"  Mix: {engine.mix:.1f}")
                elif cmd == "-":
                    engine.set_mix(max(0.0, engine.mix - 0.1))
                    print(f"  Mix: {engine.mix:.1f}")
                elif cmd.startswith("p1="):
                    try:
                        val = float(cmd.split("=")[1])
                        engine.set_cond(0, (val - p1_min) / (p1_max - p1_min))
                        print(f"  {preset['p1_name']}: {val}")
                    except ValueError:
                        pass
                elif cmd.startswith("p2=") and preset["cond_dim"] >= 2:
                    try:
                        val = float(cmd.split("=")[1])
                        p2_min, p2_max = preset["p2_range"]
                        engine.set_cond(1, (val - p2_min) / (p2_max - p2_min))
                        print(f"  {preset['p2_name']}: {val}")
                    except ValueError:
                        pass

    except KeyboardInterrupt:
        pass

    print("\n  종료.")


if __name__ == "__main__":
    main()
