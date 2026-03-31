# NNStomps CLI

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nnstomps",
        description="Neural Drive — AI saturation from plugin profiles",
    )
    sub = parser.add_subparsers(dest="command")

    # search
    s = sub.add_parser("search", help="텍스트로 새추레이션 스타일 검색")
    s.add_argument("query", help="검색 텍스트 (예: 'warm tape saturation')")
    s.add_argument("--data", "-d", default="data/", help="프로파일 데이터 경로")
    s.add_argument("--top", "-k", type=int, default=5)
    s.set_defaults(func=_search)

    # process
    p = sub.add_parser("process", help="오디오에 새추레이션 적용")
    p.add_argument("input", help="입력 오디오")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--text", "-t", help="텍스트 설명 (CLAP 검색)")
    p.add_argument("--style", "-s", type=int, default=0, help="스타일 인덱스")
    p.add_argument("--mix", type=float, default=1.0)
    p.add_argument("--data", "-d", default="data/")
    p.set_defaults(func=_process)

    # info
    i = sub.add_parser("info", help="로드된 프로파일 요약")
    i.add_argument("--data", "-d", default="data/")
    i.set_defaults(func=_info)

    return parser


def _search(args):
    from nnstomps.core.neural_drive import NeuralDrive
    drive = NeuralDrive.from_data_dir(args.data)
    results = drive.search(args.query, top_k=args.top)
    for r in results:
        print(f"  {r['similarity']:.3f}  {r['label']}")


def _process(args):
    from nnstomps.core.neural_drive import NeuralDrive
    import soundfile as sf
    import numpy as np

    drive = NeuralDrive.from_data_dir(args.data)
    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    audio = audio.T  # (ch, samples)

    if args.text:
        output, info = drive.process_by_text(audio, sr, args.text, mix=args.mix)
        print(f"Matched: {info['label']} (sim={info['similarity']:.3f})")
    else:
        output = drive.process(audio, sr, style_idx=args.style, mix=args.mix)

    sf.write(args.output, output.T, sr, subtype="PCM_24")
    print(f"Saved: {args.output}")


def _info(args):
    from nnstomps.core.neural_drive import NeuralDrive
    import json
    drive = NeuralDrive.from_data_dir(args.data)
    print(json.dumps(drive.summary(), indent=2))


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)
