"""Command-line interface for F5-TTS preprocessing and voice cloning."""

from __future__ import annotations

import argparse
import sys

from custom_tts.inference import synthesize
from custom_tts.preprocess import preprocess


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="custom-tts", description="Prepare and synthesize F5-TTS voices.")
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("preprocess", help="Prepare raw media as an F5-TTS dataset.")
    prep.add_argument("voice_folder", help="Voice folder containing a raws/ directory.")
    prep.add_argument("--whisper-model", default="base", help="Faster-Whisper model name or local model directory.")
    prep.add_argument("--asr-device", choices=("auto", "cpu", "cuda"), default="auto")
    prep.add_argument("--chunk-seconds", type=float, default=12.0, help="Maximum clip length (default: 12).")

    infer = commands.add_parser("synthesize", help="Clone a reference voice and synthesize speech.")
    infer.add_argument("-r", "--ref-audio", required=True, help="Reference audio file (preferably 3-12 seconds).")
    infer.add_argument("--ref-text", default="", help="Reference transcript; omit to transcribe automatically.")
    infer.add_argument("-t", "--text", required=True, help="Text to synthesize.")
    infer.add_argument("-o", "--output", default="output.wav", help="Output WAV path (default: output.wav).")
    infer.add_argument("--model", default="F5TTS_v1_Base", help="F5-TTS model config name.")
    infer.add_argument("--checkpoint", help="Optional local F5-TTS checkpoint.")
    infer.add_argument("--vocab", help="Optional local vocabulary file.")
    infer.add_argument("--vocoder-path", help="Local Vocos model directory for offline inference.")
    infer.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps", "xpu"),
        default="auto",
        help="Compute device (default: auto).",
    )
    infer.add_argument("--seed", type=int, help="Random seed for reproducible output.")
    infer.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0).")
    infer.add_argument("--nfe-steps", type=int, default=32, help="Denoising steps; lower is faster (default: 32).")
    infer.add_argument("--remove-silence", action="store_true", help="Trim long silence from the output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "preprocess":
            output = preprocess(
                args.voice_folder,
                whisper_model=args.whisper_model,
                asr_device=args.asr_device,
                chunk_seconds=args.chunk_seconds,
            )
        else:
            output = synthesize(
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                text=args.text,
                output=args.output,
                model=args.model,
                checkpoint=args.checkpoint,
                vocab=args.vocab,
                vocoder_path=args.vocoder_path,
                device=args.device,
                seed=args.seed,
                speed=args.speed,
                nfe_steps=args.nfe_steps,
                remove_silence=args.remove_silence,
            )
    except (FileNotFoundError, ValueError) as exc:
        parser.print_usage(sys.stderr)
        print(f"custom-tts: error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
