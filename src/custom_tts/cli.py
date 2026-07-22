"""Command-line interface for custom_tts.

Usage examples::

    custom-tts preprocess Nate
    custom-tts preprocess Nate --whisper-model small --chunk-length-ms 20000
    custom-tts synthesize --model voices/model.onnx --text "Hi there" --output hi.wav
    custom-tts preprocess Nate --config my_config.yaml

CLI flags always override values from the config file, which in turn overrides
the packaged defaults.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from custom_tts.config import load_config
from custom_tts.data.pipeline import run_pipeline
from custom_tts.inference.synthesize import synthesize
from custom_tts.utils.logging import get_logger

logger = get_logger("custom_tts.cli")


def _collect(mapping: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is ``None`` (i.e. not provided on the CLI)."""
    return {k: v for k, v in mapping.items() if v is not None}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="custom-tts",
        description="Build Piper TTS voice datasets and run synthesis.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to a YAML config file overriding the packaged defaults.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- preprocess ------------------------------------------------------
    pre = subparsers.add_parser(
        "preprocess",
        help="Run the dataset preprocessing pipeline for a voice folder.",
    )
    pre.add_argument("voice_folder", help="Path to the voice project folder.")
    pre.add_argument("--chunk-length-ms", type=int, help="Max chunk length (ms).")
    pre.add_argument("--whisper-model", help="Whisper model size for transcription.")
    pre.add_argument("--trim-top-db", type=int, help="Silence-trim threshold (dB).")
    pre.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_const",
        const=False,
        default=None,
        help="Disable peak normalization.",
    )
    pre.add_argument("--metadata-filename", help="Transcript CSV filename.")
    pre.add_argument("--sample-rate", type=int, help="Target sample rate (Hz).")

    # --- synthesize ------------------------------------------------------
    syn = subparsers.add_parser(
        "synthesize",
        help="Synthesize speech from text using a Piper voice.",
    )
    syn.add_argument("--model", help="Path to the Piper .onnx voice (or its stem).")
    syn.add_argument("--text", help="Text to synthesize.")
    syn.add_argument("--output", help="Output WAV path.")
    syn.add_argument("--volume", type=float, help="Output volume (0-1).")
    syn.add_argument("--length-scale", type=float, help="Speaking rate scale.")
    syn.add_argument("--noise-scale", type=float, help="Audio variation.")
    syn.add_argument("--noise-w-scale", type=float, help="Speaking variation.")
    syn.add_argument(
        "--normalize-audio",
        dest="normalize_audio",
        action="store_const",
        const=True,
        default=None,
        help="Normalize synthesized audio.",
    )

    return parser


def _preprocess_overrides(args: argparse.Namespace) -> dict[str, Any]:
    preprocess = _collect(
        {
            "chunk_length_ms": args.chunk_length_ms,
            "whisper_model": args.whisper_model,
            "trim_top_db": args.trim_top_db,
            "normalize": args.normalize,
            "metadata_filename": args.metadata_filename,
        }
    )
    overrides: dict[str, Any] = {}
    if preprocess:
        overrides["preprocess"] = preprocess
    if args.sample_rate is not None:
        overrides["sample_rate"] = args.sample_rate
    return overrides


def _synthesize_overrides(args: argparse.Namespace) -> dict[str, Any]:
    inference = _collect(
        {
            "model": args.model,
            "text": args.text,
            "output": args.output,
            "volume": args.volume,
            "length_scale": args.length_scale,
            "noise_scale": args.noise_scale,
            "noise_w_scale": args.noise_w_scale,
            "normalize_audio": args.normalize_audio,
        }
    )
    return {"inference": inference} if inference else {}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "preprocess":
            config = load_config(args.config, _preprocess_overrides(args))
            run_pipeline(args.voice_folder, config)
        elif args.command == "synthesize":
            config = load_config(args.config, _synthesize_overrides(args))
            synthesize(config.inference)
        else:  # pragma: no cover - argparse enforces valid commands
            parser.error(f"Unknown command: {args.command}")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
