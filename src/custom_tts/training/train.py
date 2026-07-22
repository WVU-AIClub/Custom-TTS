"""Run Piper training (``python -m piper.train fit``) for a voice folder."""

from __future__ import annotations

import os
import subprocess
import sys

from custom_tts.config import Config
from custom_tts.training.export import export_onnx, resolve_voice_name
from custom_tts.utils.logging import get_logger

logger = get_logger(__name__)


def build_train_command(voice_folder: str, config: Config) -> list[str]:
    """Build the ``piper.train fit`` command for ``voice_folder``.

    Paths are derived from the voice folder and the preprocess/train config so
    training lines up with the output of ``custom-tts preprocess``.
    """
    pre = config.preprocess
    train = config.train

    voice_name = resolve_voice_name(voice_folder, config)
    csv_path = os.path.join(voice_folder, pre.metadata_filename)
    audio_dir = os.path.join(voice_folder, pre.dirs.outputs)
    cache_dir = os.path.join(voice_folder, train.cache_dirname)
    config_filename = train.config_filename or f"{voice_name}.onnx.json"
    config_path = os.path.join(voice_folder, config_filename)

    cmd = [
        sys.executable,
        "-m",
        "piper.train",
        "fit",
        "--data.voice_name",
        voice_name,
        "--data.csv_path",
        csv_path,
        "--data.audio_dir",
        audio_dir,
        "--model.sample_rate",
        str(config.sample_rate),
        "--data.espeak_voice",
        train.espeak_voice,
        "--data.cache_dir",
        cache_dir,
        "--data.config_path",
        config_path,
        "--data.batch_size",
        str(train.batch_size),
    ]

    if train.max_epochs is not None:
        cmd += ["--trainer.max_epochs", str(train.max_epochs)]

    if train.ckpt_path:
        cmd += ["--ckpt_path", train.ckpt_path]

    return cmd


def run_training(voice_folder: str, config: Config) -> None:
    """Train a Piper voice; checkpoints are saved by Lightning under ``log_dir``.

    Training is expected to run until interrupted with ``Ctrl+C``. If
    ``train.export_after`` is set, the latest checkpoint is exported to ONNX
    after training stops.

    Args:
        voice_folder: Path to the voice project folder.
        config: Resolved :class:`~custom_tts.config.Config`.
    """
    os.makedirs(os.path.join(voice_folder, config.train.cache_dirname), exist_ok=True)

    cmd = build_train_command(voice_folder, config)
    logger.info("Starting training: %s", " ".join(cmd))
    interrupted = False
    try:
        subprocess.run(cmd, check=True)
        logger.info("Training finished. Checkpoints in '%s'.", config.train.log_dir)
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Training interrupted with Ctrl+C.")

    if config.train.export_after:
        if interrupted:
            logger.info("Exporting latest checkpoint to ONNX after interruption")
        else:
            logger.info("Exporting trained checkpoint to ONNX")
        export_onnx(voice_folder, config)
