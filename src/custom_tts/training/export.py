"""Export a trained Piper checkpoint to ONNX."""

from __future__ import annotations

import os
import subprocess
import sys

from custom_tts.config import Config
from custom_tts.training.checkpoints import find_latest_checkpoint
from custom_tts.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_voice_name(voice_folder: str, config: Config) -> str:
    """Return the configured voice name or fall back to the folder name."""
    return config.train.voice_name or os.path.basename(os.path.normpath(voice_folder))


def export_onnx(voice_folder: str, config: Config) -> str:
    """Export a checkpoint to ONNX via ``python -m piper.train.export_onnx``.

    Uses ``export.checkpoint`` when set, otherwise the latest checkpoint under
    ``export.log_dir``. The output defaults to ``<voice_folder>/<voice_name>.onnx``.

    Args:
        voice_folder: Path to the voice project folder.
        config: Resolved :class:`~custom_tts.config.Config`.

    Returns:
        Path to the written ``.onnx`` file.
    """
    exp = config.export

    checkpoint = exp.checkpoint or find_latest_checkpoint(exp.log_dir)

    voice_name = resolve_voice_name(voice_folder, config)
    output = exp.output or os.path.join(voice_folder, f"{voice_name}.onnx")

    cmd = [
        sys.executable,
        "-m",
        "piper.train.export_onnx",
        "--checkpoint",
        checkpoint,
        "--output-file",
        output,
    ]

    logger.info("Exporting checkpoint to ONNX: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Wrote %s", output)
    return output
