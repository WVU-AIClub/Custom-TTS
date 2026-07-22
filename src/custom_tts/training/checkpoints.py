"""Locate Lightning checkpoints produced during training."""

from __future__ import annotations

import glob
import os

from custom_tts.utils.logging import get_logger

logger = get_logger(__name__)


def find_latest_checkpoint(log_dir: str = "lightning_logs") -> str:
    """Return the newest ``.ckpt`` under ``log_dir/version_*/checkpoints``.

    Args:
        log_dir: Root Lightning logs directory.

    Returns:
        Path to the most recent checkpoint.

    Raises:
        FileNotFoundError: If no version directories or checkpoints exist.
    """
    version_dirs = glob.glob(os.path.join(log_dir, "version_*"))
    if not version_dirs:
        raise FileNotFoundError(
            f"No 'version_*' directories found in '{log_dir}'. Train a model first."
        )

    latest_version = max(
        version_dirs, key=lambda d: int(d.split("version_")[-1])
    )
    ckpt_dir = os.path.join(latest_version, "checkpoints")
    ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "*.ckpt")),
        key=os.path.getmtime,
    )
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in '{ckpt_dir}'.")

    latest = ckpts[-1]
    logger.info("Latest checkpoint: %s", latest)
    return latest
