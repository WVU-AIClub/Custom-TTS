"""End-to-end dataset preprocessing pipeline for a voice project folder."""

from __future__ import annotations

import os

from custom_tts.config import Config
from custom_tts.data.preprocess import (
    convert_audio,
    process_audio_files,
    split_long_audio,
)
from custom_tts.data.transcribe import transcribe
from custom_tts.utils.logging import get_logger

logger = get_logger(__name__)


def check_voice_folder(voice_folder: str, dirs) -> None:
    """Validate the voice folder and ensure required subfolders exist.

    Args:
        voice_folder: Path to the main voice project folder.
        dirs: :class:`~custom_tts.config.DirConfig` with subfolder names.

    Raises:
        FileNotFoundError: If the ``raws`` folder is missing or empty.
    """
    raws_path = os.path.join(voice_folder, dirs.raws)

    if not os.path.isdir(raws_path):
        raise FileNotFoundError(
            f"Missing '{raws_path}'. Create it and add source audio first."
        )

    files = [
        f
        for f in os.listdir(raws_path)
        if os.path.isfile(os.path.join(raws_path, f))
    ]
    if not files:
        raise FileNotFoundError(f"Please add audio data into '{raws_path}'.")

    for sub in (dirs.splits, dirs.wavs, dirs.outputs):
        os.makedirs(os.path.join(voice_folder, sub), exist_ok=True)

    logger.info("Voice folder validated: %s", voice_folder)


def run_pipeline(voice_folder: str, config: Config) -> None:
    """Run the full preprocessing pipeline for ``voice_folder``.

    Steps: validate -> split -> convert -> transcribe -> process.

    Args:
        voice_folder: Path to the main voice project folder.
        config: Resolved :class:`~custom_tts.config.Config`.
    """
    pre = config.preprocess
    dirs = pre.dirs

    raws = os.path.join(voice_folder, dirs.raws)
    splits = os.path.join(voice_folder, dirs.splits)
    wavs = os.path.join(voice_folder, dirs.wavs)
    outputs = os.path.join(voice_folder, dirs.outputs)
    metadata = os.path.join(voice_folder, pre.metadata_filename)

    check_voice_folder(voice_folder, dirs)

    logger.info("Step 1/4: splitting long audio")
    split_long_audio(raws, splits, chunk_length_ms=pre.chunk_length_ms)

    logger.info("Step 2/4: converting audio to WAV")
    convert_audio(splits, wavs)

    logger.info("Step 3/4: transcribing")
    transcribe(wavs, metadata, whisper_model=pre.whisper_model)

    logger.info("Step 4/4: processing and tagging")
    process_audio_files(
        wavs,
        outputs,
        sample_rate=config.sample_rate,
        trim_top_db=pre.trim_top_db,
        normalize=pre.normalize,
        subtype=pre.subtype,
    )

    logger.info("Pipeline complete for %s", voice_folder)
