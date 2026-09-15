"""Whisper-based transcription that writes a Piper-style ``metadata.csv``."""

from __future__ import annotations

import os

from faster_whisper import WhisperModel

from custom_tts.utils.logging import get_logger

logger = get_logger(__name__)


def transcribe(
    folder_path: str,
    transcript_file: str = "metadata.csv",
    whisper_model: str = "base",
) -> None:
    """Transcribe every ``*.wav`` in ``folder_path`` into ``transcript_file``.

    Output lines follow the ``<wav>|<text>`` format expected by Piper. Duplicate
    transcripts are logged but still written.

    Args:
        folder_path: Folder containing numbered WAV files (``1.wav`` ...).
        transcript_file: Destination metadata CSV path.
        whisper_model: Whisper model size to load.
    """
    model = WhisperModel(whisper_model, device="cuda")

    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    wav_files = sorted(wav_files, key=lambda x: int(os.path.splitext(x)[0]))

    seen: dict[str, str] = {}
    with open(transcript_file, "w", encoding="utf-8") as transcript:
        for wav_file in wav_files:
            logger.info("Transcribing %s", wav_file)
            segments, info = model.transcribe(os.path.join(folder_path, wav_file), beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip()

            if text in seen:
                logger.warning(
                    "Duplicate transcript: %s matches %s -> %r",
                    wav_file,
                    seen[text],
                    text,
                )
            else:
                seen[text] = wav_file

            transcript.write(f"{wav_file}|{text}\n")

    logger.info("Wrote transcript to %s", transcript_file)
