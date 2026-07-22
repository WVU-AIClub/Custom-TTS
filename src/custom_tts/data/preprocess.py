"""Audio preprocessing steps for building a Piper training dataset.

Each function corresponds to a stage of the pipeline:

* :func:`split_long_audio` - chunk long recordings.
* :func:`convert_audio` - normalize container format and rename sequentially.
* :func:`process_audio_files` - trim, normalize, and tag final clips.
"""

from __future__ import annotations

import os
import shutil

import librosa
import soundfile as sf
import taglib
from pydub import AudioSegment

from custom_tts.utils.logging import get_logger

logger = get_logger(__name__)


def _list_files(folder: str) -> list[str]:
    return [
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]


def split_long_audio(
    input_folder: str,
    output_folder: str,
    chunk_length_ms: int = 30000,
) -> None:
    """Split any audio files longer than ``chunk_length_ms`` into chunks.

    Files shorter than the limit are copied through unchanged.

    Args:
        input_folder: Folder containing the source audio files.
        output_folder: Folder where chunks/copies are written.
        chunk_length_ms: Maximum chunk duration in milliseconds.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file in _list_files(input_folder):
        input_file_path = os.path.join(input_folder, file)
        try:
            audio = AudioSegment.from_file(input_file_path)

            if len(audio) > chunk_length_ms:
                logger.info("Splitting long audio file: %s", file)
                stem = os.path.splitext(file)[0]
                chunks = [
                    audio[i : i + chunk_length_ms]
                    for i in range(0, len(audio), chunk_length_ms)
                ]
                for i, chunk in enumerate(chunks, start=1):
                    chunk_path = os.path.join(output_folder, f"{stem}_part{i}.wav")
                    chunk.export(chunk_path, format="wav")
                logger.info("Split %s into %d chunks.", file, len(chunks))
            else:
                logger.info("Copying short audio file: %s", file)
                shutil.copy(input_file_path, output_folder)

        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("Error processing %s: %s", file, exc)


def convert_audio(input_folder: str, output_folder: str) -> None:
    """Convert every file to WAV and rename sequentially (``1.wav``, ``2.wav`` ...).

    Args:
        input_folder: Folder containing the source audio files.
        output_folder: Folder where converted WAV files are written.
    """
    os.makedirs(output_folder, exist_ok=True)

    files = _list_files(input_folder)
    logger.info("Converting %d file(s) from %s", len(files), input_folder)

    for index, file in enumerate(files, start=1):
        input_file_path = os.path.join(input_folder, file)
        output_file_name = f"{index}.wav"
        output_file_path = os.path.join(output_folder, output_file_name)
        try:
            audio = AudioSegment.from_file(input_file_path)
            audio.export(output_file_path, format="wav")
            logger.info("Converted %s -> %s", file, output_file_name)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("Error processing %s: %s", file, exc)


def process_audio_files(
    input_folder: str,
    output_folder: str = "processed_files",
    sample_rate: int = 22050,
    trim_top_db: int = 20,
    normalize: bool = True,
    subtype: str = "PCM_16",
) -> None:
    """Trim silence, optionally normalize, and tag final WAV clips.

    Args:
        input_folder: Folder containing the WAV files to process.
        output_folder: Folder to write the processed files to.
        sample_rate: Target sample rate (Hz) for resampling on load.
        trim_top_db: Silence-trim threshold in dB below peak.
        normalize: Whether to peak-normalize the trimmed audio.
        subtype: soundfile PCM subtype for the written files.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(input_folder, filename)

        y, sr = librosa.load(filepath, sr=sample_rate)
        trimmed_audio, _ = librosa.effects.trim(y, top_db=trim_top_db)
        audio = librosa.util.normalize(trimmed_audio) if normalize else trimmed_audio

        output_filepath = os.path.join(output_folder, filename)
        sf.write(output_filepath, audio, sr, subtype=subtype)

        try:
            file_number = int(os.path.splitext(filename)[0])
        except ValueError:
            logger.warning(
                "Skipping metadata update for '%s': filename is not a number.",
                filename,
            )
            continue

        with taglib.File(output_filepath, save_on_exit=True) as tagged:
            tagged.tags["TITLE"] = [str(file_number)]
            tagged.tags["TRACKNUMBER"] = [str(file_number)]

        logger.info("Processed '%s' (track %d).", filename, file_number)

    logger.info("All .wav files processed and saved to %s.", output_folder)
