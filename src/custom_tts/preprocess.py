"""Prepare raw media as transcribed F5-TTS training clips."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence


def preprocess(
    voice_folder: str | Path,
    *,
    whisper_model: str = "base",
    asr_device: str = "auto",
    chunk_seconds: float = 12.0,
) -> Path:
    """Split ``raws/`` media into WAV clips and write F5-TTS metadata."""
    voice_path = Path(voice_folder).expanduser().resolve()
    raw_path = voice_path / "raws"
    wav_path = voice_path / "wavs"

    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw media directory not found: {raw_path}")
    raw_files = sorted(path for path in raw_path.iterdir() if path.is_file())
    if not raw_files:
        raise FileNotFoundError(f"No raw media found in: {raw_path}")
    if chunk_seconds <= 0:
        raise ValueError("Chunk seconds must be greater than zero.")

    wav_path.mkdir(parents=True, exist_ok=True)
    for old_clip in wav_path.glob("*.wav"):
        old_clip.unlink()

    max_milliseconds = round(chunk_seconds * 1000)
    clips: list[Path] = []
    for raw_file in raw_files:
        with raw_file.open("rb") as source:
            audio = AudioSegment.from_file(source).set_channels(1).set_frame_rate(24000).set_sample_width(2)
        if not audio.rms:
            continue
        silence_threshold = audio.dBFS - 16 if math.isfinite(audio.dBFS) else -40
        passages = split_on_silence(
            audio,
            min_silence_len=350,
            silence_thresh=silence_threshold,
            keep_silence=200,
        ) or [audio]
        for passage in passages:
            for start in range(0, len(passage), max_milliseconds):
                clip = passage[start : start + max_milliseconds]
                if len(clip) < 500 or not clip.rms:
                    continue
                clip_path = wav_path / f"{len(clips) + 1:05d}.wav"
                with clip_path.open("wb") as destination:
                    clip.export(destination, format="wav", codec="pcm_s16le")
                clips.append(clip_path)

    if not clips:
        raise ValueError("No usable speech clips were produced from the raw media.")

    from faster_whisper import WhisperModel

    model = WhisperModel(whisper_model, device=asr_device)
    metadata_path = voice_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata:
        writer = csv.writer(metadata, delimiter="|", lineterminator="\n")
        writer.writerow(("audio_file", "text"))
        for clip in clips:
            segments, _ = model.transcribe(str(clip), beam_size=5)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not text:
                raise ValueError(f"Transcription was empty for: {clip}")
            writer.writerow((clip.as_posix(), text))

    return metadata_path
