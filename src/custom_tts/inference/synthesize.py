"""F5-TTS synthesis backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_model(model: str, checkpoint: str, vocab: str, vocoder: str, device: str | None):
    """Load and retain the most recently used model."""
    from f5_tts.api import F5TTS

    return F5TTS(
        model=model,
        ckpt_file=checkpoint,
        vocab_file=vocab,
        vocoder_local_path=vocoder or None,
        device=device,
    )


def synthesize(
    ref_audio: str | Path,
    ref_text: str,
    text: str,
    output: str | Path = "output.wav",
    *,
    model: str = "F5TTS_v1_Base",
    checkpoint: str | Path | None = None,
    vocab: str | Path | None = None,
    vocoder_path: str | Path | None = None,
    device: str | None = None,
    seed: int | None = None,
    speed: float = 1.0,
    nfe_steps: int = 32,
    remove_silence: bool = False,
) -> str:
    """Clone ``ref_audio`` and synthesize ``text`` to a WAV file."""
    ref_path = Path(ref_audio).expanduser()
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
    vocab_path = Path(vocab).expanduser() if vocab else None
    vocoder = Path(vocoder_path).expanduser() if vocoder_path else None

    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {ref_path}")
    if checkpoint_path and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if vocab_path and not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    if vocoder and not vocoder.is_dir():
        raise FileNotFoundError(f"Vocoder directory not found: {vocoder}")
    if not text.strip():
        raise ValueError("Text to synthesize cannot be empty.")
    if speed <= 0:
        raise ValueError("Speed must be greater than zero.")
    if nfe_steps <= 0:
        raise ValueError("NFE steps must be greater than zero.")

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _load_model(
        model,
        str(checkpoint_path) if checkpoint_path else "",
        str(vocab_path) if vocab_path else "",
        str(vocoder) if vocoder else "",
        None if device in (None, "auto") else device,
    )
    engine.infer(
        ref_file=str(ref_path),
        ref_text=ref_text,
        gen_text=text,
        file_wave=str(output_path),
        seed=seed,
        speed=speed,
        nfe_step=nfe_steps,
        remove_silence=remove_silence,
    )
    return str(output_path)
