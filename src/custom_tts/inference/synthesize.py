"""Piper TTS synthesis wrapper."""

from __future__ import annotations

import wave

from piper import PiperVoice, SynthesisConfig

from custom_tts.config import InferenceConfig
from custom_tts.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_model_path(model: str) -> str:
    """Return a Piper-loadable model path.

    Piper expects the ``.onnx`` file (with a sibling ``.onnx.json``). Accepts
    either the full ``.onnx`` path or its stem.
    """
    if model.endswith(".onnx"):
        return model
    return f"{model}.onnx"


def synthesize(config: InferenceConfig) -> str:
    """Synthesize speech to a WAV file using a Piper voice.

    Args:
        config: Inference settings, including the model path and text.

    Returns:
        The path to the written WAV file.

    Raises:
        ValueError: If no model path is configured.
    """
    if not config.model:
        raise ValueError(
            "No model configured. Set 'inference.model' in a config file "
            "or pass --model on the CLI."
        )

    model_path = _resolve_model_path(config.model)

    syn_config = SynthesisConfig(
        volume=config.volume,
        length_scale=config.length_scale,
        noise_scale=config.noise_scale,
        noise_w_scale=config.noise_w_scale,
        normalize_audio=config.normalize_audio,
    )

    logger.info("Loading voice: %s", model_path)
    voice = PiperVoice.load(model_path)

    logger.info("Synthesizing to %s", config.output)
    with wave.open(config.output, "wb") as wav_file:
        voice.synthesize_wav(config.text, wav_file, syn_config=syn_config)

    logger.info("Wrote %s", config.output)
    return config.output
