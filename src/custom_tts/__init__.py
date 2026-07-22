"""custom_tts: build Piper TTS voice datasets and run synthesis.

Exposes the high-level configuration objects and the pipeline/synthesis
entry points for programmatic use.
"""

from custom_tts.config import (
    Config,
    ExportConfig,
    InferenceConfig,
    PreprocessConfig,
    TrainConfig,
    load_config,
)

__all__ = [
    "Config",
    "ExportConfig",
    "InferenceConfig",
    "PreprocessConfig",
    "TrainConfig",
    "load_config",
]

__version__ = "0.1.0"
