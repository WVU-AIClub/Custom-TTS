"""Typed configuration objects and YAML/CLI loading utilities.

Configuration is resolved in three layers, each overriding the previous:

1. Packaged defaults (``configs/default.yaml``).
2. An optional user YAML file passed via ``--config``.
3. Explicit CLI overrides (only values the user actually set).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any, get_type_hints

import yaml

# ---------------------------------------------------------------------------
# Dataclasses describing the configuration schema
# ---------------------------------------------------------------------------


@dataclass
class DirConfig:
    """Subfolder names used inside a voice project folder."""

    raws: str = "raws"
    splits: str = "splits"
    wavs: str = "wavs"
    outputs: str = "outputs"


@dataclass
class PreprocessConfig:
    """Settings for the dataset preprocessing pipeline."""

    chunk_length_ms: int = 30000
    whisper_model: str = "base"
    trim_top_db: int = 20
    normalize: bool = True
    subtype: str = "PCM_16"
    metadata_filename: str = "metadata.csv"
    dirs: DirConfig = field(default_factory=DirConfig)


@dataclass
class TrainConfig:
    """Settings for Piper training (``python -m piper.train fit``)."""

    voice_name: str | None = None  # defaults to the voice folder name
    espeak_voice: str = "en-us"
    batch_size: int = 4
    max_epochs: int | None = None  # optional cap on training epochs
    cache_dirname: str = "piper_cache"
    config_filename: str | None = None  # defaults to "<voice_name>.onnx.json"
    ckpt_path: str | None = None  # optional resume/pretrained checkpoint (path or URL)
    log_dir: str = "lightning_logs"
    export_after: bool = False  # export to ONNX after Ctrl+C or normal stop


@dataclass
class ExportConfig:
    """Settings for exporting a checkpoint to ONNX."""

    checkpoint: str | None = None  # explicit .ckpt; if null, use the latest found
    log_dir: str = "lightning_logs"
    output: str | None = None  # defaults to "<voice_folder>/<voice_name>.onnx"


@dataclass
class InferenceConfig:
    """Settings for Piper synthesis."""

    model: str | None = None
    text: str = "Hello, world."
    output: str = "output.wav"
    volume: float = 0.5
    length_scale: float = 1.4
    noise_scale: float = 1.0
    noise_w_scale: float = 1.5
    normalize_audio: bool = False


@dataclass
class Config:
    """Top-level configuration container."""

    sample_rate: int = 22050
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# ---------------------------------------------------------------------------
# Loading and merging helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_NAME = "default.yaml"


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass instance from a plain dict.

    Unknown keys raise ``ValueError`` so typos in config files fail loudly.
    """
    if not is_dataclass(cls):
        return data

    field_types = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = field_types[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` merged into ``base`` recursively."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return data


def _load_default_dict() -> dict[str, Any]:
    resource = resources.files("custom_tts.configs").joinpath(_DEFAULT_CONFIG_NAME)
    with resources.as_file(resource) as path:
        return _load_yaml(path)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from defaults, an optional YAML file, and CLI overrides.

    Args:
        config_path: Optional path to a user YAML config file.
        overrides: Nested dict of values to override (e.g. ``{"inference": {"text": "hi"}}``).

    Returns:
        A fully populated :class:`Config`.
    """
    data = _load_default_dict()

    if config_path is not None:
        data = _deep_merge(data, _load_yaml(config_path))

    if overrides:
        data = _deep_merge(data, overrides)

    return _from_dict(Config, data)


def to_dict(config: Config) -> dict[str, Any]:
    """Convert a :class:`Config` back into a plain dict (useful for logging)."""
    return dataclasses.asdict(config)
