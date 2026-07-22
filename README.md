# Custom-TTS

A Python package and CLI for building [Piper](https://github.com/OHF-Voice/piper1-gpl)
TTS voice datasets and running synthesis. Configuration lives in YAML files and
every value can be overridden from the command line.

## Installation

```bash
pip install -e .
```

## Project structure

```
src/custom_tts/
├── cli.py                # argparse CLI (custom-tts entry point)
├── config.py             # typed config + YAML/CLI loading
├── configs/default.yaml  # packaged default configuration
├── data/
│   ├── preprocess.py     # split / convert / process audio
│   ├── transcribe.py     # Whisper -> metadata.csv
│   └── pipeline.py       # full preprocessing pipeline
├── training/
│   ├── train.py          # python -m piper.train fit
│   ├── export.py         # checkpoint -> ONNX
│   └── checkpoints.py    # find latest checkpoint
├── inference/
│   └── synthesize.py     # Piper synthesis
└── utils/logging.py      # shared logging
configs/example.yaml      # example user override config
```

Each voice lives in its own folder under `voices/`, holding both the dataset and
the trained model named after the voice:

```
voices/Nate/
├── raws/                 # source recordings you provide
├── splits/ wavs/ outputs/  # generated during preprocessing
├── piper_cache/          # phoneme/spec cache used by training
├── metadata.csv          # transcript (wav|text)
├── Nate.onnx             # exported model
└── Nate.onnx.json        # model config (loaded by Piper at inference)
```

## Configuration

Values resolve in three layers, each overriding the last:

1. Packaged defaults — `src/custom_tts/configs/default.yaml`
2. A user YAML file — `--config path/to.yaml`
3. Explicit CLI flags

## CLI usage

Preprocess a voice folder (expects a `raws/` subfolder with source audio):

```bash
custom-tts preprocess voices/Nate
custom-tts preprocess voices/Nate --whisper-model small --chunk-length-ms 20000
custom-tts preprocess voices/Nate --config configs/example.yaml
```

Train a Piper voice (Lightning saves checkpoints under `lightning_logs/version_*/checkpoints/`). Training is intended to keep running until you press `Ctrl+C`; if `--export` is set, the latest checkpoint is exported right after stop:

> **Training prerequisite:** the `train` and `export` commands shell out to
> `python -m piper.train`, provided by the
> [piper1-gpl](https://github.com/OHF-voice/piper1-gpl) training install
> (`pip install -e ".[train]"` inside that repo), not the `piper-tts` inference
> package. See `Piper_Training.ipynb` for the full environment setup.

```bash
custom-tts train voices/Nate
custom-tts train voices/Nate --batch-size 8 --max-epochs 1000 --ckpt-path pretrained.ckpt
custom-tts train voices/Nate --export          # export to ONNX after Ctrl+C or normal stop
```

Export a checkpoint to ONNX (uses the latest checkpoint when `--checkpoint` is omitted):

```bash
custom-tts export voices/Nate
custom-tts export voices/Nate --checkpoint lightning_logs/version_0/checkpoints/last.ckpt
custom-tts export voices/Nate --output voices/Nate/Nate.onnx
```

Synthesize speech from a trained voice:

```bash
custom-tts synthesize --model voices/Nate/Nate.onnx --text "Hello there" --output hi.wav
custom-tts synthesize --config configs/example.yaml --text "Using config defaults"
```

You can also run it as a module: `python -m custom_tts ...`.

## Programmatic use

```python
from custom_tts import load_config
from custom_tts.data import run_pipeline
from custom_tts.training import run_training, export_onnx
from custom_tts.inference import synthesize

config = load_config(overrides={"preprocess": {"whisper_model": "small"}})
run_pipeline("voices/Nate", config)

train_config = load_config(overrides={"train": {"batch_size": 8, "export_after": True}})
run_training("voices/Nate", train_config)    # saves checkpoints, then exports
export_onnx("voices/Nate", train_config)     # or export separately

synth_config = load_config(overrides={"inference": {"model": "voices/Nate/Nate.onnx"}})
synthesize(synth_config.inference)
```
