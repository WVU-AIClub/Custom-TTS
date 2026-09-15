# Custom-TTS

A small [F5-TTS](https://github.com/SWivid/F5-TTS) voice-cloning CLI and Python API.
It generates speech from a short reference recording; no Piper model, dataset preprocessing, or per-voice training is required.

## What it supports

- Linux, Windows, and macOS.
- CPU on every platform.
- NVIDIA GPUs on Linux and Windows with PyTorch CUDA 12.8 or CUDA 13.0 wheels.
- Apple Silicon GPU acceleration through Metal Performance Shaders (MPS).
- Automatic model downloads, local F5-TTS checkpoints, deterministic seeds, and configurable inference speed/quality.

The default `F5TTS_v1_Base` checkpoint supports English and Mandarin. Its weights are downloaded from Hugging Face on first use.

## Requirements

- Python 3.10-3.12. The repository selects Python 3.12 through `.python-version`.
- [uv](https://docs.astral.sh/uv/).
- FFmpeg available on `PATH`.
- About 5 GB of free disk space for the environment and downloaded models. CPU inference works but is substantially slower than GPU inference.

### Install uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal, then confirm installation with `uv --version`.

### Install FFmpeg

Choose the command for the host OS:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# macOS with Homebrew
brew install ffmpeg
```

```powershell
# Windows PowerShell
winget install --id Gyan.FFmpeg -e
```

Fedora users can install the `ffmpeg` package after enabling RPM Fusion. Confirm every installation with `ffmpeg -version`.

## Install the project

```bash
git clone https://github.com/WVU-AIClub/Custom-TTS.git
cd Custom-TTS
uv python install 3.12
```

Install exactly one accelerator extra. Do not use `--all-extras`; the PyTorch builds conflict by design.
After syncing, the commands below use `uv run --no-sync` so uv does not replace the selected PyTorch wheel with the platform default. Rerun the chosen `uv sync --extra ...` command after dependency changes.


### CPU: Linux, Windows, or macOS

```bash
uv sync --extra cpu
```

Use `--device cpu` when running the CLI. On Apple Silicon, this same installation also contains MPS support; use `--device mps` or the default `--device auto` to use it.

### NVIDIA GPU: CUDA 13

```bash
uv sync --extra cu130
```

This installs the PyTorch CUDA 13.0 runtime. It is the recommended choice for an RTX 5060 Ti and works with a compatible CUDA 13.1 NVIDIA driver. The local CUDA toolkit is not used by normal inference because PyTorch wheels bundle their CUDA runtime.

### NVIDIA GPU: CUDA 12

```bash
uv sync --extra cu128
```

This installs the PyTorch CUDA 12.8 runtime. CUDA 12.8 is the CUDA 12 build to use for Blackwell cards such as the RTX 5060 Ti; older CUDA 12.x builds may not contain that GPU architecture. A sufficiently new CUDA 13.1 driver can also run this CUDA 12.8 wheel.

### Verify the selected backend

```bash
uv run --no-sync python -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('MPS available:', torch.backends.mps.is_available())"
```

Expected results:

- CPU: both availability values are `False`.
- NVIDIA: `CUDA available: True`; the runtime is `12.8` or `13.0`.
- Apple Silicon: `MPS available: True`.

`nvidia-smi` reports the maximum CUDA version supported by the installed driver, not necessarily the runtime bundled with PyTorch. Different values there are normal.

## Preprocess a voice dataset

Put raw, unsplit video or audio files in the voice's `raws/` directory, then run the same one-command pipeline as before:

```bash
uv run --no-sync custom-tts preprocess voices/MyVoice
```

This extracts mono 24 kHz audio, splits speech into clips under `voices/MyVoice/wavs/`, transcribes each clip, and writes F5-TTS `audio_file|text` metadata to `voices/MyVoice/metadata.csv`.

Useful overrides:

```bash
uv run --no-sync custom-tts preprocess voices/MyVoice --chunk-seconds 12 --asr-device cuda --whisper-model large-v3
uv run --no-sync custom-tts preprocess voices/MyVoice --asr-device cpu --whisper-model /path/to/offline/whisper-model
```

Model names require their normal initial download. On a network where Hugging Face is blocked, pass an approved local Faster-Whisper model directory instead.

## Prepare reference audio

For the best clone:

- Use a clean 3-12 second WAV, FLAC, MP3, or other FFmpeg-readable recording.
- Use one speaker, little background noise, and about one second of silence at the end.
- Supply an exact transcript with normal punctuation. Omitting it enables automatic transcription, which downloads another model and uses more memory.
- Use only recordings and voices you have permission to clone.

The repository includes a short example at `voices/Nate/wavs/9.wav` with the transcript `The quick brown fox jumps over the lazy dog.`

## Generate speech

Automatic device selection:

```bash
uv run --no-sync custom-tts synthesize \
  --ref-audio voices/Nate/wavs/9.wav \
  --ref-text "The quick brown fox jumps over the lazy dog." \
  --text "This sentence was generated with F5-TTS." \
  --output output.wav
```

PowerShell accepts the same command on one line:

```powershell
uv run --no-sync custom-tts synthesize --ref-audio voices/Nate/wavs/9.wav --ref-text "The quick brown fox jumps over the lazy dog." --text "This sentence was generated with F5-TTS." --output output.wav
```

Force a backend when diagnosing hardware selection:

```bash
uv run --no-sync custom-tts synthesize --ref-audio reference.wav --ref-text "Reference transcript." --text "Run on the CPU." --device cpu
uv run --no-sync custom-tts synthesize --ref-audio reference.wav --ref-text "Reference transcript." --text "Run on NVIDIA CUDA." --device cuda
uv run --no-sync custom-tts synthesize --ref-audio reference.wav --ref-text "Reference transcript." --text "Run on Apple MPS." --device mps
```

The first run downloads the F5-TTS checkpoint and Vocos vocoder. Later runs use the local cache.

### CLI options

Run `uv run --no-sync custom-tts --help` for the authoritative list.

| Option | Default | Purpose |
| --- | --- | --- |
| `--ref-audio PATH` | required | Reference voice recording. |
| `--ref-text TEXT` | automatic transcription | Exact reference transcript. |
| `--text TEXT` | required | Text to generate. |
| `--output PATH` | `output.wav` | Output WAV file; parent directories are created. |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps`, or `xpu`. |
| `--model` | `F5TTS_v1_Base` | Installed F5-TTS architecture config. |
| `--checkpoint PATH` | downloaded base model | Local F5-TTS checkpoint. |
| `--vocab PATH` | model default | Vocabulary for a custom checkpoint. |
| `--vocoder-path PATH` | downloaded Vocos model | Local `vocos-mel-24khz` directory for offline inference. |
| `--seed INTEGER` | random | Reproduce a generation. |
| `--speed FLOAT` | `1.0` | Speech speed; must be greater than zero. |
| `--nfe-steps INTEGER` | `32` | Denoising steps; fewer is faster, more can improve quality. |
| `--remove-silence` | off | Remove long silence with FFmpeg. |

Example with lower latency and reproducible output:

```bash
uv run --no-sync custom-tts synthesize --ref-audio reference.wav --ref-text "Reference transcript." --text "Repeatable output." --seed 42 --nfe-steps 16 --speed 1.1 --output generated/repeatable.wav
```

### Fine-tune a voice such as Peter

Prepare Peter's clips, then launch the maintained F5-TTS fine-tuning interface:

```bash
uv run --no-sync custom-tts preprocess voices/Peter
uv run --no-sync f5-tts_finetune-gradio
```

In the interface, create/select the `Peter` project, import `voices/Peter/wavs/` and `voices/Peter/metadata.csv`, select `F5TTS_v1_Base`, and set the pretrained checkpoint and vocabulary to the approved local bundle. Fine-tuning writes checkpoints under the F5-TTS checkpoint directory.

### Use a custom or fine-tuned checkpoint

```bash
HF_HUB_OFFLINE=1 uv run --no-sync custom-tts synthesize \
  --model F5TTS_v1_Base \
  --checkpoint models/Peter/model.safetensors \
  --vocab models/F5TTS_v1_Base/vocab.txt \
  --vocoder-path models/vocos-mel-24khz \
  --ref-audio voices/Peter/wavs/1.wav \
  --ref-text "The exact transcript of the reference clip." \
  --text "Text rendered by the Peter checkpoint." \
  --output peter.wav
```

See the upstream [training and fine-tuning guide](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/train) for training controls.

## Python API

```python
from custom_tts import synthesize

synthesize(
    ref_audio="voices/Nate/wavs/9.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    text="This call uses the same F5-TTS backend as the CLI.",
    output="output.wav",
    device="auto",
    seed=42,
)
```

The most recently selected model remains loaded inside a process, so repeated calls do not reload its weights.

## Offline model and vocoder bundle

On an authorized internet-connected machine, download the official files:

```bash
mkdir -p models/F5TTS_v1_Base models/vocos-mel-24khz
uvx --from huggingface-hub hf download SWivid/F5-TTS F5TTS_v1_Base/model_1250000.safetensors F5TTS_v1_Base/vocab.txt --local-dir models
uvx --from huggingface-hub hf download charactr/vocos-mel-24khz --local-dir models/vocos-mel-24khz
```

Transfer the complete `models/` directory through an approved channel. Keep every file in `vocos-mel-24khz`; Vocos needs its configuration and weights together. Verify the transferred files with SHA-256 checksums supplied by the downloading administrator.

On the blocked network, set offline mode and pass all three local paths:

```bash
export HF_HUB_OFFLINE=1
uv run --no-sync custom-tts synthesize --checkpoint models/F5TTS_v1_Base/model_1250000.safetensors --vocab models/F5TTS_v1_Base/vocab.txt --vocoder-path models/vocos-mel-24khz --ref-audio reference.wav --ref-text "Exact transcript." --text "Offline inference." --output output.wav
```

Windows PowerShell uses `$env:HF_HUB_OFFLINE = "1"` before the same command.

## Troubleshooting

- **`CUDA available: False`:** install the `cu130` or `cu128` extra, update the NVIDIA driver, and restart the terminal. Do not install a second PyTorch build with pip.
- **`no kernel image` or architecture errors on an RTX 50-series card:** use `cu130` or `cu128`; older CUDA wheels do not support Blackwell.
- **CUDA out of memory:** close other GPU workloads, shorten the reference/generated text, reduce `--nfe-steps`, or use `--device cpu`.
- **MPS operation error:** retry with `--device cpu`. PyTorch may fall back for operations not implemented by MPS.
- **Blank or truncated output:** install FFmpeg, keep the reference under 12 seconds, include its exact transcript, and leave silence at the end.
- **Slow first run:** model downloads and model initialization happen once. CPU generation itself remains slower than GPU generation.
- **Reset a broken environment:** remove `.venv`, then rerun the appropriate `uv sync --extra ...` command. `uv.lock` keeps versions reproducible.

## Licenses

This repository and F5-TTS code use the MIT license. The pretrained F5-TTS model weights are CC-BY-NC because of their training data; review that license before commercial use.
