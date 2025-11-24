import wave
from piper import PiperVoice, SynthesisConfig

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.4,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.5,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)
# TODO: Bug fix with loading json. Must be  .onnx.json
voice = PiperVoice.load("Nate/Nate")
with wave.open("Nate_6650.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello, I am Nate. Please give me more money.", \
                         wav_file, syn_config=syn_config)