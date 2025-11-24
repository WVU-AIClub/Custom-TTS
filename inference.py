import wave
from piper import PiperVoice, SynthesisConfig

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=2.0,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)
# TODO: Bug fix with loading json. Must be  .onnx.json
voice = PiperVoice.load("Peter/PETAH")
with wave.open("Thanksgiving.wav", "wb") as wav_file:
    voice.synthesize_wav("Happy Thanks,giving from A.I.W.V.U!", \
                         wav_file, syn_config=syn_config)