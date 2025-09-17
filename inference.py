# pip install piper-tts
import wave
from piper import PiperVoice, SynthesisConfig

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=2.0,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

voice = PiperVoice.load("voices/peter_griffin1.onnx")
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav("Welcome to the world of speech synthesis!", \
                         wav_file, syn_config=syn_config)