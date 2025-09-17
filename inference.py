# pip install piper-tts
import wave
from piper import PiperVoice, SynthesisConfig

syn_config = SynthesisConfig(
    volume=7,  # half as loud
    length_scale=0.75,  # twice as slow
    noise_scale=0.5,  # more audio variation
    noise_w_scale=1,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

voice = PiperVoice.load("voices/nate_model.onnx")
with wave.open("Nate_Tongue.wav", "wb") as wav_file:
    voice.synthesize_wav("UNIQUE NEW YORK UNEED NEW YORK YOU KNOW UNEED UNIQUE NEW YORK", \
                         wav_file, syn_config=syn_config)