from faster_whisper import WhisperModel
model = WhisperModel("base", device="cuda", compute_type="int8_float16")
print("CUDA working!")   