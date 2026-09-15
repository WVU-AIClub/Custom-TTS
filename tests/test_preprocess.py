import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydub.generators import Sine

from custom_tts.cli import main


class PreprocessTest(unittest.TestCase):
    def test_raw_media_becomes_f5_dataset(self):
        class WhisperModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, **_kwargs):
                return [SimpleNamespace(text=" Exact transcript. ")], None

        with tempfile.TemporaryDirectory() as directory:
            voice = Path(directory) / "voice"
            raws = voice / "raws"
            raws.mkdir(parents=True)
            with (raws / "source.wav").open("wb") as source:
                Sine(440).to_audio_segment(duration=1000).export(source, format="wav")

            with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=WhisperModel)}):
                exit_code = main(["preprocess", str(voice)])

            self.assertEqual(exit_code, 0)
            rows = (voice / "metadata.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "audio_file|text")
            self.assertEqual(rows[1], f"{(voice / 'wavs' / '00001.wav').as_posix()}|Exact transcript.")
            self.assertTrue((voice / "wavs" / "00001.wav").is_file())


if __name__ == "__main__":
    unittest.main()
