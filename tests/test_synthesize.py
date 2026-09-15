import unittest

from custom_tts import synthesize


class SynthesizeValidationTest(unittest.TestCase):
    def test_missing_reference_fails_before_model_load(self):
        with self.assertRaisesRegex(FileNotFoundError, "Reference audio not found"):
            synthesize("missing-reference.wav", "", "Hello")


if __name__ == "__main__":
    unittest.main()
