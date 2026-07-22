"""Enable ``python -m custom_tts``."""

import sys

from custom_tts.cli import main

if __name__ == "__main__":
    sys.exit(main())
