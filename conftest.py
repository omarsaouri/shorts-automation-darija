import sys
from pathlib import Path

# Pipeline stage scripts live in src/ as sibling modules (import processor,
# from db import get_connection, etc.) — make that importable for tests.
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Several tests import shorts_generator (or its submodules) directly to
# exercise the vendored pipeline's Darija-specific logic — make the vendored
# package importable for tests and any script run from the repo root.
sys.path.insert(
    0, str(Path(__file__).parent / "vendor" / "ai-youtube-shorts-generator")
)
