import sys
from pathlib import Path

# darija_overrides.transcriber_darija imports shorts_generator at module
# load time (to reuse the vendored .srt cache helpers) — make the vendored
# package importable for tests and any script run from the repo root.
sys.path.insert(
    0, str(Path(__file__).parent / "vendor" / "ai-youtube-shorts-generator")
)
