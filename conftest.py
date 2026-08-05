import sys
from pathlib import Path

# Everything the pipeline needs — stage scripts (processor, db, ...),
# shorts_generator (vendored base repo), and darija_overrides — lives under
# src/ as sibling packages/modules. Make that importable for tests and any
# script run from the repo root.
sys.path.insert(0, str(Path(__file__).parent / "src"))
