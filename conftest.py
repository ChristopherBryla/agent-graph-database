"""conftest.py — ensures src/ is on sys.path so tests can import genesys and sample."""

import sys
import pathlib

# Add src/ to the path so both `genesys` and `sample` packages are importable
# without installing the project.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
