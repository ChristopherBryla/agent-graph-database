"""conftest.py — ensures src/ is on sys.path so tests can import genesys and sample."""

import os
import sys
import pathlib
import tempfile

# Add src/ to the path so both `genesys` and `sample` packages are importable
# without installing the project.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

# Point the run store at a throwaway DB so test runs never pollute data/.
os.environ.setdefault(
    "GENESYS_RUNS_DB",
    os.path.join(tempfile.mkdtemp(prefix="genesys_test_"), "runs.db"),
)

# Keep model registry artifacts test-local as well.
_models_tmp = tempfile.mkdtemp(prefix="genesys_models_test_")
os.environ.setdefault(
    "GENESYS_MODELS_DB",
    os.path.join(_models_tmp, "models.db"),
)
os.environ.setdefault(
    "GENESYS_MODEL_BLOBS_DIR",
    os.path.join(_models_tmp, "blobs"),
)
