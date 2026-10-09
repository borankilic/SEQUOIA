import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent))

import dummies  # noqa: E402
import pytest  # noqa: E402

from sequoia.library.store import Library  # noqa: E402


@pytest.fixture
def registry(monkeypatch):
    dummies.reset_registry(monkeypatch)
    dummies.register_all()


@pytest.fixture
def library(tmp_path, registry):
    return Library(tmp_path / "library")
