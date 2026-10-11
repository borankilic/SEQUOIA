import dummies
import pytest

from sequoia.gui.document import SetupDocument


@pytest.fixture
def parts_library(library):
    dummies.make_parts(library)
    return library


@pytest.fixture
def doc(parts_library):
    return SetupDocument(parts_library)
