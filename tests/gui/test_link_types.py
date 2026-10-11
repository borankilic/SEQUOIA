from pathlib import Path

import pytest

from sequoia.gui.document import SetupDocument
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def doc():
    return SetupDocument.open(ROOT / "setups" / "hbt_heralded.yaml", Library(ROOT / "library"))


def test_fiber_parts_grouped_by_kind(doc):
    kinds = doc.fiber_parts_by_kind()
    assert kinds["smf"] and kinds["pm"] and kinds["mmf"]
    assert "fiber/mmf_50um" in kinds["mmf"]


def test_new_fiber_links_default_to_single_mode(doc):
    assert doc._default_fiber() in doc.fiber_parts_by_kind()["smf"]
    doc.new_fiber_ref = "fiber/pmf_810"
    assert doc._default_fiber() == "fiber/pmf_810"


def test_fiber_loss_defaults_to_zero(doc):
    link = next(link for link in doc.setup.links if link.kind == "fiber")
    assert link.connector_loss_db == 0.0


def test_air_link_becomes_fiber_and_back(doc):
    before = doc.setup
    n_inst, n_links = len(before.instances), len(before.links)
    new_id = doc.convert_link_to_fiber("s_bp", fiber="fiber/pmf_810")
    assert len(doc.setup.instances) == n_inst + 2
    assert len(doc.setup.links) == n_links + 2
    fiber = next(link for link in doc.setup.links if link.id == new_id)
    assert fiber.kind == "fiber" and fiber.fiber == "fiber/pmf_810"
    assert fiber.connector_loss_db == 0.0
    assert all(doc.component(i.id) is not None for i in doc.setup.instances)
    doc.undo_stack.undo()
    assert doc.setup == before
    doc.undo_stack.redo()
    air = doc.convert_link_to_air(new_id)
    assert len(doc.setup.instances) == n_inst
    assert len(doc.setup.links) == n_links
    assert next(link for link in doc.setup.links if link.id == air).kind == "free_space"


def test_conversion_errors_are_value_errors(doc):
    with pytest.raises(ValueError):
        doc.convert_link_to_fiber("f_signal")
    with pytest.raises(ValueError):
        doc.convert_link_to_air("f_a")
    with pytest.raises(ValueError):
        doc.convert_link_to_fiber("nope")
