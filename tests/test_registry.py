import pytest

from sequoia.components.base import (
    CATEGORIES,
    Component,
    Params,
    get_component,
    register,
    registered_types,
)
from sequoia.errors import UnknownComponentError


def test_register_sets_metadata(registry):
    cls = get_component("DummyLaser")
    assert (cls.category, cls.type_name) == ("sources", "DummyLaser")


def test_duplicate_type_name_is_rejected(registry):
    with pytest.raises(ValueError, match="already registered"):
        register("sources", "DummyLaser")(type("Other", (Component,), {}))


def test_unknown_category_is_rejected(registry):
    with pytest.raises(ValueError, match="unknown category"):
        register("widgets", "Widget")(type("Widget", (Component,), {}))


def test_unknown_type_message_lists_registered_types(registry):
    with pytest.raises(UnknownComponentError, match="DummyLaser"):
        get_component("Nope")


def test_registered_types_by_category(registry):
    assert set(registered_types("freespace")) == {"DummyMirror", "DummyBeamSplitter"}
    assert {c.category for c in registered_types().values()} <= set(CATEGORIES)


def test_params_forbid_unknown_fields():
    class P(Params):
        a: float = 1.0

    with pytest.raises(ValueError):
        P(b=2)
