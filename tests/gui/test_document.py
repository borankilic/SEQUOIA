import pytest

from sequoia.errors import SetupError
from sequoia.gui.document import SetupDocument
from sequoia.library.store import Part
from sequoia.setup.io import load_setup, save_setup
from sequoia.setup.model import Setup


class Recorder:
    def __init__(self, doc):
        self.counts = {"changed": 0, "structure": 0, "dirty": []}
        doc.changed.connect(lambda: self._bump("changed"))
        doc.structure_changed.connect(lambda: self._bump("structure"))
        doc.dirty_changed.connect(self.counts["dirty"].append)

    def _bump(self, key):
        self.counts[key] += 1

    def __getitem__(self, key):
        return self.counts[key]


def build_chain(doc):
    laser = doc.add_instance("sources/laser", [0, 0])
    mirror = doc.add_instance("freespace/mirror", [4, 0])
    coupler = doc.add_instance("fiber/coupler", [8, 0])
    det = doc.add_instance("detectors/spad", [12, 0])
    return laser, mirror, coupler, det


def test_empty_document_is_clean(doc):
    assert doc.setup == Setup()
    assert not doc.dirty
    assert doc.path is None


def test_add_instance_and_ids(doc):
    a = doc.add_instance("sources/laser", [1, 2], rotation=370)
    b = doc.add_instance("sources/laser", [3, 4])
    assert (a, b) == ("laser", "laser_2")
    inst = doc.setup.instances[0]
    assert inst.position == [1.0, 2.0] and inst.rotation == 10.0
    assert doc.add_instance("freespace/mirror", [0, 0], instance_id="m_1") == "m_1"
    with pytest.raises(ValueError, match="already used"):
        doc.add_instance("freespace/mirror", [0, 0], instance_id="m_1")
    with pytest.raises(ValueError):
        doc.add_instance("freespace/mirror", [0, 0], instance_id="1bad")
    with pytest.raises(ValueError, match="not found"):
        doc.add_instance("nope/part", [0, 0])
    with pytest.raises(ValueError, match="probe"):
        doc.add_instance("probes/powermeter", [0, 0])
    assert len(doc.setup.instances) == 3


def test_new_id_sanitises_and_avoids_all_kinds(doc):
    laser, mirror, coupler, det = build_chain(doc)
    link = doc.add_link("free_space", f"{laser}.out", f"{mirror}.in", link_id="x")
    assert doc.new_id("x") == "x_2"
    assert doc.new_id("3d part-name") == "_3d_part_name"
    assert doc.new_id("") == "item"
    assert link == "x"


def test_move_rotate_flip_and_undo(doc):
    laser = doc.add_instance("sources/laser", [0, 0])
    before = doc.setup
    doc.rotate_instance(laser, 90)
    doc.flip_instance(laser)
    inst = doc.setup.instances[0]
    assert inst.rotation == 90.0 and inst.flipped
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert doc.setup == before
    doc.undo_stack.redo()
    doc.undo_stack.redo()
    assert doc.setup.instances[0].flipped


def test_moves_merge_until_other_edit(doc):
    laser = doc.add_instance("sources/laser", [0, 0])
    mirror = doc.add_instance("freespace/mirror", [4, 0])
    n = doc.undo_stack.count()
    for x in (1, 2, 3):
        doc.move_instance(laser, [x, 0])
    assert doc.undo_stack.count() == n + 1
    doc.move_instance(mirror, [9, 9])
    doc.move_instance(laser, [5, 5])
    assert doc.undo_stack.count() == n + 3
    doc.undo_stack.undo()
    assert doc.setup.instances[0].position == [3.0, 0.0]
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert doc.setup.instances[0].position == [0.0, 0.0]


def test_move_after_save_does_not_merge_into_clean_state(doc, tmp_path):
    laser = doc.add_instance("sources/laser", [0, 0])
    doc.move_instance(laser, [1, 0])
    doc.save(tmp_path / "s.yaml")
    assert not doc.dirty
    doc.move_instance(laser, [2, 0])
    assert doc.dirty
    doc.undo_stack.undo()
    assert not doc.dirty
    assert doc.setup.instances[0].position == [1.0, 0.0]


def test_noop_edit_pushes_nothing(doc):
    laser = doc.add_instance("sources/laser", [0, 0])
    n = doc.undo_stack.count()
    doc.move_instance(laser, [0, 0])
    doc.set_name("untitled")
    assert doc.undo_stack.count() == n


def test_invalid_edit_leaves_state_intact(doc):
    laser = doc.add_instance("sources/laser", [0, 0])
    snapshot, count = doc.setup, doc.undo_stack.count()
    rec = Recorder(doc)
    with pytest.raises(ValueError):
        doc.move_instance(laser, [1, 2, 3])
    with pytest.raises(ValueError):
        doc.set_grid_cell("-3 mm")
    with pytest.raises(ValueError):
        doc.set_grid_cell("banana")
    with pytest.raises(ValueError, match="no instance"):
        doc.move_instance("ghost", [0, 0])
    with pytest.raises(ValueError):
        doc.add_link("free_space", "laser.out", "bad endpoint")
    with pytest.raises(ValueError):
        doc.add_link("warp", "laser.out", "m.in")
    assert doc.setup == snapshot
    assert doc.undo_stack.count() == count
    assert rec["changed"] == 0 and rec["structure"] == 0


def test_settings_and_overrides(doc):
    mirror = doc.add_instance("freespace/mirror", [0, 0])
    doc.set_setting(mirror, "angle", "45 deg")
    assert doc.setup.instances[0].settings == {"angle": "45 deg"}
    assert doc.component(mirror).settings.angle.value == 45.0
    doc.set_setting(mirror, "angle", None)
    assert doc.setup.instances[0].settings == {}
    with pytest.raises(ValueError):
        doc.set_setting(mirror, "angle", "banana")
    with pytest.raises(ValueError):
        doc.set_setting(mirror, "nope", 1)
    assert doc.setup.instances[0].settings == {}

    laser = doc.add_instance("sources/laser", [0, 0])
    doc.set_override(laser, "power", "5 mW")
    assert doc.component(laser).params.power.value == 5.0
    with pytest.raises(ValueError, match="unknown parameter"):
        doc.set_override(laser, "bogus", 1)
    with pytest.raises(ValueError):
        doc.set_override(laser, "power", "5 bananas")
    assert doc.setup.instances[1].overrides == {"power": "5 mW"}
    doc.set_override(laser, "power", None)
    assert doc.setup.instances[1].overrides == {}
    with pytest.raises(ValueError, match="no instance or probe"):
        doc.set_setting("ghost", "angle", 1)


def test_component_cache_and_ports(doc):
    coupler = doc.add_instance("fiber/coupler", [0, 0])
    c1 = doc.component(coupler)
    doc.move_instance(coupler, [1, 1])
    assert doc.component(coupler) is c1
    assert [(p.name, p.kind.value) for p in doc.ports(coupler)] == [
        ("in", "free_space"),
        ("out", "fiber"),
    ]
    doc.set_setting(coupler, "mode", "to_free")
    assert doc.component(coupler) is not c1
    assert [(p.name, p.kind.value) for p in doc.ports(coupler)] == [
        ("in", "fiber"),
        ("out", "free_space"),
    ]
    doc.undo_stack.undo()
    assert doc.ports(coupler)[0].kind.value == "free_space"
    assert doc.component("ghost") is None
    assert doc.ports("ghost") == ()


def test_component_none_when_invalid(parts_library):
    setup = Setup.model_validate(
        {
            "instances": [
                {"id": "a", "part": "freespace/mirror", "settings": {"angle": "banana"}},
                {"id": "b", "part": "missing/part"},
                {"id": "c", "part": "freespace/mirror"},
            ]
        }
    )
    doc = SetupDocument(parts_library, setup)
    assert doc.component("a") is None and "angle" in doc.component_error("a")
    assert doc.component("b") is None and doc.ports("b") == ()
    assert doc.component("c") is not None
    doc.set_setting("c", "angle", "10 deg")
    doc.set_setting("a", "angle", "20 deg")
    assert doc.component("a") is not None


def test_links_defaults_and_fields(doc):
    laser, mirror, coupler, det = build_chain(doc)
    fs = doc.add_link("free_space", f"{laser}.out", f"{mirror}.in")
    assert doc.setup.links[0].length.si == pytest.approx(0.1)
    assert fs == "link"
    lock = doc.add_link("free_space", f"{mirror}.out", f"{coupler}.in", lock_to_canvas=True)
    link = doc.setup.links[1]
    assert link.lock_to_canvas and link.length is None
    fib = doc.add_link("fiber", f"{coupler}.out", f"{det}.in")
    link = doc.setup.links[2]
    assert link.fiber == "fiber/smf" and link.length.si == 1.0
    assert (fs, lock, fib) == ("link", "link_2", "link_3")
    doc.set_link_field(fs, "length", "250 mm")
    assert doc.setup.links[0].length.si == pytest.approx(0.25)
    doc.set_link_field(fib, "connector_loss_db", 0.5)
    assert doc.setup.links[2].connector_loss_db == 0.5
    with pytest.raises(ValueError):
        doc.set_link_field(fib, "connector_loss_db", -1)
    with pytest.raises(ValueError):
        doc.set_link_field(fs, "length", None)
    with pytest.raises(ValueError):
        doc.set_link_field(fs, "kind", "cable")
    with pytest.raises(ValueError, match="no link"):
        doc.set_link_field("ghost", "length", "1 m")
    assert doc.setup.links[0].length.si == pytest.approx(0.25)


def test_cable_default_and_ids(doc):
    det = doc.add_instance("detectors/spad", [0, 0])
    tagger = doc.add_instance("electronics/tagger", [4, 0])
    cable = doc.add_link("cable", f"{det}.out", f"{tagger}.ch1")
    assert doc.setup.links[0].delay.si == 0.0
    doc.set_link_field(cable, "delay", "5 ns")
    assert doc.setup.links[0].delay.si == pytest.approx(5e-9)


def test_fiber_default_requires_unique_fiber(doc, parts_library):
    laser, mirror, coupler, det = build_chain(doc)
    parts_library.save_part(
        Part("fiber/smf2", "OpticalFiber", parts_library.load_part("fiber/smf").params)
    )
    with pytest.raises(ValueError, match="several"):
        doc.add_link("fiber", f"{coupler}.out", f"{det}.in")
    doc.add_link("fiber", f"{coupler}.out", f"{det}.in", fiber="fiber/smf2", length="3 m")
    assert doc.setup.links[0].fiber == "fiber/smf2"
    with pytest.raises(ValueError):
        doc.add_link("fiber", "a.b", "c.d", fiber="fiber/smf", length="-1 m")


def test_fiber_default_none_available(tmp_path, registry):
    from sequoia.library.store import Library

    doc = SetupDocument(Library(tmp_path / "empty"))
    with pytest.raises(ValueError, match="no OpticalFiber"):
        doc.add_link("fiber", "a.out", "b.in")


def test_can_connect_cases(doc):
    laser, mirror, coupler, det = build_chain(doc)
    doc.add_instance("freespace/bs", [0, 4])
    doc.add_instance("electronics/tagger", [0, 8])
    assert doc.can_connect("laser.out", "mirror.in") == ("free_space", "")
    assert doc.can_connect("coupler.out", "spad.in") == ("fiber", "")
    assert doc.can_connect("spad.out", "tagger.ch1") == ("cable", "")
    kind, reason = doc.can_connect("laser.out", "spad.in")
    assert kind is None and "free_space" in reason and "fiber" in reason
    assert doc.can_connect("mirror.out", "coupler.in")[0] == "free_space"
    kind, reason = doc.can_connect("coupler.out", "mirror.in")
    assert kind is None and "fiber" in reason
    kind, reason = doc.can_connect("laser.out", "bs.out_t")
    assert kind is None and "both out" in reason
    kind, reason = doc.can_connect("mirror.in", "bs.in_a")
    assert kind is None and "both in" in reason
    kind, reason = doc.can_connect("laser.out", "laser.out")
    assert kind is None and "itself" in reason
    kind, reason = doc.can_connect("mirror.in", "mirror.out")
    assert kind is None and "itself" in reason
    kind, reason = doc.can_connect("laser.nope", "mirror.in")
    assert kind is None and "no port 'nope'" in reason and "out" in reason
    kind, reason = doc.can_connect("ghost.out", "mirror.in")
    assert kind is None and "ghost" in reason
    assert doc.can_connect("garbage", "mirror.in")[0] is None

    link = doc.add_link("free_space", "laser.out", "mirror.in")
    kind, reason = doc.can_connect("laser.out", "bs.in_a")
    assert kind is None and link in reason
    kind, reason = doc.can_connect("bs.out_t", "mirror.in")
    assert kind is None and link in reason
    assert doc.can_connect("mirror.out", "bs.in_a")[0] == "free_space"
    doc.add_link("free_space", "mirror.out", "bs.in_a")
    kind, reason = doc.can_connect("bs.out_t", "laser.out")
    assert kind is None
    assert doc.can_connect("bs.out_t", "mirror.in")[0] is None
    assert doc.can_connect("bs.out_r", "coupler.in")[0] == "free_space"


def test_can_connect_cycle(doc):
    a = doc.add_instance("freespace/mirror", [0, 0], instance_id="m1")
    b = doc.add_instance("freespace/mirror", [4, 0], instance_id="m2")
    doc.add_link("free_space", f"{a}.out", f"{b}.in")
    kind, reason = doc.can_connect(f"{b}.out", f"{a}.in")
    assert kind is None and "cycle" in reason


def test_reversed_drag_is_normalised(doc):
    build_chain(doc)
    assert doc.can_connect("mirror.in", "laser.out") == ("free_space", "")
    assert doc.normalize_connection("mirror.in", "laser.out") == ("laser.out", "mirror.in")
    assert doc.normalize_connection("laser.out", "mirror.in") == ("laser.out", "mirror.in")
    link = doc.add_link("free_space", "mirror.in", "laser.out")
    stored = doc.setup.links[0]
    assert (stored.src, stored.dst, stored.id) == ("laser.out", "mirror.in", link)


def test_remove_instance_cascades(doc):
    laser, mirror, coupler, det = build_chain(doc)
    l1 = doc.add_link("free_space", f"{laser}.out", f"{mirror}.in")
    l2 = doc.add_link("free_space", f"{mirror}.out", f"{coupler}.in")
    probe = doc.add_probe("probes/powermeter", l1)
    keep = doc.add_probe("probes/powermeter", l2)
    before = doc.setup
    rec = Recorder(doc)
    doc.remove_instance(laser)
    assert [i.id for i in doc.setup.instances] == [mirror, coupler, det]
    assert [link.id for link in doc.setup.links] == [l2]
    assert [p.id for p in doc.setup.probes] == [keep]
    assert rec["changed"] == 1 and rec["structure"] == 1
    doc.undo_stack.undo()
    assert doc.setup == before
    assert probe in [p.id for p in doc.setup.probes]
    assert rec["changed"] == 2 and rec["structure"] == 2
    with pytest.raises(ValueError):
        doc.remove_instance("ghost")


def test_remove_link_and_probe(doc):
    laser, mirror, *_ = build_chain(doc)
    link = doc.add_link("free_space", f"{laser}.out", f"{mirror}.in")
    probe = doc.add_probe("probes/powermeter", link)
    assert probe == "powermeter"
    doc.remove_probe(probe)
    assert doc.setup.probes == []
    with pytest.raises(ValueError):
        doc.remove_probe(probe)
    doc.add_probe("probes/powermeter", link)
    doc.remove_link(link)
    assert doc.setup.links == [] and doc.setup.probes == []
    with pytest.raises(ValueError, match="no link"):
        doc.add_probe("probes/powermeter", link)
    with pytest.raises(ValueError, match="not a probe"):
        doc.add_probe("freespace/mirror", link)


def test_name_grid_analyses(doc):
    doc.set_name("My setup")
    doc.set_grid_cell("10 mm")
    spec = {"name": "g", "type": "g2", "channels": ["t.ch1", "t.ch2"], "range": "10 ns"}
    doc.set_analyses([spec])
    assert doc.setup.name == "My setup"
    assert doc.setup.grid_cell.si == pytest.approx(0.01)
    assert doc.setup.analyses == [spec]
    for _ in range(3):
        doc.undo_stack.undo()
    assert doc.setup == Setup()


def test_undo_redo_roundtrip_full_history(doc):
    snapshots = [doc.setup]
    laser, mirror, coupler, det = build_chain(doc)
    steps = [
        lambda: doc.add_link("free_space", f"{laser}.out", f"{mirror}.in"),
        lambda: doc.set_setting(mirror, "angle", "30 deg"),
        lambda: doc.rotate_instance(mirror, 45),
        lambda: doc.add_probe("probes/powermeter", "link"),
        lambda: doc.remove_instance(laser),
        lambda: doc.set_name("x"),
    ]
    states = []
    for step in steps:
        step()
        states.append(doc.setup)
    for expected in reversed(states[:-1]):
        doc.undo_stack.undo()
        assert doc.setup == expected
    while doc.undo_stack.canUndo():
        doc.undo_stack.undo()
    assert doc.setup == snapshots[0]
    while doc.undo_stack.canRedo():
        doc.undo_stack.redo()
    assert doc.setup == states[-1]


def test_macro_groups_into_one_step(doc):
    rec = Recorder(doc)
    with doc.macro("Build"):
        laser = doc.add_instance("sources/laser", [0, 0])
        mirror = doc.add_instance("freespace/mirror", [4, 0])
        doc.add_link("free_space", f"{laser}.out", f"{mirror}.in")
        doc.move_instance(laser, [1, 1])
        assert len(doc.setup.instances) == 2
        assert rec["changed"] == 0
    assert doc.undo_stack.count() == 1
    assert doc.undo_stack.undoText() == "Build"
    assert rec["changed"] == 1 and rec["structure"] == 1
    assert rec["dirty"] == [True]
    doc.undo_stack.undo()
    assert doc.setup == Setup()
    assert rec["changed"] == 2 and rec["structure"] == 2
    doc.undo_stack.redo()
    assert len(doc.setup.links) == 1
    assert doc.setup.instances[0].position == [1.0, 1.0]


def test_macro_rolls_back_on_error(doc):
    doc.add_instance("sources/laser", [0, 0])
    snapshot, count = doc.setup, doc.undo_stack.count()
    rec = Recorder(doc)
    with pytest.raises(ValueError):
        with doc.macro("Bad"):
            doc.add_instance("freespace/mirror", [4, 0])
            doc.set_grid_cell("banana")
    assert doc.setup == snapshot
    assert doc.undo_stack.count() == count
    assert rec["changed"] == 0


def test_nested_macro_and_swallowed_inner_error(doc):
    with doc.macro("Outer"):
        doc.add_instance("sources/laser", [0, 0])
        with pytest.raises(RuntimeError):
            with doc.macro("Inner"):
                doc.add_instance("freespace/mirror", [4, 0])
                raise RuntimeError
        with doc.macro("Inner2"):
            doc.add_instance("freespace/bs", [8, 0])
    assert [i.id for i in doc.setup.instances] == ["laser", "bs"]
    assert doc.undo_stack.count() == 1


def test_empty_macro_pushes_nothing(doc):
    with doc.macro("nothing"):
        pass
    assert doc.undo_stack.count() == 0


def test_signals_counts(doc):
    rec = Recorder(doc)
    laser = doc.add_instance("sources/laser", [0, 0])
    assert (rec["changed"], rec["structure"], rec["dirty"]) == (1, 1, [True])
    doc.move_instance(laser, [1, 1])
    assert (rec["changed"], rec["structure"]) == (2, 1)
    doc.move_instance(laser, [2, 2])
    assert (rec["changed"], rec["structure"]) == (3, 1)
    doc.set_name("n")
    assert (rec["changed"], rec["structure"]) == (4, 1)
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert (rec["changed"], rec["structure"], rec["dirty"]) == (7, 2, [True, False])
    doc.undo_stack.redo()
    assert rec["dirty"] == [True, False, True]


def test_save_open_equivalence(doc, tmp_path):
    laser, mirror, coupler, det = build_chain(doc)
    doc.set_name("chain")
    doc.add_link("free_space", f"{laser}.out", f"{mirror}.in", medium="air")
    doc.add_link("free_space", f"{mirror}.out", f"{coupler}.in", lock_to_canvas=True)
    doc.add_link("fiber", f"{coupler}.out", f"{det}.in", length="2 m", connector_loss_db=0.3)
    doc.add_probe("probes/powermeter", "link")
    doc.set_setting(mirror, "angle", "12 deg")
    doc.set_override(laser, "power", "2 mW")
    doc.set_analyses(
        [{"name": "g", "type": "g2", "channels": ["t.ch1", "t.ch2"], "range": "10 ns", "bins": 50}]
    )
    rec = Recorder(doc)
    path = tmp_path / "out.yaml"
    doc.save(path)
    assert doc.path == path and not doc.dirty
    assert rec["dirty"] == [False]
    reference = tmp_path / "ref.yaml"
    save_setup(doc.setup, reference)
    assert path.read_bytes() == reference.read_bytes()
    reopened = SetupDocument.open(path, doc.library)
    assert reopened.setup == doc.setup
    assert reopened.path == path and not reopened.dirty
    assert load_setup(path) == doc.setup


def test_save_without_path_and_save_as(doc, tmp_path):
    doc.add_instance("sources/laser", [0, 0])
    with pytest.raises(ValueError, match="no file path"):
        doc.save()
    paths = []
    doc.path_changed.connect(lambda: paths.append(doc.path))
    doc.save(tmp_path / "a.yaml")
    doc.set_name("b")
    doc.save()
    assert paths == [tmp_path / "a.yaml"]
    assert load_setup(tmp_path / "a.yaml").name == "b"


def test_open_errors(parts_library, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("instances: 3\n")
    with pytest.raises(SetupError):
        SetupDocument.open(bad, parts_library)
    with pytest.raises(SetupError):
        SetupDocument.open(tmp_path / "missing.yaml", parts_library)


def test_document_constructed_with_setup_is_clean(parts_library):
    setup = Setup.model_validate({"instances": [{"id": "m", "part": "freespace/mirror"}]})
    doc = SetupDocument(parts_library, setup)
    assert doc.setup is setup and not doc.dirty
    doc.set_setting("m", "angle", "5 deg")
    assert doc.dirty
    doc.undo_stack.undo()
    assert doc.setup == setup and not doc.dirty
