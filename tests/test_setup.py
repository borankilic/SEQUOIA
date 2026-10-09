import copy
import textwrap

import pytest
from dummies import make_parts

from sequoia.errors import SetupError
from sequoia.setup.io import load_setup, save_setup, setup_from_dict, setup_to_dict
from sequoia.setup.model import FiberLink, FreeSpaceLink
from sequoia.setup.validate import SetupValidationError, check_setup, validate_setup
from sequoia.yamlio import load_yaml

BASE = {
    "name": "chain",
    "instances": [
        {"id": "laser", "part": "sources/laser", "position": [0, 0]},
        {"id": "bp", "part": "filters/bp", "position": [4, 0], "rotation": 90},
        {"id": "coupler", "part": "fiber/coupler", "position": [8, 0]},
        {"id": "det", "part": "detectors/spad", "position": [12, 0], "flipped": True},
        {"id": "tagger", "part": "electronics/tagger", "position": [16, 0]},
    ],
    "links": [
        {"id": "l1", "kind": "free_space", "from": "laser.out", "to": "bp.in", "length": "100 mm"},
        {
            "id": "l2",
            "kind": "free_space",
            "from": "bp.out",
            "to": "coupler.in",
            "lock_to_canvas": True,
        },
        {
            "id": "f1",
            "kind": "fiber",
            "from": "coupler.out",
            "to": "det.in",
            "fiber": "fiber/smf",
            "length": "2 m",
            "connector_loss_db": 0.3,
            "polarization": {"kind": "random", "seed": 7},
        },
        {"id": "c1", "kind": "cable", "from": "det.out", "to": "tagger.ch1", "delay": "5 ns"},
    ],
    "probes": [{"id": "pm", "part": "probes/powermeter", "link": "l1"}],
    "analyses": [{"type": "coincidences", "channels": ["ch1", "ch2"], "window": "1 ns"}],
}


@pytest.fixture
def parts(library):
    make_parts(library)
    return library


def codes(issues, severity="error"):
    return [i.code for i in issues if i.severity == severity]


def mutated(change):
    raw = copy.deepcopy(BASE)
    change(raw)
    return raw


def link(raw, link_id):
    return next(x for x in raw["links"] if x["id"] == link_id)


def test_valid_setup_has_no_issues(parts):
    setup = setup_from_dict(BASE)
    assert check_setup(setup, parts) == []
    assert validate_setup(setup, parts) == []


def test_yaml_roundtrip_preserves_the_model_and_is_stable(tmp_path):
    setup = setup_from_dict(BASE)
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    save_setup(setup, first)
    reloaded = load_setup(first)
    assert reloaded == setup
    save_setup(reloaded, second)
    assert first.read_text() == second.read_text()


def test_saved_file_is_compact_and_readable(tmp_path):
    path = tmp_path / "s.yaml"
    save_setup(setup_from_dict(BASE), path)
    text = path.read_text()
    assert "from: laser.out" in text and "to: bp.in" in text
    assert "length: 100 mm" in text and "delay: 5 ns" in text
    assert "flipped: false" not in text and "rotation: 0" not in text
    assert "position: [4.0, 0.0]" in text
    raw = load_yaml(path)
    assert raw["analyses"] == BASE["analyses"]


def test_hand_written_file_loads(tmp_path):
    path = tmp_path / "hand.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            name: hand written
            grid_cell: 12.5 mm
            instances:
              - id: m1
                part: freespace/mirror
                position: [1, 2]
                rotation: -90
                settings: {angle: 22.5 deg}
            links: []
            """
        )
    )
    setup = load_setup(path)
    assert setup.grid_cell.si == pytest.approx(12.5e-3)
    assert setup.instances[0].rotation == 270.0
    assert setup.instances[0].settings == {"angle": "22.5 deg"}


def test_link_models_round_trip_through_dict():
    setup = setup_from_dict(BASE)
    kinds = {type(x) for x in setup.links}
    assert FreeSpaceLink in kinds and FiberLink in kinds
    assert setup.links[1].lock_to_canvas and setup.links[1].length is None
    assert setup_to_dict(setup)["links"][2]["polarization"] == {"kind": "random", "seed": 7}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["instances"][0].update(id="bad.id"), "id"),
        (lambda r: link(r, "l1").update({"from": "laserout"}), "from"),
        (lambda r: link(r, "l1").pop("length"), "needs a 'length'"),
        (lambda r: link(r, "l1").update(length="-1 mm"), "must not be negative"),
        (lambda r: link(r, "l1").update(length="3 ns"), "length quantity"),
        (lambda r: link(r, "l1").update(kind="teleport"), "kind"),
        (lambda r: link(r, "f1").update(polarization={"kind": "su2"}), "rotation_vector"),
        (lambda r: link(r, "f1").update(polarization={"kind": "random"}), "seed"),
        (lambda r: link(r, "f1").update(connector_loss_db=-1), "connector_loss_db"),
        (lambda r: link(r, "c1").update(delay="2 mm"), "time quantity"),
        (lambda r: r.update(grid_cell="0 mm"), "grid_cell must be positive"),
        (lambda r: r.update(colour="red"), "colour"),
        (lambda r: r["instances"][0].update(position=[1, 2, 3]), "position"),
    ],
)
def test_malformed_setups_are_rejected_with_the_location(change, message):
    with pytest.raises(SetupError, match=message):
        setup_from_dict(mutated(change), source="s.yaml")


def test_non_mapping_and_bad_yaml_files(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(SetupError, match="must contain a mapping"):
        load_setup(path)
    path.write_text("a: [unterminated\n")
    with pytest.raises(SetupError, match="not valid YAML"):
        load_setup(path)
    with pytest.raises(SetupError, match="cannot read"):
        load_setup(tmp_path / "missing.yaml")


def add_instance(raw, inst_id, part, **extra):
    raw["instances"].append({"id": inst_id, "part": part, **extra})


def add_loop(raw):
    add_instance(raw, "bs", "freespace/bs")
    add_instance(raw, "m", "freespace/mirror")
    link(raw, "l1")["to"] = "bs.in_a"
    raw["links"] += [
        {"id": "a", "kind": "free_space", "from": "bs.out_r", "to": "m.in", "length": "5 cm"},
        {"id": "b", "kind": "free_space", "from": "m.out", "to": "bs.in_b", "length": "5 cm"},
    ]


@pytest.mark.parametrize(
    ("change", "code", "message"),
    [
        (
            lambda r: add_instance(r, "laser", "sources/laser"),
            "duplicate_id",
            "used more than once",
        ),
        (
            lambda r: link(r, "c1").update(id="l1"),
            "duplicate_id",
            "link id",
        ),
        (
            lambda r: r["instances"][0].update(part="sources/nothing"),
            "invalid_instance",
            "not found in library",
        ),
        (
            lambda r: r["instances"][2].update(settings={"mode": "sideways"}),
            "invalid_instance",
            "invalid settings for 'coupler'",
        ),
        (
            lambda r: r["instances"][0].update(overrides={"colour": "red"}),
            "invalid_instance",
            "unknown parameter",
        ),
        (
            lambda r: r["instances"][0].update(overrides={"power": "3 florps"}),
            "invalid_instance",
            "invalid overrides for 'laser'",
        ),
        (
            lambda r: link(r, "l1").update({"to": "ghost.in"}),
            "bad_endpoint",
            "there is no instance 'ghost'",
        ),
        (
            lambda r: link(r, "l1").update({"to": "bp.input"}),
            "bad_endpoint",
            "has no port 'input' \\(ports: in, out\\)",
        ),
        (
            lambda r: link(r, "l2").update({"to": "coupler.out", "from": "bp.out"}),
            "kind_mismatch",
            "fiber coupler",
        ),
        (
            lambda r: link(r, "f1").update({"from": "coupler.in"}),
            "kind_mismatch",
            "fiber link cannot use free_space port",
        ),
        (
            lambda r: link(r, "c1").update({"to": "det.in"}),
            "kind_mismatch",
            "cable link cannot use fiber port",
        ),
        (
            lambda r: link(r, "l1").update({"from": "bp.in", "to": "laser.out"}),
            "direction",
            "links run from an output port to an input port",
        ),
        (
            lambda r: (
                add_instance(r, "bp2", "filters/bp"),
                r["links"].append(
                    {
                        "id": "dup",
                        "kind": "free_space",
                        "from": "laser.out",
                        "to": "bp2.in",
                        "length": "1 cm",
                    }
                ),
            ),
            "port_reused",
            "already connected by link 'l1'",
        ),
        (
            lambda r: r["links"].remove(link(r, "c1")),
            "missing_required",
            "port 'det.out'",
        ),
        (
            lambda r: link(r, "f1").update(fiber="detectors/spad"),
            "bad_fiber",
            "not an OpticalFiber",
        ),
        (
            lambda r: link(r, "f1").update(fiber="fiber/nothing"),
            "bad_fiber",
            "not found in library",
        ),
        (
            lambda r: r["probes"][0].update(link="ghost"),
            "bad_probe",
            "there is no link 'ghost'",
        ),
        (
            lambda r: r["probes"][0].update(part="freespace/mirror"),
            "bad_probe",
            "not a probe part",
        ),
        (
            lambda r: add_instance(r, "pm2", "probes/powermeter"),
            "wrong_category",
            "probes attach to links",
        ),
        (add_loop, "cycle", "bs -> m -> bs"),
        (
            lambda r: r["links"].append(
                {
                    "id": "self",
                    "kind": "free_space",
                    "from": "bp.out",
                    "to": "bp.in",
                    "length": "1 cm",
                }
            ),
            "cycle",
            "bp -> bp",
        ),
    ],
)
def test_validation_reports_each_problem(parts, change, code, message):
    issues = check_setup(setup_from_dict(mutated(change)), parts)
    matching = [i for i in issues if i.code == code]
    assert matching, f"expected {code}, got {[(i.code, str(i)) for i in issues]}"
    assert any(__import__("re").search(message, str(i)) for i in matching), [
        str(i) for i in matching
    ]


def test_validate_setup_raises_with_every_error_listed(parts):
    raw = mutated(
        lambda r: (link(r, "l1").update({"to": "ghost.in"}), r["links"].remove(link(r, "c1")))
    )
    with pytest.raises(SetupValidationError) as info:
        validate_setup(setup_from_dict(raw), parts)
    text = str(info.value)
    assert "there is no instance 'ghost'" in text and "det.out" in text
    assert {i.code for i in info.value.issues} >= {"bad_endpoint", "missing_required"}


def test_unconnected_outputs_are_warnings_not_errors(parts):
    raw = mutated(lambda r: add_instance(r, "bs", "freespace/bs"))
    link(raw, "l1")["to"] = "bs.in_a"
    link(raw, "l1")["id"] = "to_bs"
    raw["links"].append(
        {"id": "l1", "kind": "free_space", "from": "bs.out_t", "to": "bp.in", "length": "1 cm"}
    )
    issues = validate_setup(setup_from_dict(raw), parts)
    assert codes(issues) == []
    assert codes(issues, "warning") == ["unconnected_output"]
    assert "bs.out_r" in str(issues[0])


def test_ports_depend_on_instance_settings(parts):
    raw = mutated(lambda r: r["instances"][2].update(settings={"mode": "to_free"}))
    issues = check_setup(setup_from_dict(raw), parts)
    assert "kind_mismatch" in codes(issues)


def test_overrides_replace_part_parameters(parts):
    from sequoia.setup.resolve import instantiate

    setup = setup_from_dict(
        mutated(
            lambda r: r["instances"][0].update(
                overrides={"power": "5 mW"},
            )
        )
    )
    assert instantiate(parts, setup.instances[0]).params.power.si == pytest.approx(5e-3)
    assert parts.load_part("sources/laser").params.power.si == pytest.approx(1e-3)
