from pathlib import Path

import pytest
from optics import basic_library, fs, inst, probe, run

import sequoia.components.builtin  # noqa: F401
from sequoia.cli import main
from sequoia.engines.classical import simulate
from sequoia.engines.report import format_report
from sequoia.library.store import Library
from sequoia.setup.io import load_setup
from sequoia.setup.validate import SetupValidationError

ROOT = Path(__file__).parents[1]
LIBRARY = Library(ROOT / "library")


def test_laser_to_fiber_example():
    result = simulate(load_setup(ROOT / "setups/laser_fiber_coupling.yaml"), LIBRARY)
    fiber = result.links["f1"]
    assert 0.985 < fiber.info["coupling_efficiency"] <= 1.0
    assert fiber.end.power == pytest.approx(
        5e-3 * fiber.info["coupling_efficiency"] * 0.9984, rel=1e-3
    )
    assert result.links["s2"].end.mode.w_x == pytest.approx(1.0e-3, rel=5e-3)
    assert result.readouts["pd"]["photocurrent_A"] == pytest.approx(
        0.5 * result.links["s6"].end.power
    )
    assert not [i for i in result.issues if i.severity == "error"]
    assert result.links["s6"].end.dop == pytest.approx(0.998, abs=1e-3)


def test_mzi_example_is_dark_at_one_port():
    result = simulate(load_setup(ROOT / "setups/mzi.yaml"), LIBRARY)
    assert result.probes["p2"]["power_W"] == pytest.approx(5e-3, rel=1e-9)
    assert result.probes["p1"]["power_W"] < 1e-9 * 5e-3


def test_invalid_setup_is_refused_before_simulating(tmp_path):
    lib = basic_library(Library(tmp_path / "lib"))
    with pytest.raises(SetupValidationError, match="there is no instance 'ghost'"):
        run(lib, [inst("src", "laser")], [fs("l", "src.out", "ghost.in")])


def test_unconnected_output_loses_light_and_warns(tmp_path):
    lib = basic_library(Library(tmp_path / "lib"))
    instances = [inst("src", "laser"), inst("bs", "bs"), inst("end", "pd")]
    result = run(lib, instances, [fs("a", "src.out", "bs.in_a"), fs("b", "bs.out_a", "end.in")])
    assert result.readouts["end"]["incident_power_W"] == pytest.approx(0.5e-3)
    assert [i.code for i in result.issues] == ["unconnected_output"]


def test_probe_position_moves_along_a_free_space_link(tmp_path):
    lib = basic_library(Library(tmp_path / "lib"))
    instances = [inst("src", "laser"), inst("end", "pd")]
    links = [fs("l", "src.out", "end.in", "2 m")]
    probes = [
        {**probe("start", "l", "probes/beam"), "settings": {"position": "0 m"}},
        {**probe("mid", "l", "probes/beam"), "settings": {"position": "1 m"}},
        probe("stop", "l", "probes/beam"),
    ]
    result = run(lib, instances, links, probes)
    w = [result.probes[k]["w_x_m"] for k in ("start", "mid", "stop")]
    assert w[0] < w[1] < w[2]
    assert w[0] == pytest.approx(1e-3)


def test_locked_length_uses_canvas_distance(tmp_path):
    lib = basic_library(Library(tmp_path / "lib"))
    instances = [
        inst("src", "laser", position=[0, 0]),
        inst("end", "pd", position=[30, 40]),
    ]
    link = {"id": "l", "kind": "free_space", "from": "src.out", "to": "end.in"}
    link["lock_to_canvas"] = True
    result = run(lib, instances, [link])
    assert result.links["l"].length == pytest.approx(50 * 25e-3)


def test_report_lists_links_fibers_readouts_and_diagnostics():
    result = simulate(load_setup(ROOT / "setups/laser_fiber_coupling.yaml"), LIBRARY)
    text = format_report(result)
    for expected in ("Links", "f1", "coupling efficiency 99.", "photocurrent_A", "fiber_power"):
        assert expected in text


def test_cli_run_prints_the_link_table(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("sys.argv", ["sequoia", "run", "setups/laser_fiber_coupling.yaml"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    assert "coupling efficiency" in capsys.readouterr().out


def test_cli_reports_errors_without_a_traceback(capsys, monkeypatch, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("instances: [{id: x, part: nothing/here}]\n")
    monkeypatch.setattr(
        "sys.argv", ["sequoia", "run", str(bad), "--library", str(ROOT / "library")]
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    assert "not found in library" in capsys.readouterr().err
