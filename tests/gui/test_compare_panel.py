import json
import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.gui.compare_panel import SWEEP_MODE, CompareTab
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[2]
WAIT = 20000


@pytest.fixture
def malus():
    return load_setup(ROOT / "templates" / "malus_law.yaml"), Library(ROOT / "library")


@pytest.fixture
def tab(qtbot, malus):
    t = CompareTab(lambda: malus)
    qtbot.addWidget(t)
    yield t
    t.shutdown()


def write_fringe(path: Path, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 180, 37)
    counts = rng.poisson(500 + 400 * np.cos(2 * np.deg2rad(theta - 20)))
    lines = ["analyser angle (deg),coincidences (counts)"]
    lines += [f"{a:g},{c}" for a, c in zip(theta, counts, strict=True)]
    lines.insert(5, "40,")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(qtbot, tab):
    tab.run()
    qtbot.waitUntil(lambda: not tab.runner.busy, timeout=WAIT)
    qtbot.wait(20)


def test_load_fit_fringe_and_export(qtbot, tab, tmp_path):
    path = tmp_path / "fringe.csv"
    write_fringe(path)
    assert tab.load_file(str(path))
    assert tab.data is not None and len(tab.data) == 37
    assert tab.data.dropped == ((5, "missing or non-numeric y"),)
    assert tab.x_unit.text() == "deg" and tab.y_unit.text() == "counts"
    assert tab.preview.rowCount() > 0
    tab.model.setCurrentIndex(0)
    run(qtbot, tab)
    r = tab.result
    assert r is not None and r.identifiable
    v, se, _ = r.display("visibility")
    assert abs(v - 0.8) < 2 * se
    phase, se_phase, unit = r.display("phase")
    assert unit == "deg" and abs(phase - 20) < 2 * se_phase
    assert tab.params.rowCount() == 4
    assert tab.params.item(1, 3).text() == "deg"
    assert tab.correlation.rowCount() == 3
    assert "chi2/dof" in tab.status.text()
    assert tab.export(str(tmp_path / "out.csv"))
    rec = json.loads((tmp_path / "out.json").read_text())
    assert rec["fit"]["parameters"]["phase"]["unit"] == "deg"
    assert rec["data"]["source"]["sha256"] and rec["data"]["dropped_rows"][0]["row"] == 5
    csv_lines = (tmp_path / "out.csv").read_text().splitlines()
    assert csv_lines[0].startswith("x (deg),y (counts),model (counts)")
    assert len(csv_lines) == 38


def test_fixed_parameters_and_unidentifiable_fit(qtbot, tab, tmp_path):
    path = tmp_path / "singles.csv"
    power = np.linspace(1, 20, 15)
    rate = 2e5 * power + 500
    lines = ["pump (mW),singles (Hz),sigma (Hz)"]
    lines += [f"{p:g},{r:.6g},{math.sqrt(r):.6g}" for p, r in zip(power, rate, strict=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert tab.load_file(str(path))
    assert tab.sigma_col.currentText() == "sigma (Hz)"
    tab.model.setCurrentText("Singles vs pump power (brightness x efficiency)")
    run(qtbot, tab)
    assert not tab.result.identifiable
    assert "cannot be separated from singles alone" in tab.status.text()
    assert "not identifiable" in [tab.params.item(i, 4).text() for i in range(3)]
    tab.fixed.setText("efficiency=0.2")
    run(qtbot, tab)
    assert tab.result.identifiable
    assert tab.result.display("brightness")[0] == pytest.approx(1e6, rel=1e-3)
    tab.fixed.setText("nonsense=1")
    tab.run()
    assert "fixed parameter" in tab.status.text()


def test_bad_files_and_mappings_are_reported(qtbot, tab, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("just text\nno numbers here\n", encoding="utf-8")
    assert not tab.load_file(str(bad))
    assert "cannot load" in tab.status.text() and tab.data is None
    assert not tab.load_file(str(tmp_path / "missing.csv"))
    assert not tab.load_file(str(tmp_path / "notes.pdf"))
    xlsx = tmp_path / "broken.xlsx"
    xlsx.write_bytes(b"not a workbook")
    assert not tab.load_file(str(xlsx))
    good = tmp_path / "fringe.csv"
    write_fringe(good)
    assert tab.load_file(str(good))
    tab.y_unit.setText("parsecs per fortnight")
    tab.apply_mapping()
    assert tab.data is None and "mapping" in tab.status.text()
    assert not tab.run_button.isEnabled()
    tab.y_unit.setText("counts")
    tab.x_unit.setText("ns")
    tab.apply_mapping()
    tab.model.setCurrentIndex(0)
    run(qtbot, tab)
    assert tab.result is None and "x must be of dimension angle" in tab.status.text()


def test_compare_with_sweep_and_scale(qtbot, tab, malus, tmp_path):
    from sequoia.analysis.sweep import linspace_axis, run_sweep

    setup, library = malus
    sweep = run_sweep(
        setup,
        library,
        [linspace_axis("settings:hwp.angle", 0, 90, 10, "deg")],
        ["probe:pm.power_W"],
    )
    power = 0.8e3 * sweep.column("probe:pm.power_W")
    path = tmp_path / "malus.csv"
    lines = ["hwp (deg),power (mW),sigma (mW)"]
    lines += [f"{a:g},{p:.8g},0.01" for a, p in zip(sweep.axes[0].values, power, strict=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert tab.load_file(str(path))
    tab.model.setCurrentText(SWEEP_MODE)
    assert tab.sweep_box.isVisibleTo(tab)
    tab.target.setCurrentText("settings:hwp.angle")
    tab.target_unit.setText("deg")
    tab.metric_box.setCurrentText("probe:pm.power_W")
    tab.fit_scale.setChecked(True)
    run(qtbot, tab)
    c = tab.comparison
    assert c is not None and c.scale.values["scale"] == pytest.approx(0.8, rel=1e-4)
    assert tab.export(str(tmp_path / "cmp.csv"))
    rec = json.loads((tmp_path / "cmp.json").read_text())
    assert rec["comparison"]["details"]["target"] == "settings:hwp.angle"
    assert len(rec["setup"]["library_fingerprint"]) == 64
    tab.reset()
    assert tab.comparison is None and not tab.export_button.isEnabled()
