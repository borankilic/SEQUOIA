import json
import math
import operator
import re
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import PydanticUndefined
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sequoia import data
from sequoia.data import Curve, FileRef, Scalar, Surface

MISSING = object()
NONE_TYPE = type(None)

STYLE = """
QLineEdit[state="error"], QDoubleSpinBox[state="error"], QSpinBox[state="error"] {
    border: 1px solid #c0392b;
}
QLineEdit[state="default"], QDoubleSpinBox[state="default"], QSpinBox[state="default"] {
    color: palette(placeholder-text);
    font-style: italic;
}
QLabel[role="error"] { color: #c0392b; }
QLabel[role="muted"] { color: palette(placeholder-text); }
QLabel[role="badge"] {
    color: palette(placeholder-text);
    border: 1px solid palette(mid);
    border-radius: 6px;
    padding: 0px 4px;
}
"""

_DIMENSIONS = {
    "Length": ("length", "810 nm"),
    "Duration": ("time", "50 ns"),
    "Power": ("power", "10 mW"),
    "Temperature": ("temperature", "25 degC"),
    "Angle": ("angle", "45 deg"),
    "Frequency": ("frequency", "80 MHz"),
    "Rate": ("rate", "100 Hz"),
    "Current": ("current", "10 uA"),
    "Voltage": ("voltage", "5 V"),
    "NonlinearCoefficient": ("nonlinear coefficient", "3.9 pm/V"),
    "Dimensionless": ("dimensionless", "0.5"),
}
_BY_FUNC = {getattr(data, n).__metadata__[0].func: v for n, v in _DIMENSIONS.items()}
_BARE_NUMBER = re.compile(r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$")


@dataclass
class FieldSpec:
    kind: str
    optional: bool = False
    adapter: TypeAdapter | None = None
    literals: tuple = ()
    labels: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    inner: "FieldSpec | None" = None
    value: "FieldSpec | None" = None
    minimum: float | None = None
    maximum: float | None = None
    members: list = field(default_factory=list)


def _flatten(ann, metas=()):
    origin = get_origin(ann)
    if origin is Annotated:
        base, *extra = get_args(ann)
        return _flatten(base, metas + tuple(extra))
    if origin in (Union, types.UnionType):
        out = []
        for arg in get_args(ann):
            out.extend(_flatten(arg, metas))
        return out
    return [(ann, metas)]


def _is(base, cls) -> bool:
    return isinstance(base, type) and issubclass(base, cls)


def _union(members):
    items = tuple(Annotated[b, *m] if m else b for b, m in members)
    return TypeAdapter(reduce(operator.or_, items))


def _bound(metas, *names):
    for m in metas:
        for name in names:
            v = getattr(m, name, None)
            if isinstance(v, int | float):
                return float(v)
    return None


def analyze(annotation, metadata=()) -> FieldSpec:
    flat = _flatten(annotation, tuple(metadata))
    optional = any(b is NONE_TYPE for b, _ in flat)
    members = [(b, m) for b, m in flat if b is not NONE_TYPE]
    spec = _classify(members)
    spec.optional = optional
    spec.members = members
    return spec


def _classify(members) -> FieldSpec:
    bases = [b for b, _ in members]
    scalars = [(b, m) for b, m in members if _is(b, Scalar)]
    literals = [b for b in bases if get_origin(b) is Literal]
    lit_values = tuple(v for b in literals for v in get_args(b))
    has_curve = any(_is(b, Curve) for b in bases)
    rest = [(b, m) for b, m in members if not _is(b, Curve)]
    if any(_is(b, Surface) for b in bases):
        return FieldSpec("surface")
    if any(_is(b, FileRef) for b in bases):
        return FieldSpec("fileref")
    plain = [(b, m) for b, m in rest if b in (int, float)]
    if scalars or (has_curve and plain):
        value = FieldSpec("scalar" if scalars else "float", adapter=_union(rest))
        if scalars:
            infos = []
            for _, metas in scalars:
                for meta in metas:
                    info = getattr(meta, "func", None)
                    if info in _BY_FUNC:
                        infos.append(_BY_FUNC[info])
            value.labels = tuple(dict.fromkeys(i[0] for i in infos))
            value.examples = tuple(dict.fromkeys(i[1] for i in infos))
            value.literals = lit_values
        else:
            value.minimum = _bound([m for _, ms in plain for m in ms], "ge", "gt")
            value.maximum = _bound([m for _, ms in plain for m in ms], "le", "lt")
        if has_curve:
            return FieldSpec("curve_or_value", value=value)
        return value
    if has_curve:
        return FieldSpec("curve")
    if lit_values and len(literals) == len(bases):
        return FieldSpec("literal", literals=lit_values)
    if len(bases) == 1:
        base, metas = members[0]
        origin = get_origin(base)
        if base is bool:
            return FieldSpec("bool")
        if base is int:
            return FieldSpec(
                "int",
                minimum=_bound(metas, "ge", "gt"),
                maximum=_bound(metas, "le", "lt"),
            )
        if base is float:
            return FieldSpec(
                "float",
                minimum=_bound(metas, "ge", "gt"),
                maximum=_bound(metas, "le", "lt"),
            )
        if base is str:
            return FieldSpec("str")
        if origin is list:
            return FieldSpec("list", adapter=_union(members))
        if origin is dict and len(get_args(base)) == 2:
            inner = analyze(get_args(base)[1])
            return FieldSpec("dict", adapter=_union(members), inner=inner)
    return FieldSpec("json", adapter=_union(members) if members else None)


def error_message(exc: ValidationError) -> str:
    seen: list[str] = []
    for err in exc.errors():
        msg = err["msg"].removeprefix("Value error, ")
        if msg not in seen:
            seen.append(msg)
    return "; ".join(seen)


def _num(value: float) -> str:
    return f"{value:.12g}"


def value_text(value: Any) -> str:
    if value is None or value is MISSING:
        return ""
    if isinstance(value, Scalar):
        return _scalar_text(value.value, value.unit)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return _num(value)
    if isinstance(value, dict) and "value" in value:
        return _scalar_text(value["value"], value.get("unit", "1"))
    return str(value)


def _scalar_text(number: float, unit: str) -> str:
    if unit.strip() in ("", "1"):
        return _num(number)
    return f"{_num(number)} {unit}"


def provenance_text(value: Any) -> str | None:
    prov = None
    if isinstance(value, BaseModel):
        prov = getattr(value, "provenance", None)
        prov = prov.model_dump(exclude_none=True) if prov is not None else None
    elif isinstance(value, dict):
        prov = value.get("provenance")
    if not prov:
        return None
    parts = [f"source: {prov.get('source')}"]
    for key in ("date", "by", "reference", "note"):
        if prov.get(key):
            parts.append(f"{key}: {prov[key]}")
    return "\n".join(parts)


def value_key(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def restyle(widget: QWidget, state: str) -> None:
    if widget.property("state") != state:
        widget.setProperty("state", state)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def _axis_text(label: str, unit: str) -> str:
    return label if unit.strip() in ("", "1") else f"{label} ({unit})"


def curve_from_raw(raw: Any) -> Curve | None:
    if isinstance(raw, Curve):
        return raw
    if isinstance(raw, dict) and "x" in raw and "y" in raw:
        try:
            return Curve.model_validate(raw)
        except ValidationError:
            return None
    return None


def curve_to_raw(curve: Curve) -> dict:
    return curve.model_dump(mode="json", exclude={"file"}, exclude_none=True)


def run_import_dialog(parent: QWidget | None) -> Curve | None:
    from sequoia.gui.inspector.importer import ImportCurveDialog

    dialog = ImportCurveDialog(parent)
    if dialog.exec():
        return dialog.curve()
    return None


class Editor(QWidget):
    committed = Signal(object)
    problem = Signal(str)

    def set_raw(self, raw: Any) -> None:
        raise NotImplementedError

    def initial_raw(self) -> Any:
        return MISSING

    def set_muted(self, muted: bool) -> None:
        pass

    def set_example(self, text: str) -> None:
        pass

    def clear_value(self) -> None:
        pass


class _NoWheelMixin:
    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _Combo(_NoWheelMixin, QComboBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class _IntSpin(_NoWheelMixin, QSpinBox):
    pass


class _FloatSpin(_NoWheelMixin, QDoubleSpinBox):
    def textFromValue(self, value: float) -> str:
        return _num(value)

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        from PySide6.QtGui import QValidator

        try:
            float(text)
            return QValidator.State.Acceptable, text, pos
        except ValueError:
            return QValidator.State.Intermediate, text, pos


class ScalarEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.edit = QLineEdit()
        self._example = spec.examples[0] if spec.examples else ""
        self._refresh_placeholder()
        self.unit_label = QLabel(" / ".join(spec.labels))
        self.unit_label.setProperty("role", "muted")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        if spec.labels:
            layout.addWidget(self.unit_label)
        self._shown = ""
        self._muted = False
        self.edit.textChanged.connect(self._on_text)
        self.edit.editingFinished.connect(self.commit)

    def _refresh_placeholder(self) -> None:
        extra = ""
        if self.spec.literals:
            extra = " or " + ", ".join(str(v) for v in self.spec.literals)
        self.edit.setPlaceholderText((f"e.g. {self._example}" if self._example else "") + extra)

    def set_example(self, text: str) -> None:
        if text and not self.spec.examples:
            self._example = text
            self._refresh_placeholder()

    def force_example(self, text: str) -> None:
        self._example = text
        self._refresh_placeholder()

    def parse(self, text: str) -> tuple[Any, str]:
        stripped = text.strip()
        if not stripped:
            return MISSING, "enter a value" + (f", e.g. {self._example}" if self._example else "")
        candidate: Any = float(stripped) if _BARE_NUMBER.match(stripped) else stripped
        try:
            self.spec.adapter.validate_python(candidate)
        except ValidationError as exc:
            return MISSING, error_message(exc)
        return candidate, ""

    def _on_text(self, text: str) -> None:
        if text == self._shown and not self._muted and text != "":
            restyle(self.edit, "")
            self.problem.emit("")
            return
        _, message = self.parse(text) if text.strip() else (MISSING, "")
        restyle(self.edit, "error" if message else ("default" if self._muted else ""))
        self.problem.emit(message)

    def commit(self) -> None:
        text = self.edit.text()
        if text == self._shown and not self._muted:
            return
        raw, message = self.parse(text)
        if message:
            if text.strip():
                self.problem.emit(message)
            else:
                self.edit.setText(self._shown)
                self.problem.emit("a value is required; the previous value was kept")
            return
        self.committed.emit(raw)

    def set_raw(self, raw: Any) -> None:
        text = value_text(raw)
        self._shown = text
        if text != self.edit.text():
            pos = self.edit.cursorPosition()
            self.edit.setText(text)
            self.edit.setCursorPosition(min(pos, len(text)))
        self._on_text(text)

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        restyle(self.edit, "default" if muted else "")

    def clear_value(self) -> None:
        self._shown = ""
        self.edit.setText("")
        restyle(self.edit, "")
        self.problem.emit("")

    def text(self) -> str:
        return self.edit.text()


class StrEditor(ScalarEditor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)

    def parse(self, text: str):
        return text, ""


class ListEditor(ScalarEditor):
    def parse(self, text: str):
        stripped = text.strip()
        if not stripped:
            return MISSING, "enter a comma-separated list"
        tokens = [t for t in re.split(r"[,\s]+", stripped.strip("[]")) if t]
        values = []
        for token in tokens:
            try:
                values.append(json.loads(token))
            except ValueError:
                values.append(token)
        try:
            self.spec.adapter.validate_python(values)
        except ValidationError as exc:
            return MISSING, error_message(exc)
        return values, ""

    def set_raw(self, raw: Any) -> None:
        if isinstance(raw, list | tuple):
            raw = ", ".join(value_text(v) for v in raw)
        super().set_raw(raw)


class JsonEditor(ScalarEditor):
    def parse(self, text: str):
        if not text.strip():
            return MISSING, "enter a value (JSON)"
        try:
            value = json.loads(text)
        except ValueError as exc:
            return MISSING, f"not valid JSON: {exc}"
        if self.spec.adapter is not None:
            try:
                self.spec.adapter.validate_python(value)
            except ValidationError as exc:
                return MISSING, error_message(exc)
        return value, ""

    def set_raw(self, raw: Any) -> None:
        if raw is not None and raw is not MISSING and not isinstance(raw, str):
            raw = json.dumps(raw)
        super().set_raw(raw)


class BoolEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.check = QCheckBox()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.check)
        layout.addStretch(1)
        self._loading = False
        self.check.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        if not self._loading:
            self.committed.emit(bool(checked))

    def set_raw(self, raw: Any) -> None:
        self._loading = True
        self.check.setChecked(bool(raw) if raw is not MISSING and raw is not None else False)
        self._loading = False

    def initial_raw(self) -> Any:
        return False


class LiteralEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.combo = _Combo()
        for value in spec.literals:
            self.combo.addItem(str(value), value)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)
        self._loading = False
        self.combo.currentIndexChanged.connect(self._on_index)

    def _on_index(self, index: int) -> None:
        if not self._loading and index >= 0:
            self.committed.emit(self.combo.itemData(index))

    def set_raw(self, raw: Any) -> None:
        self._loading = True
        index = self.combo.findData(raw)
        if index < 0 and raw not in (None, MISSING):
            self.combo.addItem(f"{raw} (invalid)", raw)
            index = self.combo.count() - 1
        self.combo.setCurrentIndex(index)
        self._loading = False

    def initial_raw(self) -> Any:
        return self.spec.literals[0] if self.spec.literals else MISSING

    def set_choices(self, items: list[tuple[str, Any]]) -> None:
        current = self.combo.currentData()
        self._loading = True
        self.combo.clear()
        for text, value in items:
            self.combo.addItem(text, value)
        self.spec.literals = tuple(v for _, v in items)
        index = self.combo.findData(current)
        self.combo.setCurrentIndex(index)
        self._loading = False


class NumberEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        integer = spec.kind == "int"
        self.spin: QSpinBox | QDoubleSpinBox = _IntSpin() if integer else _FloatSpin()
        big = 2**31 - 1
        lo = spec.minimum if spec.minimum is not None else (-big if integer else -1e300)
        hi = spec.maximum if spec.maximum is not None else (big if integer else 1e300)
        if integer:
            self.spin.setRange(int(lo), int(hi))
        else:
            self.spin.setDecimals(12)
            self.spin.setRange(lo, hi)
            self.spin.setSingleStep(0.1)
        self.spin.setKeyboardTracking(False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.spin, 1)
        self._loading = False
        self.spin.valueChanged.connect(self._on_value)

    def _on_value(self, value) -> None:
        if not self._loading:
            self.committed.emit(value)

    def set_raw(self, raw: Any) -> None:
        self._loading = True
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = 0.0
        if not math.isfinite(number):
            number = 0.0
        self.spin.setValue(int(number) if self.spec.kind == "int" else number)
        self._loading = False

    def initial_raw(self) -> Any:
        value = self.spin.value()
        if self.spin.minimum() > 0:
            value = self.spin.minimum()
        return value

    def set_muted(self, muted: bool) -> None:
        restyle(self.spin, "default" if muted else "")


class DictEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._inner_spec = spec.inner
        self.rows: list[tuple[QWidget, QLineEdit, QLineEdit]] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._rows_box = QVBoxLayout()
        self._rows_box.setSpacing(2)
        self._layout.addLayout(self._rows_box)
        self.add_button = QPushButton("Add entry")
        self.add_button.clicked.connect(lambda: self._add_row("", ""))
        self._layout.addWidget(self.add_button, 0, Qt.AlignmentFlag.AlignLeft)
        self._shown: dict = {}
        self._loading = False

    def _add_row(self, key: str, value: str) -> None:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        k, v = QLineEdit(key), QLineEdit(value)
        k.setPlaceholderText("key")
        example = self._inner_spec.examples[0] if self._inner_spec.examples else ""
        v.setPlaceholderText(f"e.g. {example}" if example else "value")
        remove = QToolButton()
        remove.setText("x")
        h.addWidget(k, 1)
        h.addWidget(v, 1)
        h.addWidget(remove)
        self._rows_box.addWidget(row)
        entry = (row, k, v)
        self.rows.append(entry)
        k.editingFinished.connect(self.commit)
        v.editingFinished.connect(self.commit)
        remove.clicked.connect(lambda: self._remove(entry))

    def _remove(self, entry) -> None:
        self.rows.remove(entry)
        entry[0].setParent(None)
        entry[0].deleteLater()
        self.commit()

    def collect(self) -> tuple[dict | None, str]:
        out: dict = {}
        for _, k, v in self.rows:
            key, text = k.text().strip(), v.text().strip()
            if not key and not text:
                continue
            if not key:
                return None, "every entry needs a key"
            if key in out:
                return None, f"duplicate key '{key}'"
            candidate: Any = float(text) if _BARE_NUMBER.match(text) else text
            if self._inner_spec.adapter is not None:
                try:
                    self._inner_spec.adapter.validate_python(candidate)
                except ValidationError as exc:
                    return None, f"{key}: {error_message(exc)}"
            out[key] = candidate
        return out, ""

    def commit(self) -> None:
        if self._loading:
            return
        value, message = self.collect()
        self.problem.emit(message)
        if value is not None and value != self._shown:
            self._shown = dict(value)
            self.committed.emit(value)

    def set_raw(self, raw: Any) -> None:
        mapping = raw if isinstance(raw, Mapping) else {}
        if dict(mapping) == self._shown and (self.rows or not mapping):
            return
        self._loading = True
        for row, _, _ in self.rows:
            row.setParent(None)
            row.deleteLater()
        self.rows.clear()
        self._shown = dict(mapping)
        for key, value in mapping.items():
            self._add_row(str(key), value_text(value))
        self._loading = False
        self.problem.emit("")

    def initial_raw(self) -> Any:
        return {}


class _PlotHolder(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setVisible(False)

    @property
    def plot(self):
        if self._plot is None:
            import pyqtgraph as pg

            self._plot = pg.PlotWidget()
            self._plot.setMinimumHeight(100)
            self._plot.setMaximumHeight(130)
            self._plot.setMouseEnabled(False, False)
            self._plot.setMenuEnabled(False)
            self._plot.hideButtons()
            self._plot.setBackground(None)
            self._layout.addWidget(self._plot)
        return self._plot


class CurveEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None, importable=True) -> None:
        super().__init__(parent)
        self.spec = spec
        self.curve: Curve | None = None
        self.info = QLabel("no curve")
        self.info.setProperty("role", "muted")
        self.info.setWordWrap(True)
        self.holder = _PlotHolder()
        self.import_button = QPushButton("Import...")
        self.import_button.setVisible(importable)
        self.import_button.clicked.connect(self.import_curve)
        self.table_button = QPushButton("Edit table...")
        self.table_button.setVisible(importable)
        self.table_button.clicked.connect(self.edit_table)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.info)
        layout.addWidget(self.holder)
        buttons = QHBoxLayout()
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.table_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    def import_curve(self) -> None:
        curve = run_import_dialog(self)
        if curve is not None:
            self.committed.emit(curve_to_raw(curve))

    def edit_table(self) -> None:
        from sequoia.gui.inspector.dialogs import CurveTableDialog

        dialog = CurveTableDialog(self.curve, self)
        if dialog.exec() and dialog.curve() is not None:
            self.committed.emit(curve_to_raw(dialog.curve()))

    def set_raw(self, raw: Any) -> None:
        curve = curve_from_raw(raw)
        self.curve = curve
        if curve is None:
            self.info.setText("no curve" if raw in (None, MISSING) else "invalid curve")
            self.holder.setVisible(False)
            return
        prov = provenance_text(curve)
        src = f", {prov.splitlines()[0]}" if prov else ""
        self.info.setText(
            f"{len(curve.x)} points; {curve.x_label} ({curve.x_unit}) -> "
            f"{curve.y_label} ({curve.y_unit}){src}"
        )
        self.info.setToolTip(prov or "")
        plot = self.holder.plot
        plot.clear()
        plot.plot(curve.x, curve.y, pen="#2a7ab0")
        for side in ("bottom", "left"):
            plot.getAxis(side).enableAutoSIPrefix(False)
        plot.getAxis("left").setWidth(56)
        plot.setLabel("bottom", _axis_text(curve.x_label, curve.x_unit))
        plot.setLabel("left", _axis_text(curve.y_label, curve.y_unit))
        self.holder.setVisible(True)

    def clear_value(self) -> None:
        self.set_raw(None)


class SurfaceEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.info = QLabel("no surface")
        self.info.setProperty("role", "muted")
        self.info.setWordWrap(True)
        self.holder = _PlotHolder()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.info)
        layout.addWidget(self.holder)

    def set_raw(self, raw: Any) -> None:
        if not isinstance(raw, Surface):
            if isinstance(raw, dict) and "z" in raw:
                try:
                    raw = Surface.model_validate(raw)
                except ValidationError:
                    raw = None
            else:
                raw = None
        if raw is None:
            self.info.setText("no surface")
            self.holder.setVisible(False)
            return
        import numpy as np
        import pyqtgraph as pg

        self.info.setText(
            f"{len(raw.x)} x {len(raw.y)} grid; {raw.x_label} ({raw.x_unit}), "
            f"{raw.y_label} ({raw.y_unit}) -> {raw.z_label} ({raw.z_unit})"
        )
        plot = self.holder.plot
        plot.clear()
        image = pg.ImageItem(np.asarray(raw.z, dtype=float))
        plot.addItem(image)
        self.holder.setVisible(True)


class FileRefEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel("")
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

    def set_raw(self, raw: Any) -> None:
        if isinstance(raw, FileRef):
            raw = raw.file
        elif isinstance(raw, dict):
            raw = raw.get("file", "")
        self.label.setText(str(raw) if raw not in (None, MISSING) else "no file")


class CurveOrValueEditor(Editor):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.mode = _Combo()
        self.mode.addItem("constant", "constant")
        self.mode.addItem("curve", "curve")
        value_spec = spec.value
        self.value_editor: Editor = (
            ScalarEditor(value_spec) if value_spec.kind == "scalar" else NumberEditor(value_spec)
        )
        self.curve_editor = CurveEditor(FieldSpec("curve"))
        self.stack = QStackedWidget()
        self.stack.addWidget(self.value_editor)
        self.stack.addWidget(self.curve_editor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mode)
        layout.addWidget(self.stack)
        self.mode.currentIndexChanged.connect(self._on_mode)
        for inner in (self.value_editor, self.curve_editor):
            inner.committed.connect(self.committed)
            inner.problem.connect(self.problem)
        self._loading = False

    def _on_mode(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if self._loading:
            return
        if index == 1:
            if self.curve_editor.curve is not None:
                self.committed.emit(curve_to_raw(self.curve_editor.curve))
        elif isinstance(self.value_editor, ScalarEditor):
            self.value_editor.commit()
        else:
            self.committed.emit(self.value_editor.spin.value())

    def set_raw(self, raw: Any) -> None:
        self._loading = True
        is_curve = curve_from_raw(raw) is not None
        self.mode.setCurrentIndex(1 if is_curve else 0)
        self.stack.setCurrentIndex(1 if is_curve else 0)
        if is_curve:
            self.curve_editor.set_raw(raw)
        elif raw is not None and raw is not MISSING:
            self.value_editor.set_raw(raw)
        self._loading = False

    def set_muted(self, muted: bool) -> None:
        self.value_editor.set_muted(muted)

    def set_example(self, text: str) -> None:
        self.value_editor.set_example(text)

    def initial_raw(self) -> Any:
        return self.value_editor.initial_raw()

    def clear_value(self) -> None:
        self.value_editor.clear_value()
        self.curve_editor.clear_value()


def make_editor(spec: FieldSpec) -> Editor:
    kind = spec.kind
    if kind == "scalar":
        return ScalarEditor(spec)
    if kind == "curve_or_value":
        return CurveOrValueEditor(spec)
    if kind == "curve":
        return CurveEditor(spec)
    if kind == "surface":
        return SurfaceEditor(spec)
    if kind == "fileref":
        return FileRefEditor(spec)
    if kind == "literal":
        return LiteralEditor(spec)
    if kind == "bool":
        return BoolEditor(spec)
    if kind in ("int", "float"):
        return NumberEditor(spec)
    if kind == "str":
        return StrEditor(spec)
    if kind == "list":
        return ListEditor(spec)
    if kind == "dict":
        return DictEditor(spec)
    return JsonEditor(spec)


class FieldRow(QWidget):
    def __init__(
        self,
        name: str,
        spec: FieldSpec,
        editor: Editor,
        commit: Callable[[str, Any], None],
        *,
        description: str = "",
        track_default: bool = True,
        can_unset: bool = True,
        override_mode: bool = False,
        label: str | None = None,
        provenance_editable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self._provenance_editable = provenance_editable
        self._shown_raw: Any = MISSING
        self._explicit: Any = MISSING
        self.spec = spec
        self.editor = editor
        self._commit = commit
        self._track_default = track_default
        self._can_unset = can_unset
        self._override_mode = override_mode
        self._fallback: Any = MISSING
        self._key: str | None = None
        self._committing = False
        self.is_set = False

        self.label = QLabel(label or name)
        if description:
            self.label.setToolTip(description)
            editor.setToolTip(description)
        self.optional_box = QCheckBox("set")
        self.optional_box.setVisible(spec.optional and can_unset)
        self.reset_button = QToolButton()
        self.reset_button.setText("reset")
        self.reset_button.setAutoRaise(True)
        self.reset_button.setVisible(False)
        self.badge = QLabel("")
        self.badge.setProperty("role", "badge")
        self.badge.setVisible(False)
        self.prov_button = QToolButton()
        self.prov_button.setText("provenance...")
        self.prov_button.setAutoRaise(True)
        self.prov_button.setVisible(False)
        self.error = QLabel("")
        self.error.setProperty("role", "error")
        self.error.setWordWrap(True)
        self.error.setVisible(False)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.label)
        head.addWidget(self.badge)
        head.addWidget(self.prov_button)
        head.addStretch(1)
        head.addWidget(self.optional_box)
        head.addWidget(self.reset_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        layout.addLayout(head)
        layout.addWidget(editor)
        layout.addWidget(self.error)

        editor.committed.connect(self._on_committed)
        editor.problem.connect(self._show_error)
        self.optional_box.clicked.connect(self._on_optional)
        self.reset_button.clicked.connect(self._on_reset)
        self.prov_button.clicked.connect(self._edit_provenance)

    def _edit_provenance(self) -> None:
        from sequoia.gui.inspector.dialogs import ask_provenance
        from sequoia.library.editing import get_provenance, with_provenance

        raw = self._shown_raw
        accepted, prov = ask_provenance(self, get_provenance(raw))
        if not accepted:
            return
        try:
            new = with_provenance(raw, prov)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._on_committed(new)

    def _show_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.setVisible(bool(message))

    def _do_commit(self, value: Any) -> bool:
        self._committing = True
        try:
            self._commit(self.name, value)
        except ValueError as exc:
            self._committing = False
            self._key = None
            self._render(self._explicit)
            self._show_error(str(exc))
            return False
        finally:
            self._committing = False
        self._show_error("")
        return True

    def _on_committed(self, value: Any) -> None:
        if self._do_commit(value):
            self._key = None
            self._render(value)

    def _on_optional(self, checked: bool) -> None:
        if checked:
            self.editor.setEnabled(True)
            initial = self.editor.initial_raw()
            if initial is not MISSING:
                self._on_committed(initial)
        elif self._do_commit(None):
            self._key = None
            self._render(MISSING)

    def _on_reset(self) -> None:
        if self._do_commit(None):
            self._key = None
            self._render(MISSING)

    def refresh(self, explicit: Any = MISSING, fallback: Any = MISSING) -> None:
        if self._committing:
            return
        self._fallback = fallback
        shown = explicit if explicit not in (MISSING, None) else fallback
        key = f"{explicit is not MISSING}|{value_key(shown)}"
        if key == self._key:
            return
        self._key = key
        self._render(explicit)

    def _render(self, explicit: Any) -> None:
        fallback = self._fallback
        self._explicit = explicit
        is_set = explicit is not MISSING and explicit is not None
        shown = explicit if is_set else fallback
        present = shown is not None and shown is not MISSING
        self.is_set = is_set
        self._shown_raw = shown if present else MISSING
        if isinstance(fallback, Scalar):
            self.editor.set_example(value_text(fallback))
        if self.spec.optional:
            can = self._can_unset
            if self._override_mode:
                can = fallback is None or fallback is MISSING
            self.optional_box.setVisible(can)
            self.optional_box.setChecked(present)
            self.editor.setEnabled(present or not can)
        if present:
            self.editor.set_raw(shown)
        else:
            self.editor.clear_value()
        self.editor.set_muted(self._track_default and not is_set)
        font = self.label.font()
        font.setBold(self._track_default and is_set)
        self.label.setFont(font)
        self.reset_button.setVisible(self._track_default and is_set)
        prov = provenance_text(shown) if present else None
        self.badge.setVisible(prov is not None)
        if self._provenance_editable:
            from sequoia.library.editing import accepts_provenance

            self.prov_button.setVisible(present and accepts_provenance(shown))
        if prov:
            self.badge.setText(prov.splitlines()[0].removeprefix("source: "))
            self.badge.setToolTip(prov)
        self._show_error("")


class ModelForm(QWidget):
    def __init__(
        self,
        model_cls: type[BaseModel],
        commit: Callable[[str, Any], None],
        *,
        track_defaults: bool = True,
        skip: tuple[str, ...] = (),
        editors: Mapping[str, Callable[[FieldSpec], Editor]] | None = None,
        specs: Mapping[str, FieldSpec] | None = None,
        override_mode: bool = False,
        labels: Mapping[str, str] | None = None,
        descriptions: Mapping[str, str] | None = None,
        required: tuple[str, ...] = (),
        provenance_editable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.model_cls = model_cls
        self.rows: dict[str, FieldRow] = {}
        self.defaults: dict[str, Any] = {}
        self.setStyleSheet(STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for name, info in model_cls.model_fields.items():
            if name in skip:
                continue
            spec = (specs or {}).get(name) or analyze(info.annotation, info.metadata)
            factory = (editors or {}).get(name, make_editor)
            default = info.get_default(call_default_factory=True)
            self.defaults[name] = MISSING if default is PydanticUndefined else default
            row = FieldRow(
                name,
                spec,
                factory(spec),
                commit,
                description=(descriptions or {}).get(name) or info.description or "",
                track_default=track_defaults,
                override_mode=override_mode,
                can_unset=name not in required,
                label=(labels or {}).get(name),
                provenance_editable=provenance_editable,
            )
            layout.addWidget(row)
            self.rows[name] = row

    def refresh(self, explicit: Mapping[str, Any], fallback: Mapping[str, Any] | None = None):
        for name, row in self.rows.items():
            fb = (fallback or self.defaults).get(name, self.defaults.get(name, MISSING))
            row.refresh(explicit.get(name, MISSING), fb)


class CollapsibleGroup(QWidget):
    def __init__(self, title: str, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self.toggle = QToolButton()
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setText(title)
        self.content = content
        content.setVisible(False)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line)
        layout.addWidget(self.toggle)
        layout.addWidget(content)
        self.toggle.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        self.content.setVisible(checked)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()

    def set_title(self, title: str) -> None:
        self.toggle.setText(title)
