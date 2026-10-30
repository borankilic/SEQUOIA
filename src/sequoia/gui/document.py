import copy
import math
import re
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import Component, PortDirection, PortKind, PortSpec, get_component
from sequoia.data import Duration
from sequoia.errors import SequoiaError, format_validation_error
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, save_setup
from sequoia.setup.model import Link, Setup
from sequoia.setup.resolve import instantiate
from sequoia.setup.validate import FIBER_TYPE

LinkKind = Literal["free_space", "fiber", "cable"]

_KIND_FOR_PORT = {
    PortKind.FREE_SPACE: "free_space",
    PortKind.FIBER: "fiber",
    PortKind.ELECTRICAL: "cable",
}
_LINK_ADAPTER = TypeAdapter(Link)
_ENDPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_MOVE_ID = 1

MAX_ABS_CELLS = 1e5
UNDO_LIMIT = 200
ANALYSIS_TYPES = ("coincidences", "heralded_g2", "g2", "histogram")
_NUMBER_PREFIX = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?inf)(?:\s|$)")
_DURATION = TypeAdapter(Duration)

DEFAULT_FREE_SPACE_LENGTH = "100 mm"
DEFAULT_FIBER_LENGTH = "1 m"
DEFAULT_CABLE_DELAY = "0 ns"


class _SetupCommand(QUndoCommand):
    def __init__(self, document, text, before, after, merge_key=None):
        super().__init__(text)
        self._document = document
        self._before = before
        self.after = after
        self.merge_key = merge_key

    def id(self) -> int:
        return _MOVE_ID if self.merge_key is not None else -1

    def redo(self) -> None:
        self._document._apply(self.after)

    def undo(self) -> None:
        self._document._apply(self._before)

    def mergeWith(self, other) -> bool:
        if not isinstance(other, _SetupCommand) or other.merge_key != self.merge_key:
            return False
        self.after = other.after
        return True


def _validation_message(exc: ValidationError) -> str:
    return format_validation_error(exc)


def _nonfinite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, str):
        match = _NUMBER_PREFIX.match(value)
        if match is None:
            return False
        try:
            return not math.isfinite(float(match.group(1)))
        except ValueError:
            return False
    if isinstance(value, dict):
        if "value" in value and isinstance(value["value"], int | float):
            return _nonfinite(float(value["value"]))
        return any(_nonfinite(v) for v in value.values())
    if isinstance(value, list | tuple):
        return any(_nonfinite(v) for v in value)
    return False


def _inf_allowed(library: Library, part_ref: str, mapping: str, name: str) -> bool:
    try:
        cls = get_component(library.load_part(part_ref).type)
        model = cls.Settings if mapping == "settings" else cls.Params
        default = model.model_fields[name].default
    except Exception:
        return False
    value = getattr(default, "value", default)
    return isinstance(value, int | float) and value == math.inf


def check_physical(setup: Setup, library: Library | None = None) -> None:
    """Raise ``ValueError`` for values that make no physical or drawing sense.

    Positions must be finite and within ``MAX_ABS_CELLS``; rotations finite; link lengths, delays,
    losses and the grid cell finite. Setting and override values may be infinite only where the
    component's own default is (an iris diameter).
    """
    for inst in setup.instances:
        for axis, value in zip("xy", inst.position, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"instance '{inst.id}': position {axis} must be a finite number")
            if abs(value) > MAX_ABS_CELLS:
                raise ValueError(
                    f"instance '{inst.id}': position {axis}={value:g} cells is outside "
                    f"+-{MAX_ABS_CELLS:g} cells"
                )
        if not math.isfinite(inst.rotation):
            raise ValueError(f"instance '{inst.id}': rotation must be a finite number")
    if not math.isfinite(setup.grid_cell.si):
        raise ValueError("the grid cell must be finite")
    for link in setup.links:
        for name in ("length", "delay"):
            scalar = getattr(link, name, None)
            if scalar is not None and not math.isfinite(scalar.si):
                raise ValueError(f"link '{link.id}': {name} must be finite")
        loss = getattr(link, "connector_loss_db", 0.0)
        if not math.isfinite(loss):
            raise ValueError(f"link '{link.id}': connector_loss_db must be finite")
        pol = getattr(link, "polarization", None)
        if pol is not None and pol.rotation_vector is not None:
            if not all(math.isfinite(v) for v in pol.rotation_vector):
                raise ValueError(f"link '{link.id}': polarization rotation must be finite")
    for placed in (*setup.instances, *setup.probes):
        for mapping in ("settings", "overrides"):
            for name, value in getattr(placed, mapping).items():
                if not _nonfinite(value):
                    continue
                if library is not None and _inf_allowed(library, placed.part, mapping, name):
                    continue
                raise ValueError(f"'{placed.id}': {mapping[:-1]} '{name}' must be a finite number")


def check_unique_ids(setup: Setup) -> None:
    for what, items in (
        ("instance", setup.instances),
        ("link", setup.links),
        ("probe", setup.probes),
    ):
        seen: set[str] = set()
        for item in items:
            if item.id in seen:
                raise ValueError(f"the setup defines the {what} id '{item.id}' more than once")
            seen.add(item.id)
    both = {i.id for i in setup.instances} & {p.id for p in setup.probes}
    if both:
        raise ValueError(f"'{sorted(both)[0]}' is used by both an instance and a probe")


def check_analysis(
    spec: Any, index: int | None = None, *, default_window: Any = None, labels=()
) -> None:
    """Raise ``ValueError`` when an analysis spec cannot be run (type, channels, numbers).
    ``default_window`` (the measurement's coincidence window) stands in for a missing window;
    ``labels`` are detector ids the measurement maps to channels."""
    where = "analysis" if index is None else f"analysis #{index + 1}"
    if not isinstance(spec, dict):
        raise ValueError(f"{where} must be a mapping")
    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where} needs a non-empty name")
    where = f"analysis '{name}'"
    kind = spec.get("type")
    if kind not in ANALYSIS_TYPES:
        raise ValueError(
            f"{where}: unknown type {kind!r}; choose one of {', '.join(ANALYSIS_TYPES)}"
        )

    def duration(key: str, required: bool = True) -> float | None:
        if spec.get(key) is None:
            if required:
                raise ValueError(f"{where}: '{key}' is required (a duration such as '1 ns')")
            return None
        value = spec[key]
        if isinstance(value, bool):
            raise ValueError(f"{where}: '{key}' must be a duration such as '1 ns'")
        try:
            if isinstance(value, int | float):
                seconds = float(value)
            else:
                seconds = _DURATION.validate_python(value).si
        except ValidationError as exc:
            raise ValueError(f"{where}: '{key}': {_validation_message(exc)}") from exc
        if not math.isfinite(seconds):
            raise ValueError(f"{where}: '{key}' must be finite")
        return seconds

    def channel(key: str, value: Any) -> str:
        if isinstance(value, str) and value in labels:
            return value
        if not isinstance(value, str) or not _ENDPOINT.match(value):
            raise ValueError(f"{where}: {key} must be a channel such as 'tagger.ch1'")
        return value

    if kind == "heralded_g2":
        chans = [channel(k, spec.get(k)) for k in ("herald", "a", "b")]
    else:
        raw = spec.get("channels")
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{where}: 'channels' must list exactly two channels")
        chans = [channel("channels", c) for c in raw]
    if len(set(chans)) != len(chans):
        raise ValueError(f"{where}: the channels must all differ")
    if kind in ("coincidences", "heralded_g2"):
        if spec.get("window") is None and default_window is not None:
            pass
        elif duration("window") <= 0:
            raise ValueError(f"{where}: window must be positive")
    else:
        if duration("range") <= 0:
            raise ValueError(f"{where}: range must be positive")
        bins = spec.get("bins", 200)
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError(f"{where}: bins must be a positive integer")
    center = spec.get("center", "auto")
    if not (isinstance(center, str) and center.strip().lower() == "auto"):
        duration("center", required=False)


def check_analyses(analyses: list, measurement=None, labels=()) -> None:
    names: set[str] = set()
    window = measurement.coincidence_window if measurement is not None else None
    for index, spec in enumerate(analyses):
        check_analysis(spec, index, default_window=window, labels=labels)
        if spec["name"] in names:
            raise ValueError(f"analysis name '{spec['name']}' is used twice")
        names.add(spec["name"])


def _rename_measurement(m: dict, old: str, new: str) -> None:
    prefix = f"{old}."

    def ref(v: Any) -> Any:
        if v == old:
            return new
        if isinstance(v, str) and v.startswith(prefix):
            return new + v[len(old) :]
        return v

    m["channels"] = {ref(k): ref(v) for k, v in (m.get("channels") or {}).items()}
    clock = m.get("pulse_clock")
    if isinstance(clock, dict) and clock.get("source") == old:
        clock["source"] = new
    for cond in m.get("conditions") or []:
        cond["herald"] = [ref(v) for v in cond.get("herald", [])]
        cond["veto"] = [ref(v) for v in cond.get("veto", [])]


class SetupDocument(QObject):
    changed = Signal()
    structure_changed = Signal()
    selection_requested = Signal(str)
    dirty_changed = Signal(bool)
    path_changed = Signal()
    ids_renamed = Signal(object)

    def __init__(
        self,
        library: Library,
        setup: Setup | None = None,
        path: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.uid = uuid.uuid4().hex
        self.revision = 0
        self.changed.connect(self._bump_revision)
        self.library = library
        self._setup = setup if setup is not None else Setup()
        self.path = Path(path) if path is not None else None
        self._components: dict[tuple[str, str], tuple[Any, Component | None, str | None]] = {}
        self._macro_stack: list[Setup] = []
        self.new_fiber_ref: str | None = None
        self._renames: list[tuple[str, str]] = []
        self._structure = self._structure_key()
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(UNDO_LIMIT)
        self.undo_stack.indexChanged.connect(self._on_index_changed)
        self.undo_stack.cleanChanged.connect(self._on_clean_changed)

    @classmethod
    def open(cls, path: Path | str, library: Library) -> "SetupDocument":
        path = Path(path)
        setup = load_setup(path)
        check_unique_ids(setup)
        check_physical(setup, library)
        return cls(library, setup, path)

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("this document has no file path yet; give one to save()")
        save_setup(self._setup, target)
        moved = target != self.path
        self.path = target
        self.undo_stack.setClean()
        if moved:
            self.path_changed.emit()

    def _bump_revision(self) -> None:
        self.revision += 1

    def key(self) -> tuple:
        return (self.uid, self.revision, id(self.library))

    @property
    def setup(self) -> Setup:
        return self._setup

    @property
    def dirty(self) -> bool:
        return not self.undo_stack.isClean()

    def _structure_key(self) -> tuple:
        s = self._setup
        return (
            tuple((i.id, i.part) for i in s.instances),
            tuple((link.id, link.src, link.dst) for link in s.links),
            tuple((p.id, p.part, p.link) for p in s.probes),
        )

    def _apply(self, setup: Setup) -> None:
        self._setup = setup

    def _on_clean_changed(self, clean: bool) -> None:
        self.dirty_changed.emit(not clean)

    def _on_index_changed(self) -> None:
        key = self._structure_key()
        structural = key != self._structure
        self._structure = key
        if self._renames:
            renames, self._renames = dict(self._renames), []
            self.ids_renamed.emit(renames)
        self.changed.emit()
        if structural:
            self.structure_changed.emit()

    def _raw(self) -> dict[str, Any]:
        return self._setup.model_dump(by_alias=True)

    def _share(self, new: Setup) -> None:
        old = self._setup
        for field in ("instances", "links", "probes"):
            previous = {x.id: x for x in getattr(old, field)}
            items = getattr(new, field)
            for i, item in enumerate(items):
                same = previous.get(item.id)
                if same is not None and same == item:
                    items[i] = same

    def _commit(self, text: str, raw: dict[str, Any], merge_key=None) -> bool:
        try:
            new = Setup.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(_validation_message(exc)) from exc
        check_physical(new, self.library)
        if new == self._setup:
            return False
        self._share(new)
        if self._macro_stack:
            self._setup = new
        else:
            self.undo_stack.push(_SetupCommand(self, text, self._setup, new, merge_key))
        return True

    @contextmanager
    def macro(self, text: str, merge_key: Any = None) -> Iterator[None]:
        snapshot = self._setup
        self._macro_stack.append(snapshot)
        try:
            yield
        except BaseException:
            self._macro_stack.pop()
            self._setup = snapshot
            raise
        self._macro_stack.pop()
        if not self._macro_stack and self._setup != snapshot:
            after = self._setup
            self._setup = snapshot
            self.undo_stack.push(_SetupCommand(self, text, snapshot, after, merge_key))

    def _try_component(self, placed) -> tuple[Component | None, str | None]:
        try:
            return instantiate(self.library, placed), None
        except SequoiaError as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _lookup(self, instance_id: str):
        for inst in self._setup.instances:
            if inst.id == instance_id:
                return "i", inst
        for probe in self._setup.probes:
            if probe.id == instance_id:
                return "p", probe
        return None, None

    def _cached(self, instance_id: str):
        tag, placed = self._lookup(instance_id)
        if placed is None:
            return None, None
        key = (tag, instance_id)
        signature = (placed.part, placed.settings, placed.overrides)
        entry = self._components.get(key)
        if entry is None or entry[0] != signature:
            component, error = self._try_component(placed)
            entry = (copy.deepcopy(signature), component, error)
            self._components[key] = entry
        return entry[1], entry[2]

    def component(self, instance_id: str) -> Component | None:
        return self._cached(instance_id)[0]

    def component_error(self, instance_id: str) -> str | None:
        return self._cached(instance_id)[1]

    def invalidate_components(self) -> None:
        self._components.clear()
        self.changed.emit()

    def ports(self, instance_id: str) -> tuple[PortSpec, ...]:
        component = self.component(instance_id)
        if component is None:
            return ()
        try:
            return tuple(component.ports())
        except Exception:
            return ()

    def _all_ids(self) -> set[str]:
        s = self._setup
        return {x.id for x in (*s.instances, *s.links, *s.probes)}

    def new_id(self, base: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", base.strip()) or "item"
        if base[0].isdigit():
            base = f"_{base}"
        taken = self._all_ids()
        if base not in taken:
            return base
        n = 2
        while f"{base}_{n}" in taken:
            n += 1
        return f"{base}_{n}"

    def _find(self, raw_list: list[dict], item_id: str, what: str) -> dict:
        for item in raw_list:
            if item["id"] == item_id:
                return item
        raise ValueError(f"there is no {what} '{item_id}'")

    def _find_placed(self, raw: dict, item_id: str) -> dict:
        for item in (*raw["instances"], *raw["probes"]):
            if item["id"] == item_id:
                return item
        raise ValueError(f"there is no instance or probe '{item_id}'")

    def _check_part(self, part_ref: str) -> str:
        try:
            part = self.library.load_part(part_ref)
        except SequoiaError as exc:
            raise ValueError(str(exc)) from exc
        return get_component(part.type).category

    def _fresh_id(self, requested: str | None, base: str) -> str:
        if requested is None:
            return self.new_id(base)
        if requested in self._all_ids():
            raise ValueError(f"the id '{requested}' is already used")
        return requested

    def add_instance(
        self,
        part_ref: str,
        position: Sequence[float],
        rotation: float = 0.0,
        instance_id: str | None = None,
    ) -> str:
        category = self._check_part(part_ref)
        if category == "probes":
            raise ValueError(f"'{part_ref}' is a probe; probes attach to links (use add_probe)")
        new_id = self._fresh_id(instance_id, part_ref.rsplit("/", 1)[-1])
        raw = self._raw()
        raw["instances"].append(
            {"id": new_id, "part": part_ref, "position": list(position), "rotation": rotation}
        )
        self._commit(f"Add {new_id}", raw)
        return new_id

    def remove_instance(self, instance_id: str) -> None:
        raw = self._raw()
        self._find(raw["instances"], instance_id, "instance")
        raw["instances"] = [i for i in raw["instances"] if i["id"] != instance_id]
        dead_links = {
            link["id"]
            for link in raw["links"]
            if instance_id in (link["from"].split(".")[0], link["to"].split(".")[0])
        }
        raw["links"] = [link for link in raw["links"] if link["id"] not in dead_links]
        raw["probes"] = [p for p in raw["probes"] if p["link"] not in dead_links]
        self._commit(f"Remove {instance_id}", raw)

    def _edit_instance(self, instance_id: str, text: str, merge_key=None, **fields) -> None:
        raw = self._raw()
        self._find(raw["instances"], instance_id, "instance").update(fields)
        self._commit(text, raw, merge_key)

    def move_instance(
        self, instance_id: str, position: Sequence[float], gesture: Any = None
    ) -> None:
        self._edit_instance(
            instance_id,
            f"Move {instance_id}",
            merge_key=("move", instance_id, gesture),
            position=list(position),
        )

    def rotate_instance(self, instance_id: str, rotation_deg: float) -> None:
        self._edit_instance(instance_id, f"Rotate {instance_id}", rotation=rotation_deg)

    def flip_instance(self, instance_id: str) -> None:
        raw = self._raw()
        item = self._find(raw["instances"], instance_id, "instance")
        item["flipped"] = not item["flipped"]
        self._commit(f"Flip {instance_id}", raw)

    def _set_mapping(self, field: str, instance_id: str, name: str, value: Any) -> None:
        raw = self._raw()
        item = self._find_placed(raw, instance_id)
        if value is None:
            item[field].pop(name, None)
        else:
            item[field][name] = copy.deepcopy(value)
        new = self._validated(raw)
        placed = next(p for p in (*new.instances, *new.probes) if p.id == instance_id)
        if self.component(instance_id) is not None:
            _, error = self._try_component(placed)
            if error is not None:
                raise ValueError(error)
        self._commit(f"Set {instance_id}.{name}", raw)

    def _validated(self, raw: dict) -> Setup:
        try:
            return Setup.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(_validation_message(exc)) from exc

    def set_setting(self, instance_id: str, name: str, value: Any) -> None:
        self._set_mapping("settings", instance_id, name, value)

    def set_override(self, instance_id: str, name: str, value: Any) -> None:
        self._set_mapping("overrides", instance_id, name, value)

    def _check_endpoint_syntax(self, *endpoints: str) -> None:
        for endpoint in endpoints:
            if not _ENDPOINT.match(endpoint):
                raise ValueError(f"'{endpoint}' is not an 'instance.port' endpoint")

    def normalize_connection(self, a: str, b: str) -> tuple[str, str]:
        port_a, port_b = self._port_of(a), self._port_of(b)
        if (
            port_a is not None
            and port_b is not None
            and port_a.direction == PortDirection.IN
            and port_b.direction == PortDirection.OUT
        ):
            return b, a
        return a, b

    def _port_of(self, endpoint: str) -> PortSpec | None:
        if not _ENDPOINT.match(endpoint):
            return None
        inst_id, port_name = endpoint.split(".")
        for port in self.ports(inst_id):
            if port.name == port_name:
                return port
        return None

    def _link_using(self, endpoint: str) -> str | None:
        for link in self._setup.links:
            if endpoint in (link.src, link.dst):
                return link.id
        return None

    def _would_cycle(self, src_inst: str, dst_inst: str) -> bool:
        successors: dict[str, set[str]] = {}
        for link in self._setup.links:
            successors.setdefault(link.src.split(".")[0], set()).add(link.dst.split(".")[0])
        seen, stack = set(), [dst_inst]
        while stack:
            node = stack.pop()
            if node == src_inst:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(successors.get(node, ()))
        return False

    def can_connect(self, src: str, dst: str) -> tuple[LinkKind | None, str]:
        for endpoint in (src, dst):
            if not _ENDPOINT.match(endpoint):
                return None, f"'{endpoint}' is not an 'instance.port' endpoint"
        if src == dst:
            return None, f"cannot connect port '{src}' to itself"
        ports = {}
        for endpoint in (src, dst):
            inst_id, port_name = endpoint.split(".")
            tag, _ = self._lookup(inst_id)
            if tag != "i":
                return None, f"there is no instance '{inst_id}'"
            if self.component(inst_id) is None:
                return None, f"instance '{inst_id}' is invalid: {self.component_error(inst_id)}"
            port = self._port_of(endpoint)
            if port is None:
                names = ", ".join(p.name for p in self.ports(inst_id)) or "none"
                return None, f"instance '{inst_id}' has no port '{port_name}' (ports: {names})"
            ports[endpoint] = port
        p_src, p_dst = ports[src], ports[dst]
        if p_src.kind != p_dst.kind:
            hint = ""
            if {p_src.kind, p_dst.kind} == {PortKind.FREE_SPACE, PortKind.FIBER}:
                hint = "; a beam needs a fiber coupler to enter or leave a fiber"
            return None, f"'{src}' is a {p_src.kind} port but '{dst}' is a {p_dst.kind} port{hint}"
        if p_src.direction == p_dst.direction:
            return None, (
                f"'{src}' and '{dst}' are both {p_src.direction} ports; "
                "a link joins an output port to an input port"
            )
        out, inp = (src, dst) if p_src.direction == PortDirection.OUT else (dst, src)
        for endpoint in (out, inp):
            used = self._link_using(endpoint)
            if used is not None:
                return None, f"port '{endpoint}' is already connected by link '{used}'"
        out_inst, in_inst = out.split(".")[0], inp.split(".")[0]
        if out_inst == in_inst:
            return None, f"cannot connect instance '{out_inst}' to itself"
        if self._would_cycle(out_inst, in_inst):
            return None, f"linking '{out}' to '{inp}' would form a cycle"
        return _KIND_FOR_PORT[p_src.kind], ""

    def _fiber_kind(self, ref: str) -> str | None:
        try:
            params = self.library.load_part(ref).params
        except SequoiaError:
            return None
        return getattr(params, "kind", None)

    def fiber_parts_by_kind(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"smf": [], "pm": [], "mmf": []}
        for ref in self.fiber_parts():
            kind = self._fiber_kind(ref)
            if kind in out:
                out[kind].append(ref)
        return out

    def convert_link_to_fiber(self, link_id: str, fiber: str | None = None) -> str:
        """Replace a free-space link by launch coupler, fiber and receive coupler.

        Returns the id of the new fiber link."""
        raw = self._raw()
        link = self._find(raw["links"], link_id, "link")
        if link["kind"] != "free_space":
            raise ValueError(f"link '{link_id}' is not a free-space link")
        couplers = [i.ref for i in self.library.index() if i.type == "FiberCoupler"]
        if not couplers:
            raise ValueError("the library has no FiberCoupler part to launch into a fiber")
        coupler = next((c for c in couplers if c.endswith("810")), couplers[0])
        fiber = fiber or next(
            iter(self.fiber_parts_by_kind().get("smf") or self.fiber_parts()), None
        )
        if fiber is None:
            raise ValueError("the library has no fiber part")
        src_inst, dst_inst = link["from"].split(".")[0], link["to"].split(".")[0]
        positions = {i["id"]: i["position"] for i in raw["instances"]}
        a, b = positions[src_inst], positions[dst_inst]
        launch_at = [round((a[0] * 2 + b[0]) / 3 * 2) / 2, round((a[1] * 2 + b[1]) / 3 * 2) / 2]
        recv_at = [round((a[0] + b[0] * 2) / 3 * 2) / 2, round((a[1] + b[1] * 2) / 3 * 2) / 2]
        length = link.get("length", "1 m")
        with self.macro(f"Fiber-couple {link_id}"):
            launch = self.add_instance(coupler, launch_at)
            recv = self.add_instance(coupler, recv_at)
            self.set_setting(recv, "mode", "to_free")
            free_in = next(
                p.name
                for p in self.ports(launch)
                if p.kind == PortKind.FREE_SPACE and p.direction == PortDirection.IN
            )
            free_out = next(
                p.name
                for p in self.ports(recv)
                if p.kind == PortKind.FREE_SPACE and p.direction == PortDirection.OUT
            )
            fib_out = next(
                p.name
                for p in self.ports(launch)
                if p.kind == PortKind.FIBER and p.direction == PortDirection.OUT
            )
            fib_in = next(
                p.name
                for p in self.ports(recv)
                if p.kind == PortKind.FIBER and p.direction == PortDirection.IN
            )
            medium = link.get("medium", "air")
            original_src, original_dst = link["from"], link["to"]
            self.remove_link(link_id)
            self.add_link(
                "free_space", original_src, f"{launch}.{free_in}", length="50 mm", medium=medium
            )
            fiber_link = self.add_link(
                "fiber",
                f"{launch}.{fib_out}",
                f"{recv}.{fib_in}",
                fiber=fiber,
                length=length if isinstance(length, str) else "1 m",
            )
            self.add_link(
                "free_space", f"{recv}.{free_out}", original_dst, length="50 mm", medium=medium
            )
        return fiber_link

    def convert_link_to_air(self, link_id: str) -> str:
        """Inverse of ``convert_link_to_fiber`` for a fiber link between two couplers."""
        raw = self._raw()
        fiber = self._find(raw["links"], link_id, "link")
        if fiber["kind"] != "fiber":
            raise ValueError(f"link '{link_id}' is not a fiber link")
        launch = fiber["from"].split(".")[0]
        recv = fiber["to"].split(".")[0]
        kinds = {i["id"]: self.component(i["id"]) for i in raw["instances"]}
        for inst in (launch, recv):
            if type(kinds.get(inst)).__name__ != "FiberCoupler":
                raise ValueError(
                    "only fibers between two fiber couplers can be turned back into an air link"
                )
        upstream = [
            link
            for link in raw["links"]
            if link["kind"] == "free_space" and link["to"].split(".")[0] == launch
        ]
        downstream = [
            link
            for link in raw["links"]
            if link["kind"] == "free_space" and link["from"].split(".")[0] == recv
        ]
        if len(upstream) != 1 or len(downstream) != 1:
            raise ValueError("the couplers of this fiber are not connected in a simple chain")
        up, down = upstream[0], downstream[0]
        for inst in (launch, recv):
            others = [
                x
                for x in raw["links"]
                if inst in (x["from"].split(".")[0], x["to"].split(".")[0])
                and x["id"] not in (up["id"], down["id"], link_id)
            ]
            if others:
                raise ValueError(f"coupler '{inst}' has other connections")
        length = fiber.get("length", "100 mm")
        with self.macro(f"Replace fiber {link_id} by air"):
            src, dst = up["from"], down["to"]
            medium = up.get("medium", "air")
            for lid in (up["id"], link_id, down["id"]):
                self.remove_link(lid)
            self.remove_instance(launch)
            self.remove_instance(recv)
            return self.add_link("free_space", src, dst, length=length, medium=medium)

    def _default_fiber(self) -> str:
        fibers = [info.ref for info in self.library.index() if info.type == FIBER_TYPE]
        if self.new_fiber_ref in fibers:
            return self.new_fiber_ref
        smf = self.fiber_parts_by_kind().get("smf") or []
        if len(fibers) == 1 or smf:
            return (smf or fibers)[0]
        if not fibers:
            raise ValueError(f"the library has no {FIBER_TYPE} part to use for a fiber link")
        raise ValueError(
            f"the library has several {FIBER_TYPE} parts ({', '.join(fibers)}); choose one with "
            "fiber=..."
        )

    def add_link(self, kind: LinkKind, src: str, dst: str, **fields: Any) -> str:
        if kind not in _KIND_FOR_PORT.values():
            raise ValueError(f"unknown link kind {kind!r}; choose free_space, fiber or cable")
        self._check_endpoint_syntax(src, dst)
        src, dst = self.normalize_connection(src, dst)
        fields = dict(fields)
        link_id = self._fresh_id(fields.pop("link_id", None), "link")
        if kind == "free_space":
            if "length" not in fields and not fields.get("lock_to_canvas"):
                fields["length"] = DEFAULT_FREE_SPACE_LENGTH
        elif kind == "fiber":
            if fields.get("fiber") is None:
                fields["fiber"] = self._default_fiber()
            fields.setdefault("length", DEFAULT_FIBER_LENGTH)
        else:
            fields.setdefault("delay", DEFAULT_CABLE_DELAY)
        entry = {"id": link_id, "kind": kind, "from": src, "to": dst, **fields}
        try:
            _LINK_ADAPTER.validate_python(entry)
        except ValidationError as exc:
            raise ValueError(_validation_message(exc)) from exc
        raw = self._raw()
        raw["links"].append(entry)
        self._commit(f"Add link {link_id}", raw)
        return link_id

    def remove_link(self, link_id: str) -> None:
        raw = self._raw()
        self._find(raw["links"], link_id, "link")
        raw["links"] = [link for link in raw["links"] if link["id"] != link_id]
        raw["probes"] = [p for p in raw["probes"] if p["link"] != link_id]
        self._commit(f"Remove link {link_id}", raw)

    def set_link_field(self, link_id: str, name: str, value: Any) -> None:
        if name in ("id", "kind"):
            raise ValueError(f"the link field '{name}' cannot be edited")
        raw = self._raw()
        link = self._find(raw["links"], link_id, "link")
        name = {"src": "from", "dst": "to"}.get(name, name)
        if value is None:
            link.pop(name, None)
        else:
            link[name] = copy.deepcopy(value)
        self._commit(f"Set {link_id}.{name}", raw)

    def add_probe(self, part_ref: str, link_id: str, probe_id: str | None = None) -> str:
        category = self._check_part(part_ref)
        if category != "probes":
            raise ValueError(f"'{part_ref}' is not a probe part")
        if all(link.id != link_id for link in self._setup.links):
            raise ValueError(f"there is no link '{link_id}'")
        new_id = self._fresh_id(probe_id, part_ref.rsplit("/", 1)[-1])
        raw = self._raw()
        raw["probes"].append({"id": new_id, "part": part_ref, "link": link_id})
        self._commit(f"Add probe {new_id}", raw)
        return new_id

    def remove_probe(self, probe_id: str) -> None:
        raw = self._raw()
        self._find(raw["probes"], probe_id, "probe")
        raw["probes"] = [p for p in raw["probes"] if p["id"] != probe_id]
        self._commit(f"Remove probe {probe_id}", raw)

    def set_name(self, name: str) -> None:
        raw = self._raw()
        raw["name"] = name
        self._commit("Rename setup", raw)

    def set_description(self, description: str) -> None:
        raw = self._raw()
        raw["description"] = description
        self._commit("Edit description", raw)

    def set_grid_cell(self, scalar: Any) -> None:
        raw = self._raw()
        raw["grid_cell"] = scalar
        self._commit("Set grid cell", raw)

    def rename_instance(self, old_id: str, new_id: str) -> str:
        if new_id == old_id:
            return old_id
        raw = self._raw()
        item = self._find_placed(raw, old_id)
        if new_id in self._all_ids():
            raise ValueError(f"the id '{new_id}' is already used")
        item["id"] = new_id
        prefix = f"{old_id}."
        for link in raw["links"]:
            for end in ("from", "to"):
                if link[end].startswith(prefix):
                    link[end] = f"{new_id}.{link[end][len(prefix) :]}"
        for spec in raw["analyses"]:
            for key, value in list(spec.items()):
                if value == old_id and key in ("herald", "a", "b"):
                    spec[key] = new_id
                elif isinstance(value, str) and value.startswith(prefix):
                    spec[key] = f"{new_id}.{value[len(prefix) :]}"
                elif isinstance(value, list):
                    spec[key] = [
                        new_id
                        if v == old_id and key == "channels"
                        else f"{new_id}.{v[len(prefix) :]}"
                        if isinstance(v, str) and v.startswith(prefix)
                        else v
                        for v in value
                    ]
        m = raw.get("measurement")
        if m:
            _rename_measurement(m, old_id, new_id)
        self._renames.append((old_id, new_id))
        try:
            self._commit(f"Rename {old_id} to {new_id}", raw)
        except BaseException:
            self._renames.pop()
            raise
        return new_id

    def rename_link(self, old_id: str, new_id: str) -> str:
        if new_id == old_id:
            return old_id
        raw = self._raw()
        link = self._find(raw["links"], old_id, "link")
        if new_id in self._all_ids():
            raise ValueError(f"the id '{new_id}' is already used")
        link["id"] = new_id
        for probe in raw["probes"]:
            if probe["link"] == old_id:
                probe["link"] = new_id
        self._renames.append((old_id, new_id))
        try:
            self._commit(f"Rename link {old_id} to {new_id}", raw)
        except BaseException:
            self._renames.pop()
            raise
        return new_id

    def fiber_parts(self) -> list[str]:
        return sorted(info.ref for info in self.library.index() if info.type == FIBER_TYPE)

    def measurement_labels(self) -> set[str]:
        from sequoia.measurement import channel_map

        return set(channel_map(self._setup))

    def set_analyses(self, analyses: list[dict[str, Any]]) -> None:
        check_analyses(list(analyses), self._setup.measurement, self.measurement_labels())
        raw = self._raw()
        raw["analyses"] = copy.deepcopy(list(analyses))
        self._commit("Edit analyses", raw)

    def set_measurement(self, measurement: Any) -> None:
        """Replace the measurement definition (a ``Measurement``, its mapping, or None). The
        edit bumps the document key, so results computed before it become outdated."""
        from sequoia.measurement import Measurement

        if isinstance(measurement, Measurement):
            data = measurement.model_dump(mode="json", exclude_defaults=True)
        else:
            data = copy.deepcopy(measurement)
        from sequoia.measurement import channel_map

        parsed = None
        if data is not None:
            try:
                parsed = Measurement.model_validate(data)
            except ValidationError as exc:
                raise ValueError(_validation_message(exc)) from exc
        trial = self._setup.model_copy(update={"measurement": parsed})
        check_analyses(list(trial.analyses), parsed, set(channel_map(trial)))
        raw = self._raw()
        raw["measurement"] = data
        self._commit("Edit measurement", raw)
