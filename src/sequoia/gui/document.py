import copy
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from sequoia.components.base import Component, PortDirection, PortKind, PortSpec, get_component
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


class SetupDocument(QObject):
    changed = Signal()
    structure_changed = Signal()
    dirty_changed = Signal(bool)
    path_changed = Signal()

    def __init__(
        self,
        library: Library,
        setup: Setup | None = None,
        path: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.library = library
        self._setup = setup if setup is not None else Setup()
        self.path = Path(path) if path is not None else None
        self._components: dict[tuple[str, str], tuple[Any, Component | None, str | None]] = {}
        self._macro_stack: list[Setup] = []
        self._structure = self._structure_key()
        self.undo_stack = QUndoStack(self)
        self.undo_stack.indexChanged.connect(self._on_index_changed)
        self.undo_stack.cleanChanged.connect(self._on_clean_changed)

    @classmethod
    def open(cls, path: Path | str, library: Library) -> "SetupDocument":
        path = Path(path)
        return cls(library, load_setup(path), path)

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
        self.changed.emit()
        if structural:
            self.structure_changed.emit()

    def _raw(self) -> dict[str, Any]:
        return self._setup.model_dump(by_alias=True)

    def _commit(self, text: str, raw: dict[str, Any], merge_key=None) -> bool:
        try:
            new = Setup.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(_validation_message(exc)) from exc
        if new == self._setup:
            return False
        if self._macro_stack:
            self._setup = new
        else:
            self.undo_stack.push(_SetupCommand(self, text, self._setup, new, merge_key))
        return True

    @contextmanager
    def macro(self, text: str) -> Iterator[None]:
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
            self.undo_stack.push(_SetupCommand(self, text, snapshot, after))

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

    def move_instance(self, instance_id: str, position: Sequence[float]) -> None:
        self._edit_instance(
            instance_id,
            f"Move {instance_id}",
            merge_key=("move", instance_id),
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

    def _default_fiber(self) -> str:
        fibers = [info.ref for info in self.library.index() if info.type == FIBER_TYPE]
        if len(fibers) == 1:
            return fibers[0]
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

    def set_analyses(self, analyses: list[dict[str, Any]]) -> None:
        raw = self._raw()
        raw["analyses"] = copy.deepcopy(list(analyses))
        self._commit("Edit analyses", raw)
