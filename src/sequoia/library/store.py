import csv
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from pydantic import ValidationError

from sequoia.components.base import Params, get_component
from sequoia.data import Curve, Surface, format_axis, parse_axis
from sequoia.errors import (
    PartError,
    SequoiaError,
    UnknownComponentError,
    format_validation_error,
)
from sequoia.library.calibration import (
    CalibrationRecord,
    fingerprint_files,
    load_calibration,
    provenance_rank,
    select_records,
)
from sequoia.library.importers.table import read_table
from sequoia.yamlio import dump_yaml, load_yaml

PART_FILE = "part.yaml"
_HEADER_KEYS = {"type", "manufacturer", "model", "serial", "description", "params", "attachments"}
_CURVE_KEYS = {"curve", "x", "y", "provenance", "extrapolation"}
_SURFACE_KEYS = {"surface", "x", "y", "z", "provenance", "extrapolation"}


@dataclass
class Part:
    ref: str
    type: str
    params: Params
    manufacturer: str | None = None
    model: str | None = None
    serial: str | None = None
    description: str | None = None
    attachments: list[str] = field(default_factory=list)
    calibration: list[CalibrationRecord] = field(default_factory=list)
    calibrated: dict[str, str] = field(default_factory=dict)
    fingerprint: str | None = None


@dataclass(frozen=True)
class PartInfo:
    ref: str
    type: str | None
    category: str | None
    manufacturer: str | None = None
    model: str | None = None
    serial: str | None = None
    error: str | None = None


def _write_csv(path: Path, header: list[str], columns: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in zip(*columns, strict=True):
            writer.writerow([repr(float(v)) for v in row])


class Library:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def part_dir(self, ref: str) -> Path:
        pure = PurePosixPath(ref)
        if (
            not ref
            or "\\" in ref
            or pure.is_absolute()
            or any(p in ("", ".", "..") for p in ref.split("/"))
        ):
            raise PartError(f"invalid part reference {ref!r}")
        return self.root.joinpath(*pure.parts)

    def exists(self, ref: str) -> bool:
        return (self.part_dir(ref) / PART_FILE).is_file()

    def _file(self, directory: Path, name: str, ref: str) -> Path:
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents:
            raise PartError(f"{ref}: data file {name!r} must be inside the part folder")
        return path

    def load_part(self, ref: str, *, calibrated: bool = True) -> Part:
        """Load a part. Its calibration records (calibration.yaml) replace a nominal value when
        they rank higher (measured > datasheet/vendor nominal > model > ideal default; a value
        without provenance ranks as nominal); ``calibrated=False`` gives the nominal part."""
        directory = self.part_dir(ref)
        path = directory / PART_FILE
        if not path.is_file():
            raise PartError(f"part {ref!r} not found in library {self.root}")
        try:
            raw = load_yaml(path)
        except Exception as exc:
            raise PartError(f"{ref}: cannot read {PART_FILE}: {exc}") from exc
        if not isinstance(raw, dict):
            raise PartError(f"{ref}: {PART_FILE} must contain a mapping")
        unknown = set(raw) - _HEADER_KEYS
        if unknown:
            raise PartError(f"{ref}: unknown key(s) {sorted(unknown)} in {PART_FILE}")
        if "type" not in raw:
            raise PartError(f"{ref}: {PART_FILE} has no 'type'")
        try:
            cls = get_component(raw["type"])
        except UnknownComponentError as exc:
            raise PartError(f"{ref}: {exc}") from exc
        params_raw = raw.get("params") or {}
        if not isinstance(params_raw, dict):
            raise PartError(f"{ref}: 'params' must be a mapping")
        resolved = {
            name: self._resolve(directory, ref, name, value) for name, value in params_raw.items()
        }
        records = load_calibration(directory, ref)
        applied: dict[str, str] = {}
        if calibrated:
            chosen = select_records(records, set(cls.Params.model_fields), ref)
            for name, record in chosen.items():
                current = provenance_rank(resolved[name]) if name in resolved else -1
                if name in cls.Params.model_fields and record.rank > current:
                    resolved[name] = record.applied_value()
                    applied[name] = record.id
        try:
            params = cls.Params.model_validate(resolved)
        except ValidationError as exc:
            raise PartError(f"{ref}: invalid parameters:\n{format_validation_error(exc)}") from exc
        return Part(
            ref=ref,
            type=raw["type"],
            params=params,
            manufacturer=raw.get("manufacturer"),
            model=raw.get("model"),
            serial=None if raw.get("serial") is None else str(raw["serial"]),
            description=raw.get("description"),
            attachments=list(raw.get("attachments") or []),
            calibration=records,
            calibrated=applied,
            fingerprint=fingerprint_files(directory, set(raw.get("attachments") or [])),
        )

    def part_fingerprint(self, ref: str) -> str:
        """SHA-256 of the part's files (part.yaml, calibration.yaml and data files; attachments
        such as datasheets are left out). Any change of a nominal value or calibration changes
        it."""
        directory = self.part_dir(ref)
        if not (directory / PART_FILE).is_file():
            raise PartError(f"part {ref!r} not found in library {self.root}")
        try:
            attachments = set((load_yaml(directory / PART_FILE) or {}).get("attachments") or [])
        except Exception:
            attachments = set()
        return fingerprint_files(directory, attachments)

    def fingerprint(self, refs=None) -> str:
        """SHA-256 over the part fingerprints of ``refs`` (default: every part)."""
        import hashlib

        refs = sorted(refs) if refs is not None else [info.ref for info in self.index()]
        digest = hashlib.sha256()
        for ref in refs:
            digest.update(f"{ref}:{self.part_fingerprint(ref)}\n".encode())
        return digest.hexdigest()

    def _resolve(self, directory: Path, ref: str, name: str, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        where = f"{ref}.{name}"
        if "curve" in value:
            return self._read_curve(directory, where, ref, value)
        if "surface" in value:
            return self._read_surface(directory, where, ref, value)
        if "file" in value and set(value) <= {"file", "provenance"}:
            if not self._file(directory, str(value["file"]), ref).is_file():
                raise PartError(f"{where}: file {value['file']!r} not found in the part folder")
        return value

    def _read_columns(self, directory: Path, where: str, ref: str, filename: str, ncols: int):
        path = self._file(directory, filename, ref)
        try:
            table = read_table(path)
        except SequoiaError as exc:
            raise PartError(f"{where}: {exc}") from exc
        if len(table.columns) < ncols:
            raise PartError(
                f"{where}: {filename} needs {ncols} numeric columns, found {len(table.columns)}"
            )
        return [c.values for c in table.columns[:ncols]]

    def _axes(self, where: str, value: dict, keys: tuple[str, ...], allowed: set[str]) -> list:
        unknown = set(value) - allowed
        if unknown:
            raise PartError(f"{where}: unknown key(s) {sorted(unknown)}")
        try:
            return [parse_axis(value[k]) for k in keys]
        except KeyError as exc:
            raise PartError(f"{where}: missing key {exc}") from exc
        except ValueError as exc:
            raise PartError(f"{where}: {exc}") from exc

    def _read_curve(self, directory: Path, where: str, ref: str, value: dict) -> Curve:
        (x_label, x_unit), (y_label, y_unit) = self._axes(where, value, ("x", "y"), _CURVE_KEYS)
        xs, ys = self._read_columns(directory, where, ref, str(value["curve"]), 2)
        try:
            return Curve(
                x=xs.tolist(),
                y=ys.tolist(),
                x_label=x_label,
                x_unit=x_unit,
                y_label=y_label,
                y_unit=y_unit,
                provenance=value.get("provenance"),
                extrapolation=value.get("extrapolation", "warn_clamp"),
                file=str(value["curve"]),
            )
        except ValidationError as exc:
            raise PartError(f"{where}: invalid curve:\n{format_validation_error(exc)}") from exc

    def _read_surface(self, directory: Path, where: str, ref: str, value: dict) -> Surface:
        axes = self._axes(where, value, ("x", "y", "z"), _SURFACE_KEYS)
        (x_label, x_unit), (y_label, y_unit), (z_label, z_unit) = axes
        xs, ys, zs = self._read_columns(directory, where, ref, str(value["surface"]), 3)
        grid_x, grid_y = np.unique(xs), np.unique(ys)
        z = np.full((len(grid_x), len(grid_y)), np.nan)
        z[np.searchsorted(grid_x, xs), np.searchsorted(grid_y, ys)] = zs
        if len(xs) != z.size or np.isnan(z).any():
            raise PartError(
                f"{where}: {value['surface']} is not a complete rectangular x-y grid "
                f"({len(xs)} rows for {len(grid_x)} x {len(grid_y)} grid points)"
            )
        try:
            return Surface(
                x=grid_x.tolist(),
                y=grid_y.tolist(),
                z=z.tolist(),
                x_label=x_label,
                x_unit=x_unit,
                y_label=y_label,
                y_unit=y_unit,
                z_label=z_label,
                z_unit=z_unit,
                provenance=value.get("provenance"),
                extrapolation=value.get("extrapolation", "warn_clamp"),
                file=str(value["surface"]),
            )
        except ValidationError as exc:
            raise PartError(f"{where}: invalid surface:\n{format_validation_error(exc)}") from exc

    def save_part(self, part: Part) -> Path:
        if part.calibrated:
            raise PartError(
                f"{part.ref}: this copy carries calibration values "
                f"({', '.join(sorted(part.calibrated))}); load it with calibrated=False to edit "
                "the nominal part"
            )
        directory = self.part_dir(part.ref)
        directory.mkdir(parents=True, exist_ok=True)
        explicit = part.params.model_fields_set
        dumped = part.params.model_dump(mode="python", exclude_none=True, include=set(explicit))
        for name in type(part.params).model_fields:
            if name not in explicit:
                continue
            value = getattr(part.params, name)
            if isinstance(value, Curve):
                dumped[name] = self._write_curve(directory, part.ref, name, value)
            elif isinstance(value, Surface):
                dumped[name] = self._write_surface(directory, part.ref, name, value)
        header: dict[str, Any] = {"type": part.type}
        for key in ("manufacturer", "model", "serial", "description"):
            if getattr(part, key) is not None:
                header[key] = getattr(part, key)
        header["params"] = dumped
        if part.attachments:
            header["attachments"] = list(part.attachments)
        path = directory / PART_FILE
        dump_yaml(header, path)
        return path

    def _write_curve(self, directory: Path, ref: str, name: str, curve: Curve) -> dict:
        filename = curve.file or f"{name}.csv"
        _write_csv(
            self._file(directory, filename, ref),
            [format_axis(curve.x_label, curve.x_unit), format_axis(curve.y_label, curve.y_unit)],
            [curve.x, curve.y],
        )
        entry: dict[str, Any] = {
            "curve": filename,
            "x": format_axis(curve.x_label, curve.x_unit),
            "y": format_axis(curve.y_label, curve.y_unit),
        }
        return self._with_extras(entry, curve)

    def _write_surface(self, directory: Path, ref: str, name: str, surface: Surface) -> dict:
        filename = surface.file or f"{name}.csv"
        rows = [
            (x, y, surface.z[i][j])
            for i, x in enumerate(surface.x)
            for j, y in enumerate(surface.y)
        ]
        _write_csv(
            self._file(directory, filename, ref),
            [
                format_axis(surface.x_label, surface.x_unit),
                format_axis(surface.y_label, surface.y_unit),
                format_axis(surface.z_label, surface.z_unit),
            ],
            [list(col) for col in zip(*rows, strict=True)],
        )
        entry: dict[str, Any] = {
            "surface": filename,
            "x": format_axis(surface.x_label, surface.x_unit),
            "y": format_axis(surface.y_label, surface.y_unit),
            "z": format_axis(surface.z_label, surface.z_unit),
        }
        return self._with_extras(entry, surface)

    @staticmethod
    def _with_extras(entry: dict, item: Curve | Surface) -> dict:
        if item.provenance is not None:
            entry["provenance"] = item.provenance.model_dump(mode="python", exclude_none=True)
        if item.extrapolation != "warn_clamp":
            entry["extrapolation"] = item.extrapolation
        return entry

    def duplicate_part(self, ref: str, new_ref: str) -> Path:
        source = self.part_dir(ref)
        target = self.part_dir(new_ref)
        if not (source / PART_FILE).is_file():
            raise PartError(f"part {ref!r} not found in library {self.root}")
        if target.exists():
            raise PartError(f"part {new_ref!r} already exists")
        shutil.copytree(source, target)
        return target

    def rename_part(self, ref: str, new_ref: str) -> Path:
        source = self.part_dir(ref)
        target = self.part_dir(new_ref)
        if not (source / PART_FILE).is_file():
            raise PartError(f"part {ref!r} not found in library {self.root}")
        if target.exists():
            raise PartError(f"part {new_ref!r} already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return target

    def delete_part(self, ref: str) -> None:
        directory = self.part_dir(ref)
        if not (directory / PART_FILE).is_file():
            raise PartError(f"part {ref!r} not found in library {self.root}")
        shutil.rmtree(directory)
        parent = directory.parent
        while parent != self.root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

    def attach_file(self, ref: str, source: Path | str) -> str:
        """Copy ``source`` into the part folder and return the stored file name."""
        directory = self.part_dir(ref)
        source = Path(source)
        if not source.is_file():
            raise PartError(f"{source}: not a file")
        directory.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", source.name) or "attachment"
        if name == PART_FILE:
            raise PartError(f"an attachment cannot be called {PART_FILE}")
        target = directory / name
        if target.resolve() != source.resolve():
            shutil.copyfile(source, target)
        return name

    def index(self) -> list[PartInfo]:
        if not self.root.is_dir():
            return []
        infos = []
        for path in sorted(self.root.rglob(PART_FILE)):
            ref = path.parent.relative_to(self.root).as_posix()
            try:
                raw = load_yaml(path)
                type_name = raw["type"]
                try:
                    category = get_component(type_name).category
                except UnknownComponentError:
                    category = None
                infos.append(
                    PartInfo(
                        ref,
                        type_name,
                        category,
                        raw.get("manufacturer"),
                        raw.get("model"),
                        None if raw.get("serial") is None else str(raw["serial"]),
                    )
                )
            except Exception as exc:
                infos.append(PartInfo(ref, None, None, error=f"cannot read {PART_FILE}: {exc}"))
        return infos


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "part"
