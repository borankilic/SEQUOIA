import datetime
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, ValidationError

from sequoia.components.base import get_component
from sequoia.data import Curve, Provenance, Scalar, Source, Surface
from sequoia.errors import PartError, format_validation_error
from sequoia.library.store import Library, Part, slugify

PROVENANCE_SOURCES: tuple[str, ...] = get_args(Source)
HEADER_FIELDS = ("manufacturer", "model", "serial", "description")


def draft_from_part(part: Part) -> dict[str, Any]:
    """Raw (YAML-like) values of the parameters the part file sets explicitly."""
    draft: dict[str, Any] = {}
    for name in type(part.params).model_fields:
        if name not in part.params.model_fields_set:
            continue
        value = getattr(part.params, name)
        if value is None:
            continue
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", exclude_none=True)
        draft[name] = value
    return draft


def build_part(ref: str, type_name: str, header: dict[str, Any], draft: dict[str, Any]) -> Part:
    cls = get_component(type_name)
    try:
        params = cls.Params.model_validate(draft)
    except ValidationError as exc:
        raise PartError(f"invalid parameters:\n{format_validation_error(exc)}") from exc
    values = {k: (header.get(k) or None) for k in HEADER_FIELDS}
    return Part(
        ref=ref,
        type=type_name,
        params=params,
        attachments=list(header.get("attachments") or []),
        **values,
    )


def data_files(part: Part) -> set[str]:
    files = set()
    for name in type(part.params).model_fields:
        value = getattr(part.params, name)
        if isinstance(value, Curve | Surface):
            files.add(value.file or f"{name}.csv")
    return files


def save_edited(library: Library, part: Part, previous: Part | None = None) -> Path:
    """Save ``part`` and drop curve files that the previous version used and this one does not."""
    path = library.save_part(part)
    if previous is not None:
        directory = library.part_dir(part.ref)
        for name in data_files(previous) - data_files(part):
            stale = directory / name
            if stale.is_file() and name not in part.attachments:
                stale.unlink()
    return path


def get_provenance(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, BaseModel):
        prov = getattr(raw, "provenance", None)
        return prov.model_dump(mode="json", exclude_none=True) if prov is not None else None
    if isinstance(raw, dict):
        prov = raw.get("provenance")
        if isinstance(prov, Provenance):
            return prov.model_dump(mode="json", exclude_none=True)
        return dict(prov) if prov else None
    return None


def with_provenance(raw: Any, provenance: dict[str, Any] | None) -> Any:
    """``raw`` (quantity text, number, Scalar, curve or surface) carrying ``provenance``."""
    if isinstance(raw, Curve | Surface):
        raw = raw.model_dump(mode="json", exclude_none=True)
    if isinstance(raw, Scalar):
        raw = raw.model_dump(mode="json", exclude_none=True)
        if not isinstance(raw, dict):
            raw = Scalar.model_validate(raw).model_dump(mode="json", exclude_none=True)
    if isinstance(raw, str) or (isinstance(raw, int | float) and not isinstance(raw, bool)):
        scalar = Scalar.model_validate(raw)
        raw = {"value": scalar.value, "unit": scalar.unit}
    if not isinstance(raw, dict):
        raise ValueError("only quantities, curves and surfaces carry provenance")
    out = {k: v for k, v in raw.items() if k != "provenance"}
    if provenance:
        out["provenance"] = Provenance.model_validate(provenance).model_dump(
            mode="json", exclude_none=True
        )
    return out


def accepts_provenance(raw: Any) -> bool:
    if isinstance(raw, Curve | Surface | Scalar):
        return True
    if isinstance(raw, bool):
        return False
    if isinstance(raw, int | float):
        return True
    if isinstance(raw, dict):
        return "value" in raw or "x" in raw
    if isinstance(raw, str):
        try:
            Scalar.model_validate(raw)
        except ValueError:
            return False
        return True
    return False


def keep_provenance(old: Any, new: Any) -> Any:
    """A newly typed plain value keeps the provenance of the value it replaces."""
    prov = get_provenance(old)
    if prov is None or isinstance(new, dict):
        return new
    try:
        return with_provenance(new, prov)
    except ValueError:
        return new


def suggest_ref(library: Library, type_name: str, model: str | None) -> str:
    category = get_component(type_name).category
    base = slugify(model or type_name)
    ref = f"{category}/{base}"
    n = 2
    while library.exists(ref) or library.part_dir(ref).exists():
        ref = f"{category}/{base}_{n}"
        n += 1
    return ref


def today() -> str:
    return datetime.date.today().isoformat()
