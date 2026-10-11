from typing import Annotated, Any

from pydantic import TypeAdapter, ValidationError

from sequoia.components.base import Component, Params, get_component
from sequoia.errors import SetupError, format_validation_error
from sequoia.library.store import Library
from sequoia.setup.model import Instance, ProbeSpec


def apply_overrides(params: Params, overrides: dict[str, Any]) -> Params:
    fields = type(params).model_fields
    unknown = sorted(set(overrides) - set(fields))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; this part has {sorted(fields)}")
    update = {}
    for name, value in overrides.items():
        info = fields[name]
        annotation = (
            Annotated[info.annotation, *info.metadata] if info.metadata else info.annotation
        )
        adapter = TypeAdapter(annotation)
        try:
            update[name] = adapter.validate_python(value)
        except ValidationError as exc:
            raise ValueError(f"{name}: {format_validation_error(exc)}") from exc
    updated = params.model_copy(update=update)
    try:
        return type(params).model_validate(dict(updated))
    except ValidationError as exc:
        raise ValueError(format_validation_error(exc)) from exc


def instantiate(library: Library, placed: Instance | ProbeSpec) -> Component:
    part = library.load_part(placed.part)
    cls = get_component(part.type)
    try:
        settings = cls.Settings.model_validate(placed.settings)
    except ValidationError as exc:
        raise SetupError(
            f"invalid settings for '{placed.id}':\n{format_validation_error(exc)}"
        ) from exc
    try:
        params = apply_overrides(part.params, placed.overrides)
    except ValueError as exc:
        raise SetupError(f"invalid overrides for '{placed.id}': {exc}") from exc
    return cls(params, settings)
