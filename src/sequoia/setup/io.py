from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sequoia.errors import SetupError, format_validation_error
from sequoia.setup.model import Setup
from sequoia.yamlio import dump_yaml, load_yaml


def setup_from_dict(raw: Any, source: str = "setup") -> Setup:
    if not isinstance(raw, dict):
        raise SetupError(f"{source}: a setup file must contain a mapping")
    try:
        return Setup.model_validate(raw)
    except ValidationError as exc:
        raise SetupError(f"{source}: invalid setup\n{format_validation_error(exc)}") from exc


def setup_to_dict(setup: Setup) -> dict[str, Any]:
    return setup.model_dump(mode="python", by_alias=True, exclude_defaults=True)


def load_setup(path: Path | str) -> Setup:
    path = Path(path)
    try:
        raw = load_yaml(path)
    except OSError as exc:
        raise SetupError(f"{path}: cannot read setup: {exc}") from exc
    except Exception as exc:
        raise SetupError(f"{path}: not valid YAML: {exc}") from exc
    return setup_from_dict(raw, source=path.name)


def save_setup(setup: Setup, path: Path | str) -> None:
    dump_yaml(setup_to_dict(setup), Path(path))
