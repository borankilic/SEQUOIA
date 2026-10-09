import os
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def load_yaml(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return YAML(typ="safe", pure=True).load(f)


def dump_yaml(data: Any, path: Path) -> None:
    path = Path(path)
    yaml = YAML(typ="rt")
    yaml.default_flow_style = None
    yaml.allow_unicode = True
    yaml.width = 120
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
    os.replace(tmp, path)
