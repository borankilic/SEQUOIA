from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError


class SequoiaError(Exception):
    pass


class UnknownComponentError(SequoiaError):
    pass


class PartError(SequoiaError):
    pass


class SetupError(SequoiaError):
    pass


class TableImportError(SequoiaError):
    pass


@dataclass(frozen=True)
class Issue:
    severity: Literal["error", "warning"]
    code: str
    subject: str
    message: str
    engine: str | None = None
    run_id: str | None = None

    def for_run(self, engine: str, run_id: str | None) -> "Issue":
        return Issue(self.severity, self.code, self.subject, self.message, engine, run_id)

    def __str__(self) -> str:
        return f"{self.subject}: {self.message}"


def format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"{loc}: {msg}" if loc else msg)
    return "\n".join(lines)
