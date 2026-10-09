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


def format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"{loc}: {msg}" if loc else msg)
    return "\n".join(lines)
