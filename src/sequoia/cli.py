import argparse
import sys
from pathlib import Path


def resolve_library(library_path: str, setup_path: str) -> Path:
    """A relative library path that does not exist from the working directory is looked up
    in the setup file's directory and then in its parents."""
    path = Path(library_path)
    if path.is_absolute() or path.exists():
        return path
    setup_dir = Path(setup_path).resolve().parent
    for directory in (setup_dir, *setup_dir.parents):
        candidate = directory / path
        if candidate.exists():
            return candidate
    return path


def _run(setup_path: str, library_path: str) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.engines.cache import RunCache
    from sequoia.engines.classical import simulate
    from sequoia.engines.rates import has_rate_devices, simulate_rates
    from sequoia.engines.report import format_rates_report, format_report
    from sequoia.errors import SequoiaError
    from sequoia.library.store import Library
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(setup_path)
        library_root = resolve_library(library_path, setup_path)
        if not library_root.is_dir():
            print(
                f"error: library folder {library_path!r} not found (looked in the working "
                f"directory and in the folders of {setup_path})",
                file=sys.stderr,
            )
            return 1
        library = Library(library_root)
        cache = RunCache()
        result = simulate(setup, library, cache=cache)
        rates = (
            simulate_rates(setup, library, classical=result, cache=cache)
            if has_rate_devices(setup, library, cache, result.network)
            else None
        )
    except SequoiaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(result))
    if rates is not None:
        print()
        print(format_rates_report(rates))
    return 0


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        parser = argparse.ArgumentParser(
            prog="sequoia run", description="simulate a setup file and print the results"
        )
        parser.add_argument("setup")
        parser.add_argument("--library", default="library")
        args = parser.parse_args(argv[1:])
        sys.exit(_run(args.setup, args.library))
    from sequoia.gui.app import run as run_gui

    sys.exit(run_gui(argv))
