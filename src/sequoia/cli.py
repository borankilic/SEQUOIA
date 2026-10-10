import argparse
import sys
from pathlib import Path


def _run(setup_path: str, library_path: str) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.engines.classical import simulate
    from sequoia.engines.rates import has_rate_devices, simulate_rates
    from sequoia.engines.report import format_rates_report, format_report
    from sequoia.errors import SequoiaError
    from sequoia.library.store import Library
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(setup_path)
        library = Library(Path(library_path))
        result = simulate(setup, library)
        rates = (
            simulate_rates(setup, library, classical=result)
            if has_rate_devices(setup, library)
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
    parser = argparse.ArgumentParser(prog="sequoia")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="simulate a setup file and print the results")
    run.add_argument("setup")
    run.add_argument("--library", default="library")
    args = parser.parse_args()
    if args.command == "run":
        sys.exit(_run(args.setup, args.library))
    from sequoia.gui.app import run as run_gui

    sys.exit(run_gui())
