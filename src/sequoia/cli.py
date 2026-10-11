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


def _run_mc(
    setup_path: str, library_path: str, duration: float, seed: int | None, tags_path: str | None
) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.analysis.correlation import analyse_tags, write_tags
    from sequoia.engines.cache import RunCache
    from sequoia.engines.montecarlo import compare_with_rates, simulate_timetags
    from sequoia.engines.rates import simulate_rates
    from sequoia.engines.report import format_mc_report
    from sequoia.errors import SequoiaError
    from sequoia.library.store import Library
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(setup_path)
        library_root = resolve_library(library_path, setup_path)
        if not library_root.is_dir():
            print(f"error: library folder {library_path!r} not found", file=sys.stderr)
            return 1
        library = Library(library_root)
        cache = RunCache()
        tags = simulate_timetags(setup, library, duration, seed=seed, cache=cache)
        analyses = analyse_tags(tags.channels, tags.duration, setup.analyses, tags.repetition_rates)
        rates = simulate_rates(setup, library, cache=cache)
        comparisons = compare_with_rates(tags, analyses, rates)
        if tags_path:
            write_tags(tags_path, tags.channels, tags.duration)
    except (SequoiaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_mc_report(tags, analyses, comparisons))
    return 0


def _run_gaussian(setup_path: str, library_path: str, gain_scale: float) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.engines.gaussian import simulate_gaussian
    from sequoia.engines.report import format_gaussian_report
    from sequoia.errors import SequoiaError
    from sequoia.library.store import Library
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(setup_path)
        library_root = resolve_library(library_path, setup_path)
        if not library_root.is_dir():
            print(f"error: library folder {library_path!r} not found", file=sys.stderr)
            return 1
        result = simulate_gaussian(setup, Library(library_root), gain_scale=gain_scale)
    except (SequoiaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_gaussian_report(result))
    return 0


def _analyze_tags(tag_path: str, setup_path: str, duration: float | None) -> int:
    from sequoia.analysis.correlation import analyse_tags, read_tags
    from sequoia.engines.report import format_mc_report
    from sequoia.errors import SequoiaError
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(setup_path)
        channels, file_duration = read_tags(tag_path)
        span = duration or file_duration
        if span is None:
            span = max((float(t[-1]) for t in channels.values() if len(t)), default=0.0)
        if span <= 0:
            print("error: cannot determine the record duration; pass --duration", file=sys.stderr)
            return 1
        analyses = analyse_tags(channels, span, setup.analyses)
    except (SequoiaError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    class _Tags:
        def __init__(self) -> None:
            self.duration = span
            self.seed = None

        def counts(self):
            return {k: len(v) for k, v in channels.items()}

    print(format_mc_report(_Tags(), analyses))
    return 0


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        parser = argparse.ArgumentParser(
            prog="sequoia run", description="simulate a setup file and print the results"
        )
        parser.add_argument("setup")
        parser.add_argument("--library", default="library")
        parser.add_argument("--engine", choices=("rates", "mc", "gaussian"), default="rates")
        parser.add_argument(
            "--gain-scale", type=float, default=1.0, help="pump power factor (gaussian)"
        )
        parser.add_argument("--duration", type=float, default=10.0, help="seconds (mc)")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--tags", default=None, help="write the time tags to this .npz (mc)")
        args = parser.parse_args(argv[1:])
        if args.engine == "gaussian":
            sys.exit(_run_gaussian(args.setup, args.library, args.gain_scale))
        if args.engine == "mc":
            sys.exit(_run_mc(args.setup, args.library, args.duration, args.seed, args.tags))
        sys.exit(_run(args.setup, args.library))
    if argv and argv[0] == "analyze-tags":
        parser = argparse.ArgumentParser(
            prog="sequoia analyze-tags",
            description="apply a setup's analyses to a time-tag file (.npz or channel,time_s CSV)",
        )
        parser.add_argument("tags")
        parser.add_argument("--setup", required=True)
        parser.add_argument("--duration", type=float, default=None)
        args = parser.parse_args(argv[1:])
        sys.exit(_analyze_tags(args.tags, args.setup, args.duration))
    from sequoia.gui.app import run as run_gui

    sys.exit(run_gui(argv))
