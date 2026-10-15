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
    setup_path: str,
    library_path: str,
    duration: float,
    seed: int | None,
    tags_path: str | None,
    schmidt: float | None = None,
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
        tags = simulate_timetags(
            setup, library, duration, seed=seed, cache=cache, schmidt_number=schmidt
        )
        analyses = analyse_tags(tags.channels, tags.duration, setup.analyses, tags.clock)
        rates = simulate_rates(setup, library, cache=cache)
        comparisons = compare_with_rates(tags, analyses, rates)
        if tags_path:
            write_tags(tags_path, tags.channels, tags.duration, tags.clock, tags.metadata())
    except (SequoiaError, ValueError, OSError) as exc:
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


def _run_chsh(args) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.analysis.chsh import format_chsh, run_chsh
    from sequoia.errors import SequoiaError
    from sequoia.library.store import Library
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(args.setup)
        library_root = resolve_library(args.library, args.setup)
        if not library_root.is_dir():
            print(f"error: library folder {args.library!r} not found", file=sys.stderr)
            return 1
        names = [n.strip() for n in args.outcomes.split(",")]
        if len(names) != 4:
            print("error: --outcomes needs four names", file=sys.stderr)
            return 1
        result = run_chsh(
            setup,
            Library(library_root),
            args.a,
            args.b,
            [float(x) for x in args.a_values.split(",")],
            [float(x) for x in args.b_values.split(",")],
            args.unit,
            [names[:2], names[2:]],
            args.quantity,
            args.time,
        )
    except (SequoiaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_chsh(result))
    return 0


def _analyze_tags(
    tag_path: str,
    setup_path: str,
    duration: float | None,
    rep_rate: float | None = None,
    cw: bool = False,
) -> int:
    from sequoia.analysis.correlation import analyse_tags, read_tag_record
    from sequoia.engines.report import format_mc_report
    from sequoia.errors import SequoiaError
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(setup_path)
        record = read_tag_record(tag_path)
        channels, file_duration = record.channels, record.duration
        span = duration or file_duration
        if span is None:
            span = max((float(t[-1]) for t in channels.values() if len(t)), default=0.0)
        if span <= 0:
            print("error: cannot determine the record duration; pass --duration", file=sys.stderr)
            return 1
        if rep_rate:
            clock = {name: rep_rate for name in channels}
        elif cw:
            clock = {}
        else:
            clock = record.repetition_rates
            if not clock:
                print(
                    "note: the file records no pulse clock, so accidentals use the CW singles "
                    "product; pass --rep-rate HZ for pulsed data or --cw to silence this",
                    file=sys.stderr,
                )
        analyses = analyse_tags(channels, span, setup.analyses, clock)
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
            "--gain-scale",
            type=float,
            default=1.0,
            help="pump power factor (gaussian): scales pair gain and classical leakage, "
            "dark counts stay fixed",
        )
        parser.add_argument(
            "--schmidt",
            type=float,
            default=None,
            help="effective Schmidt number K for thermal pulse statistics (mc)",
        )
        parser.add_argument("--duration", type=float, default=10.0, help="seconds (mc)")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--tags", default=None, help="write the time tags to this .npz (mc)")
        args = parser.parse_args(argv[1:])
        if args.engine == "gaussian":
            sys.exit(_run_gaussian(args.setup, args.library, args.gain_scale))
        if args.engine == "mc":
            sys.exit(
                _run_mc(args.setup, args.library, args.duration, args.seed, args.tags, args.schmidt)
            )
        sys.exit(_run(args.setup, args.library))
    if argv and argv[0] == "chsh":
        parser = argparse.ArgumentParser(
            prog="sequoia chsh", description="CHSH S from four analyser settings (rate engine)"
        )
        parser.add_argument("setup")
        parser.add_argument("--library", default="library")
        parser.add_argument("--a", required=True, help="arm-a target, e.g. settings:hwp_a.angle")
        parser.add_argument("--b", required=True, help="arm-b target, e.g. settings:hwp_b.angle")
        parser.add_argument("--a-values", required=True, help="two values, e.g. 0,22.5")
        parser.add_argument("--b-values", required=True, help="two values, e.g. 11.25,33.75")
        parser.add_argument("--unit", default="deg")
        parser.add_argument(
            "--outcomes", default="tt,tr,rt,rr", help="four coincidence analysis names"
        )
        parser.add_argument("--quantity", choices=("true_hz", "total_hz"), default="true_hz")
        parser.add_argument("--time", type=float, default=10.0, help="integration time per setting")
        args = parser.parse_args(argv[1:])
        sys.exit(_run_chsh(args))
    if argv and argv[0] == "fock":
        from sequoia.fock.cli import main as fock_main

        sys.exit(fock_main(argv[1:]))
    if argv and argv[0] == "analyze-tags":
        parser = argparse.ArgumentParser(
            prog="sequoia analyze-tags",
            description="apply a setup's analyses to a time-tag file (.npz or channel,time_s CSV)",
        )
        parser.add_argument("tags")
        parser.add_argument("--setup", required=True)
        parser.add_argument("--duration", type=float, default=None)
        parser.add_argument("--rep-rate", type=float, default=None, help="pulse clock in Hz")
        parser.add_argument("--cw", action="store_true", help="treat the record as continuous")
        args = parser.parse_args(argv[1:])
        sys.exit(_analyze_tags(args.tags, args.setup, args.duration, args.rep_rate, args.cw))
    from sequoia.gui.app import run as run_gui

    sys.exit(run_gui(argv))
