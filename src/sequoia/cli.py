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


def _export(record, directory: str | None) -> None:
    if directory:
        from sequoia.runs import write_bundle

        write_bundle(record, directory)
        print(f"\nexported run {record.run_id} to {directory}")


def _run(setup_path: str, library_path: str, export: str | None = None) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.engines.cache import RunCache
    from sequoia.engines.classical import simulate
    from sequoia.engines.rates import has_rate_devices, simulate_rates
    from sequoia.engines.report import format_rates_report, format_report
    from sequoia.errors import SequoiaError
    from sequoia.library.store import Library
    from sequoia.measurement import resolve_setup
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
        run = resolve_setup(setup, library)
        result = simulate(run, library, cache=cache)
        rates = (
            simulate_rates(run, library, classical=result, cache=cache)
            if has_rate_devices(run, library, cache, result.network)
            else None
        )
    except SequoiaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(result))
    if rates is not None:
        print()
        print(format_rates_report(rates))
    if export:
        from sequoia.engines.outcome import SimulationOutcome
        from sequoia.runs import new_run_id, record_from_outcome

        run_id = new_run_id("rates")
        issues = rates.issues if rates is not None else result.issues
        outcome = SimulationOutcome(
            0, [i.for_run("rates", run_id) for i in issues], result, rates, run_id=run_id
        )
        try:
            _export(record_from_outcome(outcome, setup, library), export)
        except (OSError, SequoiaError) as exc:
            print(f"error: cannot export the run: {exc}", file=sys.stderr)
            return 1
    return 0


def _run_mc(
    setup_path: str,
    library_path: str,
    duration: float | None,
    seed: int | None,
    tags_path: str | None,
    schmidt: float | None = None,
    export: str | None = None,
) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.analysis.correlation import write_tags
    from sequoia.engines.report import format_mc_report
    from sequoia.errors import SequoiaError
    from sequoia.library.store import Library
    from sequoia.measurement import acquisition_duration
    from sequoia.runs import run_montecarlo, write_bundle
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(setup_path)
        library_root = resolve_library(library_path, setup_path)
        if not library_root.is_dir():
            print(f"error: library folder {library_path!r} not found", file=sys.stderr)
            return 1
        library = Library(library_root)
        span = duration if duration is not None else acquisition_duration(setup, 10.0)
        out = run_montecarlo(setup, library, duration_s=span, seed=seed, schmidt_number=schmidt)
        tags = out.tags
        if tags_path:
            write_tags(
                tags_path,
                tags.channels,
                tags.duration,
                tags.clock,
                tags.metadata(),
                tags.clock_ids,
            )
        if export:
            write_bundle(out.record, export)
    except (SequoiaError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_mc_report(tags, out.analyses, out.comparisons))
    if export:
        print(f"\nexported run {out.record.run_id} to {export}")
    return 0


def _run_gaussian(
    setup_path: str, library_path: str, gain_scale: float, export: str | None = None
) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.engines.gaussian import simulate_gaussian
    from sequoia.engines.report import format_gaussian_report
    from sequoia.errors import SequoiaError
    from sequoia.library.store import Library
    from sequoia.measurement import resolve_setup
    from sequoia.runs import run_gaussian, write_bundle
    from sequoia.setup.io import load_setup

    try:
        setup = load_setup(setup_path)
        library_root = resolve_library(library_path, setup_path)
        if not library_root.is_dir():
            print(f"error: library folder {library_path!r} not found", file=sys.stderr)
            return 1
        library = Library(library_root)
        if export:
            out = run_gaussian(setup, library, pump_scale=gain_scale)
            if out.record is None:
                simulate_gaussian(resolve_setup(setup, library), library)
                print("error: the Gaussian run produced no result to export", file=sys.stderr)
                return 1
            result = out.result
            write_bundle(out.record, export)
        else:
            result = simulate_gaussian(
                resolve_setup(setup, library), library, gain_scale=gain_scale
            )
    except (SequoiaError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_gaussian_report(result))
    if export:
        print(f"\nexported run {out.record.run_id} to {export}")
    return 0


def _rerun(directory: str) -> int:
    import sequoia.components.builtin  # noqa: F401
    from sequoia.errors import SequoiaError
    from sequoia.runs import _short, rerun_bundle

    try:
        res = rerun_bundle(directory)
    except (SequoiaError, ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    c = res.comparison
    print(f"reran {res.original.get('run_id')} ({res.record.engine}) as {res.record.run_id}")
    for d in c.input_differences:
        print(f"input differs: {d}")
    changed = [row for row in c.rows if row[3] != 0.0]
    print(f"{len(c.rows)} stored outputs, {len(changed)} differ")
    for key, a, b, d in changed[:40]:
        print(f"  {key}: stored {_short(a)}, rerun {_short(b)}, difference {_short(d)}")
    print("reproduced" if c.matches else "NOT reproduced")
    return 0 if c.matches else 2


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
    from sequoia.measurement import analyse_measured_tags, effective_analyses
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
            clock = {name: 0.0 for name in channels}
        else:
            clock = record.repetition_rates
            if clock is None:
                clock = {}
                print(
                    "note: the file records no pulse clock, so accidentals use the CW singles "
                    "product; pass --rep-rate HZ for pulsed data or --cw to silence this",
                    file=sys.stderr,
                )
        ids = record.clock_ids if not (rep_rate or cw) else None
        if rep_rate or cw:
            analyses = analyse_tags(channels, span, effective_analyses(setup), clock, ids)
        else:
            analyses = analyse_measured_tags(setup, channels, span, clock, ids)
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
        parser.add_argument(
            "--duration",
            type=float,
            default=None,
            help="acquisition duration in seconds (mc; default: the measurement's, else 10)",
        )
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--tags", default=None, help="write the time tags to this .npz (mc)")
        parser.add_argument(
            "--export-run",
            default=None,
            metavar="DIR",
            help="write a run bundle (run.json, setup, library parts, data) that reruns alone",
        )
        args = parser.parse_args(argv[1:])
        if args.engine == "gaussian":
            sys.exit(_run_gaussian(args.setup, args.library, args.gain_scale, args.export_run))
        if args.engine == "mc":
            sys.exit(
                _run_mc(
                    args.setup,
                    args.library,
                    args.duration,
                    args.seed,
                    args.tags,
                    args.schmidt,
                    args.export_run,
                )
            )
        sys.exit(_run(args.setup, args.library, args.export_run))
    if argv and argv[0] == "rerun":
        parser = argparse.ArgumentParser(
            prog="sequoia rerun",
            description="rerun an exported run bundle from its own files and compare outputs",
        )
        parser.add_argument("bundle")
        args = parser.parse_args(argv[1:])
        sys.exit(_rerun(args.bundle))
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
        parser.add_argument("--quantity", choices=("total_hz", "true_hz"), default="total_hz")
        parser.add_argument(
            "--time", type=float, default=10.0, help="integration time per setting pair (s)"
        )
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
