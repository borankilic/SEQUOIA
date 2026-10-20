import argparse
import csv
import json
import sys
from pathlib import Path

from sequoia.errors import SequoiaError
from sequoia.fock.analysis import format_state


def _load(path: str, occupations: str | None):
    from sequoia.fock.experiment import load_experiment
    from sequoia.fock.import_perceval_gui import import_perceval_gui

    p = Path(path)
    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "components" in raw and "mode_losses" in raw:
            if occupations is None:
                raise SequoiaError(
                    "a Perceval GUI circuit file needs --input, for example --input 1,1,0,0"
                )
            imp = import_perceval_gui(p, [int(x) for x in occupations.split(",")], strict=False)
            for w in imp.warnings:
                print(f"warning: {w}", file=sys.stderr)
            return imp.experiment
    exp = load_experiment(p)
    if occupations is not None:
        exp.occupations = [int(x) for x in occupations.split(",")]
    return exp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sequoia fock", description="photon-number circuit experiment (YAML/JSON)"
    )
    parser.add_argument("file")
    parser.add_argument("--input", help="input occupations, e.g. 1,1,0 (overrides the file)")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", choices=("native", "perceval"), default=None)
    parser.add_argument("--top", type=int, default=20, help="rows printed")
    parser.add_argument("--json", dest="json_out", help="write the full result as JSON")
    parser.add_argument("--csv", dest="csv_out", help="write the probability table as CSV")
    args = parser.parse_args(argv)
    from sequoia.fock.experiment import run_experiment

    try:
        exp = _load(args.file, args.input)
        if args.shots is not None:
            exp.shots = args.shots
        if args.seed is not None:
            exp.seed = args.seed
        if args.backend is not None:
            exp.backend = args.backend
        result = run_experiment(exp)
    except (SequoiaError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"{exp.name}: {exp.circuit.n_modes} modes, {result.resources['photons']} photons, "
        f"backend {result.backend}"
    )
    rows = result.table()
    header = ["state", "probability"] + (["counts"] if result.samples else [])
    print("  ".join(f"{h:>14}" for h in header))
    for row in rows[: args.top]:
        cells = [row["state"], f"{row['probability']:.6g}"]
        if result.samples:
            cells.append(str(row["counts"]))
        print("  ".join(f"{c:>14}" for c in cells))
    if len(rows) > args.top:
        print(f"... {len(rows) - args.top} more states")
    print(f"total probability {result.mass:.12g}")
    if result.samples and result.samples.unresolved_probability:
        print(
            f"unresolved (truncated input, probability "
            f"{result.samples.unresolved_probability:.3g}): {result.samples.unresolved} shots"
        )
    for name, cond in result.conditionals.items():
        print(f"condition {name}: success probability {cond.success_probability:.6g}")
        for s, p in sorted(cond.probs.items(), key=lambda kv: -kv[1])[: args.top]:
            print(f"  {format_state(s)}  {p:.6g}")
    for m in result.messages:
        print(f"note: {m}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
    return 0
