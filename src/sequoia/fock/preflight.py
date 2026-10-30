import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from sequoia.fock.analysis import MAX_SEQUENCE, MAX_SHOTS
from sequoia.fock.engine import Limits, source_mixture, transfer_isometry
from sequoia.fock.experiment import Experiment
from sequoia.fock.inputs import PureState

MAX_PATTERNS = 2_000_000
MAX_MEMORY = 4e9
WARN_WORK = 2e8
WARN_MEMORY = 1e9
EXACT_BRANCHES = 20_000
SHOWN_ROWS = 500
PERM_BUFFER = 64e6


@dataclass(frozen=True)
class Estimate:
    """Resource estimate of a Fock run before it is computed. ``work`` and ``memory_bytes``
    follow the native engine's own accounting; the work limit applies per input configuration
    (``branch_work``), as in the engine."""

    modes: int
    loss_modes: int
    photons: int
    sectors: tuple[int, ...]
    branches: int
    branches_bounded: bool
    work: float
    branch_work: float
    memory_bytes: float
    output_patterns: int
    shots: int
    backend: str
    limits: Limits
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        sectors = (
            ", ".join(map(str, self.sectors))
            if len(self.sectors) <= 6
            else (f"{self.sectors[0]}..{self.sectors[-1]}")
        )
        text = (
            f"{self.modes} modes + {self.loss_modes} loss modes, up to {self.photons} photons "
            f"(sectors {sectors or '0'}), {'<= ' if self.branches_bounded else ''}"
            f"{self.branches} input configuration(s), work {self.work:.3g} "
            f"(limit {self.limits.max_work:.3g} per configuration), memory "
            f"~{_bytes(self.memory_bytes)}, <= {self.output_patterns:,} output patterns"
        )
        if self.shots:
            text += f", {self.shots:,} shots (counts only)"
        if self.backend != "native":
            text += f"; {self.backend} backend: native limits not applied"
        return text


def _bytes(b: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= scale:
            return f"{b / scale:.3g} {unit}"
    return f"{b:.0f} B"


def _multisets(width: int, total: int) -> int:
    if total == 0:
        return 1
    return math.comb(width + total - 1, total) if width > 0 else 0


def _class_cost(n_sys: int, env: int, n_in: int) -> tuple[float, float]:
    if n_in == 0:
        return 0.0, 0.0
    work = 0.0
    peak = 0.0
    for k in range(n_in + 1):
        if k > 0 and env == 0:
            break
        rows = _multisets(n_sys, n_in - k) * (_multisets(env, k) if k else 1)
        step = rows * (2 ** max(n_in - 1, 0)) * max(n_in, 1)
        work += step
        peak = max(
            peak, rows * (16.0 * n_in * n_in + 8.0 * n_in + 48.0) + min(16.0 * step, PERM_BUFFER)
        )
    return work, peak


def _photon_list_cost(n: int, env: int, photons: Sequence[tuple[int, Hashable]]):
    groups: dict = {}
    for _m, tag in photons:
        groups[tag] = groups.get(tag, 0) + 1
    work, peak = 0.0, 0.0
    for count in groups.values():
        w, p = _class_cost(n, env, count)
        work += w
        peak = max(peak, p)
    return work, peak


def _pure_cost(n: int, env: int, state: PureState) -> tuple[float, float]:
    counts = [c for c in state.tag_counts.values() if c > 0]
    if not counts:
        return 0.0, 0.0
    width = n + env
    terms = len(state.terms)
    sizes = [_multisets(width, c) for c in counts]
    full = float(math.prod(sizes))
    work = full * terms
    amp = 0.0
    for c, rows in zip(counts, sizes, strict=True):
        step = rows * (2 ** max(c - 1, 0)) * c
        work += step * terms
        amp = max(amp, rows * 16.0 * c * c + min(16.0 * step, PERM_BUFFER))
    return work, full * 40.0 + amp


def _patterns(n: int, sectors: Sequence[int], lossy: bool) -> int:
    if not sectors:
        return 1
    if lossy:
        return math.comb(n + max(sectors), max(sectors))
    return sum(math.comb(n + s - 1, s) if s else 1 for s in set(sectors))


def estimate_experiment(
    exp: Experiment, *, shots: int | None = None, backend: str | None = None
) -> Estimate:
    """Modes, photon sectors, loss modes, input configurations, permanent work, memory and the
    number of output patterns of ``exp``, without computing it. Errors are requests the run
    would refuse (or that would exhaust memory); each names a remedy."""
    limits = exp.limits
    shots = exp.shots if shots is None else int(shots)
    backend = exp.backend if backend is None else backend
    n = exp.circuit.n_modes
    env = transfer_isometry(exp.circuit.matrix()).shape[0] - n
    costs: list[tuple[float, float]] = []
    photon_counts: list[int] = []
    bounded = False
    total_work = 0.0
    if exp.input_mixture is not None:
        for w, state in exp.input_mixture.branches:
            if w == 0.0:
                continue
            photon_counts.append(sum(state.tag_counts.values()))
            if len(state.terms) == 1:
                costs.append(_photon_list_cost(n, env, state.terms[0].photons))
            else:
                costs.append(_pure_cost(n, env, state))
        branches = len(costs)
    else:
        photons = exp.photons()
        b, i = exp.brightness, exp.indistinguishability
        if b == 1.0 and i == 1.0:
            configs = [list(photons)]
        elif 3 ** len(photons) <= EXACT_BRANCHES:
            configs = [ph for _, ph in source_mixture(photons, b, i)]
        else:
            configs = []
            bounded = True
        if bounded:
            full = _photon_list_cost(n, env, photons)
            branches = 3 ** len(photons)
            costs = [full]
            photon_counts = list(range(len(photons) + 1)) if b < 1.0 else [len(photons)]
            total_work = full[0] * branches
        else:
            for ph in configs:
                costs.append(_photon_list_cost(n, env, ph))
                photon_counts.append(len(ph))
            branches = len(configs)
    work = total_work if bounded else sum(c[0] for c in costs)
    branch_work = max((c[0] for c in costs), default=0.0)
    memory = max((c[1] for c in costs), default=0.0)
    sectors = tuple(sorted(set(photon_counts)))
    patterns = _patterns(n, sectors, env > 0)
    max_ph = max(photon_counts, default=0)
    errors: list[str] = []
    warnings: list[str] = []
    native = backend == "native"
    if native and max_ph > limits.max_photons:
        errors.append(
            f"{max_ph} photons exceed the photon limit of {limits.max_photons}; lower max pairs, "
            "remove input photons or raise max photons"
        )
    if native and branch_work > limits.max_work:
        errors.append(
            f"estimated work {branch_work:.3g} per input configuration exceeds the limit "
            f"{limits.max_work:.3g} ({max_ph} photons, {n} modes, {env} loss modes); lower max "
            "pairs or photons, reduce modes or simplify the circuit"
        )
    if native and memory > MAX_MEMORY:
        errors.append(
            f"about {_bytes(memory)} of working memory needed (budget {_bytes(MAX_MEMORY)}); "
            "lower max pairs or photons, or reduce modes"
        )
    if patterns > MAX_PATTERNS:
        errors.append(
            f"up to {patterns:,} output patterns (budget {MAX_PATTERNS:,}); reduce modes or photons"
        )
    if shots > MAX_SHOTS:
        errors.append(f"{shots:,} shots exceed the limit of {MAX_SHOTS:g}; reduce shots")
    if native and not errors and work > WARN_WORK:
        warnings.append(f"estimated work {work:.3g}: the run may take minutes")
    if native and not errors and memory > WARN_MEMORY:
        warnings.append(f"about {_bytes(memory)} of working memory needed")
    if patterns > SHOWN_ROWS:
        warnings.append(
            f"up to {patterns:,} output patterns: tables list the most probable {SHOWN_ROWS} "
            "and the probability of the rest"
        )
    if shots > MAX_SEQUENCE:
        warnings.append("shots are drawn as counts only (no per-shot sequence)")
    return Estimate(
        n,
        env,
        max_ph,
        sectors,
        branches,
        bounded,
        float(work),
        float(branch_work),
        float(memory),
        int(patterns),
        shots,
        backend,
        limits,
        tuple(errors),
        tuple(warnings),
    )
