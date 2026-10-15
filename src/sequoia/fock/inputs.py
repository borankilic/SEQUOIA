import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product
from typing import Any

from sequoia.fock.circuit import FockError

Photons = tuple[tuple[int, int], ...]


def _canon(photons: Sequence[Sequence[int]]) -> Photons:
    try:
        return tuple(sorted((int(m), int(t)) for m, t in photons))
    except (TypeError, ValueError) as exc:
        raise FockError(f"photons must be (mode, tag) integer pairs: {exc}") from exc


def _tag_counts(photons: Photons) -> dict[int, int]:
    return dict(Counter(t for _, t in photons))


@dataclass(frozen=True)
class Term:
    amplitude: complex
    photons: Photons

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude", complex(self.amplitude))
        object.__setattr__(self, "photons", _canon(self.photons))
        if any(m < 0 for m, _ in self.photons):
            raise FockError("negative mode index in a photon")

    def to_dict(self) -> dict[str, Any]:
        return {
            "amplitude": [self.amplitude.real, self.amplitude.imag],
            "photons": [list(p) for p in self.photons],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Term":
        try:
            a = d["amplitude"]
            amp = complex(a[0], a[1]) if isinstance(a, (list, tuple)) else complex(a)
            return cls(amp, tuple(tuple(p) for p in d["photons"]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise FockError(f"invalid term {d!r}: {exc}") from exc


@dataclass(frozen=True)
class PureState:
    """Coherent superposition of Fock configurations. Each term is a normalised species-resolved
    Fock state (occupation of every (mode, tag)); identical configurations are merged. The
    state is not normalised on construction; use ``normalised()`` (the engine normalises)."""

    terms: tuple[Term, ...]

    def __post_init__(self) -> None:
        merged: dict[Photons, complex] = {}
        for t in self.terms:
            merged[t.photons] = merged.get(t.photons, 0j) + t.amplitude
        terms = tuple(Term(a, p) for p, a in sorted(merged.items()))
        object.__setattr__(self, "terms", terms)
        if not terms:
            raise FockError("a pure state needs at least one term")
        ref = _tag_counts(terms[0].photons)
        for t in terms[1:]:
            if _tag_counts(t.photons) != ref:
                raise FockError(
                    "all terms of a superposition must hold the same number of photons in every "
                    f"tag class; got {ref} and {_tag_counts(t.photons)}"
                )
        if self.norm2 <= 0.0:
            raise FockError("pure state has zero norm")

    @property
    def norm2(self) -> float:
        return float(sum(abs(t.amplitude) ** 2 for t in self.terms))

    @property
    def tag_counts(self) -> dict[int, int]:
        return _tag_counts(self.terms[0].photons)

    @classmethod
    def fock(cls, photons: Sequence[Sequence[int]]) -> "PureState":
        return cls((Term(1.0, tuple(map(tuple, photons))),))

    def normalised(self) -> "PureState":
        s = math.sqrt(self.norm2)
        return PureState(tuple(Term(t.amplitude / s, t.photons) for t in self.terms))

    def to_dict(self) -> dict[str, Any]:
        return {"terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PureState":
        try:
            return cls(tuple(Term.from_dict(t) for t in d["terms"]))
        except (KeyError, TypeError) as exc:
            raise FockError(f"invalid pure state {d!r}: {exc}") from exc


@dataclass(frozen=True)
class InputMixture:
    """Classical mixture of pure states. Weights sum to at most 1; the remainder is
    ``omitted_weight`` (for example pair numbers beyond the truncation)."""

    branches: tuple[tuple[float, PureState], ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        br = tuple((float(w), s) for w, s in self.branches)
        if any(w < 0.0 for w, _ in br):
            raise FockError("mixture weights must be non-negative")
        if sum(w for w, _ in br) > 1.0 + 1e-9:
            raise FockError("mixture weights sum to more than 1")
        object.__setattr__(self, "branches", br)
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def total_weight(self) -> float:
        return float(sum(w for w, _ in self.branches))

    @property
    def omitted_weight(self) -> float:
        return max(0.0, 1.0 - self.total_weight)

    @classmethod
    def from_state(cls, state: PureState) -> "InputMixture":
        return cls(((1.0, state),))

    def to_dict(self) -> dict[str, Any]:
        return {
            "branches": [{"weight": w, "state": s.to_dict()} for w, s in self.branches],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InputMixture":
        try:
            br = tuple((float(b["weight"]), PureState.from_dict(b["state"])) for b in d["branches"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FockError(f"invalid input mixture: {exc}") from exc
        return cls(br, tuple(d.get("notes", ())))


def _as_mixture(x: "InputMixture | PureState") -> InputMixture:
    return x if isinstance(x, InputMixture) else InputMixture.from_state(x)


def bell_pair(
    mode_s_h: int,
    mode_s_v: int,
    mode_i_h: int,
    mode_i_v: int,
    phase: float = 0.0,
    visibility: float = 1.0,
    kind: str = "phi",
    tag: int = 0,
) -> InputMixture:
    """kind 'phi': (|HH> + e^{i phase}|VV>)/sqrt2; 'psi': (|HV> + e^{i phase}|VH>)/sqrt2 (signal
    first, idler second).
    Werner form: rho = V |B><B| + (1 - V) I/4, with I/4 the maximally mixed state of the four
    product states |HH>, |HV>, |VH>, |VV> (mixture weights V and (1 - V)/4 each)."""
    if kind not in ("phi", "psi"):
        raise FockError(f"kind must be 'phi' or 'psi', got {kind!r}")
    if not 0.0 <= visibility <= 1.0:
        raise FockError("visibility must lie in [0, 1]")
    s = {"H": mode_s_h, "V": mode_s_v}
    i = {"H": mode_i_h, "V": mode_i_v}

    def ph(a: str, b: str) -> Photons:
        return ((s[a], tag), (i[b], tag))

    e = complex(math.cos(phase), math.sin(phase))
    r = 1.0 / math.sqrt(2.0)
    pair = ("HH", "VV") if kind == "phi" else ("HV", "VH")
    bell = PureState((Term(r, ph(*pair[0])), Term(r * e, ph(*pair[1]))))
    branches = [(visibility, bell)] if visibility > 0.0 else []
    if visibility < 1.0:
        w = (1.0 - visibility) / 4.0
        branches += [(w, PureState.fock(ph(a, b))) for a in "HV" for b in "HV"]
    return InputMixture(tuple(branches))


def product_pair(mode_s: int, mode_i: int, tag: int = 0) -> InputMixture:
    return InputMixture.from_state(PureState.fock(((mode_s, tag), (mode_i, tag))))


def _product_state(states: Sequence[PureState]) -> PureState:
    """Creation-operator product of the factors applied to the vacuum, so photons sharing a
    (mode, tag) bunch with the correct sqrt(m!) factors; normalised afterwards."""
    acc: dict[Photons, complex] = {}
    factor_fact = [
        [math.prod(math.factorial(c) for c in Counter(t.photons).values()) for t in s.terms]
        for s in states
    ]
    for combo in product(*[range(len(s.terms)) for s in states]):
        amp = 1.0 + 0j
        photons: list[tuple[int, int]] = []
        denom = 1
        for s, k, ff in zip(states, combo, factor_fact, strict=True):
            amp *= s.terms[k].amplitude
            photons += s.terms[k].photons
            denom *= ff[k]
        key = _canon(photons)
        num = math.prod(math.factorial(c) for c in Counter(key).values())
        acc[key] = acc.get(key, 0j) + amp * math.sqrt(num / denom)
    return PureState(tuple(Term(a, p) for p, a in acc.items())).normalised()


def _mode_mixture(
    builder: Callable[[int], "InputMixture | PureState"], tag: int, pairs: int
) -> list[tuple[float, PureState]]:
    single = _as_mixture(builder(tag))
    br = single.branches
    total = single.total_weight
    if total <= 0.0:
        raise FockError("single-pair state has zero weight")
    out = []
    for combo in combinations_with_replacement(range(len(br)), pairs):
        mult = math.factorial(pairs) // math.prod(
            math.factorial(c) for c in Counter(combo).values()
        )
        w = float(mult) * math.prod(br[i][0] / total for i in combo)
        out.append((w, _product_state([br[i][1] for i in combo])))
    return out


def _partitions(n: int, max_parts: int, largest: int | None = None):
    if n == 0:
        yield ()
        return
    largest = n if largest is None else largest
    if max_parts == 0:
        return
    for first in range(min(n, largest), 0, -1):
        for rest in _partitions(n - first, max_parts - 1, first):
            yield (first, *rest)


def _log_nb(n: int, mu: float, k: float) -> float:
    x = mu / k
    return (
        math.lgamma(n + k)
        - math.lgamma(k)
        - math.lgamma(n + 1)
        + n * math.log(x)
        - (k + n) * math.log1p(x)
    )


def pair_number_pmf(
    n: int, mean_pairs: float, schmidt_number: float = 1.0, statistics: str = "auto"
) -> float:
    if statistics == "auto":
        statistics = "thermal"
    if mean_pairs == 0.0:
        return 1.0 if n == 0 else 0.0
    if statistics == "poisson":
        return math.exp(-mean_pairs + n * math.log(mean_pairs) - math.lgamma(n + 1))
    if statistics == "thermal":
        return math.exp(_log_nb(n, mean_pairs, schmidt_number))
    raise FockError(f"statistics must be 'thermal', 'poisson' or 'auto', got {statistics!r}")


def pair_number_mixture(
    single_pair_state_builder: Callable[[int], "InputMixture | PureState"],
    mean_pairs: float,
    schmidt_number: float = 1.0,
    n_max: int = 4,
    statistics: str = "auto",
) -> InputMixture:
    """Mixture over the number n of pairs in one detection window, n = 0..n_max.

    ``single_pair_state_builder(tag)`` returns the one-pair state with all its photons in tag
    class ``tag``. Pair number: thermal (negative binomial, mean ``mean_pairs``, K =
    ``schmidt_number`` modes; K = 1 is single-mode thermal) or Poisson. Given n pairs, the pair
    occupations of the K equal-weight Schmidt modes are Bose-Einstein distributed (thermal) or
    multinomial (Poisson); every Schmidt mode is one tag class shared by both photons of a pair
    and by all pairs in that mode (bunching). The tail weight beyond ``n_max`` is the result's
    ``omitted_weight``."""
    if mean_pairs < 0.0:
        raise FockError("mean_pairs must not be negative")
    if schmidt_number < 1.0:
        raise FockError("schmidt_number must be at least 1")
    if n_max < 0:
        raise FockError("n_max must not be negative")
    stat = "thermal" if statistics == "auto" else statistics
    if stat not in ("thermal", "poisson"):
        raise FockError(f"statistics must be 'thermal', 'poisson' or 'auto', got {statistics!r}")
    k = max(1, round(schmidt_number))
    notes: list[str] = []
    if abs(schmidt_number - k) > 1e-12:
        notes.append(
            f"Schmidt number {schmidt_number:g} rounded to {k} Schmidt modes for tag "
            "assignment; pair-number weights keep the exact K"
        )
    probe = _as_mixture(single_pair_state_builder(0))
    if len(probe.branches) > 1 and n_max >= 2:
        notes.append(
            "mixed single-pair state: pairs in one Schmidt mode are combined as independent "
            "branch choices, then symmetrised"
        )
    exact_k = schmidt_number
    branches: list[tuple[float, PureState]] = []
    for n in range(n_max + 1):
        pn = pair_number_pmf(n, mean_pairs, exact_k, stat)
        if pn <= 0.0:
            continue
        if n == 0:
            branches.append((pn, PureState((Term(1.0, ()),))))
            continue
        n_comp = math.comb(n + k - 1, n)
        for part in _partitions(n, k):
            ell = len(part)
            slots = math.prod(range(k, k - ell, -1)) // math.prod(
                math.factorial(c) for c in Counter(part).values()
            )
            if stat == "thermal":
                wc = slots / n_comp
            else:
                wc = (
                    slots
                    * math.factorial(n)
                    / math.prod(math.factorial(c) for c in part)
                    / float(k) ** n
                )
            per_mode = [_mode_mixture(single_pair_state_builder, j, c) for j, c in enumerate(part)]
            for combo in product(*per_mode):
                w = wc * math.prod(c[0] for c in combo)
                state = combo[0][1] if len(combo) == 1 else _product_state([c[1] for c in combo])
                branches.append((pn * w, state))
    return InputMixture(tuple(branches), tuple(notes))


def product_mixture(mixtures: Sequence["InputMixture | PureState"]) -> InputMixture:
    """Independent sources: branch weights multiply, states are creation-operator products
    (photons sharing a (mode, tag) bunch)."""
    mixes = [_as_mixture(m) for m in mixtures]
    if not mixes:
        raise FockError("product of no input states")
    notes = tuple(n for m in mixes for n in m.notes)
    out: list[tuple[float, PureState]] = []
    for combo in product(*[m.branches for m in mixes]):
        w = math.prod(c[0] for c in combo)
        if w <= 0.0:
            continue
        out.append((w, combo[0][1] if len(combo) == 1 else _product_state([c[1] for c in combo])))
    return InputMixture(tuple(out), notes)
