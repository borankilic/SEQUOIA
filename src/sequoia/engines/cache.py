"""Run-time support shared by the classical and rate engines.

- ``Cancelled`` and ``check_cancel``: cooperative cancellation between stages.
- ``record_warnings``: thread-safe capture of the data-layer warnings. A process-wide
  ``warnings.showwarning`` hook (installed once) routes warnings of the recorded categories to the
  recorder of the thread that raised them, so concurrent runs never see each other's warnings and
  no global filter is toggled per run.
- ``RunCache``: parsed parts and ``emit_pairs`` results reused across runs. A component opts in to
  emission caching with two optional methods that this module reads with ``getattr``:
  ``emit_cache_key(lights)`` returns a hashable key for everything that changes the emission
  except the parameters and settings (which are always part of the key), or None to disable
  caching; ``emit_rate_scale(lights)`` (default 1.0) returns the factor by which the emitted pair
  rate scales linearly (for example the pump power), so a cached emission is reused when only that
  factor changed. A component whose rate scale is one of its settings lists that setting in
  ``emit_scale_settings`` so that it is left out of the key. The key also contains the context's
  ``min_power_for_warnings``; issues replayed from a cached emission are re-labelled with the
  subject of the current instance, and their text must not depend on the scale (the issues of
  the first run are kept, so a diagnostic that depends on the scale itself must not be emitted
  by a component that declares ``emit_rate_scale``).
"""

import math
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from sequoia.components.base import Component, Context
from sequoia.components.detectors import DetectorConfigWarning
from sequoia.data import ExtrapolationWarning
from sequoia.errors import Issue, SetupError
from sequoia.library.store import PART_FILE, Library, Part
from sequoia.light.pairs import PairEmission

DEFAULT_MEMORY_LIMIT = 3 * 2**30
RECORDED = (ExtrapolationWarning, DetectorConfigWarning)


class Cancelled(Exception):
    """Raised by an engine when its ``should_cancel`` callable returns True."""


class ResourceLimitError(SetupError):
    pass


def check_cancel(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise Cancelled("simulation cancelled")


def check_memory(needed: float, limit: float, what: str, advice: str) -> None:
    if needed > limit:
        raise ResourceLimitError(
            f"{what} needs about {needed / 2**30:.3g} GiB of memory, above the limit of "
            f"{limit / 2**30:.3g} GiB; {advice}"
        )


class CapturedWarning(NamedTuple):
    category: type
    message: str


_local = threading.local()
_hook_lock = threading.Lock()
_original_showwarning: Any = None


def _dispatch(message, category, filename, lineno, file=None, line=None):
    stack = getattr(_local, "stack", None)
    if stack and issubclass(category, RECORDED):
        stack[-1].append(CapturedWarning(category, str(message)))
        return
    original = _original_showwarning
    if original is not None:
        original(message, category, filename, lineno, file, line)


def _ensure_hook() -> None:
    global _original_showwarning
    with _hook_lock:
        if warnings.showwarning is not _dispatch:
            _original_showwarning = warnings.showwarning
            warnings.showwarning = _dispatch
        front = warnings.filters[: len(RECORDED)]
        if not all(
            f[0] == "always" and f[1] is None and f[2] in RECORDED and f[3] is None for f in front
        ) or {f[2] for f in front} != set(RECORDED):
            for category in RECORDED:
                warnings.filterwarnings("always", category=category)


@contextmanager
def record_warnings() -> Iterator[list[CapturedWarning]]:
    _ensure_hook()
    stack = _local.__dict__.setdefault("stack", [])
    bucket: list[CapturedWarning] = []
    stack.append(bucket)
    try:
        yield bucket
    finally:
        stack.pop()


def _directory_stamp(path) -> tuple:
    entries = []
    try:
        for p in sorted(path.rglob("*")):
            if p.is_file():
                st = p.stat()
                entries.append((str(p.relative_to(path)), st.st_mtime_ns, st.st_size))
    except OSError:
        return ()
    return tuple(entries)


@dataclass
class _Emission:
    emission: PairEmission
    scale: float
    issues: list[Issue]
    warnings: list[CapturedWarning]
    subject: str = ""


@dataclass
class RunCache:
    parts: dict = field(default_factory=dict)
    emissions: dict = field(default_factory=dict)
    part_hits: int = 0
    part_misses: int = 0
    emission_hits: int = 0
    emission_misses: int = 0
    max_emissions: int = 16
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def library(self, library: Library) -> Library:
        if isinstance(library, CachedLibrary) and library.cache is self:
            return library
        return CachedLibrary(library, self)

    def load_part(self, library: Library, ref: str) -> Part:
        directory = library.part_dir(ref)
        key = (str(directory.resolve()), ref)
        stamp = _directory_stamp(directory) if (directory / PART_FILE).exists() else None
        with self._lock:
            hit = self.parts.get(key)
            if hit is not None and stamp is not None and hit[0] == stamp:
                self.part_hits += 1
                return hit[1]
        part = library.load_part(ref)
        with self._lock:
            self.part_misses += 1
            self.parts[key] = (stamp, part)
        return part

    def emit_key(self, comp: Component, lights, ctx: Context | None = None) -> tuple | None:
        hook = getattr(comp, "emit_cache_key", None)
        if hook is None:
            return None
        extra = hook(lights)
        if extra is None:
            return None
        left_out = set(getattr(comp, "emit_scale_settings", ()))
        threshold = None if ctx is None else ctx.min_power_for_warnings
        return (
            type(comp).__qualname__,
            comp.params.model_dump_json(),
            comp.settings.model_dump_json(exclude=left_out),
            extra,
            threshold,
        )

    def lookup_emission(self, key, comp, lights) -> _Emission | None:
        with self._lock:
            entry = self.emissions.get(key)
        if entry is None:
            return None
        scale_hook = getattr(comp, "emit_rate_scale", None)
        now = float(scale_hook(lights)) if scale_hook is not None else 1.0
        if now == entry.scale:
            with self._lock:
                self.emission_hits += 1
            return entry
        if not entry.scale or not math.isfinite(entry.scale) or not math.isfinite(now):
            return None
        with self._lock:
            self.emission_hits += 1
        factor = now / entry.scale
        source = replace(entry.emission.source, rate=entry.emission.source.rate * factor)
        return _Emission(
            replace(entry.emission, source=source), now, entry.issues, entry.warnings, entry.subject
        )

    def store_emission(
        self, key, comp, lights, emission, issues, caught, subject: str = ""
    ) -> None:
        scale_hook = getattr(comp, "emit_rate_scale", None)
        scale = float(scale_hook(lights)) if scale_hook is not None else 1.0
        with self._lock:
            self.emission_misses += 1
            if len(self.emissions) >= self.max_emissions:
                self.emissions.pop(next(iter(self.emissions)))
            self.emissions[key] = _Emission(emission, scale, list(issues), list(caught), subject)


class CachedLibrary(Library):
    def __init__(self, inner: Library, cache: RunCache) -> None:
        super().__init__(inner.root)
        self.inner = inner
        self.cache = cache

    def load_part(self, ref: str) -> Part:
        return self.cache.load_part(self.inner, ref)


def run_library(library: Library, cache: RunCache | None) -> Library:
    return library if cache is None else cache.library(library)


def emit_with_cache(
    cache: RunCache | None, comp: Component, lights, ctx: Context
) -> tuple[PairEmission | None, list[CapturedWarning]]:
    """Call ``comp.emit_pairs`` through the cache; returns the emission and captured warnings."""
    key = cache.emit_key(comp, lights, ctx) if cache is not None else None
    if key is not None:
        entry = cache.lookup_emission(key, comp, lights)
        if entry is not None:
            ctx.issues.extend(
                replace(i, subject=ctx.subject) if i.subject == entry.subject else i
                for i in entry.issues
            )
            return entry.emission, list(entry.warnings)
    before = len(ctx.issues)
    with record_warnings() as caught:
        emission = comp.emit_pairs(lights, ctx)
    if key is not None and emission is not None:
        cache.store_emission(key, comp, lights, emission, ctx.issues[before:], caught, ctx.subject)
    return emission, list(caught)
