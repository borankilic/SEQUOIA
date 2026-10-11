"""Wall-clock benchmark of the rate engine on the three example setups.

    uv run python scripts/benchmark_m3.py

Columns: no cache; a cold RunCache; a warm RunCache with the emission cache emptied (only the
parsed parts are reused); a warm RunCache (parts and emission); and the same warm cache after
doubling the pump power (the crystal emission is reused and its rate rescaled).
"""

import statistics
import time
from pathlib import Path

import sequoia.components.builtin  # noqa: F401
from sequoia.engines.cache import RunCache
from sequoia.engines.rates import simulate_rates
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[1]
REPEATS = 3
EXAMPLES = {
    "hbt_heralded": ("pump", "overrides", "power"),
    "hom": ("pump", "overrides", "power"),
    "bell_test": ("source", "settings", "pump_power"),
}


def load(name):
    return load_setup(ROOT / "setups" / f"{name}.yaml").model_copy(deep=True)


def doubled(name, inst_id, kind, field):
    setup = load(name)
    inst = next(i for i in setup.instances if i.id == inst_id)
    current = getattr(inst, kind).get(field)
    if current is None:
        from sequoia.setup.resolve import instantiate

        comp = instantiate(Library(ROOT / "library"), inst)
        holder = comp.params if kind == "overrides" else comp.settings
        current = getattr(holder, field).si
        value = f"{2.0 * current * 1e3} mW"
    else:
        number, unit = str(current).split()
        value = f"{2.0 * float(number)} {unit}"
    setattr(inst, kind, {**getattr(inst, kind), field: value})
    return setup


def once(setup, library, cache=None):
    start = time.perf_counter()
    simulate_rates(setup, library, cache=cache)
    return time.perf_counter() - start


def main():
    library = Library(ROOT / "library")
    print(f"seconds, median of {REPEATS}; wall clock, one process, includes the classical pass")
    header = ("example", "no cache", "cold cache", "warm parts", "warm emission", "power x2")
    print(f"{header[0]:14}" + "".join(f"{h:>15}" for h in header[1:]))
    for name, (inst_id, kind, field) in EXAMPLES.items():
        setup = load(name)
        once(setup, library)
        no_cache = statistics.median(once(setup, library) for _ in range(REPEATS))
        cold = statistics.median(once(setup, library, RunCache()) for _ in range(REPEATS))
        cache = RunCache()
        once(setup, library, cache)
        parts_only = []
        for _ in range(REPEATS):
            cache.emissions.clear()
            parts_only.append(once(setup, library, cache))
        once(setup, library, cache)
        warm = statistics.median(once(setup, library, cache) for _ in range(REPEATS))
        twice = doubled(name, inst_id, kind, field)
        hits = cache.emission_hits
        power = statistics.median(once(twice, library, cache) for _ in range(REPEATS))
        note = "" if cache.emission_hits - hits >= REPEATS else "  (emission NOT reused)"
        print(
            f"{name:14}{no_cache:15.2f}{cold:15.2f}{statistics.median(parts_only):15.2f}"
            f"{warm:15.2f}{power:15.2f}{note}"
        )


if __name__ == "__main__":
    main()
