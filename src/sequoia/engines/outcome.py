import time
import uuid
from dataclasses import dataclass, field

import sequoia.components.builtin  # noqa: F401  (registers the component types)
from sequoia.engines.cache import Cancelled, RunCache
from sequoia.engines.classical import SimulationResult, simulate
from sequoia.engines.rates import RateResult, has_rate_devices, simulate_rates
from sequoia.errors import Issue, SequoiaError
from sequoia.library.store import Library
from sequoia.measurement import resolve_setup
from sequoia.setup.model import Setup
from sequoia.setup.validate import SetupValidationError


@dataclass
class SimulationOutcome:
    revision: int
    issues: list[Issue] = field(default_factory=list)
    classical: SimulationResult | None = None
    rates: RateResult | None = None
    elapsed: float = 0.0
    run_id: str = ""

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def ok(self) -> bool:
        return self.classical is not None and not self.has_errors


def compute_outcome(
    setup: Setup,
    library: Library,
    cache: RunCache | None,
    should_cancel,
    revision: int = 0,
) -> SimulationOutcome:
    """Validate and simulate; user-level problems become error issues, never exceptions.

    Raises ``Cancelled`` when ``should_cancel`` fires and re-raises unexpected exceptions.
    """
    started = time.perf_counter()
    run_id = f"rates-r{revision}-{uuid.uuid4().hex[:8]}"

    def outcome(issues, **kwargs) -> SimulationOutcome:
        return SimulationOutcome(
            revision=revision,
            elapsed=time.perf_counter() - started,
            issues=[i.for_run("rates", run_id) for i in issues],
            run_id=run_id,
            **kwargs,
        )

    setup = resolve_setup(setup, library)

    try:
        classical = simulate(setup, library, should_cancel=should_cancel, cache=cache)
    except SetupValidationError as exc:
        return outcome(issues=list(exc.issues))
    except Cancelled:
        raise
    except SequoiaError as exc:
        return outcome(issues=[Issue("error", "simulation_failed", "setup", str(exc))])
    issues = list(classical.issues)
    rates = None
    try:
        if has_rate_devices(setup, library, cache, classical.network):
            rates = simulate_rates(
                setup, library, classical=classical, should_cancel=should_cancel, cache=cache
            )
            issues = list(rates.issues)
    except Cancelled:
        raise
    except SequoiaError as exc:
        issues.append(Issue("error", "rates_failed", "setup", str(exc)))
    return outcome(issues=issues, classical=classical, rates=rates)
