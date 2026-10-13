class RunGuard:
    """Ties an analysis tab's results to the project revision they were computed for."""

    def _guard_init(self) -> None:
        self._generation = 0
        self._pending: int | None = None
        self._latest_revision: int | None = None
        self._run_revision: int | None = None

    def begin_run(self) -> None:
        self._pending = self._generation
        self._run_revision = self._latest_revision

    def run_is_current(self) -> bool:
        return self._pending == self._generation

    def invalidate(self) -> None:
        self._generation += 1
        self._run_revision = None
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.cancel()

    def track_revision(self, outcome) -> bool:
        """Record the latest simulated revision; True when the shown results are out of date."""
        if outcome is not None:
            self._latest_revision = getattr(outcome, "revision", None)
        return self._run_revision is not None and self._latest_revision != self._run_revision

    def stale_text(self, text: str) -> str:
        return f"outdated - the setup changed since this run. {text}".strip()
