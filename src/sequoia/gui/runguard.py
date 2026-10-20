class RunGuard:
    """Ties an analysis tab's results to the inputs they were computed for.

    A run captures, when it is launched, the document key (``revision_source``: document
    identity, edit revision and library) and the tab's own settings (``config_key``). Results
    are shown as current only while both still match; a run reset or cancelled before it
    finishes is dropped (``run_is_current``).
    """

    revision_source = None

    def _guard_init(self) -> None:
        self._generation = 0
        self._pending: int | None = None
        self._launch_key = None
        self._shown_key = None

    def document_key(self):
        return self.revision_source() if self.revision_source is not None else None

    def config_key(self):
        return None

    def current_key(self):
        return (self.document_key(), self.config_key())

    def begin_run(self) -> None:
        self._pending = self._generation
        self._launch_key = self.current_key()

    def run_is_current(self) -> bool:
        return self._pending == self._generation

    def accept_run(self) -> bool:
        """Mark the finished run's results as shown; True when its inputs changed meanwhile."""
        self._shown_key = self._launch_key
        return self.results_outdated()

    def results_outdated(self) -> bool:
        return self._shown_key is not None and self._shown_key != self.current_key()

    def invalidate(self) -> None:
        self._generation += 1
        self._shown_key = None
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.cancel()

    def track_revision(self, _outcome=None) -> bool:
        return self.results_outdated()

    def refresh_stale(self, *_args) -> None:
        self.set_outcome(None)

    def stale_text(self, text: str) -> str:
        return f"outdated - the setup or the settings changed since this run. {text}".strip()
