# M4 specification: GUI shell and schematic editor (PySide6)

Read `docs/PLAN.md` sections 5 (GUI), 6 (layout), 7 (M4), `docs/conventions.md`, and the existing
code: `src/sequoia/gui/` (empty shell), `src/sequoia/setup/{model,io,validate,resolve}.py`,
`src/sequoia/library/store.py`, `src/sequoia/components/base.py`, `src/sequoia/engines/`.
M0-M3 are done and green (`uv run pytest`, ~140 s). All GUI code lives under `src/sequoia/gui/`,
all GUI tests under `tests/gui/` (pytest-qt, `QT_QPA_PLATFORM=offscreen` is set by tests/conftest.py;
add `tests/gui/__init__.py`-free helpers under `tests/gui/`).

## Ground rules for every agent

- Touch only your files. Other agents edit other files concurrently. Never git add/commit/checkout/
  stash. Re-read files before editing; targeted edits; format only your own files.
- No comments/docstrings unless explaining something non-obvious. Match existing style.
- The physics engines are not your concern: use `simulate`/`simulate_rates` and the registry
  through the APIs. Do not edit src outside `src/sequoia/gui/`.
- GUI code must never block the UI thread with physics; keep widgets testable headless.
- Report: what you built, how you tested it (screenshots via `QWidget.grab()` saved to the scratchpad
  directory are encouraged and you should look at them), open issues.

## Architecture

```
SetupDocument (QObject, owns Setup + QUndoStack + Library)   gui/document.py
   ^ edits via commands                    ^ signals changed / dirty_changed
CanvasScene/CanvasView (QGraphicsScene)  gui/canvas/*.py   Palette gui/palette.py
Inspector (schema-driven forms)          gui/inspector/*.py
SimulationController (debounce, worker)  gui/simulation.py ; ResultsDock gui/results.py
MainWindow wires everything             gui/main_window.py
```

### SetupDocument contract (gui/document.py)

```python
class SetupDocument(QObject):
    changed = Signal()            # after any edit/undo/redo/load
    dirty_changed = Signal(bool)
    structure_changed = Signal()  # instances/links/probes added or removed
    def __init__(self, library: Library, setup: Setup | None = None, path: Path | None = None)
    setup: Setup                  # read-only property; never mutate in place
    library: Library
    path: Path | None; dirty: bool; undo_stack: QUndoStack
    def component(self, instance_id) -> Component | None      # cached instantiate; None if invalid
    def ports(self, instance_id) -> tuple[PortSpec, ...]
    def new_id(self, base: str) -> str                         # unique, valid identifier
    # edits: each pushes one QUndoCommand (mergeable where noted), returns affected id
    def add_instance(self, part_ref, position, rotation=0.0, instance_id=None) -> str
    def remove_instance(self, instance_id)         # also removes links/probes that reference it
    def move_instance(self, instance_id, position)  # merge consecutive moves of the same id
    def rotate_instance(self, instance_id, rotation_deg); def flip_instance(self, instance_id)
    def set_setting(self, instance_id, name, value); def set_override(self, instance_id, name, value)
    def add_link(self, kind, src, dst, **fields) -> str    # kind: free_space | fiber | cable
    def remove_link(self, link_id); def set_link_field(self, link_id, name, value)
    def add_probe(self, part_ref, link_id, probe_id=None) -> str; def remove_probe(self, probe_id)
    def set_name(self, name); def set_grid_cell(self, scalar)
    def set_analyses(self, analyses: list[dict])
    def can_connect(self, src, dst) -> tuple[LinkKind | None, str]   # kind chosen from port kinds
    def save(self, path=None); @classmethod def open(cls, path, library)
    def macro(self, text) -> context manager grouping edits into one undo step
```

Values are the raw file-level values (strings such as `"45 deg"`, numbers) validated through the
same pydantic models as the file loader (`Setup.model_validate`): an edit that makes the
document invalid raises `ValueError` with the message and changes nothing. Setup validation errors
that do not stop the model from existing (dangling ports etc.) are not edit errors.

### Canvas geometry contract (gui/canvas)

- Scene units are grid cells (1 cell = 40 px). `Instance.position` is the item's centre in cells,
  `rotation` degrees counter-clockwise on screen, snapping optional (toggle, default on, 0.5 cell).
- A component item is a rounded rectangle with the type name and instance id. Unrotated: input
  ports on the left edge, output ports on the right edge, evenly spaced in `ports()` order;
  electrical ports are drawn as small squares, free-space as circles, fiber as diamonds, with
  colours per kind. Ports rotate with the item (rotation in 90 deg steps via shortcut `R` and
  `Shift+R`, free rotation by an inspector field with 15 deg snapping in the canvas; `F` flips).
- Links: free-space = straight line (width 2) between port anchors, colour by kind; fiber =
  smooth cubic curve; cable = thin dashed line. Each link shows a length label (`100 mm`, `2 m`,
  `5 ns`) at its midpoint; double-click edits it. Free-space lock-to-canvas draws the label in
  italics with the canvas-derived length.
- Link tool: drag from a port to another port. Allowed iff `SetupDocument.can_connect` says so
  (port kinds match, out -> in, free ports); on release over empty canvas nothing is created.
  Invalid targets are highlighted red during the drag with the reason in the status bar.
- Selection: click, rubber band, Delete removes, copy/paste duplicates instances (new ids,
  offset one cell) with links among them, Undo/Redo via the document's stack (Ctrl+Z / Ctrl+Shift+Z).
- The scene is a view of the document: it rebuilds or patches items on `structure_changed` and
  updates geometry on `changed`; it writes edits back only through document methods.

### Inspector contract (gui/inspector)

Schema-driven forms from `Component.Params`/`Component.Settings` pydantic models and link models.
Field widgets: `Scalar` (number + unit combo/free text validated through pint, dimension hint from
the annotation), `Scalar | Curve` (switch between constant and curve), `Curve`/`Surface`
(read-only pyqtgraph preview + 'Import...' using `sequoia.library.importers.table`, editable table
later), `Literal` (combo), bool (checkbox), float/int, `list[int]`, `dict[str, Scalar]`, optional
fields (clear button), `FileRef`. Provenance shown as a tooltip/badge. Instance inspector edits
settings (and overrides, in an expandable group) through the document; link inspector edits
length/medium/fiber/connector loss/polarization/delay.

### Simulation controller contract (gui/simulation.py, gui/results.py)

On `SetupDocument.changed`, debounce ~150 ms, then run `simulate` (classical) and, when the setup
has rate devices (`engines.rates.has_rate_devices`), `simulate_rates`, in a background worker
(use `QThreadPool`/`QRunnable` with the engines' `should_cancel` hook; a newer request cancels the
older one; results delivered by signal on the UI thread). Validation errors are shown, not raised.
The engines accept a `RunCache` (see `sequoia.engines.cache`): keep one per document.
Results dock tabs for M4: Diagnostics (issues; click selects the component/link), Links (power,
mode sizes, DOP table), Counts (channels singles, analyses). Richer plots are M5.

## Work packages

- **G1** `gui/document.py` + `tests/gui/test_document.py`: the document, undo/redo, errors,
  open/save equivalence (`save` then `open` equals the model; saved YAML equals what
  `setup.io.save_setup` writes), grouped macros, `can_connect`.
- **G2** `gui/canvas/` + `tests/gui/test_canvas.py`: items, ports, links, link tool, snapping,
  rotation, flip, selection, copy/paste, scene <-> document sync.
- **G3** `gui/palette.py`, `gui/inspector/`, `tests/gui/test_inspector.py`,
  `tests/gui/test_palette.py`: palette (category tree from `Library.index()` + search, drag onto
  the canvas using a MIME type `application/x-sequoia-part` carrying the part ref; double-click
  also adds at view centre via a signal), schema-driven inspector.
- **G4** `gui/simulation.py`, `gui/results.py`, `tests/gui/test_simulation.py`: controller,
  worker, results dock.
- **G5 (supervisor/integration)** `gui/main_window.py`, `gui/app.py`, `cli.py` file argument,
  menus/shortcuts/recent files/status bar/dock layout persistence, end-to-end test building
  `setups/hbt_heralded.yaml` from scratch through the document + canvas API and comparing the
  saved YAML with the example.

## Done when (plan M4)

`hbt_heralded` can be built from scratch in the GUI and saving gives a YAML equivalent to the
example (same instances, parts, links, lengths, settings); the pytest-qt tests pass; the app
starts (`uv run sequoia [setup.yaml]`), edits live-update diagnostics without freezing the UI.
