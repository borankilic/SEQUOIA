from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QTabWidget, QVBoxLayout, QWidget

from sequoia.fock.from_setup import ComponentRow, InputDescription
from sequoia.gui.fock_common import ResultTable, num
from sequoia.gui.fock_results import SetupResultsView


class SetupWorkspace(QWidget):
    """Setup mode: the component inventory and photon inputs of the drawn circuit (collapsed by
    default) above the result views."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.components_table = ResultTable(
            ["Id", "Type", "Category", "Part", "Settings and key parameters", "Ports (H, V modes)"],
            "Import from setup to list the components and links of the drawn circuit.",
        )
        self.inputs_table = ResultTable(
            [
                "Source",
                "Kind",
                "Photons",
                "Mean pairs\nper window",
                "Schmidt K",
                "Window\ns",
                "Omitted weight",
                "Description",
            ],
            "Import from setup to list the photon inputs found in the drawn circuit.",
            (
                "",
                "fock: N photons per pulse; pair: thermal pair-number statistics",
                "Photon numbers simulated",
                "Mean number of pairs per detection window",
                "Schmidt number (equal-weight modes)",
                "Detection window (pulse period or 1 / bandwidth)",
                "Input weight beyond the pair-number truncation, not simulated",
                "",
            ),
        )
        self.info_tabs = QTabWidget()
        self.info_tabs.addTab(self.components_table, "Components")
        self.info_tabs.addTab(self.inputs_table, "Inputs")
        self.info_tabs.setVisible(False)
        self.results = SetupResultsView()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.info_tabs)
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def clear(self) -> None:
        self.components_table.clear_rows()
        self.inputs_table.clear_rows()
        self.results.clear()

    def show_inventory(
        self, components: list[ComponentRow], inputs: list[InputDescription]
    ) -> None:
        self.components_table.set_rows(
            components,
            lambda _r, row: tuple(
                (t, None)
                for t in (row.id, row.type, row.category, row.part, row.summary, row.ports)
            ),
        )
        self.inputs_table.set_rows(
            inputs,
            lambda _r, d: (
                (d.source, None),
                (d.kind, None),
                (d.photons, None),
                ("" if d.mean_pairs is None else num(d.mean_pairs), d.mean_pairs),
                ("" if d.schmidt is None else num(d.schmidt), d.schmidt),
                (f"{d.window_s:.6g} s", d.window_s),
                (num(d.omitted), d.omitted),
                (d.text, None),
            ),
        )
