"""
foremost_annotator.py — QGIS plugin entry point.
"""

import os
from qgis.PyQt.QtWidgets import QAction, QToolBar
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsProject


class ForemorAnnotatorPlugin:
    """FOREMOST Annotator QGIS plugin."""

    def __init__(self, iface):
        self.iface = iface
        self.dock  = None
        self.action = None
        self._prev_tool = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        self.action = QAction(
            QIcon(icon_path),
            "FOREMOST Annotator",
            self.iface.mainWindow(),
        )
        self.action.setCheckable(True)
        self.action.setToolTip("Open / close the FOREMOST Annotator panel")
        self.action.triggered.connect(self._toggle_dock)

        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&FOREMOST", self.action)

    def unload(self):
        self.iface.removePluginMenu("&FOREMOST", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None

    # ── dock ──────────────────────────────────────────────────────────────────

    def _toggle_dock(self, checked: bool):
        if checked:
            self._open_dock()
        else:
            self._close_dock()

    def _open_dock(self):
        if self.dock is None:
            from .annotator_dock import AnnotatorDock
            self.dock = AnnotatorDock(self.iface, parent=self.iface.mainWindow())
            self.dock.closed.connect(self._on_dock_closed)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.show()
        self.action.setChecked(True)

    def _close_dock(self):
        if self.dock is not None:
            self.dock.hide()
        self.action.setChecked(False)

    def _on_dock_closed(self):
        self.action.setChecked(False)
