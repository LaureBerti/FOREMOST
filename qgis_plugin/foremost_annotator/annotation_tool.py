"""
annotation_tool.py — QgsMapTool for click-and-drag cell painting.
"""

from qgis.gui import QgsMapTool
from qgis.core import QgsPointXY
from qgis.PyQt.QtCore import Qt


class AnnotationTool(QgsMapTool):
    """
    Map tool that translates mouse clicks / drags to grid-cell edits.

    The dock widget is responsible for:
      - holding a reference to ``GridManager``
      - deciding which class_code / cost to apply
      - calling ``grid.set_cell()`` after editing starts
    """

    def __init__(self, canvas, grid_manager, get_paint_params, on_cell_changed=None):
        """
        Parameters
        ----------
        canvas           : QgsMapCanvas
        grid_manager     : GridManager
        get_paint_params : callable() → (class_code: int, cost: float) | None
            Called each time a cell is painted; return None to suppress painting.
        on_cell_changed  : callable() | None
            Called once after a paint stroke is committed (mouse release).
        """
        super().__init__(canvas)
        self._gm = grid_manager
        self._get_params = get_paint_params
        self._on_cell_changed = on_cell_changed
        self._painting = False
        self._last_rc  = None          # (row, col) of the last painted cell

    # ── QgsMapTool interface ──────────────────────────────────────────────────

    def canvasPressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._painting = True
        self._last_rc  = None
        self._paint_at(event.mapPoint())

    def canvasMoveEvent(self, event):
        if not self._painting:
            return
        self._paint_at(event.mapPoint())

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._painting = False
            self._last_rc  = None
            if self._gm.layer is not None:
                self._gm.layer.commitChanges()
            if self._on_cell_changed is not None:
                self._on_cell_changed()

    def deactivate(self):
        self._painting = False
        self._last_rc  = None
        super().deactivate()

    # ── internal ─────────────────────────────────────────────────────────────

    def _paint_at(self, map_point: QgsPointXY):
        rc = self._gm.row_col(map_point)
        if rc is None or rc == self._last_rc:
            return
        feat = self._gm.feature_at(map_point)
        if feat is None:
            return
        params = self._get_params()
        if params is None:
            return
        class_code, cost = params
        if self._gm.layer.isEditable() is False:
            self._gm.layer.startEditing()
        self._gm.set_cell(feat, class_code, cost)
        self._last_rc = rc
