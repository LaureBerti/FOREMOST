"""
auto_label_raster_dialog.py — Non-modal dialog for value-mapping auto-label.

Workflow:
  1. Select a raster layer and band.
  2. Click "Scan raster values" → unique integer values appear as mapping rows.
  3. Map each value to a class via the dropdown.
  4. Click "Auto-label from Raster" → labels all UNLABELED cells.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton,
    QComboBox, QScrollArea, QWidget, QDialogButtonBox, QSizePolicy,
    QProgressBar, QApplication,
)
from qgis.PyQt.QtCore import Qt, QEventLoop
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsCoordinateTransform, QgsPointXY,
)

from .constants import CLASS_NONE, CLASS_HAB, CLASS_RA, CLASS_NR, CLASS_LABEL, FLD_CLASS


class AutoLabelRasterDialog(QDialog):
    """Non-modal floating dialog for band-value → class mapping."""

    def __init__(self, grid_manager, status_fn, parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("FOREMOST — Auto-label from Raster")
        self.setMinimumWidth(380)
        self.setSizeGripEnabled(True)
        self.resize(420, 400)
        self._gm             = grid_manager
        self._status         = status_fn
        self._stop_requested = False
        self._value_mappings: list[tuple[int, QComboBox]] = []
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        # layer + band row
        row_top = QHBoxLayout()
        row_top.addWidget(QLabel("Layer:"))
        self._raster_combo = QComboBox()
        self._raster_combo.setToolTip("Classified raster to read band values from")
        row_top.addWidget(self._raster_combo)
        btn_r = QPushButton("↺")
        btn_r.setFixedWidth(28)
        btn_r.setToolTip("Refresh layer list")
        btn_r.clicked.connect(self._refresh_raster_combo)
        row_top.addWidget(btn_r)
        row_top.addWidget(QLabel("Band:"))
        self._band_spin = QSpinBox()
        self._band_spin.setRange(1, 99)
        self._band_spin.setValue(1)
        self._band_spin.setFixedWidth(44)
        row_top.addWidget(self._band_spin)
        root.addLayout(row_top)

        # scan button
        btn_scan = QPushButton("Scan raster values")
        btn_scan.setToolTip(
            "Sample all grid cells and find unique integer band values.\n"
            "Creates one mapping row per unique value."
        )
        btn_scan.clicked.connect(self._scan_values)
        root.addWidget(btn_scan)

        # inline status label — always visible inside the dialog
        self._dlg_status = QLabel("")
        self._dlg_status.setWordWrap(True)
        self._dlg_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addWidget(self._dlg_status)

        # scrollable mapping rows
        self._mappings_widget = QWidget()
        self._mappings_layout = QVBoxLayout(self._mappings_widget)
        self._mappings_layout.setContentsMargins(0, 0, 0, 0)
        self._mappings_layout.setSpacing(2)

        mappings_scroll = QScrollArea()
        mappings_scroll.setWidgetResizable(True)
        mappings_scroll.setWidget(self._mappings_widget)
        mappings_scroll.setMinimumHeight(60)
        mappings_scroll.setFrameShape(QScrollArea.NoFrame)
        root.addWidget(mappings_scroll, stretch=1)

        # other-values fallback
        row_other = QHBoxLayout()
        row_other.addWidget(QLabel("Other values →"))
        self._other_combo = self._class_combo(CLASS_NONE)
        row_other.addWidget(self._other_combo)
        root.addLayout(row_other)

        # apply button
        btn_apply = QPushButton("Auto-label from Raster")
        btn_apply.setToolTip(
            "Assign each UNLABELED cell the class mapped to its dominant band value."
        )
        btn_apply.clicked.connect(self._apply)
        root.addWidget(btn_apply)

        # wrap content in a scroll area so the dialog is fully scrollable
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setWidget(content)
        content_scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(content_scroll)

        # progress bar + stop button — shown during scan and apply
        prog_row = QHBoxLayout()
        prog_row.setContentsMargins(8, 0, 8, 0)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%  (%v / %m cells)")
        self._progress.setFixedHeight(18)
        self._progress.hide()
        prog_row.addWidget(self._progress)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedWidth(52)
        self._stop_btn.setToolTip("Stop the current operation")
        self._stop_btn.clicked.connect(self._request_stop)
        self._stop_btn.hide()
        prog_row.addWidget(self._stop_btn)
        outer.addLayout(prog_row)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.hide)
        outer.addWidget(btns)

        self._refresh_raster_combo()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _class_combo(self, default=CLASS_NONE) -> QComboBox:
        cb = QComboBox()
        for cls in (CLASS_NONE, CLASS_HAB, CLASS_RA, CLASS_NR):
            cb.addItem(CLASS_LABEL[cls], userData=cls)
        cb.setCurrentIndex(cb.findData(default))
        return cb

    def _refresh_raster_combo(self):
        current = self._raster_combo.currentData()
        self._raster_combo.clear()
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer) and lyr.isValid():
                self._raster_combo.addItem(lyr.name(), lyr.id())
        idx = self._raster_combo.findData(current)
        if idx >= 0:
            self._raster_combo.setCurrentIndex(idx)

    def _request_stop(self):
        self._stop_requested = True
        self._stop_btn.setEnabled(False)

    def _begin_progress(self, total: int):
        self._stop_requested = False
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.setFormat("%p%  (%v / %m cells)")
        self._progress.show()
        self._stop_btn.setEnabled(True)
        self._stop_btn.show()

    def _end_progress(self, done: int):
        self._progress.setValue(done)
        self._progress.hide()
        self._stop_btn.hide()

    def _msg(self, text: str, error: bool = False):
        """Show *text* in the inline dialog status label (always visible)."""
        self._dlg_status.setText(text)
        colour = "#c0392b" if error else "#27ae60"
        self._dlg_status.setStyleSheet(f"color: {colour}; font-style: italic;")
        self._status(text)   # also forward to the dock status bar

    def _get_ctx(self):
        """Return (provider, xform) or None; shows error in the dialog label."""
        if not self._gm.active:
            self._msg("⚠ Create a grid first (use 'Create / Reset Grid' in the dock).", error=True)
            return None
        lid = self._raster_combo.currentData()
        if not lid:
            self._msg("⚠ No raster layer selected.", error=True)
            return None
        raster = QgsProject.instance().mapLayer(lid)
        if raster is None or not raster.isValid():
            self._msg("⚠ Selected raster is no longer valid.", error=True)
            return None
        xform = QgsCoordinateTransform(
            self._gm.layer.crs(), raster.crs(), QgsProject.instance()
        )
        return raster.dataProvider(), xform

    def _dominant_value(self, feat, provider, band: int, xform) -> "int | None":
        """Modal band value over the cell centroid + 3×3 interior sample grid."""
        tally: dict[int, int] = {}
        bbox = feat.geometry().boundingBox()
        cx0, cy0 = bbox.xMinimum(), bbox.yMinimum()
        dx = (bbox.xMaximum() - cx0) / 4
        dy = (bbox.yMaximum() - cy0) / 4
        for pi in range(1, 4):
            for pj in range(1, 4):
                try:
                    pt    = xform.transform(QgsPointXY(cx0 + pi * dx, cy0 + pj * dy))
                    v, ok = provider.sample(pt, band)
                    if ok and v == v:   # v == v catches NaN
                        key = int(round(v))
                        tally[key] = tally.get(key, 0) + 1
                except Exception:
                    pass
        return max(tally, key=tally.__getitem__) if tally else None

    # ── slots ─────────────────────────────────────────────────────────────────

    def _scan_values(self):
        ctx = self._get_ctx()
        if ctx is None:
            return
        provider, xform = ctx
        band  = self._band_spin.value()
        total = self._gm.N ** 2
        self._msg("Scanning raster values…")

        self._begin_progress(total)

        unique: set[int] = set()
        n_feats = n_sampled = 0
        stopped = False
        try:
            for feat in self._gm.layer.getFeatures():
                if self._stop_requested:
                    stopped = True
                    break
                n_feats += 1
                v = self._dominant_value(feat, provider, band, xform)
                if v is not None:
                    unique.add(v)
                    n_sampled += 1
                if n_feats % 50 == 0:
                    self._progress.setValue(n_feats)
                    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        except Exception as exc:
            self._end_progress(n_feats)
            self._msg(f"⚠ Scan error: {exc}", error=True)
            return

        self._end_progress(n_feats)

        # clear old rows
        for _, cb in self._value_mappings:
            cb.setParent(None)
        while self._mappings_layout.count():
            item = self._mappings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._value_mappings.clear()

        if not unique:
            self._msg(
                f"⚠ No values found ({n_feats} cells sampled). "
                "Check that the raster overlaps the grid and the band is correct.",
                error=True,
            )
            return

        for val in sorted(unique):
            row = QHBoxLayout()
            row.setContentsMargins(0, 1, 0, 1)
            lbl = QLabel(f"Value {val}  →")
            lbl.setFixedWidth(80)
            row.addWidget(lbl)
            cb = self._class_combo(CLASS_NONE)
            row.addWidget(cb)
            self._value_mappings.append((val, cb))
            container = QWidget()
            container.setLayout(row)
            container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._mappings_layout.addWidget(container)

        self._msg(
            f"Found {len(unique)} unique values: "
            f"{', '.join(str(v) for v in sorted(unique))}  "
            f"({n_sampled}/{n_feats} cells sampled). "
            "Set mappings then click Auto-label from Raster."
        )

    def _apply(self):
        ctx = self._get_ctx()
        if ctx is None:
            return
        if not self._value_mappings:
            self._msg("⚠ Run 'Scan raster values' first.", error=True)
            return

        provider, xform = ctx
        band     = self._band_spin.value()
        fallback = self._other_combo.currentData()
        mapping  = {val: cb.currentData() for val, cb in self._value_mappings}

        total = self._gm.N ** 2
        self._begin_progress(total)

        layer = self._gm.layer
        layer.startEditing()
        counts  = {CLASS_NONE: 0, CLASS_HAB: 0, CLASS_RA: 0, CLASS_NR: 0}
        skipped = done = 0
        stopped = False
        for feat in layer.getFeatures():
            if self._stop_requested:
                stopped = True
                break
            done += 1
            if int(feat[FLD_CLASS]) != CLASS_NONE:
                skipped += 1
            else:
                dom = self._dominant_value(feat, provider, band, xform)
                cls = mapping.get(dom, fallback) if dom is not None else fallback
                layer.changeAttributeValue(feat.id(), FLD_CLASS, cls)
                counts[cls] = counts.get(cls, 0) + 1
            if done % 50 == 0:
                self._progress.setValue(done)
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        layer.commitChanges()
        layer.triggerRepaint()
        self._end_progress(done)

        suffix = " (stopped early)" if stopped else ""
        self._msg(
            f"Auto-label: {counts[CLASS_HAB]} Habitat, "
            f"{counts[CLASS_RA]} Restorable, "
            f"{counts[CLASS_NR]} Non-Restorable "
            f"({skipped} already labeled, skipped){suffix}."
        )
