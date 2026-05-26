"""
auto_label_threshold_dialog.py — Non-modal dialog for brightness-threshold auto-label.

Labels UNLABELED cells based on the fraction of bright pixels (any band value > 20).
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton,
    QComboBox, QDialogButtonBox, QProgressBar, QApplication,
)
from qgis.PyQt.QtCore import Qt, QEventLoop
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsCoordinateTransform, QgsPointXY,
)

from .constants import CLASS_NONE, CLASS_HAB, CLASS_RA, CLASS_NR, CLASS_LABEL, FLD_CLASS


class AutoLabelThresholdDialog(QDialog):
    """Non-modal floating dialog for brightness-threshold based auto-label."""

    def __init__(self, grid_manager, status_fn, parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("FOREMOST — Threshold-based Auto-label")
        self.setMinimumWidth(420)
        self._gm             = grid_manager
        self._status         = status_fn
        self._stop_requested = False
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # raster layer selector
        row_layer = QHBoxLayout()
        row_layer.addWidget(QLabel("Layer:"))
        self._raster_combo = QComboBox()
        self._raster_combo.setToolTip(
            "Raster used to compute brightness (max of all bands per pixel)"
        )
        row_layer.addWidget(self._raster_combo)
        btn_r = QPushButton("↺")
        btn_r.setFixedWidth(28)
        btn_r.setToolTip("Refresh layer list")
        btn_r.clicked.connect(self._refresh_raster_combo)
        row_layer.addWidget(btn_r)
        root.addLayout(row_layer)

        def _spin(default):
            s = QSpinBox()
            s.setRange(1, 100)
            s.setValue(default)
            s.setSuffix(" %")
            s.setFixedWidth(68)
            return s

        # Habitat row
        row_hab = QHBoxLayout()
        row_hab.addWidget(QLabel("Habitat   ≥"))
        self._hab_thr = _spin(50)
        self._hab_thr.setToolTip("Label unlabeled cell Habitat when bright-pixel fraction ≥ this value")
        row_hab.addWidget(self._hab_thr)
        row_hab.addStretch()
        btn_hab = QPushButton("Auto-label Habitat")
        btn_hab.clicked.connect(lambda: self._apply(CLASS_HAB))
        row_hab.addWidget(btn_hab)
        root.addLayout(row_hab)

        # Restorable row
        row_ra = QHBoxLayout()
        row_ra.addWidget(QLabel("Restorable"))
        self._ra_min = _spin(20)
        self._ra_min.setToolTip("Lower bound of Restorable brightness range (inclusive)")
        row_ra.addWidget(self._ra_min)
        row_ra.addWidget(QLabel("–"))
        self._ra_max = _spin(50)
        self._ra_max.setToolTip("Upper bound of Restorable range (exclusive)")
        row_ra.addWidget(self._ra_max)
        row_ra.addStretch()
        btn_ra = QPushButton("Auto-label Restorable")
        btn_ra.clicked.connect(lambda: self._apply(CLASS_RA))
        row_ra.addWidget(btn_ra)
        root.addLayout(row_ra)

        # Non-Restorable row
        row_nr = QHBoxLayout()
        row_nr.addWidget(QLabel("Non-Rest.  <"))
        self._nr_thr = _spin(20)
        self._nr_thr.setToolTip("Label unlabeled cell Non-Restorable when bright-pixel fraction < this value")
        row_nr.addWidget(self._nr_thr)
        row_nr.addStretch()
        btn_nr = QPushButton("Auto-label Non-Restorable")
        btn_nr.clicked.connect(lambda: self._apply(CLASS_NR))
        row_nr.addWidget(btn_nr)
        root.addLayout(row_nr)

        prog_row = QHBoxLayout()
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
        self._stop_btn.setToolTip("Stop the current auto-label operation")
        self._stop_btn.clicked.connect(self._request_stop)
        self._stop_btn.hide()
        prog_row.addWidget(self._stop_btn)
        root.addLayout(prog_row)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.hide)
        root.addWidget(btns)

        self._refresh_raster_combo()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _refresh_raster_combo(self):
        current = self._raster_combo.currentData()
        self._raster_combo.clear()
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer) and lyr.isValid():
                self._raster_combo.addItem(lyr.name(), lyr.id())
        idx = self._raster_combo.findData(current)
        if idx >= 0:
            self._raster_combo.setCurrentIndex(idx)

    def _bright_fraction(self, feat, provider, n_bands: int, xform) -> float:
        """Fraction of 5×5 sampled pixels where any band value > 20."""
        bbox     = feat.geometry().boundingBox()
        cx0, cy0 = bbox.xMinimum(), bbox.yMinimum()
        dx       = (bbox.xMaximum() - cx0) / 6
        dy       = (bbox.yMaximum() - cy0) / 6
        bright = total = 0
        for pi in range(1, 6):
            for pj in range(1, 6):
                pt = xform.transform(QgsPointXY(cx0 + pi * dx, cy0 + pj * dy))
                for band in range(1, n_bands + 1):
                    v, ok = provider.sample(pt, band)
                    if ok and v > 20:
                        bright += 1
                        break
                total += 1
        return bright / total if total else 0.0

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

    def _end_progress(self):
        self._progress.hide()
        self._stop_btn.hide()

    # ── slot ─────────────────────────────────────────────────────────────────

    def _apply(self, class_code: int):
        if not self._gm.active:
            self._status("Create a grid first.")
            return
        lid = self._raster_combo.currentData()
        if not lid:
            self._status("No raster layer selected.")
            return
        raster = QgsProject.instance().mapLayer(lid)
        if raster is None or not raster.isValid():
            self._status("Selected raster is no longer valid.")
            return

        provider = raster.dataProvider()
        n_bands  = raster.bandCount()
        xform    = QgsCoordinateTransform(
            self._gm.layer.crs(), raster.crs(), QgsProject.instance()
        )

        hab_thr = self._hab_thr.value() / 100.0
        ra_min  = self._ra_min.value()  / 100.0
        ra_max  = self._ra_max.value()  / 100.0
        nr_thr  = self._nr_thr.value()  / 100.0

        def _matches(bf: float) -> bool:
            if class_code == CLASS_HAB: return bf >= hab_thr
            if class_code == CLASS_RA:  return ra_min <= bf < ra_max
            if class_code == CLASS_NR:  return bf < nr_thr
            return False

        total = self._gm.N ** 2
        self._begin_progress(total)

        layer = self._gm.layer
        layer.startEditing()
        labeled = done = 0
        stopped = False
        for feat in layer.getFeatures():
            if self._stop_requested:
                stopped = True
                break
            done += 1
            if int(feat[FLD_CLASS]) != CLASS_NONE:
                pass
            elif _matches(self._bright_fraction(feat, provider, n_bands, xform)):
                layer.changeAttributeValue(feat.id(), FLD_CLASS, class_code)
                labeled += 1
            if done % 50 == 0:
                self._progress.setValue(done)
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        layer.commitChanges()
        layer.triggerRepaint()

        self._progress.setValue(done)
        self._end_progress()
        suffix = " (stopped early)" if stopped else ""
        self._status(
            f"Threshold auto-label: {labeled} cells → {CLASS_LABEL[class_code]}{suffix}."
        )
