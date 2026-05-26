"""
solution_picker_dialog.py — Browse Pareto solutions and load a mask into QGIS.

Auto-discovers all pareto_*.csv files in the output directory.  For each solution
in the selected objective the user can:
  • Load as Vector Layer  — creates a polygon memory layer (one feature per
    restored cell) from the selection grid stored in the companion .npz file;
    requires a session JSON in the same directory for georeferencing.
  • Load as Raster        — adds the best/knee mask TIF as a raster layer.

Green row = knee (balanced) solution.
"""

import csv
import glob
import json
import os

import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QFrame,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsField, QgsFields, QgsProject,
    QgsCoordinateReferenceSystem,
)
from qgis.PyQt.QtCore import QVariant


# ── objective catalogue ────────────────────────────────────────────────────────
# csv_suffix → (display_label, mask_suffix)
_OBJ_MAP = {
    "pareto_full":       ("FULL  (3-way: MESH × IIC × Cost)", "best_pareto3d_mask"),
    "pareto_mesh_cost":  ("MESH × Cost",                       "best_mesh_cost_mask"),
    "pareto_mesh_iic":   ("MESH × IIC",                        "best_mesh_iic_mask"),
    "pareto_iic_cost":   ("IIC × Cost",                        "best_iic_cost_mask"),
    "pareto_mesh":       ("MESH  (single-objective)",           "best_mesh_mask"),
    "pareto_iic":        ("IIC  (single-objective)",            "best_iic_mask"),
    "pareto_cost":       ("Cost  (single-objective)",           "best_cost_mask"),
}

_DISPLAY_COLS = [
    ("rank",        "Rank"),
    ("mesh",        "MESH"),
    ("iic",         "IIC"),
    ("cost",        "Cost (R$)"),
    ("cost_per_ha", "Cost/ha"),
    ("n_cells",     "Cells"),
    ("is_knee",     "Knee ★"),
]


class SolutionPickerDialog(QDialog):
    """Show Pareto solutions and load the selected one into QGIS."""

    def __init__(self, iface, out_dir: str, stem: str, parent=None):
        super().__init__(parent, Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("FOREMOST — Load Optimization Results")
        self.setMinimumSize(820, 520)
        self._iface     = iface
        self._out_dir   = out_dir
        self._stem      = stem
        self._solutions: list[dict] = []
        self._build_ui()
        self._discover_csvs()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        title = QLabel("Select a Pareto Solution to Load")
        font  = QFont()
        font.setBold(True)
        font.setPointSize(12)
        title.setFont(font)
        lay.addWidget(title)

        # objective selector
        obj_row = QHBoxLayout()
        obj_row.addWidget(QLabel("Objective:"))
        self._obj_combo = QComboBox()
        self._obj_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._obj_combo.currentIndexChanged.connect(self._on_obj_changed)
        obj_row.addWidget(self._obj_combo)
        obj_row.addStretch()
        lay.addLayout(obj_row)

        lay.addWidget(QLabel(
            "Green row = knee (balanced) solution  •  "
            "Click to select · Double-click to generate and load as raster."
        ))

        self._table = QTableWidget()
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        lay.addWidget(self._table, stretch=1)

        # separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        lay.addWidget(line)

        # TIF file picker (fallback / raster load)
        tif_row = QHBoxLayout()
        tif_row.addWidget(QLabel("Raster mask:"))
        self._tif_combo = QComboBox()
        self._tif_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        tif_row.addWidget(self._tif_combo, stretch=1)
        lay.addLayout(tif_row)

        self._info_lbl = QLabel("")
        self._info_lbl.setWordWrap(True)
        self._info_lbl.setStyleSheet("color: #888; font-style: italic;")
        lay.addWidget(self._info_lbl)

        # buttons
        btn_row = QHBoxLayout()
        self._vec_btn = QPushButton("Load Selected as Vector Layer")
        self._vec_btn.setToolTip(
            "Creates a polygon memory layer for the selected Pareto solution.\n"
            "Requires the _selections.npz file and a session JSON for georeferencing."
        )
        self._vec_btn.setDefault(True)
        self._vec_btn.clicked.connect(self._on_load_vector)
        btn_row.addWidget(self._vec_btn)

        self._raster_btn = QPushButton("Load Best as Raster")
        self._raster_btn.setToolTip(
            "Loads the knee/best-solution TIF mask as a raster layer."
        )
        self._raster_btn.clicked.connect(self._on_load_raster)
        btn_row.addWidget(self._raster_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    # ── discovery ────────────────────────────────────────────────────────────

    def _discover_csvs(self):
        """Populate the objective combo with every available pareto CSV."""
        found = []
        for suffix, (label, _mask_sfx) in _OBJ_MAP.items():
            path = os.path.join(self._out_dir, f"{self._stem}_{suffix}.csv")
            if os.path.exists(path):
                found.append((label, suffix))

        if not found:
            # Try any _pareto_*.csv matching the stem
            for path in sorted(glob.glob(
                    os.path.join(self._out_dir, f"{self._stem}_pareto_*.csv"))):
                base = os.path.splitext(os.path.basename(path))[0]
                suffix = base[len(self._stem) + 1:]  # strip "stem_"
                if suffix not in _OBJ_MAP:
                    found.append((suffix.replace("_", " ").upper(), suffix))

        if not found:
            self._info_lbl.setText(
                f"No pareto CSV files found in:\n  {self._out_dir}\n"
                "Run the optimizer first, then come back here."
            )
            self._vec_btn.setEnabled(False)
            self._raster_btn.setEnabled(False)
            return

        self._obj_combo.blockSignals(True)
        for label, suffix in found:
            self._obj_combo.addItem(label, userData=suffix)
        self._obj_combo.blockSignals(False)
        self._on_obj_changed()

    def _on_obj_changed(self):
        suffix = self._obj_combo.currentData()
        if not suffix:
            return
        csv_path = os.path.join(self._out_dir, f"{self._stem}_{suffix}.csv")
        self._load_csv(csv_path)
        self._refresh_tif_combo(suffix)

    def _load_csv(self, csv_path: str):
        self._solutions.clear()
        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline="", encoding="utf-8") as fh:
                    lines = [l for l in fh if not l.startswith("#")]
                self._solutions = list(csv.DictReader(lines))
                self._populate_table()
                self._info_lbl.setText(
                    f"{len(self._solutions)} solutions  —  {os.path.basename(csv_path)}"
                )
            except Exception as exc:
                self._info_lbl.setText(f"Could not read CSV: {exc}")
        else:
            self._info_lbl.setText(f"CSV not found: {csv_path}")

        # enable/disable vector load based on selections.npz availability
        npz_path = self._npz_path_for_suffix(self._obj_combo.currentData() or "")
        has_npz = os.path.exists(npz_path)
        has_georef = bool(self._find_session_json())
        self._vec_btn.setEnabled(bool(self._solutions) and has_npz and has_georef)
        if self._solutions and not has_npz:
            self._info_lbl.setText(
                self._info_lbl.text() +
                "\n  (selections.npz not found — re-run optimizer to enable vector load)"
            )
        elif self._solutions and not has_georef:
            self._info_lbl.setText(
                self._info_lbl.text() +
                "\n  (session JSON not found — georef unavailable; vector load disabled)"
            )

    def _refresh_tif_combo(self, suffix: str):
        self._tif_combo.clear()
        mask_sfx = _OBJ_MAP.get(suffix, (None, None))[1]
        preferred = []
        if mask_sfx:
            p = os.path.join(self._out_dir, f"{self._stem}_{mask_sfx}.tif")
            if os.path.exists(p):
                preferred.append(p)
        # also show all mask TIFs in the dir as fallback
        all_masks = sorted(glob.glob(os.path.join(self._out_dir, f"{self._stem}*mask*.tif")))
        for path in preferred + [p for p in all_masks if p not in preferred]:
            self._tif_combo.addItem(os.path.basename(path), path)
        if self._tif_combo.count() == 0:
            self._tif_combo.addItem("(no TIF files found)")
            self._raster_btn.setEnabled(False)
        else:
            self._raster_btn.setEnabled(True)

    def _npz_path_for_suffix(self, suffix: str) -> str:
        return os.path.join(self._out_dir, f"{self._stem}_{suffix}_selections.npz")

    def _find_session_json(self) -> "str | None":
        """Return path to the most recent session JSON in out_dir (or parent dirs)."""
        for search_dir in [self._out_dir,
                           os.path.dirname(self._out_dir)]:
            hits = sorted(glob.glob(os.path.join(search_dir,
                                                  f"{self._stem}_session_*.json")),
                          reverse=True)
            if hits:
                return hits[0]
            # also try without stem prefix
            hits = sorted(glob.glob(os.path.join(search_dir, "*_session_*.json")),
                          reverse=True)
            if hits:
                return hits[0]
        return None

    # ── table ─────────────────────────────────────────────────────────────────

    def _populate_table(self):
        if not self._solutions:
            self._table.setRowCount(0)
            return
        keys, headers = zip(*_DISPLAY_COLS)
        self._table.setColumnCount(len(keys))
        self._table.setHorizontalHeaderLabels(list(headers))
        self._table.setRowCount(len(self._solutions))

        knee_row = None
        for row, sol in enumerate(self._solutions):
            is_knee = sol.get("is_knee", "0") == "1"
            if is_knee:
                knee_row = row
            for col, key in enumerate(keys):
                val = sol.get(key, "")
                try:
                    if key in ("mesh", "iic"):
                        val = f"{float(val):.4f}"
                    elif key == "cost":
                        val = f"{float(val):,.0f}"
                    elif key == "cost_per_ha":
                        val = f"{float(val):,.2f}"
                    elif key == "is_knee":
                        val = "★" if is_knee else ""
                except (ValueError, TypeError):
                    pass
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if is_knee:
                    item.setBackground(QColor(180, 240, 180))
                self._table.setItem(row, col, item)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)

        if knee_row is not None:
            self._table.selectRow(knee_row)
            self._table.scrollToItem(self._table.item(knee_row, 0))
        elif self._table.rowCount() > 0:
            self._table.selectRow(0)

    # ── load handlers ─────────────────────────────────────────────────────────

    def _selected_row(self) -> int:
        rows = self._table.selectedItems()
        return self._table.row(rows[0]) if rows else 0

    def _on_load_vector(self):
        row = self._selected_row()
        suffix = self._obj_combo.currentData() or ""
        npz_path = self._npz_path_for_suffix(suffix)
        session_json = self._find_session_json()

        if not os.path.exists(npz_path):
            self._info_lbl.setText("selections.npz not found. Re-run the optimizer.")
            return
        if not session_json:
            self._info_lbl.setText("No session JSON found — cannot determine grid georef.")
            return

        try:
            with open(session_json, encoding="utf-8") as fh:
                sess = json.load(fh)
            georef = sess.get("georef", {})
            ext = georef.get("extent")
            crs_auth = georef.get("crs", "EPSG:3857")
            N = int(sess.get("N", 0))
            if not ext or len(ext) != 4 or N <= 0:
                self._info_lbl.setText("Session JSON missing georef data.")
                return
        except Exception as exc:
            self._info_lbl.setText(f"Could not read session JSON: {exc}")
            return

        try:
            npz = np.load(npz_path)
            key = f"sel_{row}"
            if key not in npz:
                self._info_lbl.setText(
                    f"Solution #{row + 1} not in selections.npz "
                    f"(available: 0–{len(npz.files) - 1})."
                )
                return
            sel_grid = npz[key]   # (N, N) uint8
        except Exception as exc:
            self._info_lbl.setText(f"Could not load selections.npz: {exc}")
            return

        sol = self._solutions[row] if row < len(self._solutions) else {}
        rank = sol.get("rank", row + 1)
        cost = sol.get("cost", "?")
        n_cells = sol.get("n_cells", "?")
        obj_label = self._obj_combo.currentText()
        layer_name = (f"FOREMOST {obj_label} rank{rank} "
                      f"({n_cells} cells, R${float(cost):,.0f})"
                      if cost != "?" else f"FOREMOST {obj_label} rank{rank}")

        try:
            vlyr = self._build_vector_layer(sel_grid, ext, N, crs_auth, layer_name, sol)
        except Exception as exc:
            self._info_lbl.setText(f"Could not build vector layer: {exc}")
            return

        # Store solution metadata as custom properties BEFORE addMapLayer so that
        # the dock's _on_active_layer_changed() sees them the moment the signal fires.
        try:
            vlyr.setCustomProperty("foremost_solution_cost",    cost_val)
            vlyr.setCustomProperty("foremost_solution_rank",    rank)
            vlyr.setCustomProperty("foremost_solution_n_cells", int(sel_grid.sum()))
        except Exception:
            pass

        QgsProject.instance().addMapLayer(vlyr)
        self._info_lbl.setText(
            f"✓ Loaded: '{layer_name}'  ({int(sel_grid.sum())} cells)"
        )

    def _build_vector_layer(self, sel_grid, ext, N, crs_auth, layer_name, sol) -> QgsVectorLayer:
        """Create an in-memory polygon layer with one feature per restored cell."""
        xmin, ymin, xmax, ymax = ext
        cell_w = (xmax - xmin) / N
        cell_h = (ymax - ymin) / N

        vlyr = QgsVectorLayer("Polygon", layer_name, "memory")
        vlyr.setCrs(QgsCoordinateReferenceSystem(crs_auth))

        pr = vlyr.dataProvider()
        fields = QgsFields()
        fields.append(QgsField("row",      QVariant.Int))
        fields.append(QgsField("col",      QVariant.Int))
        fields.append(QgsField("rank",     QVariant.Int))
        fields.append(QgsField("cost",     QVariant.Double))
        fields.append(QgsField("mesh",     QVariant.Double))
        fields.append(QgsField("iic",      QVariant.Double))
        pr.addAttributes(fields)
        vlyr.updateFields()

        rank = int(sol.get("rank", 0))
        try: cost_val  = float(sol.get("cost",  0))
        except: cost_val = 0.0
        try: mesh_val  = float(sol.get("mesh",  0))
        except: mesh_val = 0.0
        try: iic_val   = float(sol.get("iic",   0))
        except: iic_val = 0.0

        feats = []
        rows_idx, cols_idx = np.where(sel_grid == 1)
        for r, c in zip(rows_idx.tolist(), cols_idx.tolist()):
            x0 = xmin + c * cell_w
            y1 = ymax - r * cell_h         # top edge (row 0 = top)
            y0 = y1 - cell_h               # bottom edge
            x1 = x0 + cell_w
            pts = [QgsPointXY(x0, y0), QgsPointXY(x1, y0),
                   QgsPointXY(x1, y1), QgsPointXY(x0, y1),
                   QgsPointXY(x0, y0)]
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolygonXY([pts]))
            feat.setAttributes([r, c, rank, cost_val, mesh_val, iic_val])
            feats.append(feat)

        pr.addFeatures(feats)
        vlyr.updateExtents()
        return vlyr

    def _on_row_double_clicked(self, row: int, _col: int):
        """Double-click → generate a per-solution GeoTIFF and load it."""
        self._generate_and_load_raster(row)

    def _generate_and_load_raster(self, row: int):
        """Build a GeoTIFF for solution *row* from the .npz selection grid."""
        suffix   = self._obj_combo.currentData() or ""
        npz_path = self._npz_path_for_suffix(suffix)

        if not os.path.exists(npz_path):
            self._info_lbl.setText(
                "selections.npz not found — re-run the optimizer to enable raster generation."
            )
            return

        try:
            npz = np.load(npz_path)
            key = f"sel_{row}"
            if key not in npz:
                self._info_lbl.setText(
                    f"Solution #{row + 1} not in selections.npz "
                    f"(available: 0–{len(npz.files) - 1})."
                )
                return
            sel_grid = npz[key].astype(np.uint8)
        except Exception as exc:
            self._info_lbl.setText(f"Could not load selections.npz: {exc}")
            return

        sol      = self._solutions[row] if row < len(self._solutions) else {}
        rank     = sol.get("rank", row + 1)
        out_name = f"{self._stem}_solution_rank{rank}_mask"
        out_path = os.path.join(self._out_dir, f"{out_name}.tif")

        georef = self._get_georef_for_raster()
        try:
            self._write_geotiff(sel_grid, out_path, georef)
        except Exception as exc:
            self._info_lbl.setText(f"Could not write raster: {exc}")
            return

        from qgis.core import QgsRasterLayer
        lyr = QgsRasterLayer(out_path, out_name)
        if not lyr.isValid():
            self._info_lbl.setText(f"Generated TIF could not be loaded: {out_path}")
            return

        try:
            lyr.setCustomProperty("foremost_solution_cost",
                                  float(sol.get("cost", 0)))
            lyr.setCustomProperty("foremost_solution_rank",
                                  int(sol.get("rank", 0)))
            n = sol.get("n_cells", "")
            if n:
                lyr.setCustomProperty("foremost_solution_n_cells", int(n))
        except Exception:
            pass

        QgsProject.instance().addMapLayer(lyr)
        self._info_lbl.setText(
            f"✓ Generated and loaded: '{out_name}'  ({int(sel_grid.sum())} cells)"
        )

    def _get_georef_for_raster(self):
        """Return (gdal_geotransform, wkt_projection) or None."""
        # 1. Session JSON (most complete)
        session_json = self._find_session_json()
        if session_json:
            try:
                with open(session_json, encoding="utf-8") as fh:
                    sess = json.load(fh)
                georef  = sess.get("georef", {})
                ext     = georef.get("extent")
                crs_str = georef.get("crs", "EPSG:3857")
                N       = int(sess.get("N", 0))
                if ext and N > 0:
                    xmin, ymin, xmax, ymax = ext
                    cell_w = (xmax - xmin) / N
                    cell_h = (ymax - ymin) / N
                    transform = (xmin, cell_w, 0, ymax, 0, -cell_h)
                    from osgeo import osr
                    srs = osr.SpatialReference()
                    srs.SetFromUserInput(crs_str)
                    return transform, srs.ExportToWkt()
            except Exception:
                pass

        # 2. Existing best-mask TIF (copy its geotransform/projection)
        tif = self._tif_combo.currentData()
        if tif and os.path.isfile(tif):
            try:
                from osgeo import gdal
                ds = gdal.Open(tif)
                if ds:
                    result = ds.GetGeoTransform(), ds.GetProjection()
                    ds = None
                    return result
            except Exception:
                pass

        return None

    def _write_geotiff(self, sel_grid: np.ndarray, out_path: str, georef):
        """Write a binary N×N uint8 array as a GeoTIFF."""
        from osgeo import gdal
        N      = sel_grid.shape[0]
        driver = gdal.GetDriverByName("GTiff")
        ds     = driver.Create(out_path, N, N, 1, gdal.GDT_Byte,
                               ["COMPRESS=LZW", "TILED=YES"])
        if georef:
            transform, wkt = georef
            ds.SetGeoTransform(transform)
            if wkt:
                ds.SetProjection(wkt)
        band = ds.GetRasterBand(1)
        band.WriteArray(sel_grid)
        band.SetNoDataValue(0)
        ds.FlushCache()
        ds = None

    def _on_load_raster(self):
        tif = self._tif_combo.currentData()
        if tif and os.path.isfile(tif):
            from qgis.core import QgsRasterLayer
            name = os.path.splitext(os.path.basename(tif))[0]
            lyr  = QgsRasterLayer(tif, name)
            if lyr.isValid():
                # Find the knee/best solution for this raster's metadata
                sol_dict = next(
                    (s for s in self._solutions if s.get("is_knee", "0") == "1"),
                    self._solutions[0] if self._solutions else {},
                )
                try:
                    lyr.setCustomProperty("foremost_solution_cost",
                                         float(sol_dict.get("cost", 0)))
                    lyr.setCustomProperty("foremost_solution_rank",
                                         int(sol_dict.get("rank", 0)))
                    n = sol_dict.get("n_cells", "")
                    if n:
                        lyr.setCustomProperty("foremost_solution_n_cells", int(n))
                except Exception:
                    pass
                QgsProject.instance().addMapLayer(lyr)
                self._info_lbl.setText(f"✓ Loaded raster: '{name}'")
                return
        self._info_lbl.setText("Could not load the selected raster file. Check the path.")
