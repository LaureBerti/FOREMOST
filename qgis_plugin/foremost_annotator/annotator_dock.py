"""
annotator_dock.py — QgsDockWidget housing the full FOREMOST annotation UI.

Layout (top → bottom):
  ┌─ Grid Setup ────────────────────────────────────────────────────────┐
  │  N (spin)   [Create / Reset Grid]                                   │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─ Auto-Label ────────────────────────────────────────────────────────┐
  │  [Auto-label from Raster…]   [Threshold-based Auto-label…]          │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─ Manual Cell Annotation ────────────────────────────────────────────┐
  │  ○ Unlabeled   ○ Habitat   ○ Restorable   ○ Non-Restorable         │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─ Cost Settings ─────────────────────────────────────────────────────┐
  │  [Compute costs from layers]   [Cost by default]                    │
  │  [Configure Cost Parameters…]                                       │
  │  [Show Cost Map]                                                    │
  │  <color legend>                                                     │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─ Actions ───────────────────────────────────────────────────────────┐
  │  [Reset All Cells]                                                  │
  │  [Save Session]      [Load Session]                                 │
  │  [Export .npy Arrays]   [Export GeoPackage]                         │
  │  Stem: [___________________________] ↻                              │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─ Statistics ────────────────────────────────────────────────────────┐
  │  Last action: …                                                     │
  │  Coverage:    42.3%  (423/1000: Hab 200 | RA 150 | NR 73)          │
  │  Total cost:  $3,450,000                                            │
  │  Avg/cell:    $23,000                                               │
  │  Avg/m²:      $2.30                                                 │
  └─────────────────────────────────────────────────────────────────────┘
  Status bar
"""

import os

from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QLabel, QSpinBox, QComboBox, QSizePolicy,
    QPushButton, QRadioButton, QButtonGroup,
    QFileDialog, QMessageBox, QInputDialog,
    QScrollArea, QFrame, QProgressBar, QApplication,
)
from qgis.PyQt.QtCore import Qt, QTimer, QEventLoop, pyqtSignal
from qgis.PyQt.QtGui import QFont
from qgis.core import (
    QgsProject, QgsCoordinateReferenceSystem,
    QgsRectangle, QgsRasterLayer, QgsVectorLayer, QgsPointXY,
    QgsCoordinateTransform,
)

import numpy as np

from .constants import CLASS_NONE, CLASS_HAB, CLASS_RA, CLASS_NR, CLASS_LABEL
from .grid_manager import GridManager
from .annotation_tool import AnnotationTool
from .cost_model import compute_cell_cost, compute_grid_cost
from .cost_params_dialog import CostParamsDialog
from .npy_exporter import export_arrays, export_gpkg
from .session_manager import save_session, load_session, session_path

# YlOrRd ramp colors matching grid_manager.refresh_cost_layer()
_COST_RAMP = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]


class AnnotatorDock(QDockWidget):
    """Main panel for the FOREMOST Annotator plugin."""

    closed = pyqtSignal()

    def __init__(self, iface, parent=None):
        super().__init__("FOREMOST Annotation & Restoration Optimization", parent)
        self.iface        = iface
        self.gm           = GridManager()
        self._tool        = None
        self._cost_dialog = CostParamsDialog(self)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)

        self._raster_label_dialog    = None
        self._threshold_label_dialog = None
        self._opt_dialog             = None
        self._last_npy_path_stem     = ""   # kept in sync after export / session load
        self._legend_label           = None  # no cost-map legend in current layout
        self._build_ui()

        # force repaint of memory layers on every pan/zoom
        self.iface.mapCanvas().extentsChanged.connect(self._on_extent_changed)

        # detect when user manually removes a FOREMOST layer from the Layers panel
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)

        # Populate the raster combo when layers change
        QgsProject.instance().layersAdded.connect(self._refresh_raster_combo)
        QgsProject.instance().layersRemoved.connect(self._refresh_raster_combo)

        # Show optimizer solution stats when a solution layer is selected
        self.iface.layerTreeView().currentLayerChanged.connect(
            self._on_active_layer_changed
        )

    # ── closeEvent ────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._deactivate_tool()
        try:
            self.iface.mapCanvas().extentsChanged.disconnect(self._on_extent_changed)
        except Exception:
            pass
        try:
            QgsProject.instance().layersRemoved.disconnect(self._on_layers_removed)
        except Exception:
            pass
        self.closed.emit()
        super().closeEvent(event)

    def _on_extent_changed(self):
        """Repaint FOREMOST memory layers after every pan/zoom."""
        if self.gm.layer is not None and self.gm.active:
            self.gm.layer.triggerRepaint()
        if self.gm.cost_layer is not None:
            try:
                if self.gm.cost_layer.isValid():
                    self.gm.cost_layer.triggerRepaint()
            except RuntimeError:
                self.gm.cost_layer = None

    def _on_layers_removed(self, layer_ids: list):
        """Nullify gm references when FOREMOST layers are removed by the user."""
        for attr in ("layer", "cost_layer"):
            lyr = getattr(self.gm, attr)
            if lyr is None:
                continue
            try:
                if lyr.id() in layer_ids:
                    setattr(self.gm, attr, None)
            except RuntimeError:
                setattr(self.gm, attr, None)
        if not self.gm.active:
            self._btn_activate.setEnabled(False)
            self._refresh_stats()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setSpacing(6)
        lay.setContentsMargins(6, 6, 6, 6)

        # Plugin banner
        banner = QLabel("FOREMOST")
        banner_font = QFont()
        banner_font.setBold(True)
        banner_font.setPointSize(16)
        banner.setFont(banner_font)
        banner.setAlignment(Qt.AlignCenter)
        banner.setStyleSheet(
            "color: #1b4332; padding: 4px 0 2px 0;"
        )
        lay.addWidget(banner)
        sub = QLabel("Annotation & Restoration Optimization")
        sub_font = QFont()
        sub_font.setPointSize(9)
        sub_font.setItalic(True)
        sub.setFont(sub_font)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #555; margin-bottom: 4px;")
        lay.addWidget(sub)

        # Row 1 — main raster selector
        raster_row = QHBoxLayout()
        raster_row.addWidget(QLabel("Main raster:"))
        self._main_raster_combo = QComboBox()
        self._main_raster_combo.setToolTip(
            "Select the reference raster that defines the grid extent and CRS"
        )
        self._main_raster_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        raster_row.addWidget(self._main_raster_combo, stretch=1)
        btn_refresh_r = QPushButton("↺")
        btn_refresh_r.setFixedWidth(28)
        btn_refresh_r.setToolTip("Refresh raster list")
        btn_refresh_r.clicked.connect(self._refresh_raster_combo)
        raster_row.addWidget(btn_refresh_r)
        lay.addLayout(raster_row)

        # Row 2 — grid size + create
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid N×N:"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(5, 500)
        self._n_spin.setValue(100)
        self._n_spin.setToolTip("Number of rows / columns in the planning grid")
        grid_row.addWidget(self._n_spin)
        btn_create = QPushButton("Create / Reset Grid")
        btn_create.setToolTip(
            "Create the N×N grid over the extent of the selected main raster"
        )
        btn_create.clicked.connect(self._on_create_grid)
        grid_row.addWidget(btn_create, stretch=1)
        lay.addLayout(grid_row)

        self._refresh_raster_combo()

        lay.addWidget(self._section_header("Annotation"))
        lay.addLayout(self._build_annotation_layout())

        lay.addWidget(self._section_header("Settings"))
        lay.addLayout(self._build_settings_layout())

        lay.addWidget(self._section_header("Cost Computation"))
        lay.addLayout(self._build_cost_computation_layout())

        lay.addWidget(self._section_header("Actions"))
        lay.addLayout(self._build_actions_layout())

        lay.addWidget(self._section_header("Statistics"))
        lay.addLayout(self._build_stats_layout())

        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)
        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QFrame.NoFrame)
        self.setWidget(scroll)

    # ── layout helpers ────────────────────────────────────────────────────────

    def _section_header(self, text: str) -> QLabel:
        lbl  = QLabel(text)
        font = QFont()
        font.setBold(True)
        font.setPointSize(13)
        lbl.setFont(font)
        return lbl

    def _build_annotation_layout(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        # Left column: auto-label + clear
        left = QVBoxLayout()
        btn_raster = QPushButton("Raster-based Auto-Label")
        btn_raster.clicked.connect(self._open_raster_label_dialog)
        left.addWidget(btn_raster)
        btn_thr = QPushButton("Threshold-based Auto-Label")
        btn_thr.clicked.connect(self._open_threshold_label_dialog)
        left.addWidget(btn_thr)
        btn_reset = QPushButton("Clear All Cells")
        btn_reset.clicked.connect(self._on_reset_all)
        left.addWidget(btn_reset)
        left.addStretch()
        row.addLayout(left)

        # Right column: activate button + 2×2 radio grid
        right = QVBoxLayout()
        self._btn_activate = QPushButton("Activate Manual Annotation")
        self._btn_activate.setToolTip(
            "Re-activate the FOREMOST cell-painting tool.\n"
            "Click this after using the QGIS pan / zoom tools."
        )
        self._btn_activate.clicked.connect(self._on_activate_tool)
        self._btn_activate.setEnabled(False)
        right.addWidget(self._btn_activate)

        self._class_group = QButtonGroup(self)
        radio_grid = QGridLayout()
        radio_grid.setSpacing(4)
        rb_hab  = QRadioButton("Habitat")
        rb_ra   = QRadioButton("Restorable")
        rb_nr   = QRadioButton("Non Restorable")
        rb_none = QRadioButton("Unlabelled")
        self._class_group.addButton(rb_hab,  CLASS_HAB)
        self._class_group.addButton(rb_ra,   CLASS_RA)
        self._class_group.addButton(rb_nr,   CLASS_NR)
        self._class_group.addButton(rb_none, CLASS_NONE)
        radio_grid.addWidget(rb_hab,  0, 0)
        radio_grid.addWidget(rb_ra,   0, 1)
        radio_grid.addWidget(rb_nr,   1, 0)
        radio_grid.addWidget(rb_none, 1, 1)
        right.addLayout(radio_grid)
        right.addStretch()
        row.addLayout(right)
        return row

    def _build_settings_layout(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        btn_cost = QPushButton("Cost Parameters")
        btn_cost.clicked.connect(self._open_cost_dialog)
        row.addWidget(btn_cost)
        btn_opt = QPushButton("Optimization Parameters")
        btn_opt.clicked.connect(self._open_opt_dialog)
        row.addWidget(btn_opt)
        return row

    def _build_cost_computation_layout(self) -> QVBoxLayout:
        lay = QVBoxLayout()
        lay.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(8)
        btn_layer = QPushButton("Layer-Based")
        btn_layer.setToolTip(
            "Compute restoration cost for all cells\n"
            "using the spatial formula (elevation / road / water layers)."
        )
        btn_layer.clicked.connect(self._on_compute_cost)
        row.addWidget(btn_layer)
        btn_fixed = QPushButton("Fixed")
        btn_fixed.setToolTip(
            "Fill all Restorable cells with zero cost with a uniform value."
        )
        btn_fixed.clicked.connect(self._on_fill_defaults)
        row.addWidget(btn_fixed)
        lay.addLayout(row)
        btn_clear = QPushButton("Clear Costs")
        btn_clear.setToolTip("Reset all cell cost values to zero.")
        btn_clear.clicked.connect(self._on_clear_costs)
        lay.addWidget(btn_clear)

        self._cost_progress = QProgressBar()
        self._cost_progress.setRange(0, 100)
        self._cost_progress.setValue(0)
        self._cost_progress.setTextVisible(True)
        self._cost_progress.setFormat("%p%  (%v / %m cells)")
        self._cost_progress.setFixedHeight(16)
        self._cost_progress.hide()
        lay.addWidget(self._cost_progress)
        return lay

    def _build_actions_layout(self) -> QVBoxLayout:
        lay = QVBoxLayout()
        lay.setSpacing(6)
        row1 = QHBoxLayout()
        btn_load = QPushButton("Load Session")
        btn_load.clicked.connect(self._on_load_session)
        row1.addWidget(btn_load)
        btn_save = QPushButton("Save Session")
        btn_save.clicked.connect(self._on_save_session)
        row1.addWidget(btn_save)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        btn_npy = QPushButton("Export .npy files")
        btn_npy.clicked.connect(self._on_export_npy)
        row2.addWidget(btn_npy)
        btn_gpkg = QPushButton("Export GeoPackage")
        btn_gpkg.clicked.connect(self._on_export_gpkg)
        row2.addWidget(btn_gpkg)
        lay.addLayout(row2)
        btn_launch = QPushButton("Launch FOREMOST Optimizer")
        btn_launch.setToolTip(
            "Run foremost.py --mode 1 with the parameters configured in\n"
            "'Optimization Parameters'. Shows real-time output in QGIS."
        )
        btn_launch.clicked.connect(self._on_launch_foremost)
        lay.addWidget(btn_launch)

        btn_load_sol = QPushButton("Load Best Solution")
        btn_load_sol.setToolTip(
            "Browse for a folder containing previous optimizer results\n"
            "and open the Pareto solution picker."
        )
        btn_load_sol.clicked.connect(self._on_load_best_solution)
        lay.addWidget(btn_load_sol)
        return lay

    def _build_stats_layout(self) -> QFormLayout:
        form = QFormLayout()
        form.setSpacing(3)
        self._stat_cell_area  = QLabel("—")
        self._stat_coverage   = QLabel("—")
        self._stat_total_cost = QLabel("—")
        self._stat_avg_cell   = QLabel("—")
        self._stat_avg_m2     = QLabel("—")
        self._stat_solution   = QLabel("—")
        for lbl in (self._stat_cell_area, self._stat_coverage,
                    self._stat_total_cost, self._stat_avg_cell, self._stat_avg_m2,
                    self._stat_solution):
            lbl.setWordWrap(True)
        form.addRow("Cell area:",  self._stat_cell_area)
        form.addRow("Coverage:",   self._stat_coverage)
        form.addRow("Total cost:", self._stat_total_cost)
        form.addRow("Avg/cell:",   self._stat_avg_cell)
        form.addRow("Avg/m²:",     self._stat_avg_m2)
        self._stat_solution_lbl = QLabel("Cost of selected solution:")
        form.addRow(self._stat_solution_lbl, self._stat_solution)
        return form

    def _open_raster_label_dialog(self):
        if self._raster_label_dialog is None:
            from .auto_label_raster_dialog import AutoLabelRasterDialog
            self._raster_label_dialog = AutoLabelRasterDialog(
                self.gm, self._status_and_refresh, self
            )
        self._raster_label_dialog.show()
        self._raster_label_dialog.raise_()

    def _open_threshold_label_dialog(self):
        if self._threshold_label_dialog is None:
            from .auto_label_threshold_dialog import AutoLabelThresholdDialog
            self._threshold_label_dialog = AutoLabelThresholdDialog(
                self.gm, self._status_and_refresh, self
            )
        self._threshold_label_dialog.show()
        self._threshold_label_dialog.raise_()

    def _status_and_refresh(self, msg: str):
        """Status callback for auto-label dialogs — updates status AND statistics."""
        self._refresh_stats(msg)

    def _open_cost_dialog(self):
        self._cost_dialog._refresh_spatial_combos()
        self._cost_dialog.show()
        self._cost_dialog.raise_()

    def _open_opt_dialog(self):
        if self._opt_dialog is None:
            from .optimization_settings_dialog import OptimizationSettingsDialog
            self._opt_dialog = OptimizationSettingsDialog(self.gm, self.iface, self)
        if self._last_npy_path_stem:
            self._opt_dialog.prefill_npy_path_stem(self._last_npy_path_stem)
        self._opt_dialog.show()
        self._opt_dialog.raise_()

    def _on_launch_foremost(self):
        import glob as _glob

        # Ensure the settings dialog is initialised so we can read its saved state
        if self._opt_dialog is None:
            from .optimization_settings_dialog import OptimizationSettingsDialog
            self._opt_dialog = OptimizationSettingsDialog(self.gm, self.iface, self)

        # Determine the npy stem: prefer what the opt dialog says,
        # fall back to the last exported stem.
        npy_stem = self._last_npy_path_stem
        dlg_stem = self._opt_dialog._npy_path_edit.text().strip()
        if dlg_stem:
            npy_stem = dlg_stem

        # Also accept launch when explicit per-array paths are configured
        _has_explicit = any(
            edit.text().strip()
            for edit in [
                self._opt_dialog._npy_habitat_edit,
                self._opt_dialog._npy_restorable_edit,
                self._opt_dialog._npy_accessible_edit,
                self._opt_dialog._npy_cost_edit,
            ]
        )

        if not _has_explicit and (not npy_stem or not _glob.glob(npy_stem + "*.npy")):
            QMessageBox.warning(
                self, "Export .npy arrays first",
                "The optimizer cannot be launched because no .npy arrays have been "
                "exported yet.\n\n"
                "Annotate your grid, compute costs, then click 'Export .npy files' "
                "before launching the optimizer.\n\n"
                "Alternatively, set individual .npy file paths in Optimization Parameters.",
            )
            return

        if self._last_npy_path_stem:
            self._opt_dialog.prefill_npy_path_stem(self._last_npy_path_stem)
        cmd = self._opt_dialog.get_command()
        if cmd is None:
            self._status.setText("foremost.py not found — check Optimization Parameters.")
            self._opt_dialog.show()
            self._opt_dialog.raise_()
            return

        # Build the unified session payload (grid state + cost + optimizer params).
        # Passed to launch() so it can write one single JSON — no opt_config file.
        session_data = None
        if self.gm.active:
            try:
                session_data = self.gm.to_dict()
                session_data.update(self._params_extra())
            except Exception as _exc:
                print(f"[FOREMOST] Could not build session data for launch: {_exc}")

        ok, info = self._opt_dialog.launch(session_data=session_data)
        self._status.setText(f"Optimizer running — {info}" if ok else f"Launch failed: {info}")

    def _on_load_best_solution(self):
        import glob as _glob
        folder = QFileDialog.getExistingDirectory(
            self, "Select Optimization Results Folder"
        )
        if not folder:
            return

        # Discover stem from pareto CSV files in the folder
        csv_files = sorted(_glob.glob(os.path.join(folder, "*_pareto_*.csv")))
        if not csv_files:
            QMessageBox.warning(
                self, "No Results Found",
                f"No pareto CSV files found in:\n{folder}\n\n"
                "Run the optimizer first to generate results.",
            )
            return

        # Strip "_pareto_*" suffix to recover the stem (e.g. "run1_pareto_full.csv" → "run1")
        first_name = os.path.basename(csv_files[0])
        idx = first_name.find("_pareto_")
        stem = first_name[:idx] if idx > 0 else os.path.splitext(first_name)[0]

        from .solution_picker_dialog import SolutionPickerDialog
        self._solution_picker = SolutionPickerDialog(
            self.iface, folder, stem, parent=None
        )
        self._solution_picker.show()
        self._solution_picker.raise_()

    def _sync_opt_npy(self, path_stem: str):
        """Store the current npy path+stem and push it to the opt dialog if open."""
        self._last_npy_path_stem = path_stem
        if self._opt_dialog is not None:
            self._opt_dialog.prefill_npy_path_stem(path_stem)

    def _refresh_stats(self, last_action: str = ""):
        """Recompute and display grid statistics."""
        try:
            self._status.objectName()   # raises RuntimeError if C++ widget deleted
        except RuntimeError:
            return
        if not self.gm.active:
            return

        total   = self.gm.N ** 2
        labeled = hab = ra = nr = 0
        total_cost   = 0.0
        ra_with_cost = 0

        for feat in self.gm.layer.getFeatures():
            cls  = int(feat["class_code"] or 0)
            cost = float(feat["cost"] or 0.0)
            if cls == CLASS_HAB:
                labeled += 1; hab += 1
            elif cls == CLASS_RA:
                labeled += 1; ra  += 1
                if cost > 0:
                    total_cost   += cost
                    ra_with_cost += 1
            elif cls == CLASS_NR:
                labeled += 1; nr  += 1

        csm          = self.gm.cell_size_m()
        cell_area_m2 = csm ** 2
        cov_pct      = labeled / total * 100 if total else 0.0
        avg_cell     = total_cost / ra_with_cost if ra_with_cost else 0.0
        avg_m2       = avg_cell / cell_area_m2   if cell_area_m2 > 0 else 0.0

        try:
            _, currency = self._cost_params()
        except Exception:
            currency = "R$"

        if last_action:
            self._status.setText(last_action)

        # Cell area — format with appropriate unit (ha when ≥ 10 000 m²)
        if cell_area_m2 >= 10_000:
            area_txt = f"{cell_area_m2:,.0f} m²  ({cell_area_m2 / 10_000:.2f} ha)"
        else:
            area_txt = f"{cell_area_m2:,.0f} m²"
        self._stat_cell_area.setText(area_txt)

        self._stat_coverage.setText(
            f"{cov_pct:.1f}%  ({labeled}/{total}:  "
            f"Hab {hab} | RA {ra} | NR {nr})"
        )
        self._stat_total_cost.setText(f"{currency}{total_cost:,.0f}")
        self._stat_avg_cell.setText(f"{currency}{avg_cell:,.0f}")
        self._stat_avg_m2.setText(f"{currency}{avg_m2:.2f}")

    def _on_active_layer_changed(self, layer):
        """Update 'Cost of selected solution' when a FOREMOST solution layer is selected."""
        try:
            self._stat_solution.objectName()   # guard deleted widget
        except RuntimeError:
            return

        sol_cost = sol_rank = sol_n_cells = None

        if layer is not None:
            # Primary path: custom property set by solution_picker_dialog before addMapLayer
            try:
                prop = layer.customProperty("foremost_solution_cost")
                if prop is not None:
                    sol_cost    = float(prop)
                    sol_rank    = int(layer.customProperty("foremost_solution_rank") or 0)
                    sol_n_cells = int(layer.customProperty("foremost_solution_n_cells") or 0)
            except Exception:
                pass

            # Fallback: vector layer with cost + rank fields (backward compat)
            if sol_cost is None and hasattr(layer, "fields"):
                field_names = [f.name() for f in layer.fields()]
                if "cost" in field_names and "rank" in field_names:
                    try:
                        feats = list(layer.getFeatures())
                        if feats:
                            sol_cost    = float(feats[0]["cost"])
                            sol_rank    = int(feats[0]["rank"])
                            sol_n_cells = len(feats)
                    except Exception:
                        pass

        if sol_cost is None:
            self._stat_solution.setText("—")
            return

        try:
            _, currency = self._cost_params()
        except Exception:
            currency = "R$"

        csm          = self.gm.cell_size_m() if self.gm.active else 0.0
        cell_ha      = csm ** 2 / 10_000 if csm > 0 else 0.0
        total_ha     = (sol_n_cells or 0) * cell_ha
        cost_per_ha  = sol_cost / total_ha if total_ha > 0 else 0.0

        rank_str    = f"rank {sol_rank} · " if sol_rank else ""
        cells_str   = f"{sol_n_cells} cells · " if sol_n_cells else ""
        ha_str      = f"{currency}{cost_per_ha:,.0f}/ha" if cost_per_ha else ""
        self._stat_solution.setText(
            f"{currency}{sol_cost:,.0f}  ({rank_str}{cells_str}{ha_str})".rstrip("  ()")
        )

    # ── raster combo helpers ──────────────────────────────────────────────────

    def _refresh_raster_combo(self, *_):
        """Repopulate the main-raster combo from current project rasters."""
        current_id = self._main_raster_combo.currentData()
        self._main_raster_combo.clear()
        self._main_raster_combo.addItem("— select main raster —", None)
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer) and lyr.isValid():
                if lyr.id() not in {
                    getattr(self.gm.layer, "id", lambda: None)(),
                    getattr(self.gm.cost_layer, "id", lambda: None)(),
                }:
                    self._main_raster_combo.addItem(lyr.name(), lyr.id())
        idx = self._main_raster_combo.findData(current_id)
        if idx >= 0:
            self._main_raster_combo.setCurrentIndex(idx)

    def _selected_main_raster(self) -> "QgsRasterLayer | None":
        lid = self._main_raster_combo.currentData()
        if not lid:
            return None
        lyr = QgsProject.instance().mapLayer(lid)
        return lyr if isinstance(lyr, QgsRasterLayer) and lyr.isValid() else None

    # ── slot: create grid ─────────────────────────────────────────────────────

    def _on_create_grid(self):
        raster = self._selected_main_raster()
        if raster is None:
            self._status.setText(
                "Select a main raster first — the grid will match its extent and CRS."
            )
            return
        N      = self._n_spin.value()
        crs    = raster.crs()
        extent = raster.extent()
        self.gm.create(extent, N, crs)
        self._activate_tool()
        # Zoom canvas to the new grid
        canvas = self.iface.mapCanvas()
        padded = QgsRectangle(extent)
        padded.scale(1.05)
        canvas.setExtent(padded)
        canvas.refresh()
        csm = self.gm.cell_size_m()
        msg = (
            f"Grid created: {N}×{N} cells over '{raster.name()}', "
            f"~{csm:,.0f} m/cell ({crs.authid()})."
        )
        self._status.setText(msg)
        self._refresh_stats(f"Grid created {N}×{N}")

    # ── slot: annotation tool ─────────────────────────────────────────────────

    def _activate_tool(self):
        self._deactivate_tool()
        canvas = self.iface.mapCanvas()
        self._tool = AnnotationTool(
            canvas, self.gm, self._get_paint_params,
            on_cell_changed=lambda: self._refresh_stats("Cell painted"),
        )
        canvas.setMapTool(self._tool)
        # detect when another tool takes over
        try:
            canvas.mapToolSet.disconnect(self._on_map_tool_changed)
        except Exception:
            pass
        canvas.mapToolSet.connect(self._on_map_tool_changed)
        self._btn_activate.setEnabled(False)
        self._status.setText("Annotation tool active — click cells to paint.")

    def _on_activate_tool(self):
        if not self.gm.active:
            self._status.setText("Create a grid first.")
            return
        self._activate_tool()

    def _on_map_tool_changed(self, new_tool, _old_tool):
        """Called by QGIS when the active map tool changes."""
        if new_tool is not self._tool:
            self._btn_activate.setEnabled(True)
            self._status.setText(
                "Annotation tool inactive — click 'Activate Manual Annotation' to resume painting."
            )

    def _deactivate_tool(self):
        if self._tool is not None:
            self.iface.mapCanvas().unsetMapTool(self._tool)
            self._tool = None

    def _get_paint_params(self) -> "tuple[int, float] | None":
        try:
            cls = self._class_group.checkedId()
        except RuntimeError:
            return None  # C++ object deleted (e.g. during plugin reload)
        if cls == -1:
            return None   # no class selected — suppress painting
        return cls, 0.0   # costs filled afterward via Compute / Cost by default

    # ── cost helpers ──────────────────────────────────────────────────────────

    def _cost_params(self) -> tuple[dict, str]:
        """Return (kwargs for cost functions, currency_symbol)."""
        try:
            p = self._cost_dialog.get_params()
        except (RuntimeError, AttributeError):
            from .cost_params_dialog import CostParamsDialog
            p = dict(CostParamsDialog._DEFAULTS)
        sym = p.pop("currency_symbol", "R$")
        p.pop("currency_name", None)
        return p, sym

    # ── slot: reset all ───────────────────────────────────────────────────────

    def _on_reset_all(self):
        if not self.gm.active:
            return
        reply = QMessageBox.question(
            self, "Clear All Cells",
            "Clear all cells to Unlabeled?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.gm.set_class_bulk(CLASS_NONE)
            self._refresh_stats("All cells reset to Unlabeled.")

    # ── slot: cost by default (fixed uniform value) ───────────────────────────

    def _on_fill_defaults(self):
        if not self.gm.active:
            return
        from .constants import FLD_CLASS, FLD_COST

        params, sym = self._cost_params()
        csm         = self.gm.cell_size_m()
        estimate    = compute_cell_cost(csm, **params)

        val, ok = QInputDialog.getDouble(
            self, "Cost by default",
            f"Default cost per Restorable cell ({sym}):\n"
            f"(pre-filled from: {csm:,.0f} m cell, "
            f"{params.get('tree_spacing_m', 2.5)} m spacing, "
            f"{sym}{params.get('tree_unit_cost', 15)}/tree — no terrain factors)",
            value=estimate,
            min=0.0,
            max=1e15,
            decimals=2,
        )
        if not ok:
            return

        total = self.gm.N ** 2
        self._cost_progress.setRange(0, total)
        self._cost_progress.setValue(0)
        self._cost_progress.setFormat("%p%  (%v / %m cells)")
        self._cost_progress.show()

        layer = self.gm.layer
        attrs = {}
        done = 0
        for feat in layer.getFeatures():
            done += 1
            if int(feat[FLD_CLASS]) == CLASS_RA and float(feat[FLD_COST]) == 0.0:
                attrs[feat.id()] = {FLD_COST: val}
            if done % 100 == 0:
                self._cost_progress.setValue(done)
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

        self._cost_progress.setValue(total)
        self._cost_progress.hide()

        if not attrs:
            self._status.setText("No Restorable cells with missing cost.")
            return
        layer.dataProvider().changeAttributeValues(attrs)
        layer.reload()
        layer.triggerRepaint()
        self.gm.refresh_cost_layer()
        msg = f"Filled {len(attrs)} RA cells with {sym}{val:,.0f}."
        self._status.setText(msg)
        self._refresh_stats(msg)

    # ── slot: clear all costs ─────────────────────────────────────────────────

    def _on_clear_costs(self):
        if not self.gm.active:
            return
        from .constants import FLD_COST
        layer = self.gm.layer
        attrs = {feat.id(): {FLD_COST: 0.0} for feat in layer.getFeatures()}
        layer.dataProvider().changeAttributeValues(attrs)
        layer.reload()
        layer.triggerRepaint()
        self.gm.refresh_cost_layer()
        self._refresh_stats("All costs cleared.")

    # ── progress helpers ──────────────────────────────────────────────────────

    def _cost_progress_update(self, value: int, total: int):
        """Push a progress value and pump the event loop."""
        self._cost_progress.setValue(value)
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    # ── slot: compute costs from spatial layers (vectorized formula) ──────────

    def _on_compute_cost(self):
        if not self.gm.active:
            return
        from .constants import FLD_CLASS, FLD_COST

        N           = self.gm.N
        params, sym = self._cost_params()
        csm         = self.gm.cell_size_m()

        # 4 stages × N² each → total bar range = 4 × N²
        stage_size = N * N
        total_steps = 4 * stage_size
        self._cost_progress.setRange(0, total_steps)
        self._cost_progress.setValue(0)
        self._cost_progress.setFormat("Stage %v / " + str(total_steps))
        self._cost_progress.show()

        # Stage 1 — build ra_mask
        self._status.setText("Stage 1/4: Reading cell classes…")
        ra_mask  = np.zeros((N, N), dtype=bool)
        feat_ids: dict[tuple[int, int], int] = {}
        for i, feat in enumerate(self.gm.layer.getFeatures()):
            r, c = int(feat["row"]), int(feat["col"])
            feat_ids[(r, c)] = feat.id()
            if int(feat[FLD_CLASS]) == CLASS_RA:
                ra_mask[r, c] = True
            if i % 100 == 0:
                self._cost_progress_update(i, total_steps)
        self._cost_progress_update(stage_size, total_steps)

        if not ra_mask.any():
            self._cost_progress.hide()
            self._status.setText("No Restorable cells to compute cost for.")
            return

        ids = self._cost_dialog.get_spatial_layer_ids()

        # Stage 2 — elevation
        self._status.setText("Stage 2/4: Sampling elevation…")
        elev_norm = self._sample_elevation_grid(
            ids.get("elevation"), prog_offset=stage_size, prog_total=total_steps
        )

        # Stage 3 — road distances
        self._status.setText("Stage 3/4: Sampling road distances…")
        dist_road = self._sample_vector_distances(
            ids.get("road"), prog_offset=2 * stage_size, prog_total=total_steps
        )

        # Stage 4 — water distances
        self._status.setText("Stage 4/4: Sampling water distances…")
        dist_water = self._sample_vector_distances(
            ids.get("water"), prog_offset=3 * stage_size, prog_total=total_steps
        )

        self._cost_progress.setValue(total_steps)
        self._cost_progress.hide()

        if elev_norm  is None: print("[cost] No elevation layer — elevation factor = 1.0")
        if dist_road  is None: print("[cost] No road layer — road penalty = 0")
        if dist_water is None: print("[cost] No water layer — water penalty = 0")

        cost_grid = compute_grid_cost(
            N, csm, ra_mask,
            elev_norm=elev_norm, dist_road=dist_road, dist_water=dist_water,
            **params,
        )

        layer = self.gm.layer
        attrs = {}
        for (r, c), fid in feat_ids.items():
            if ra_mask[r, c]:
                attrs[fid] = {FLD_COST: float(cost_grid[r, c])}

        layer.dataProvider().changeAttributeValues(attrs)
        layer.reload()
        layer.triggerRepaint()
        self.gm.refresh_cost_layer()

        n     = int(ra_mask.sum())
        total = float(cost_grid[ra_mask].sum())
        avg   = total / n if n else 0.0
        msg   = (
            f"Computed {n} RA cell costs — "
            f"total {sym}{total:,.0f} | avg {sym}{avg:,.0f}/cell "
            f"| ~{csm:,.0f} m/cell"
        )
        self._status.setText(msg)
        self._refresh_stats(msg)

    # ── spatial sampling helpers ──────────────────────────────────────────────

    def _sample_elevation_grid(
        self, layer_id: "str | None",
        prog_offset: int = 0, prog_total: int = 0,
    ) -> "np.ndarray | None":
        """
        Sample *layer_id* (raster) at N×N cell centroids.
        Returns N×N float64 in [0, 1] (2nd–98th percentile normalised), or None.
        """
        if not layer_id:
            if prog_total:
                self._cost_progress_update(prog_offset + self.gm.N ** 2, prog_total)
            return None
        raster = QgsProject.instance().mapLayer(layer_id)
        if not isinstance(raster, QgsRasterLayer) or not raster.isValid():
            if prog_total:
                self._cost_progress_update(prog_offset + self.gm.N ** 2, prog_total)
            return None

        N        = self.gm.N
        provider = raster.dataProvider()
        xform    = QgsCoordinateTransform(
            self.gm.layer.crs(), raster.crs(), QgsProject.instance()
        )

        raw = np.full((N, N), np.nan, dtype=np.float64)
        for i, feat in enumerate(self.gm.layer.getFeatures()):
            r, c     = int(feat["row"]), int(feat["col"])
            centroid = feat.geometry().centroid().asPoint()
            try:
                pt       = xform.transform(centroid)
                val, ok  = provider.sample(pt, 1)
                if ok and val == val:
                    raw[r, c] = val
            except Exception:
                pass
            if prog_total and i % 100 == 0:
                self._cost_progress_update(prog_offset + i, prog_total)

        if prog_total:
            self._cost_progress_update(prog_offset + N * N, prog_total)

        valid = ~np.isnan(raw)
        if not valid.any():
            return None
        lo, hi = np.percentile(raw[valid], [2, 98])
        if hi <= lo:
            return np.zeros((N, N), dtype=np.float64)
        result = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
        result[np.isnan(result)] = 0.0
        return result

    def _sample_vector_distances(
        self, layer_id: "str | None",
        prog_offset: int = 0, prog_total: int = 0,
    ) -> "np.ndarray | None":
        """
        Build a spatial index from all vertices of *layer_id* (vector),
        then return an N×N float64 array of metres from each cell centroid
        to the nearest vertex.  Returns None if the layer is missing or
        scipy is unavailable.
        """
        N = self.gm.N
        if not layer_id:
            if prog_total:
                self._cost_progress_update(prog_offset + N * N, prog_total)
            return None
        vlayer = QgsProject.instance().mapLayer(layer_id)
        if not isinstance(vlayer, QgsVectorLayer) or not vlayer.isValid():
            if prog_total:
                self._cost_progress_update(prog_offset + N * N, prog_total)
            return None

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            print("[cost] scipy not found — install it for road/water distance sampling")
            if prog_total:
                self._cost_progress_update(prog_offset + N * N, prog_total)
            return None

        xform = QgsCoordinateTransform(
            vlayer.crs(), self.gm.layer.crs(), QgsProject.instance()
        )

        pts: list[list[float]] = []
        for feat in vlayer.getFeatures():
            for v in feat.geometry().vertices():
                try:
                    pt = xform.transform(QgsPointXY(v.x(), v.y()))
                    pts.append([pt.x(), pt.y()])
                except Exception:
                    pass

        if not pts:
            if prog_total:
                self._cost_progress_update(prog_offset + N * N, prog_total)
            return None

        tree = cKDTree(pts)
        dist = np.zeros((N, N), dtype=np.float64)

        for i, feat in enumerate(self.gm.layer.getFeatures()):
            r, c     = int(feat["row"]), int(feat["col"])
            centroid = feat.geometry().centroid().asPoint()
            d, _     = tree.query([centroid.x(), centroid.y()])
            dist[r, c] = d
            if prog_total and i % 100 == 0:
                self._cost_progress_update(prog_offset + i, prog_total)

        if prog_total:
            self._cost_progress_update(prog_offset + N * N, prog_total)
        return dist

    # ── session parameter helpers ─────────────────────────────────────────────

    def _params_extra(self) -> dict:
        """Return cost + optimizer param dicts to embed in session JSON."""
        extra = {}
        try:
            extra["cost_params"] = self._cost_dialog.get_params()
        except Exception:
            pass
        # Always include optimizer params — create the dialog if not yet opened
        # (the constructor reads QSettings, so current defaults are captured).
        if self._opt_dialog is None:
            from .optimization_settings_dialog import OptimizationSettingsDialog
            self._opt_dialog = OptimizationSettingsDialog(self.gm, self.iface, self)
        try:
            extra["optimizer_params"] = self._opt_dialog.config_as_dict()
        except Exception:
            pass
        return extra

    # ── slot: session save / load ─────────────────────────────────────────────

    def _on_save_session(self):
        if not self.gm.active:
            self._status.setText("No active grid to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", f"session_N{self.gm.N}.json", "JSON (*.json)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        save_session(self.gm, path, extra=self._params_extra())
        msg = f"Session saved → {os.path.basename(path)}"
        self._status.setText(msg)
        self._refresh_stats(msg)

    def _on_load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            N      = data.get("N", 0)
            georef = data.get("georef", {})

            ext_vals = georef.get("extent")
            crs_auth = georef.get("crs", "EPSG:4326")
            if not ext_vals or N == 0:
                self._status.setText(
                    "Session file has no georef metadata — "
                    "create a matching grid manually first."
                )
                return
            extent = QgsRectangle(*ext_vals)
            crs    = QgsCoordinateReferenceSystem(crs_auth)

            # Recreate the grid whenever size OR extent differ from the current one.
            # This prevents loading annotation onto a grid at a different geographic
            # position (e.g. the auto-created grid from a different canvas view).
            extent_mismatch = (
                self.gm.extent is None
                or not self.gm.extent.equals(extent)
            )
            if not self.gm.active or self.gm.N != N or extent_mismatch:
                self.gm.create(extent, N, crs)
                self._n_spin.setValue(N)
                self._activate_tool()

            self.gm.from_dict(data)

            # Bring FOREMOST Grid to top of the layer tree and make it active.
            if self.gm.layer is not None:
                root = QgsProject.instance().layerTreeRoot()
                node = root.findLayer(self.gm.layer.id())
                if node:
                    node.setItemVisibilityChecked(True)
                    clone = node.clone()
                    root.insertChildNode(0, clone)
                    node.parent().removeChildNode(node)
                self.iface.setActiveLayer(self.gm.layer)

            # Zoom canvas to the session extent so the grid is always in view.
            padded = QgsRectangle(extent)
            padded.scale(1.05)
            self.iface.mapCanvas().setExtent(padded)
            self.iface.mapCanvas().refresh()
            self.gm.refresh_cost_layer()
            msg = f"Session loaded ← {os.path.basename(path)}"
            self._status.setText(msg)
            self._refresh_stats(msg)

            # Restore cost + optimizer params if present in JSON
            if "cost_params" in data:
                self._cost_dialog.apply_params(data["cost_params"])
            if "optimizer_params" in data:
                if self._opt_dialog is None:
                    from .optimization_settings_dialog import OptimizationSettingsDialog
                    self._opt_dialog = OptimizationSettingsDialog(
                        self.gm, self.iface, self
                    )
                self._opt_dialog.apply_params(data["optimizer_params"])

            # Derive npy path+stem from session filename and sync to opt dialog.
            # Session files are named  {stem}_session_N{N}.json, so strip that suffix.
            import re as _re
            sess_base = os.path.splitext(os.path.basename(path))[0]
            sess_stem = _re.sub(r"_session_N\d+$", "", sess_base)
            self._sync_opt_npy(os.path.join(os.path.dirname(path), sess_stem))
        except Exception as exc:
            self._status.setText(f"Load failed: {exc}")

    # ── slot: export ──────────────────────────────────────────────────────────

    def _on_export_npy(self):
        if not self.gm.active:
            self._status.setText("No active grid to export.")
            return

        # Check whether Restorable cells have costs computed
        ra_total = ra_costed = 0
        for feat in self.gm.layer.getFeatures():
            if int(feat["class_code"] or 0) == CLASS_RA:
                ra_total += 1
                if float(feat["cost"] or 0.0) > 0:
                    ra_costed += 1
        if ra_total > 0 and ra_costed == 0:
            reply = QMessageBox.warning(
                self, "Cost not computed",
                f"There are {ra_total} Restorable cells but no cost values have been "
                "computed yet.\n\n"
                "The exported cost.npy will be all zeros, which will make the optimizer "
                "ignore cost as an objective.\n\n"
                "Use 'Layer-Based' or 'Fixed' (Cost Computation section) before exporting.\n\n"
                "Export anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        folder = self._folder_or_ask()
        if not folder:
            return
        stem, ok = QInputDialog.getText(
            self, "File stem",
            f"Stem (prefix) for all exported .npy files in:\n{folder}",
            text=self._guess_stem(),
        )
        if not ok or not stem.strip():
            return
        stem = stem.strip()
        try:
            paths = export_arrays(self.gm, folder, stem)
            msg = f"Exported {len(paths)} .npy arrays to {folder}"
            self._status.setText(msg)
            self._refresh_stats(msg)
            # auto-save session JSON alongside (includes cost + optimizer params)
            save_session(self.gm, session_path(folder, stem, self.gm.N),
                         extra=self._params_extra())
            self._sync_opt_npy(os.path.join(folder, stem))
        except Exception as exc:
            self._status.setText(f"Export failed: {exc}")

    def _on_export_gpkg(self):
        if not self.gm.active:
            self._status.setText("No active grid to export.")
            return
        folder = self._folder_or_ask()
        if not folder:
            return
        stem = self._guess_stem()
        try:
            path = export_gpkg(self.gm, folder, stem)
            msg = f"GeoPackage saved → {os.path.basename(path)}"
            self._status.setText(msg)
            self._refresh_stats(msg)
        except Exception as exc:
            self._status.setText(f"GPKG export failed: {exc}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _folder_or_ask(self) -> "str | None":
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        return folder or None

    def _guess_stem(self) -> str:
        """
        Derive output file stem.
        Priority: last confirmed stem → active raster layer → first raster → 'foremost'.
        """
        # 1. Use the stem from the last export / loaded session
        if self._last_npy_path_stem:
            return os.path.basename(self._last_npy_path_stem)
        # 2. Active layer in QGIS if it's a raster
        active = self.iface.activeLayer()
        if isinstance(active, QgsRasterLayer) and active.isValid():
            return os.path.splitext(active.name())[0]
        # 3. First raster that is NOT a cost/elevation helper layer
        _skip = {"foremost cost", "foremost grid"}
        for lyr in QgsProject.instance().mapLayers().values():
            if (isinstance(lyr, QgsRasterLayer) and lyr.isValid()
                    and lyr.name().lower() not in _skip):
                return os.path.splitext(lyr.name())[0]
        return "foremost"
