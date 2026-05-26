"""
cost_params_dialog.py — Floating, non-modal dialog for all cost parameters.

Open with CostParamsDialog.show(); read values with .get_params().
The dialog stays on top of QGIS while the user annotates.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QDoubleSpinBox, QLabel, QPushButton,
    QDialogButtonBox, QComboBox, QScrollArea, QWidget, QFrame,
)
from qgis.PyQt.QtCore import Qt, QSettings
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

# (symbol, name) pairs — symbol shown in UI labels; name shown in dropdown
CURRENCIES = [
    ("R$",  "BRL — Brazilian Real"),
    ("$",   "USD — US Dollar"),
    ("€",   "EUR — Euro"),
    ("£",   "GBP — British Pound"),
    ("A$",  "AUD — Australian Dollar"),
    ("C$",  "CAD — Canadian Dollar"),
    ("¥",   "JPY — Japanese Yen"),
    ("CHF", "CHF — Swiss Franc"),
    ("MX$", "MXN — Mexican Peso"),
    ("S/",  "PEN — Peruvian Sol"),
    ("CLP$","CLP — Chilean Peso"),
    ("COP$","COP — Colombian Peso"),
    ("UYU$","UYU — Uruguayan Peso"),
    ("PYG₲","PYG — Paraguayan Guaraní"),
]


class CostParamsDialog(QDialog):
    """
    Non-modal floating dialog exposing every cost-model parameter.

    Usage
    -----
    dialog = CostParamsDialog(parent)
    dialog.show()
    params = dialog.get_params()   # dict of all current values
    cost   = compute_cell_cost(cell_size_m, **params)
    """

    _SETTINGS_KEY = "foremost_annotator/cost_params"

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("FOREMOST — Cost Parameters")
        self.setMinimumWidth(360)
        self._build_ui()
        self._load_settings()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        root = QVBoxLayout(content)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(self._group_currency())
        root.addWidget(self._group_tree())
        root.addWidget(self._group_access())
        root.addWidget(self._group_elevation())
        root.addWidget(self._group_road())
        root.addWidget(self._group_water())
        root.addWidget(self._group_noise())
        root.addWidget(self._group_spatial_layers())

        btn_reset = QPushButton("Reset to defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        root.addWidget(btn_reset)
        root.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        # Save & Close button always visible outside the scroll area
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.button(QDialogButtonBox.Close).setText("Save && Close")
        btns.rejected.connect(self._save_and_close)
        outer.addWidget(btns)

    # ── parameter groups ──────────────────────────────────────────────────────

    def _group_currency(self) -> QGroupBox:
        gb   = QGroupBox("Monetary unit")
        form = QFormLayout(gb)
        self._currency_combo = QComboBox()
        for symbol, label in CURRENCIES:
            self._currency_combo.addItem(label, userData=symbol)
        self._currency_combo.setToolTip(
            "Currency used for all cost values — affects labels only, not the calculation"
        )
        form.addRow("Currency:", self._currency_combo)
        return gb

    def _group_tree(self) -> QGroupBox:
        gb = QGroupBox("Tree planting model")
        form = QFormLayout(gb)

        self._tree_unit_cost = self._dspin(15.0, 0.01, 1e6, 2,   "USD per planted tree")
        self._tree_spacing   = self._dspin(2.5,  0.5,  100, 1,   "Metres between trees")
        form.addRow("Unit cost ($/tree):",  self._tree_unit_cost)
        form.addRow("Tree spacing (m):",    self._tree_spacing)
        return gb

    def _group_access(self) -> QGroupBox:
        gb = QGroupBox("Accessibility")
        form = QFormLayout(gb)

        self._inacc_surcharge = self._dspin(0.40, 0.0, 10.0, 3,
            "Fractional cost increase for inaccessible cells (0.40 = +40 %)")
        form.addRow("Inaccessible surcharge:", self._inacc_surcharge)
        return gb

    def _group_elevation(self) -> QGroupBox:
        gb = QGroupBox("Elevation")
        form = QFormLayout(gb)

        self._elev_base   = self._dspin(0.0,    0.0,  5000.0, 1,
            "Reference elevation in metres (no penalty below this)")
        self._elev_slope  = self._dspin(0.005,  0.0,  0.1,    4,
            "Fractional cost increase per metre above base (0.005 = +0.5 %/m)")
        self._max_elev    = self._dspin(1000.0, 1.0,  9000.0, 1,
            "Maps normalised elevation [0, 1] → metres")
        self._elev_slope.setSingleStep(0.001)
        form.addRow("Elevation base (m):",       self._elev_base)
        form.addRow("Elevation slope (per m):",  self._elev_slope)
        form.addRow("Max elevation (m):",         self._max_elev)
        return gb

    def _group_road(self) -> QGroupBox:
        gb = QGroupBox("Road distance penalty")
        form = QFormLayout(gb)

        self._road_ref    = self._dspin(500.0,  0.0, 50000.0, 1,
            "No road penalty within this distance (m)")
        self._road_slope  = self._dspin(0.0002, 0.0, 0.01,    5,
            "Fractional cost increase per metre beyond road_ref_dist (0.0002 = +0.02 %/m)")
        self._road_slope.setSingleStep(0.0001)
        form.addRow("Ref. distance (m):",    self._road_ref)
        form.addRow("Penalty slope:",        self._road_slope)
        return gb

    def _group_water(self) -> QGroupBox:
        gb = QGroupBox("Water distance penalty")
        form = QFormLayout(gb)

        self._water_ref   = self._dspin(200.0,  0.0, 50000.0, 1,
            "No water penalty within this distance (m)")
        self._water_slope = self._dspin(0.0001, 0.0, 0.01,    5,
            "Fractional cost increase per metre beyond water_ref_dist")
        self._water_slope.setSingleStep(0.0001)
        form.addRow("Ref. distance (m):",    self._water_ref)
        form.addRow("Penalty slope:",        self._water_slope)
        return gb

    def _group_spatial_layers(self) -> QGroupBox:
        gb   = QGroupBox("Spatial layers for cost computation")
        form = QFormLayout(gb)

        def _layer_combo(raster_only=False) -> QComboBox:
            cb = QComboBox()
            cb.setToolTip(
                "Layer sampled during 'Compute costs from layers'.\n"
                "Leave as '— none —' to skip this factor (factor = 1.0)."
            )
            cb.addItem("— none —", None)
            return cb

        self._elev_combo  = _layer_combo(raster_only=True)
        self._road_combo  = _layer_combo()
        self._water_combo = _layer_combo()

        def _row(label, combo):
            row = QHBoxLayout()
            row.addWidget(combo)
            btn = QPushButton("↺")
            btn.setFixedWidth(28)
            btn.setToolTip("Refresh layer list")
            btn.clicked.connect(self._refresh_spatial_combos)
            row.addWidget(btn)
            form.addRow(label, row)

        _row("Elevation raster:", self._elev_combo)
        _row("Road vector:",      self._road_combo)
        _row("Water vector:",     self._water_combo)

        self._refresh_spatial_combos()
        return gb

    def _refresh_spatial_combos(self):
        """Repopulate elevation/road/water combos from the current QGIS project."""
        layers = list(QgsProject.instance().mapLayers().values())
        raster_items = [(l.name(), l.id()) for l in layers
                        if isinstance(l, QgsRasterLayer) and l.isValid()]
        vector_items = [(l.name(), l.id()) for l in layers
                        if isinstance(l, QgsVectorLayer) and l.isValid()]

        for combo, items in [
            (self._elev_combo,  raster_items),
            (self._road_combo,  vector_items),
            (self._water_combo, vector_items),
        ]:
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("— none —", None)
            for name, lid in items:
                combo.addItem(name, lid)
            idx = combo.findData(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def get_spatial_layer_ids(self) -> dict:
        """Return {'elevation': id|None, 'road': id|None, 'water': id|None}."""
        return {
            "elevation": self._elev_combo.currentData(),
            "road":      self._road_combo.currentData(),
            "water":     self._water_combo.currentData(),
        }

    def _group_noise(self) -> QGroupBox:
        gb = QGroupBox("Cost noise")
        form = QFormLayout(gb)

        self._noise_sigma = self._dspin(0.05, 0.0, 1.0, 3,
            "Multiplicative noise σ — set to 0 for deterministic costs")
        self._noise_sigma.setSingleStep(0.005)
        form.addRow("Noise σ:", self._noise_sigma)
        return gb

    # ── public API ────────────────────────────────────────────────────────────

    # Default values returned when C++ widgets have been deleted (plugin reload)
    _DEFAULTS = dict(
        tree_unit_cost         = 15.0,
        tree_spacing_m         = 2.5,
        inaccessible_surcharge = 0.40,
        elevation_base_m       = 0.0,
        elevation_slope        = 0.005,
        max_elevation_m        = 1000.0,
        road_ref_dist_m        = 500.0,
        road_penalty_slope     = 0.0002,
        water_ref_dist_m       = 200.0,
        water_penalty_slope    = 0.0001,
        noise_sigma            = 0.05,
        currency_symbol        = "R$",
        currency_name          = "BRL — Brazilian Real",
    )

    def get_params(self) -> dict:
        """
        Return a dict with all cost-model keyword arguments plus display fields.
        Falls back to defaults if the C++ widgets have been deleted.
        """
        try:
            return dict(
                tree_unit_cost         = self._tree_unit_cost.value(),
                tree_spacing_m         = self._tree_spacing.value(),
                inaccessible_surcharge = self._inacc_surcharge.value(),
                elevation_base_m       = self._elev_base.value(),
                elevation_slope        = self._elev_slope.value(),
                max_elevation_m        = self._max_elev.value(),
                road_ref_dist_m        = self._road_ref.value(),
                road_penalty_slope     = self._road_slope.value(),
                water_ref_dist_m       = self._water_ref.value(),
                water_penalty_slope    = self._water_slope.value(),
                noise_sigma            = self._noise_sigma.value(),
                currency_symbol        = self._currency_combo.currentData(),
                currency_name          = self._currency_combo.currentText(),
            )
        except RuntimeError:
            return dict(self._DEFAULTS)

    def currency_symbol(self) -> str:
        """Convenience accessor — returns the active currency symbol."""
        try:
            return self._currency_combo.currentData()
        except RuntimeError:
            return self._DEFAULTS["currency_symbol"]

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dspin(value, lo, hi, decimals, tooltip="") -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(value)
        if tooltip:
            w.setToolTip(tooltip)
        return w

    def _save_and_close(self):
        self._save_settings()
        self.hide()

    def _save_settings(self):
        s = QSettings()
        s.beginGroup(self._SETTINGS_KEY)
        s.setValue("tree_unit_cost",  self._tree_unit_cost.value())
        s.setValue("tree_spacing",    self._tree_spacing.value())
        s.setValue("inacc_surcharge", self._inacc_surcharge.value())
        s.setValue("elev_base",       self._elev_base.value())
        s.setValue("elev_slope",      self._elev_slope.value())
        s.setValue("max_elev",        self._max_elev.value())
        s.setValue("road_ref",        self._road_ref.value())
        s.setValue("road_slope",      self._road_slope.value())
        s.setValue("water_ref",       self._water_ref.value())
        s.setValue("water_slope",     self._water_slope.value())
        s.setValue("noise_sigma",     self._noise_sigma.value())
        s.setValue("currency_idx",    self._currency_combo.currentIndex())
        s.endGroup()

    def _load_settings(self):
        s = QSettings()
        s.beginGroup(self._SETTINGS_KEY)
        def _f(k, d):
            try: return float(s.value(k, d))
            except: return d
        def _i(k, d):
            try: return int(s.value(k, d))
            except: return d
        self._tree_unit_cost.setValue(_f("tree_unit_cost",  15.0))
        self._tree_spacing.setValue(  _f("tree_spacing",    2.5))
        self._inacc_surcharge.setValue(_f("inacc_surcharge", 0.40))
        self._elev_base.setValue(     _f("elev_base",       0.0))
        self._elev_slope.setValue(    _f("elev_slope",      0.005))
        self._max_elev.setValue(      _f("max_elev",        1000.0))
        self._road_ref.setValue(      _f("road_ref",        500.0))
        self._road_slope.setValue(    _f("road_slope",      0.0002))
        self._water_ref.setValue(     _f("water_ref",       200.0))
        self._water_slope.setValue(   _f("water_slope",     0.0001))
        self._noise_sigma.setValue(   _f("noise_sigma",     0.05))
        idx = _i("currency_idx", 0)
        if 0 <= idx < self._currency_combo.count():
            self._currency_combo.setCurrentIndex(idx)
        s.endGroup()

    def apply_params(self, data: dict):
        """Restore widget values from a previously saved cost-params dict."""
        def _f(k, d):
            try: return float(data.get(k, d))
            except: return d
        try:
            self._tree_unit_cost.setValue(_f("tree_unit_cost",          15.0))
            self._tree_spacing.setValue(  _f("tree_spacing_m",          2.5))
            self._inacc_surcharge.setValue(_f("inaccessible_surcharge", 0.40))
            self._elev_base.setValue(     _f("elevation_base_m",        0.0))
            self._elev_slope.setValue(    _f("elevation_slope",         0.005))
            self._max_elev.setValue(      _f("max_elevation_m",         1000.0))
            self._road_ref.setValue(      _f("road_ref_dist_m",         500.0))
            self._road_slope.setValue(    _f("road_penalty_slope",      0.0002))
            self._water_ref.setValue(     _f("water_ref_dist_m",        200.0))
            self._water_slope.setValue(   _f("water_penalty_slope",     0.0001))
            self._noise_sigma.setValue(   _f("noise_sigma",             0.05))
            sym = data.get("currency_symbol", "R$")
            for i in range(self._currency_combo.count()):
                if self._currency_combo.itemData(i) == sym:
                    self._currency_combo.setCurrentIndex(i)
                    break
        except RuntimeError:
            pass

    def _reset_defaults(self):
        self._currency_combo.setCurrentIndex(0)   # BRL
        self._tree_unit_cost.setValue(15.0)
        self._tree_spacing.setValue(2.5)
        self._inacc_surcharge.setValue(0.40)
        self._elev_base.setValue(0.0)
        self._elev_slope.setValue(0.005)
        self._max_elev.setValue(1000.0)
        self._road_ref.setValue(500.0)
        self._road_slope.setValue(0.0002)
        self._water_ref.setValue(200.0)
        self._water_slope.setValue(0.0001)
        self._noise_sigma.setValue(0.05)
