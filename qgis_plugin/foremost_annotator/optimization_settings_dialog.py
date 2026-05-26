"""
optimization_settings_dialog.py — Floating dialog to configure and launch foremost.py.

Groups:
  Data        — npy folder, output dir, prefix
  Optimizer   — algo, objective, pop_size, n_gen, seed, iic_max_dist
  Constraints — min/max_restore, max_diameter, max_nb_cc, max_cost
  Cost model  — cell_size_m (read from grid), tree cost, spacing, surcharge, noise

foremost.py and the Python interpreter are auto-detected (no user input needed).
"""

import json
import os
import sys

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QListWidgetItem,
    QDialogButtonBox, QScrollArea, QWidget, QFrame,
    QFileDialog, QMessageBox,
)
from qgis.PyQt.QtCore import Qt, QSettings


class OptimizationSettingsDialog(QDialog):
    """Floating dialog for configuring and launching the FOREMOST optimizer."""

    _SETTINGS_KEY = "foremost_annotator/opt_settings"

    def __init__(self, grid_manager, iface=None, parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("FOREMOST — Optimization Settings")
        self.setMinimumWidth(460)
        self.setSizeGripEnabled(True)
        self.resize(500, 480)
        self._gm    = grid_manager
        self._iface = iface
        self._script_path, self._python_path = self._detect_paths()
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

        root.addWidget(self._group_data())
        root.addWidget(self._group_optimizer())
        root.addWidget(self._group_constraints())
        root.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.setContentsMargins(8, 4, 8, 8)
        btns.button(QDialogButtonBox.Close).setText("Save && Close")
        btns.rejected.connect(self._save_and_close)
        outer.addWidget(btns)

    # ── groups ────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_paths() -> "tuple[str, str]":
        """Auto-detect foremost.py and the best available Python interpreter."""
        script = ""
        for candidate in [
            os.path.expanduser("~/Projects/Restauration/foremost.py"),
            os.path.normpath(os.path.join(os.path.dirname(__file__),
                                          "../../../../foremost.py")),
            os.path.normpath(os.path.join(os.path.dirname(__file__),
                                          "../../../foremost.py")),
        ]:
            if os.path.isfile(candidate):
                script = candidate
                break

        # Prefer the project venv Python (has pymoo/rasterio) over QGIS's Python
        python = sys.executable
        if script:
            project_dir = os.path.dirname(script)
            for venv_python in [
                os.path.join(project_dir, ".venv", "bin", "python"),
                os.path.join(project_dir, "venv",  "bin", "python"),
                os.path.join(project_dir, ".venv", "bin", "python3"),
            ]:
                if os.path.isfile(venv_python):
                    python = venv_python
                    break

        return script, python

    def _group_data(self) -> QGroupBox:
        gb   = QGroupBox("Data")
        form = QFormLayout(gb)

        self._npy_path_edit = self._path_field(
            form, ".npy files path:", self._browse_npy,
            "Select any .npy file from the export folder — the stem is extracted\n"
            "automatically (e.g. selecting  /outputs/V2_habitat_N100.npy  sets\n"
            "npy folder = /outputs/   stem = V2).\n"
            "You can also type the path+stem directly: /outputs/V2",
        )
        self._out_edit = self._path_field(
            form, "Output dir:", self._browse_out,
            "Directory where optimizer figures and result files are written",
        )

        # ── explicit per-array file overrides ─────────────────────────────────
        sep = QLabel("— Individual .npy overrides (optional, take priority over folder) —")
        sep.setStyleSheet("color: #888; font-size: 10px;")
        sep.setAlignment(Qt.AlignCenter)
        form.addRow(sep)

        self._npy_habitat_edit    = self._npy_file_field(form, "Habitat:")
        self._npy_restorable_edit = self._npy_file_field(form, "Restorable:")
        self._npy_accessible_edit = self._npy_file_field(form, "Accessible:")
        self._npy_cost_edit       = self._npy_file_field(form, "Cost:")

        return gb

    def _group_optimizer(self) -> QGroupBox:
        gb   = QGroupBox("Optimizer")
        form = QFormLayout(gb)

        self._algo_combo = QComboBox()
        for a in ("NSGA2", "NSGA3", "CTAEA", "RNSGA3", "GA"):
            self._algo_combo.addItem(a)
        self._algo_combo.setToolTip(
            "NSGA2  — fast default, good for 2–3 objectives\n"
            "NSGA3  — reference-direction NSGA for 3+ objectives\n"
            "CTAEA  — constrained two-archive EA (best for hard constraints)\n"
            "RNSGA3 — aspiration-point guided NSGA-III\n"
            "GA     — single-objective (use with MESH / IIC / COST objectives only)"
        )
        form.addRow("Algorithm:", self._algo_combo)

        self._obj_list = QListWidget()
        self._obj_list.setToolTip(
            "Check one or more objectives to run.\n"
            "FULL      = 3-way Pareto: MESH × IIC × Cost\n"
            "MESH_IIC  = 2-way Pareto: MESH × IIC\n"
            "MESH_COST = 2-way Pareto: MESH × Cost\n"
            "IIC_COST  = 2-way Pareto: IIC × Cost\n"
            "MESH / IIC / COST = single-objective"
        )
        _obj_labels = {
            "FULL":      "FULL  — 3-way Pareto (MESH × IIC × Cost)",
            "MESH_IIC":  "MESH_IIC  — 2-way Pareto (MESH × IIC)",
            "MESH_COST": "MESH_COST  — 2-way Pareto (MESH × Cost)",
            "IIC_COST":  "IIC_COST  — 2-way Pareto (IIC × Cost)",
            "MESH":      "MESH  — single-objective",
            "IIC":       "IIC  — single-objective",
            "COST":      "COST  — single-objective",
        }
        for key, label in _obj_labels.items():
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if key == "FULL" else Qt.Unchecked)
            self._obj_list.addItem(item)
        self._obj_list.setFixedHeight(
            self._obj_list.sizeHintForRow(0) * len(_obj_labels) + 4
        )
        form.addRow("Objectives:", self._obj_list)

        self._pop_spin = self._ispin(80, 10, 2000,
            "Population size — larger = richer Pareto front, slower per generation")
        form.addRow("Population:", self._pop_spin)

        self._gen_spin = self._ispin(120, 10, 5000,
            "Number of generations — budget: total evaluations = pop × gen")
        form.addRow("Generations:", self._gen_spin)

        self._seed_spin = self._ispin(42, 0, 99999,
            "Random seed for reproducibility")
        form.addRow("Seed:", self._seed_spin)

        self._iic_dist_spin = self._ispin(10, 1, 1000,
            "Max dispersal distance for IIC graph edges (cells).\n"
            "Scale with N to maintain physical equivalence: N=100→10, N=200→20, N=300→30")
        form.addRow("IIC max dist (cells):", self._iic_dist_spin)

        return gb

    def _group_constraints(self) -> QGroupBox:
        gb   = QGroupBox("Constraints")
        form = QFormLayout(gb)

        self._min_restore = self._dspin(2.0,   0.0, 1e6,  1,
            "Minimum number of restored cells (lower bound on restored area)")
        self._max_restore = self._dspin(100.0,  0.0, 1e7,  1,
            "Maximum number of restored cells (upper bound on restored area)")
        self._max_diam    = self._ispin(9,   1, 10000,
            "Maximum spatial diameter of the restored zone in cells\n"
            "(controls compactness; scale with N)")
        self._max_cc      = self._ispin(10,  1,  1000,
            "Maximum number of disconnected restored patches\n"
            "(1 = must be contiguous; higher = fragmented OK)")
        self._max_cost    = self._dspin(0.0, 0.0, 1e15, 0,
            "Budget cap — total restoration cost must be ≤ this value.\n"
            "Set to 0 for unlimited (no budget constraint).")

        form.addRow("Min restore (cells):", self._min_restore)
        form.addRow("Max restore (cells):", self._max_restore)
        form.addRow("Max diameter (cells):", self._max_diam)
        form.addRow("Max patches:", self._max_cc)
        form.addRow("Max cost (0 = ∞):", self._max_cost)

        return gb

    # ── public API ────────────────────────────────────────────────────────────

    def _selected_objectives(self) -> list:
        """Return list of checked objective keys; falls back to ['FULL']."""
        result = []
        for i in range(self._obj_list.count()):
            item = self._obj_list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.data(Qt.UserRole))
        return result or ["FULL"]

    def config_as_dict(self) -> dict:
        """Return all current settings as a plain dict (JSON-serialisable)."""
        mc = self._max_cost.value()
        selected = self._selected_objectives()
        return {
            "npy_path_stem":   self._npy_path_edit.text().strip(),
            "output_dir":      self._out_edit.text().strip(),
            "npy_habitat":     self._npy_habitat_edit.text().strip(),
            "npy_restorable":  self._npy_restorable_edit.text().strip(),
            "npy_accessible":  self._npy_accessible_edit.text().strip(),
            "npy_cost":        self._npy_cost_edit.text().strip(),
            "optimizer": {
                "algo":         self._algo_combo.currentText(),
                "objectives":   selected,
                "objective":    selected[0],
                "pop_size":     self._pop_spin.value(),
                "n_gen":        self._gen_spin.value(),
                "seed":         self._seed_spin.value(),
                "iic_max_dist": self._iic_dist_spin.value(),
            },
            "constraints": {
                "min_restore":    self._min_restore.value(),
                "max_restore":    self._max_restore.value(),
                "max_diameter":   self._max_diam.value(),
                "max_nb_cc":      self._max_cc.value(),
                "max_cost":       None if mc == 0 else mc,
            },
            "cost": {
                "cell_size_m": round(self._gm.cell_size_m(), 2) if self._gm.active else None,
            },
            "script_path":  self._script_path,
            "python_path":  self._python_path,
        }

    def apply_params(self, data: dict):
        """Restore widget values from a previously saved optimizer config dict."""
        opt = data.get("optimizer", {})
        con = data.get("constraints", {})
        def _f(d, k, default):
            try: return float(d.get(k, default))
            except: return default
        def _i(d, k, default):
            try: return int(d.get(k, default))
            except: return default
        try:
            npy = data.get("npy_path_stem", "")
            if npy: self._npy_path_edit.setText(npy)
            out = data.get("output_dir", "")
            if out: self._out_edit.setText(out)
            for key, edit in [
                ("npy_habitat",    self._npy_habitat_edit),
                ("npy_restorable", self._npy_restorable_edit),
                ("npy_accessible", self._npy_accessible_edit),
                ("npy_cost",       self._npy_cost_edit),
            ]:
                v = data.get(key, "")
                if v: edit.setText(v)
            algo_val = opt.get("algo", "NSGA2")
            idx = self._algo_combo.findText(algo_val)
            if idx >= 0: self._algo_combo.setCurrentIndex(idx)
            saved_objs = opt.get("objectives") or []
            if not saved_objs:
                single = opt.get("objective", "FULL")
                saved_objs = [single] if single else ["FULL"]
            saved_set = set(saved_objs)
            for i in range(self._obj_list.count()):
                item = self._obj_list.item(i)
                key  = item.data(Qt.UserRole)
                item.setCheckState(Qt.Checked if key in saved_set else Qt.Unchecked)
            self._pop_spin.setValue(   _i(opt, "pop_size",     80))
            self._gen_spin.setValue(   _i(opt, "n_gen",       120))
            self._seed_spin.setValue(  _i(opt, "seed",         42))
            self._iic_dist_spin.setValue(_i(opt, "iic_max_dist", 10))
            self._min_restore.setValue(_f(con, "min_restore",   2.0))
            self._max_restore.setValue(_f(con, "max_restore", 100.0))
            self._max_diam.setValue(   _i(con, "max_diameter",   9))
            self._max_cc.setValue(     _i(con, "max_nb_cc",     10))
            mc = con.get("max_cost")
            self._max_cost.setValue(0.0 if mc is None else float(mc))
        except RuntimeError:
            pass

    def _save_and_close(self):
        self._save_settings()
        self.hide()

    def prefill_npy_path_stem(self, path_stem: str):
        """Push the current session's npy path+stem into the field (always overwrites)."""
        if path_stem:
            self._npy_path_edit.setText(path_stem)
            self._save_settings()

    def get_command(self) -> "list[str] | None":
        """Return the full subprocess command list, or None if foremost.py not found."""
        if not self._script_path or not os.path.isfile(self._script_path):
            # re-detect in case the project was moved
            self._script_path, self._python_path = self._detect_paths()
        if not self._script_path or not os.path.isfile(self._script_path):
            return None
        cmd = [self._python_path, self._script_path, "--mode", "1"]
        cmd += self._hydra_overrides()
        return cmd

    def launch(self, session_data: dict = None) -> "tuple[bool, str]":
        """
        Validate settings, launch foremost.py via QProcess.
        When *session_data* is provided (built by the dock), it is serialised as a
        single unified session JSON — no separate opt_config file is written.
        Returns (True, runner_info) on success, (False, error_msg) on failure.
        """
        self._save_settings()

        # ── npy existence check ───────────────────────────────────────────────
        # Skip when explicit per-array paths cover at least the 4 core arrays.
        _explicit_paths = [
            self._npy_habitat_edit.text().strip(),
            self._npy_restorable_edit.text().strip(),
            self._npy_accessible_edit.text().strip(),
            self._npy_cost_edit.text().strip(),
        ]
        _has_explicit = all(_explicit_paths)   # all 4 core arrays specified
        _has_any_explicit = any(_explicit_paths)

        npy_stem = self._npy_path_edit.text().strip()
        if not _has_explicit:
            import glob as _glob
            if not npy_stem and not _has_any_explicit:
                QMessageBox.warning(
                    self, "No .npy path",
                    "The '.npy files path' field is empty and no individual array files "
                    "are specified.\n\n"
                    "Export the grid arrays first (Export .npy Arrays), or set individual "
                    "file paths in the Data section.",
                )
                return False, "no npy path"
            if npy_stem and not _glob.glob(npy_stem + "*.npy") and not _has_any_explicit:
                reply = QMessageBox.warning(
                    self, "No .npy files found",
                    f"No .npy files matching:\n  {npy_stem}*.npy\n\n"
                    "Export the grid arrays first (Export .npy Arrays).\n\n"
                    "Launch anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return False, "npy files missing"

        # ── script check ─────────────────────────────────────────────────────
        if not self._script_path or not os.path.isfile(self._script_path):
            self._script_path, self._python_path = self._detect_paths()
        if not self._script_path or not os.path.isfile(self._script_path):
            err = (
                "foremost.py could not be found automatically.\n"
                f"Expected: {os.path.expanduser('~/Projects/Restauration/foremost.py')}"
            )
            QMessageBox.warning(self, "Script not found", err)
            return False, err

        import datetime
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir    = self._out_edit.text().strip()
        stem       = os.path.basename(npy_stem) or "foremost"
        npy_folder = os.path.dirname(npy_stem) if npy_stem else ""
        config_dir = npy_folder or out_dir or os.path.expanduser("~")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Write ONE unified JSON.  If full session data was supplied by the dock
        # (grid state + cost params + optimizer params), use that; otherwise fall
        # back to a standalone optimizer config so the run is still logged.
        N = self._gm.N if self._gm.active else 0
        if session_data is not None:
            payload   = dict(session_data)
            payload["launched_at"] = timestamp
            sess_name = (f"{stem}_session_N{N}.json" if N
                         else f"{stem}_session.json")
            cfg_path  = os.path.join(config_dir, sess_name)
            try:
                os.makedirs(config_dir, exist_ok=True)
                with open(cfg_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
            except Exception:
                cfg_path = sess_name
        else:
            cfg = self.config_as_dict()
            cfg["launched_at"] = timestamp
            try:
                os.makedirs(config_dir, exist_ok=True)
                cfg_path = os.path.join(config_dir,
                                        f"{stem}_opt_config_{timestamp}.json")
                with open(cfg_path, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh, indent=2)
            except Exception:
                cfg_path = f"{stem}_opt_config.json"

        # Build extra args (Hydra overrides, without python / script / --mode 1)
        extra_args = self._hydra_overrides()

        # Launch via QProcess runner (shows output in QGIS, no Terminal needed)
        try:
            from .optimizer_runner_dialog import OptimizerRunnerDialog
            # Store on self so the Python wrapper is not GC'd while the window is open
            self._runner_dialog = OptimizerRunnerDialog(
                self._iface,
                self._python_path,
                self._script_path,
                extra_args,
                out_dir or npy_folder or os.path.expanduser("~"),
                stem,
                parent=self.parent(),
            )
            return True, f"runner started — config: {os.path.basename(cfg_path)}"
        except Exception as exc:
            QMessageBox.critical(self, "Launch Error", str(exc))
            return False, str(exc)

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_settings(self):
        s = QSettings()
        s.beginGroup(self._SETTINGS_KEY)
        s.setValue("npy_path_stem", self._npy_path_edit.text())
        s.setValue("output_dir",   self._out_edit.text())
        s.setValue("npy_habitat",    self._npy_habitat_edit.text())
        s.setValue("npy_restorable", self._npy_restorable_edit.text())
        s.setValue("npy_accessible", self._npy_accessible_edit.text())
        s.setValue("npy_cost",       self._npy_cost_edit.text())
        s.setValue("algo",           self._algo_combo.currentText())
        s.setValue("objectives",     json.dumps(self._selected_objectives()))
        s.setValue("pop_size",     self._pop_spin.value())
        s.setValue("n_gen",        self._gen_spin.value())
        s.setValue("seed",         self._seed_spin.value())
        s.setValue("iic_max_dist", self._iic_dist_spin.value())
        s.setValue("min_restore",  self._min_restore.value())
        s.setValue("max_restore",  self._max_restore.value())
        s.setValue("max_diam",     self._max_diam.value())
        s.setValue("max_cc",       self._max_cc.value())
        s.setValue("max_cost",     self._max_cost.value())
        s.endGroup()

    def _load_settings(self):
        s = QSettings()
        s.beginGroup(self._SETTINGS_KEY)

        def _str(k, default=""): return s.value(k, default) or default
        def _int(k, d):
            try: return int(s.value(k, d))
            except: return d
        def _flt(k, d):
            try: return float(s.value(k, d))
            except: return d

        self._npy_path_edit.setText(_str("npy_path_stem"))
        self._out_edit.setText(_str("output_dir"))
        self._npy_habitat_edit.setText(_str("npy_habitat"))
        self._npy_restorable_edit.setText(_str("npy_restorable"))
        self._npy_accessible_edit.setText(_str("npy_accessible"))
        self._npy_cost_edit.setText(_str("npy_cost"))

        algo_val = _str("algo", "NSGA2")
        idx = self._algo_combo.findText(algo_val)
        if idx >= 0:
            self._algo_combo.setCurrentIndex(idx)

        try:
            saved_objs = json.loads(_str("objectives", "[]"))
        except Exception:
            saved_objs = []
        if not saved_objs:
            single = _str("objective", "FULL")
            saved_objs = [single] if single else ["FULL"]
        saved_set = set(saved_objs)
        for i in range(self._obj_list.count()):
            item = self._obj_list.item(i)
            key  = item.data(Qt.UserRole)
            item.setCheckState(Qt.Checked if key in saved_set else Qt.Unchecked)

        self._pop_spin.setValue(_int("pop_size",    80))
        self._gen_spin.setValue(_int("n_gen",       120))
        self._seed_spin.setValue(_int("seed",        42))
        self._iic_dist_spin.setValue(_int("iic_max_dist", 10))
        self._min_restore.setValue(_flt("min_restore",  2.0))
        self._max_restore.setValue(_flt("max_restore", 100.0))
        self._max_diam.setValue(_int("max_diam",    9))
        self._max_cc.setValue(_int("max_cc",       10))
        self._max_cost.setValue(_flt("max_cost",    0.0))

        s.endGroup()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _hydra_overrides(self) -> list:
        ov = []
        npy_path_stem = self._npy_path_edit.text().strip()
        npy_folder = ""
        if npy_path_stem:
            npy_folder = os.path.dirname(npy_path_stem)
            stem       = os.path.basename(npy_path_stem)
            if npy_folder:
                ov.append(f"data.npy_folder={npy_folder}")
            if stem:
                ov.append(f"output.prefix={stem}")

        out = self._out_edit.text().strip()
        # Always resolve to an absolute path so Hydra never tries to create
        # its run directory relative to the read-only QGIS app bundle CWD.
        abs_out = (os.path.abspath(out) if out
                   else npy_folder or os.path.expanduser("~"))
        ov.append(f"output.dir={abs_out}")
        # Override Hydra's own run-dir (default: outputs/YYYY-MM-DD/HH-MM-SS
        # relative to CWD) with the same absolute writable path.
        ov.append(f"hydra.run.dir={abs_out}")

        selected = self._selected_objectives()
        if len(selected) == 1:
            obj_overrides = [f"optimizer.objective={selected[0]}"]
        else:
            joined = ",".join(selected)
            obj_overrides = [
                f"++optimizer.objectives=[{joined}]",
                f"optimizer.objective={selected[0]}",
            ]

        for attr, edit in [
            ("habitat_path",    self._npy_habitat_edit),
            ("restorable_path", self._npy_restorable_edit),
            ("accessible_path", self._npy_accessible_edit),
            ("cost_path",       self._npy_cost_edit),
        ]:
            val = edit.text().strip()
            if val:
                ov.append(f"data.{attr}={val}")

        ov += [
            "output.fig_saved=true",
            f"optimizer.algo={self._algo_combo.currentText()}",
        ] + obj_overrides + [
            f"optimizer.pop_size={self._pop_spin.value()}",
            f"optimizer.n_gen={self._gen_spin.value()}",
            f"optimizer.seed={self._seed_spin.value()}",
            f"optimizer.iic_max_dist={self._iic_dist_spin.value()}",
            f"constraints.min_restore={self._min_restore.value()}",
            f"constraints.max_restore={self._max_restore.value()}",
            f"constraints.max_diameter={self._max_diam.value()}",
            f"constraints.max_nb_cc={self._max_cc.value()}",
        ]

        mc = self._max_cost.value()
        ov.append(f"constraints.max_cost={'inf' if mc == 0 else mc}")

        # cell_size_m may not be in the YAML schema, so use ++ (add-or-override)
        if self._gm.active:
            ov.append(f"++cost.cell_size_m={self._gm.cell_size_m():.2f}")

        return ov

    def _npy_file_field(self, form, label: str) -> QLineEdit:
        """One-line file picker for an individual .npy array."""
        row  = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText("(use folder discovery)")
        edit.setToolTip("Absolute path to a .npy file.  Overrides keyword-based folder discovery.")
        row.addWidget(edit)
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.setToolTip("Browse for .npy file")
        btn.clicked.connect(lambda _=False, e=edit: self._browse_npy_file(e))
        row.addWidget(btn)
        clr = QPushButton("✕")
        clr.setFixedWidth(22)
        clr.setToolTip("Clear this field (revert to folder discovery)")
        clr.clicked.connect(lambda _=False, e=edit: e.clear())
        row.addWidget(clr)
        form.addRow(label, row)
        return edit

    def _browse_npy_file(self, edit: "QLineEdit"):
        p, _ = QFileDialog.getOpenFileName(self, "Select .npy file", "", "NumPy arrays (*.npy)")
        if p:
            edit.setText(p)

    def _path_field(self, form, label: str, browse_fn, tooltip: str = "") -> QLineEdit:
        row  = QHBoxLayout()
        edit = QLineEdit()
        edit.setToolTip(tooltip)
        row.addWidget(edit)
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.clicked.connect(browse_fn)
        row.addWidget(btn)
        form.addRow(label, row)
        return edit

    def _browse_script(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select foremost.py", "", "Python (*.py)")
        if p:
            self._script_edit.setText(p)

    def _browse_python(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Python interpreter", "", "All (*)")
        if p:
            self._python_edit.setText(p)

    def _browse_npy(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select any exported .npy file", "", "NumPy arrays (*.npy)"
        )
        if p:
            self._npy_path_edit.setText(self._npy_to_path_stem(p))

    @staticmethod
    def _npy_to_path_stem(filepath: str) -> str:
        """
        Given e.g. /outputs/V2_habitat_N100.npy → /outputs/V2
        Strips the class-suffix and N-suffix added by the plugin exporter.
        Falls back to the full path without extension.
        """
        folder = os.path.dirname(filepath)
        name   = os.path.splitext(os.path.basename(filepath))[0]  # V2_habitat_N100
        for keyword in ("_habitat", "_restorable", "_accessible",
                        "_cost", "_elevation", "_session"):
            idx = name.find(keyword)
            if idx > 0:
                name = name[:idx]
                break
        return os.path.join(folder, name)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._out_edit.setText(d)

    @staticmethod
    def _dspin(value, lo, hi, decimals, tooltip="") -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(value)
        if tooltip:
            w.setToolTip(tooltip)
        return w

    @staticmethod
    def _ispin(value, lo, hi, tooltip="") -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(value)
        if tooltip:
            w.setToolTip(tooltip)
        return w
