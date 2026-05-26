"""
optimizer_runner_dialog.py — Runs foremost.py via QProcess (no Terminal window).

Shows real-time stdout/stderr inside QGIS, then lets the user open the
solution picker via the "Display Best Solutions" button when done.
"""

import os

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
)
from qgis.PyQt.QtCore import Qt, QProcess, QProcessEnvironment
from qgis.PyQt.QtGui import QFont


class OptimizerRunnerDialog(QDialog):
    """Non-modal log window that runs foremost.py and opens result picker on finish."""

    def __init__(self, iface, python_path, script_path, extra_args,
                 out_dir, stem, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("FOREMOST — Optimizer")
        self.setMinimumSize(680, 460)
        self._iface       = iface
        self._out_dir     = out_dir
        self._stem        = stem
        self._proc        = None
        self._build_ui()
        self._start(python_path, script_path, extra_args)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        self._status_lbl = QLabel("Running optimizer…")
        lay.addWidget(self._status_lbl)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self._log.setFont(mono)
        lay.addWidget(self._log)

        row = QHBoxLayout()
        self._stop_btn = QPushButton("Stop optimizer")
        self._stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self._stop_btn)

        self._load_btn = QPushButton("Display Best Solutions")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load_solutions)
        row.addWidget(self._load_btn)

        row.addStretch()
        lay.addLayout(row)

    # ── process management ────────────────────────────────────────────────────

    def _start(self, python: str, script: str, extra_args: list):
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        venv_bin = os.path.dirname(python)
        env.insert("PATH", venv_bin + os.pathsep + env.value("PATH", ""))
        env.insert("PYTHONUNBUFFERED", "1")
        self._proc.setProcessEnvironment(env)
        self._proc.readyRead.connect(self._on_read)
        self._proc.finished.connect(self._on_finished)
        self._proc.start(python, ["-u", script, "--mode", "1"] + extra_args)
        self.show()
        self.raise_()

    def _on_read(self):
        raw  = bytes(self._proc.readAll()).decode("utf-8", errors="replace")
        text = raw.rstrip("\n")
        if text:
            self._log.appendPlainText(text)

    # ── finish ────────────────────────────────────────────────────────────────

    def _on_stop(self):
        if self._proc and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()

    def _on_finished(self, exit_code: int, _status):
        self._stop_btn.setEnabled(False)
        self._load_btn.setEnabled(True)
        if exit_code == 0:
            self._status_lbl.setText("Optimizer finished successfully.")
        else:
            self._status_lbl.setText(
                f"Optimizer stopped (exit code {exit_code}). "
                "Review the log, then load any available solutions."
            )

    def _on_load_solutions(self):
        try:
            from .solution_picker_dialog import SolutionPickerDialog
            self._picker = SolutionPickerDialog(
                self._iface, self._out_dir, self._stem, parent=None
            )
            self._picker.show()
            self._picker.raise_()
        except Exception as exc:
            self._status_lbl.setText(f"Could not open solution picker: {exc}")

    def closeEvent(self, event):
        if self._proc and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()
            self._proc.waitForFinished(3000)
        super().closeEvent(event)
