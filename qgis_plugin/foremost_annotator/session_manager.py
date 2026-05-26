"""
session_manager.py — Save / load annotation sessions as JSON.
"""

import json
import os
from datetime import datetime


def save_session(grid_manager, path: str, extra: dict | None = None):
    """Serialise the grid state and write it to *path* (JSON)."""
    data = grid_manager.to_dict()
    data["saved_at"] = datetime.now().isoformat(timespec="seconds")
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_session(grid_manager, path: str) -> dict:
    """Read *path* and restore grid labels/costs. Returns the raw dict."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    grid_manager.from_dict(data)
    return data


def session_path(output_folder: str, stem: str, N: int) -> str:
    return os.path.join(output_folder, f"{stem}_session_N{N}.json")
