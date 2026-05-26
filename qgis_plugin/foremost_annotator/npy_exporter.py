"""
npy_exporter.py — Export grid annotation to NumPy .npy arrays.

Output arrays (all N×N, dtype float64 unless noted):
    habitat     — 1 where CLASS_HAB, else 0
    restorable  — 1 where CLASS_RA,  else 0
    accessible  — 1 where CLASS_RA,  else 0  (RA = restorable AND accessible)
    cost        — restoration cost per cell (CLASS_RA only, else 0.0)
    class_code  — raw integer class labels (int32)

All arrays are saved under *output_folder* as:
    {stem}_habitat_N{N}.npy
    {stem}_restorable_N{N}.npy
    {stem}_accessible_N{N}.npy
    {stem}_cost_N{N}.npy
    {stem}_class_code_N{N}.npy
"""

import os
import numpy as np

from .constants import CLASS_HAB, CLASS_RA, FLD_ROW, FLD_COL, FLD_CLASS, FLD_COST


def export_arrays(
    grid_manager,
    output_folder: str,
    stem: str = "foremost",
) -> dict[str, str]:
    """
    Build and save .npy arrays from the current grid state.

    Returns a dict mapping array name → saved file path.
    Raises RuntimeError if no active grid layer.
    """
    gm = grid_manager
    if gm.layer is None or not gm.layer.isValid():
        raise RuntimeError("No active grid layer to export.")

    N = gm.N
    habitat    = np.zeros((N, N), dtype=np.float64)
    restorable = np.zeros((N, N), dtype=np.float64)
    accessible = np.zeros((N, N), dtype=np.float64)
    cost       = np.zeros((N, N), dtype=np.float64)
    class_code = np.zeros((N, N), dtype=np.int32)

    for feat in gm.layer.getFeatures():
        r   = int(feat[FLD_ROW])
        c   = int(feat[FLD_COL])
        cls = int(feat[FLD_CLASS])
        cst = float(feat[FLD_COST])

        class_code[r, c] = cls
        cost[r, c]       = cst   # full cost surface for all cells
        if cls == CLASS_HAB:
            habitat[r, c] = 1.0
        elif cls == CLASS_RA:
            restorable[r, c] = 1.0
            accessible[r, c] = 1.0

    os.makedirs(output_folder, exist_ok=True)

    paths = {}
    for name, arr in [
        ("habitat",    habitat),
        ("restorable", restorable),
        ("accessible", accessible),
        ("cost",       cost),
        ("class_code", class_code),
    ]:
        fpath = os.path.join(output_folder, f"{stem}_{name}_N{N}.npy")
        np.save(fpath, arr)
        paths[name] = fpath

    return paths


def export_gpkg(grid_manager, output_folder: str, stem: str = "foremost") -> str:
    """
    Save the grid layer as a GeoPackage alongside the .npy files.
    Requires qgis.core (runs inside QGIS).
    Returns the path of the saved .gpkg.
    """
    from qgis.core import QgsVectorFileWriter, QgsProject

    gm = grid_manager
    if gm.layer is None:
        raise RuntimeError("No active grid layer.")

    gpkg_path = os.path.join(output_folder, f"{stem}_N{gm.N}.gpkg")
    os.makedirs(output_folder, exist_ok=True)

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"

    err, msg = QgsVectorFileWriter.writeAsVectorFormatV2(
        gm.layer,
        gpkg_path,
        QgsProject.instance().transformContext(),
        options,
    )
    if err != QgsVectorFileWriter.NoError:
        raise RuntimeError(f"GeoPackage export failed: {msg}")
    return gpkg_path
