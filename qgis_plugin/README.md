# FOREMOST Annotator — QGIS Plugin
@LaureBerti

Interactive N×N grid annotation tool for forest restoration planning.
Overlay a configurable planning grid on any raster, label cells by class,
compute per-cell restoration costs, run the FOREMOST multi-objective optimizer,
and browse Pareto solutions directly inside QGIS.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| QGIS | ≥ 3.16 | |
| Python | ≥ 3.9 | bundled with QGIS |
| NumPy | any | bundled with QGIS |
| SciPy | optional | road/water distance sampling in Layer-Based cost |

SciPy is needed only for the **Layer-Based** cost computation.  On most
systems it is already present in QGIS's Python environment; if not:

```bash
# macOS / Linux — use the Python that ships with QGIS
/path/to/qgis/python3 -m pip install scipy
```

The **FOREMOST optimizer** (`foremost.py`) requires an independent Python
environment with `pymoo`, `rasterio`, `geopandas`, and `hydra-core`.
This is the `.venv/` in the `Restauration/` project folder.

---

## Installation

1. Copy (or symlink) the `foremost_annotator/` folder into your QGIS
   plugins directory:

   | Platform | Path |
   |---|---|
   | macOS | `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/` |
   | Linux | `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/` |
   | Windows | `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` |

2. In QGIS: **Plugins → Manage and Install Plugins → Installed →** tick
   **FOREMOST Annotator**.

3. The panel opens via **Plugins → FOREMOST → FOREMOST Annotator**.

---

## Quick-start workflow

```
1. Load a raster (satellite image, land-cover map, etc.) into QGIS
2. Select it as the Main raster in the plugin panel
3. Set Grid N×N (e.g. 100 for a 100×100 grid)
4. Click Create / Reset Grid
5. Label cells by painting or auto-labeling
6. Configure cost parameters → click Layer-Based or Fixed
7. Export .npy files
8. Click Launch FOREMOST Optimizer → monitor log → Display Best Solutions
   — or — click Load Best Solution to browse a previous run
```

---

## Panel layout

### Main raster

Select the reference raster that defines the **grid extent and CRS**.
The grid is created to match that raster exactly — not the current zoom level.
Click **↺** to refresh the list when layers are added after opening the panel.

---

### Grid N×N

Number of rows and columns.  The planning area is divided into N² square
cells of equal size.  **Create / Reset Grid** creates a new grid (or
replaces the existing one) over the selected raster.

---

### Annotation

#### Raster-based Auto-Label
Classifies cells automatically from pixel values of a loaded raster.
Opens a dialog to configure thresholds and a target class.

#### Threshold-based Auto-Label
Labels cells based on a numerical threshold applied to a chosen band.

#### Clear All Cells
Resets every cell to **Unlabeled** (prompts for confirmation).

#### Activate Manual Annotation
Re-engages the cell-painting map tool after using QGIS pan/zoom tools.
The button is grayed-out while the annotation tool is active; it becomes
available whenever another QGIS tool (pan, zoom, identify) is selected.

#### Cell classes

| Radio button | Code | Colour | Meaning |
|---|---|---|---|
| Habitat | HAB | Green | Existing natural habitat — do not restore |
| Restorable | RA | Orange | Can be restored — receives a cost value |
| Non Restorable | NR | Grey | Unsuitable for restoration |
| Unlabelled | — | Transparent | Not yet classified |

Click a radio button then **click or drag** on the map to paint cells.

---

### Settings

#### Cost Parameters
Floating non-modal dialog with all cost-model parameters.
Settings persist between QGIS sessions via QSettings and are embedded in
every saved session JSON.

| Group | Parameters |
|---|---|
| Monetary unit | Currency (R$, $, €, …) |
| Tree planting model | Unit cost ($/tree), tree spacing (m) |
| Accessibility | Inaccessible surcharge (fraction) |
| Elevation | Base (m), slope (per m), max elevation (m) |
| Road distance penalty | Reference distance (m), penalty slope |
| Water distance penalty | Reference distance (m), penalty slope |
| Cost noise | σ — set to 0 for deterministic costs |
| Spatial layers | Elevation raster, road vector, water vector |

#### Optimization Parameters
Configures every aspect of the optimizer run:

| Group | Parameters |
|---|---|
| Data | .npy files path + stem, output directory, individual array overrides |
| Optimizer | Algorithm (NSGA2/NSGA3/CTAEA/RNSGA3/GA), objectives, population, generations, seed, IIC max distance |
| Constraints | Min/max restored cells, max diameter, max patches, budget cap |

Settings are saved to QSettings **and** embedded into every session JSON,
so reloading a session fully restores the optimizer configuration used at
export time.

---

### Cost Computation

#### Layer-Based
Computes a per-cell restoration cost for every **Restorable** cell:

```
n_trees      = cell_area_m² / tree_spacing_m²
base         = n_trees × unit_cost

access_f     = 1 + surcharge       if dist_road > road_ref else 1
elev_f       = 1 + elev_slope × max(0, elev_m − elev_base)
road_f       = 1 + road_slope × max(0, dist_road − road_ref)
water_f      = 1 + water_slope × max(0, dist_water − water_ref)
noise        = clip(1 + N(0, σ²), 0.5, 2.5)

cost[cell]   = base × access_f × elev_f × road_f × water_f × noise
```

Requires layers assigned in **Cost Parameters → Spatial layers**:
- **Elevation raster** — any single-band raster (DEM, SRTM, etc.)
- **Road vector** — line layer of the road network
- **Water vector** — line or polygon layer of rivers / water bodies

Any unassigned layer uses a factor of 1.0 (no penalty).
Requires SciPy for road/water distance sampling.

#### Fixed
Fills every Restorable cell that has **no cost yet** with a uniform value.
Pre-fills the dialog with a formula estimate from current parameters.

#### Clear Costs
Resets all cell cost values to zero.

---

### Actions

#### Load Session / Save Session
Sessions are JSON files that store and fully restore:
- Grid size N and georeferenced extent (CRS, bounding box)
- All cell labels and cost values
- All **Cost Parameters** settings
- All **Optimization Parameters** settings (algorithm, objectives, paths, constraints)

Loading a session:
1. Recreates the grid at the exact stored geographic position
2. Restores all cell labels with correct colours in the map canvas
3. Zooms the canvas to the grid extent
4. Restores the cost and optimizer parameter dialogs to the saved values

> **Note:** Session files are named `{stem}_session_N{N}.json`.
> The npy path stem is derived automatically from the filename on load.

#### Export .npy files
Saves NumPy arrays under a chosen folder:

| File | Content | dtype |
|---|---|---|
| `{stem}_habitat_N{N}.npy` | 1 where HAB, else 0 | float64 |
| `{stem}_restorable_N{N}.npy` | 1 where RA, else 0 | float64 |
| `{stem}_accessible_N{N}.npy` | 1 where RA, else 0 | float64 |
| `{stem}_cost_N{N}.npy` | Restoration cost per RA cell | float64 |
| `{stem}_class_code_N{N}.npy` | Raw integer class labels | int32 |

A session JSON (`{stem}_session_N{N}.json`) is also saved automatically
alongside, embedding the current cost and optimizer settings.

#### Export GeoPackage
Saves the grid as a `.gpkg` vector file with attributes: row, col,
class\_code, cost.

#### Launch FOREMOST Optimizer
Runs `foremost.py --mode 1` inside QGIS using the project's `.venv`
Python — no Terminal window needed.

- Uses the algorithm, objectives, and constraints from **Optimization Parameters**
- Streams stdout/stderr in real time to a log window
- The **Stop optimizer** button kills the process if needed
- When the run finishes, **Display Best Solutions** becomes active and opens
  the Pareto solution picker automatically

The optimizer settings are embedded into the session JSON
(`{stem}_session_N{N}.json`) that was written at export time.  No separate
config snapshot file is created.

> **Prerequisite:** Export .npy files before launching.  The button warns
> if no arrays are found at the configured path.

#### Load Best Solution
Browse for a folder containing previous optimizer output and open the
Pareto solution picker without running the optimizer again.

The folder is scanned for `*_pareto_*.csv` files; the stem is extracted
automatically.  Useful for comparing runs or loading results from another
machine.

---

### Pareto Solution Picker

Opens automatically after the optimizer finishes, or via **Load Best Solution**.

| Control | Description |
|---|---|
| Objective dropdown | Switch between available Pareto fronts (FULL 3-way, MESH×Cost, IIC×Cost, etc.) |
| Table | All Pareto solutions with Rank, MESH, IIC, Cost, Cost/ha, Cells, Knee ★ |
| Green row | Knee (balanced) solution — selected by default |
| Load Selected as Vector Layer | Creates a polygon memory layer for the selected row; requires `_selections.npz` and a session JSON for georeferencing |
| Load Best as Raster | Adds the knee/best-solution TIF mask as a raster layer |
| **Double-click any row** | Generates a georeferenced GeoTIFF for that solution and loads it as a raster layer — no pre-existing TIF required |

**Double-click raster generation** writes `{stem}_solution_rank{rank}_mask.tif` to the
output directory, then loads it in QGIS with the solution's cost metadata attached.
Georeferencing is read from the session JSON (`{stem}_session_N{N}.json`) if present,
or copied from the existing knee-solution mask TIF.  The generated raster is immediately
selectable in the Layers panel and updates the **Cost of selected solution** statistic.

When any FOREMOST solution layer is selected in the QGIS Layers panel, the
**Statistics** panel updates the **Cost of selected solution** row with
that solution's cost, rank, cell count, and cost per hectare.

---

### Statistics

Updated after every labeling, cost, or session operation.

| Field | Description |
|---|---|
| Cell area | Physical area of one cell in m² and ha |
| Coverage | % labeled, with per-class counts (Hab / RA / NR) |
| Total cost | Sum of all RA cell costs in the annotation grid |
| Avg/cell | Mean cost per Restorable cell |
| Avg/m² | Mean cost per square metre |
| Cost of selected solution | Cost of the currently active solution layer (rank · cells · cost/ha); updates automatically when any FOREMOST solution layer is selected in the Layers panel; shows `—` otherwise |

---

## Output files

```
{output_folder}/
├── {stem}_habitat_N{N}.npy
├── {stem}_restorable_N{N}.npy
├── {stem}_accessible_N{N}.npy
├── {stem}_cost_N{N}.npy
├── {stem}_class_code_N{N}.npy
├── {stem}_session_N{N}.json          ← auto-saved on every .npy export and optimizer launch
├── {stem}_N{N}.gpkg                  ← optional GeoPackage export
├── {stem}_pareto_full.csv            ← 3-way Pareto front
├── {stem}_pareto_mesh_cost.csv       ← MESH × Cost front
├── {stem}_pareto_*.csv               ← one file per objective
├── {stem}_pareto_full_selections.npz ← per-solution cell masks
├── {stem}_best_pareto3d_mask.tif     ← knee-solution raster mask
└── {stem}_solution_rank{rank}_mask.tif  ← per-solution raster (generated on double-click)
```

---

## Cost model defaults

| Parameter | Default | Meaning |
|---|---|---|
| Unit cost | R$ 15 / tree | Cost per planted seedling |
| Tree spacing | 2.5 m | ≈ 1 600 trees/ha |
| Inaccessible surcharge | 40 % | Applied when dist\_road > road\_ref |
| Elevation base | 0 m | No penalty below this elevation |
| Elevation slope | 0.5 % / m | +0.5 % cost per metre of elevation |
| Max elevation | 1 000 m | Normalised [0, 1] → metres scale |
| Road ref. distance | 500 m | No road penalty within 500 m |
| Road penalty slope | 0.02 % / m | +20 % per additional 100 m from road |
| Water ref. distance | 200 m | No water penalty within 200 m |
| Water penalty slope | 0.01 % / m | Additional penalty beyond 200 m |
| Noise σ | 5 % | Reproducible per-cell multiplicative noise |

---

## Optimizer algorithms

| Name | Type | Notes |
|---|---|---|
| NSGA2 | Multi-objective EA | Fast default; good for 2–3 objectives |
| NSGA3 | Reference-direction NSGA | Recommended for 3 objectives (FULL) |
| CTAEA | Constrained two-archive EA | Best for hard budget/area constraints |
| RNSGA3 | Aspiration-point NSGA-III | Guided search toward target solutions |
| GA | Single-objective GA | Use with MESH / IIC / COST only |

---

## Troubleshooting

**Grid appears blank after Load Session**
Cell colours are restored by rebuilding the categorized renderer from scratch
on load.  If the grid still shows white or transparent cells, try toggling
the FOREMOST Grid layer off and on in the Layers panel to force a repaint.

**Cost of selected solution shows `—` after loading a solution layer**
The solution metadata (cost, rank, cell count) is stored as a custom
layer property at load time.  Only layers loaded via **Load Selected as
Vector Layer**, **Load Best as Raster**, or **double-click raster generation**
carry this metadata.  Layers loaded manually from disk (e.g. via the QGIS
Layer menu) will not show a solution cost.

**Optimizer log window is blank while running**
The process runs with unbuffered output (`-u` flag + `PYTHONUNBUFFERED=1`).
If no output appears, check that the correct venv Python is detected in
**Optimization Parameters** and that `foremost.py` is found at the
expected path (`~/Projects/Restauration/foremost.py`).

**Display Best Solutions button stays greyed out**
The button is enabled only when the optimizer process finishes (exit code
0 or non-zero).  If the log window was closed prematurely, use
**Load Best Solution** to browse results manually.

**Layer-Based cost button does nothing**
- Ensure the grid has at least one Restorable (RA) cell.
- Open **Cost Parameters** and assign layers in the *Spatial layers* group.
  All three layers are optional — any unassigned factor defaults to 1.0.
- Check the QGIS Python console for error messages.

**Road/water distance sampling requires SciPy**
If SciPy is not installed the distance arrays fall back to zero (no
road/water penalty).  Install it into QGIS's Python environment:
```bash
/path/to/qgis/python3 -m pip install scipy
```

**Load Best Solution finds no CSV files**
The selected folder must contain at least one `*_pareto_*.csv` file
produced by the optimizer.  Ensure the output directory configured in
**Optimization Parameters** matches the folder you are browsing.

**Double-click raster generation fails or produces an ungeoreferenced raster**
The double-click feature requires a georef source — either:
- A session JSON (`{stem}_session_N{N}.json`) in the same folder as the `.npz` file, **or**
- The knee-solution mask TIF (`{stem}_best_pareto3d_mask.tif`) already present.

If neither is found the TIF is written without a CRS or geotransform.
Export `.npy` files (which auto-saves the session JSON) before loading solutions,
or run the optimizer at least once so the mask TIF exists.

**Double-click: "selections.npz not found"**
The `.npz` file (`{stem}_pareto_full_selections.npz`) must exist in the output folder.
It is produced by the optimizer alongside the CSV files.  Without it neither
**Load Selected as Vector Layer** nor double-click raster generation will work.
