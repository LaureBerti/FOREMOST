# FOREMOST

**FOrest Restoration with Evolutionary Multiobjective Optimization STrategies**

FOREMOST is a Python pipeline for planning ecological forest restoration at the landscape scale. Starting from satellite imagery or GIS data, it produces Pareto-optimal restoration scenarios that simultaneously maximize habitat connectivity and minimize intervention costs — using state-of-the-art multi-objective evolutionary algorithms.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![pymoo](https://img.shields.io/badge/optimizer-pymoo%200.6-green)](https://pymoo.org/)
[![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

---

## Overview

Deciding *which* degraded land patches to restore is a constrained combinatorial problem: budgets are limited, access is uneven, and two ecologists rarely agree on what "best" means. FOREMOST frames this as a multi-objective optimization problem over a binary decision grid and solves it with evolutionary algorithms, giving decision-makers a Pareto front of non-dominated solutions rather than a single arbitrary answer.

The pipeline has two loosely coupled stages:

```
Satellite image / GeoPackage
        │
        ▼
  [foremost.annotator]          ← interactive N×N grid labelling GUI
        │
  habitat.npy  restorable.npy  accessible.npy  cost.npy  elevation.npy
        │
        ▼
  [foremost.core]               ← multi-objective optimization
        │
  Pareto-optimal restoration plans  +  publication-quality figures
```

## Features

- **Five evolutionary algorithms** — GA, NSGA-II, NSGA-III, CTAEA, RNSGA-III via [pymoo](https://pymoo.org/)
- **Seven objective combinations** — single-objective (MESH / IIC / COST) and multi-objective (up to 3-way Pareto front)
- **Landscape connectivity metrics** — Effective Mesh Size (MESH, Jaeger 2000) and Integral Index of Connectivity (IIC, Pascual-Hortal & Saura 2006)
- **Ecological cost model** — tree density × accessibility surcharge × elevation penalty × noise
- **Interactive annotation GUI** — label GeoTIFF / GeoPackage data on an N×N grid with 4 overlay layers (elevation, roads, cadastral, hydrology)
- **Hydra configuration** — every parameter overridable from the CLI with no code changes
- **Three run modes** — synthetic demo, load existing `.npy` arrays, or annotate from scratch

## Installation

```bash
git clone https://github.com/yourorg/foremost
cd foremost
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For development (tests + docs):

```bash
pip install -e ".[dev,docs]"
```

> [!NOTE]
> The annotation GUI requires `tkinter` (included in standard CPython builds).
> The core optimizer (`foremost.core`) works headlessly without it.

## Quick Start

**Mode 0 — synthetic landscape (no data required):**

```bash
foremost --mode 0
```

**Mode 1 — load your own `.npy` arrays:**

```bash
foremost --mode 1 --npy-folder outputs/
```

**Mode 2 — annotate a satellite image then optimize:**

```bash
foremost --mode 2 --image zone.tif
foremost --mode 2 --gpkg zones.gpkg
```

**Override any parameter via Hydra dot-notation:**

```bash
foremost optimizer.algo=CTAEA optimizer.pop_size=150 optimizer.n_gen=200
foremost constraints.max_diameter=12 constraints.max_nb_cc=3
```

**Generate a config template:**

```bash
foremost --write-config    # creates conf/foremost.yaml
annotate --write-config    # creates conf/annotator.yaml
```

## Python API

```python
from foremost import HabitatData, ForemostProblemBuilder

# Build a restoration problem from existing arrays
data = HabitatData(
    habitat=habitat_arr,
    restorable=restorable_arr,
    accessible=accessible_arr,
    cost=cost_arr,
    cell_area=1.0,
)

# Fluent builder API — mirrors the restopt R-package interface
result = (
    ForemostProblemBuilder(data)
    .set_full_objective()                               # 3-way Pareto: MESH × IIC × Cost
    .add_restorable_constraint(min_restore=20.0, max_restore=200.0)
    .add_compactness_constraint(max_diameter=9)
    .add_connected_constraint(max_nb_cc=3)
    .add_budget_constraint(max_cost=20_000.0)
    .solve(pop_size=80, n_gen=120, algo="NSGA2")
)

for sol in result["solutions"]:
    print(f"MESH={sol['mesh']:.3f}  IIC={sol['iic']:.4f}  cost={sol['total_cost']:.0f}")
```

## Objectives

| Name | Description |
|------|-------------|
| `MESH` | Maximize Effective Mesh Size — measures landscape permeability |
| `IIC` | Maximize Integral Index of Connectivity — graph-based habitat reachability |
| `COST` | Minimize total restoration cost |
| `MESH_IIC` | 2-objective Pareto: MESH + IIC |
| `MESH_COST` | 2-objective Pareto: MESH + Cost |
| `IIC_COST` | 2-objective Pareto: IIC + Cost |
| `FULL` | 3-objective Pareto: MESH + IIC + Cost |

## Outputs

Each run saves to `outputs/` (configurable):

| File | Description |
|------|-------------|
| `*_0_cost_surface.png` | 3-panel cost decomposition (elevation / restorable area / final cost) |
| `*_1_mesh.png` | Best MESH solution — 2×3 panel with metrics |
| `*_2_iic.png` | Best IIC solution |
| `*_3_cost.png` | Minimum-cost solution |
| `*_4_comparison.png` | Side-by-side comparison of all three |
| `*_5_pareto_mesh_cost.png` | 2-D Pareto front (MESH vs Cost) |
| `*_6_pareto_iic_cost.png` | 2-D Pareto front (IIC vs Cost) |
| `*_7_pareto_3d.png` | 3-D Pareto front (MESH × IIC × Cost) |

## Configuration Reference

Key sections in `conf/foremost.yaml`:

```yaml
optimizer:
  algo: NSGA2          # GA | NSGA2 | NSGA3 | CTAEA | RNSGA3
  objective: FULL      # MESH | IIC | COST | MESH_IIC | MESH_COST | IIC_COST | FULL
  pop_size: 80
  n_gen: 120

constraints:
  min_restore: 2.0     # minimum restorable area
  max_restore: 18.0    # maximum restorable area
  max_diameter: 9      # compactness (max spatial extent in cells)
  max_nb_cc: 1         # maximum number of connected components
  max_cost: .inf       # budget cap

cost:
  tree_unit_cost: 30.0          # € per tree
  tree_spacing_m: 2.0           # planting density
  inaccessible_surcharge: 0.40  # +40% for inaccessible cells
  elevation_slope: 0.005        # +0.5%/m above base elevation
```

## Development

```bash
# Run the full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=foremost --cov-report=term-missing

# Build documentation locally
mkdocs serve
```

> [!TIP]
> Solver tests use a 5×5 synthetic landscape with 1 generation and 4 individuals and complete in under 1 second. The full pipeline on a 30×30 grid with 120 generations takes 2–5 minutes depending on the algorithm.

## Project Structure

```
foremost/
├── foremost/               # installable package
│   ├── __init__.py         # public API
│   ├── core.py             # optimizer, cost model, landscape indices
│   ├── annotator.py        # interactive annotation GUI
│   └── conf/               # default YAML configs (package data)
├── tests/                  # pytest test suite (36 tests)
├── docs/                   # MkDocs documentation source
├── conf/                   # project-level Hydra config overrides
├── input/                  # geospatial input data (GeoTIFF / GPKG)
├── outputs/                # generated figures and arrays
└── pyproject.toml
```

## Reference

> Justeau-Allaire, D. et al. (2021). *Constrained optimization of landscape indices for conservation planning.* **Journal of Applied Ecology**, 58(4), 744–754. <https://doi.org/10.1111/1365-2664.13803>
