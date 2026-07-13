"""
FOREMOST — FOrest Restoration with Evolutionary Multiobjective Optimization STrategies.

Ecological restoration optimization pipeline using evolutionary algorithms.
"""

__version__ = "0.1.0"
__author__ = "Laure Berti-Equille"

# --- Configuration -----------------------------------------------------------
from foremost.core import (
    ForemostConfig,
    DataConfig,
    CostConfig,
    ConstraintsConfig,
    OptimizerConfig,
    OutputConfig,
)

# --- Data layer --------------------------------------------------------------
from foremost.core import HabitatData

# --- Ecological cost function ------------------------------------------------
from foremost.core import compute_restoration_cost, compute_slope

# --- Landscape indices -------------------------------------------------------
from foremost.core import mesh, iic, compute_patches

# --- Optimization ------------------------------------------------------------
from foremost.core import (
    ObjectiveType,
    RestorationConstraints,
    RestorationProblem,
    solve,
)

# --- High-level builder API --------------------------------------------------
from foremost.core import ForemostProblemBuilder

# --- Entry points ------------------------------------------------------------
from foremost.core import run_demo, load_npy_arrays, load_habitatdata_from_npy

# --- Annotator (optional — requires tkinter) ---------------------------------
try:
    from foremost.annotator import (  # type: ignore[import]
        AnnotatorConfig,
        annotate,
        annotate_gpkg,
        compute_accessibility_from_roads,
        extract_layers_from_osm,
    )

    _HAS_ANNOTATOR = True
except Exception:
    _HAS_ANNOTATOR = False

__all__ = [
    "__version__",
    # config
    "ForemostConfig",
    "DataConfig",
    "CostConfig",
    "ConstraintsConfig",
    "OptimizerConfig",
    "OutputConfig",
    # data
    "HabitatData",
    # cost
    "compute_restoration_cost",
    "compute_slope",
    # indices
    "mesh",
    "iic",
    "compute_patches",
    # optimization
    "ObjectiveType",
    "RestorationConstraints",
    "RestorationProblem",
    "solve",
    # high-level API
    "ForemostProblemBuilder",
    "run_demo",
    "load_npy_arrays",
    "load_habitatdata_from_npy",
    # annotator (conditional)
    "AnnotatorConfig",
    "annotate",
    "annotate_gpkg",
    "compute_accessibility_from_roads",
    "extract_layers_from_osm",
]
