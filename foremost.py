"""
foremost.py
================
FOREMOST (FOrest Restoration with Evolutionary Multiobjective Optimization STrategies) code is a Python tool for
studying the feasibility of forest restoration scenarios.

Initially, it was inspired by the restopt R package (Justeau-Allaire et al. 2021/2023)
Original Java library : https://github.com/dimitri-justeau/restopt
Reference paper       : Justeau-Allaire et al. (2021) J. Applied Ecology 58(4):744-754
 
But it has the following main differences:
- It uses several constrained evolutionary multiobjective optimisation algorithms (GA, NSGA2, NSGA3, CTAEA, RNSGA3) 
    with pymoo Python library instead of Choco CP-solver
- It includes an advanced annotation tool
- It adds  COST and BERI (bioclimatic ecosystem resilience indicator) as optimization criteria in addition
    to MESH and IIC
- It can generate a synthetic landscape for testing, including a synthetic elevation DEM (dome-shaped hill).
- It proposes a cost model for computing the cost of restoration (synthetic)
- It considers elevation data to compute the accessibility and cost of restoration (synthetic)
- It considers the hydrologic and road networks to compute an accessibility and cost matrices
- It considers the cadastral data to add realistic constraints to the study area and scenarios
- It enables the thorough study and comparison of several restoration scenarios (using hydra experiment configuration)
  prescribed by the literature and policies but never verified in terms of feasibility (Brazil's Forest Code,
  Brazilian National Biodiversity Strategy and Action Plan, Land sparing, Land sharing)
 

New features
--------------------------
1. **Hydra** configuration  (pip install hydra-core omegaconf)
   Every experiment parameter is declared in  conf/foremost.yaml  and
   overridable from the CLI:
       python foremost.py optimizer.algo=CTAEA optimizer.pop_size=120
 
2. **CTAEA & RNSGA-III** multi-objective algorithms (pymoo ≥ 0.6)
   algo in {"GA", "NSGA2", "NSGA3", "CTAEA", "RNSGA3"}
 
3. **Ecological cost function** compute_restoration_cost()
   Generates an N×N restoration-cost matrix from:
     • restorable area per cell  (larger area → more trees → higher cost)
     • accessibility             (inaccessible cells bear a logistics surcharge)
     • elevation                 (high-altitude sites cost more to work)
     • unit cost per tree        (configurable, e.g. 15 $/tree)
 
4. **Improved plot_solution()**
   • optional satellite/raster background in transparency (bg_image parameter)
   • algorithm name displayed in every subplot title


The problem
-----------
Given a binary habitat raster, a "restorable area" raster, an "accessible
cells" raster and a "restoration cost" raster, find the set of cells to
restore so as to optimise one or several landscape objectives subject to:
  - compactness  : max diameter of the restored zone (in cell units)
  - restorable   : total restored area in [min_restore, max_restore]
  - locked-out   : certain cells cannot be selected
  - connectivity : restored zone must form <= max_nb_cc connected components
  - budget       : total restoration cost <= max_cost  (optional)

Objectives available
--------------------
  Single-objective
  ----------------
  MESH      : maximize Effective Mesh Size (Jaeger 2000)
  IIC       : maximize Integral Index of Connectivity (Pascual-Hortal & Saura 2006)
  COST      : minimize total restoration cost

   ** >>TO DO: add BERI : maximize the CSIRO Bioclimatic Ecosystem Resilience Index index (Harwood et al.  2020
    https://doi.org/10.25919/4vvz-4j96

  Multi-objective  (Pareto front via NSGA-II)
  -------------------------------------------
  MESH_IIC  : maximize MESH  x  maximize IIC
  MESH_COST : maximize MESH  x  minimize cost
  IIC_COST  : maximize IIC   x  minimize cost
  FULL      : maximize MESH  x  maximize IIC  x  minimize cost  (3-obj NSGA-II)

 ** >>TO DO: add the other combinaisons


Dependencies
------------
    pip install pymoo numpy scipy rasterio matplotlib networkx
    pip install hydra-core omegaconf        # optional but recommended
"""
import shapely
from annotation import annotate, CRS_WEB_MERCATOR, AnnotatorConfig
import os
import re
import sys
import textwrap
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# from matplotlib.colors import Normalize
import networkx as nx
from scipy.ndimage import (
    label as ndimage_label,
    generate_binary_structure,
    distance_transform_edt,
)
from scipy.spatial import cKDTree as _cKDTree

# ── pymoo core (required) ─────────────────────────────────────────────────────
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ── pymoo extended algorithms (optional: NSGA3, CTAEA, RNSGA3) ───────────────
try:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions

    _HAS_NSGA3 = True
except ImportError:
    _HAS_NSGA3 = False

try:
    from pymoo.algorithms.moo.ctaea import CTAEA

    _HAS_CTAEA = True
except ImportError:
    _HAS_CTAEA = False

try:
    from pymoo.algorithms.moo.rnsga3 import RNSGA3

    _HAS_RNSGA3 = True
except ImportError:
    _HAS_RNSGA3 = False

# ── Hydra / OmegaConf (optional) ─────────────────────────────────────────────
try:
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf

    _HAS_HYDRA = True
except ImportError:
    _HAS_HYDRA = False
    DictConfig = dict  # type alias for typing only

# ── colour palette ────────────────────────────────────────────────────────────
_C = dict(
    habitat="#2d6a4f",
    restored="#f4a261",
    nodata="#cccccc",
    other="#f0f0f0",
    accent="#e63946",
    bg="#ffffff",
)

SUPPORTED_ALGOS = ("GA", "NSGA2", "NSGA3", "CTAEA", "RNSGA3")


def _fmt_cost(v: float, decimals: int = 2) -> str:
    """Format a cost value in scientific notation with 4 significant figures.

    Example: 123_400_000 → '1,234 × 10⁸ B$'
    Comma is used as decimal separator; Unicode superscripts for the exponent.
    """
    import math as _math
    if abs(v) < 1e-9:
        return "0 B$"
    exp = int(_math.floor(_math.log10(abs(v))))
    mantissa = round(v / (10 ** exp), 3)
    if abs(mantissa) >= 10:          # floating-point edge case e.g. 9.9995 → 10.000
        mantissa = round(mantissa / 10, 3)
        exp += 1
    sup = str(exp).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return f"{mantissa:.3f}".replace(".", ",") + f" × 10{sup} B$"


def _cost_scale_formatter(values):
    """Shared-scale axis formatter for a collection of cost values.

    Returns (FuncFormatter, label_suffix) where:
      - FuncFormatter shows the mantissa only (e.g. '9,750') using a common
        power-of-10 scale derived from min(values), so the smallest tick has
        a mantissa in [1, 10) and all ticks share the same exponent.
      - label_suffix is the "× 10^N B$" string that belongs once in the
        axis or colorbar label (e.g. ' (× 10¹⁰ B$)').

    This avoids repeating the exponent on every tick.
    """
    import math as _math
    finite = [abs(v) for v in values if v == v and abs(v) < float("inf") and abs(v) > 0]
    if not finite:
        return plt.FuncFormatter(lambda v, _: "0"), " (B$)"
    # Anchor on the minimum so the smallest tick reads ≥ 1 (e.g. 8,750 not 0,875)
    exp = int(_math.floor(_math.log10(min(finite))))
    scale = 10 ** exp
    sup = str(exp).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    label_suffix = f" (× 10{sup} B$)"
    fmt = plt.FuncFormatter(lambda v, _: f"{v / scale:.3f}".replace(".", ","))
    return fmt, label_suffix


# =============================================================================
# 0.  HYDRA CONFIGURATION
# =============================================================================

# ── Python-dataclass config schema (mirrors conf/foremost.yaml) ────────────────


@dataclass
class DataConfig:
    """Paths and parameters for input rasters."""

    mode: int = 2
    nrows: int = 30
    ncols: int = 30
    habitat_fraction: float = 0.28
    cell_area: float = 1.0
    seed_gen: int = 7
    # Real raster paths (empty string → use synthetic data)
    image_path: str = ""
    gpkg_path: str = ""
    habitat_path: str = ""
    restorable_path: str = ""
    accessible_path: str = ""
    cost_path: str = ""  # leave empty to use compute_restoration_cost()
    locked_out_path: str = ""
    elevation_path: str = ""  # used by compute_restoration_cost()
    # Mode 1: folder containing pre-exported .npy arrays
    npy_folder: str = ""  # leave empty → user is prompted via file dialog


@dataclass
class CostConfig:
    """Parameters for compute_restoration_cost()."""

    tree_unit_cost: float = 15.0  # $ per tree
    tree_spacing_m: float = 2.5  # metres between trees (~1600 trees/ha)
    cell_size_m: float = 100.0  # fallback for synthetic mode; set to landscape_extent_m / N for real runs
    inaccessible_surcharge: float = 0.40  # +40 % for inaccessible cells
    elevation_base_m: float = 0.0  # reference elevation (m)
    elevation_slope: float = 0.005  # cost increase per metre elevation
    noise_sigma: float = 0.05  # relative noise on the cost surface


@dataclass
class ConstraintsConfig:
    min_restore: float = 20.0
    max_restore: float = 100.0
    max_diameter: int = 9
    max_nb_cc: int = 10
    min_proportion: float = 0.0
    max_cost: float = float("inf")
    min_app_compliance: float = 0.0   # fraction of restored cells in APPs (Forest Code Art. 61-A)
    max_slope_deg: float = float("inf")  # max terrain slope for restoration (degrees)
    # NC1: minimum cells per connected component (0 = disabled)
    min_patch_size: int = 0
    # NC2: maximum edge-to-area ratio of restored zone (1.0 = disabled)
    max_edge_ratio: float = 1.0
    # NC3: minimum IIC delta above pre-restoration baseline (0.0 = disabled)
    min_iic_delta: float = 0.0
    # NC4: maximum Chebyshev distance from restored cell to existing habitat (0 = disabled)
    max_distance_to_habitat: int = 0
    # NC7: minimum total carbon stock delivered in tCO2 (0.0 = disabled)
    min_carbon_stock: float = 0.0
    carbon_rate_tco2_ha_yr: float = 3.0   # Atlantic Forest average (Chazdon et al. 2016)
    carbon_horizon_yr: int = 20
    # NC8: minimum corridor width in cells via morphological erosion (0 = disabled)
    min_corridor_width: int = 0


@dataclass
class OptimizerConfig:
    """Algorithm and hyper-parameters."""

    algo: str = "NSGA2"  # GA | NSGA2 | NSGA3 | CTAEA | RNSGA3
    objective: str = "FULL"  # single objective (used when objectives list is empty)
    objectives: list = field(default_factory=list)  # list of objective strings; when non-empty overrides objective
    pop_size: int = 80
    n_gen: int = 120
    seed: int = 42
    verbose: bool = True
    iic_max_dist: int = 10
    penalty: float = 1e6
    mu: float = 0.1
    pop_per_ref_point: int = 50


@dataclass
class OutputConfig:
    dir: str = "outputs"
    prefix: str = "foremost"
    dpi: int = 150
    show: bool = False  # set True for interactive display
    fig_saved: bool = True  # set False to skip figure generation (faster batch runs)


@dataclass
class ForemostConfig:
    data: DataConfig = field(default_factory=DataConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Run mode  (see run_demo docstring for details)
    #   0 → synthetic data from YAML config
    #   1 → load .npy arrays from data.npy_folder (or prompted folder)
    #   2 → launch annotator GUI, use returned arrays
    mode: int = 0


def write_default_yaml(config_dir: Union[str, Path] = "conf") -> Path:
    """
    Write the default conf/foremost.yaml file.
    Call once to bootstrap a project:  python foremost.py --write-config
    """
    cfg = ForemostConfig()
    yaml_text = textwrap.dedent(
        f"""\
    # foremost — Hydra configuration
    # Override any field via CLI:  python foremost.py optimizer.algo=CTAEA
    # ─────────────────────────────────────────────────────────────────────────
 
    data:
      nrows:            {cfg.data.nrows}
      ncols:            {cfg.data.ncols}
      habitat_fraction: {cfg.data.habitat_fraction}
      cell_area:        {cfg.data.cell_area}
      seed:             {cfg.data.seed_gen}
      # Leave paths empty to use synthetic / computed data
      habitat_path:     {cfg.data.habitat_path}
      restorable_path:  {cfg.data.restorable_path}
      accessible_path:  {cfg.data.accessible_path}
      cost_path:        ""   # leave empty → uses compute_restoration_cost()
      locked_out_path:  ""
      elevation_path:   {cfg.data.elevation_path}   # optional DEM raster
 
    cost:
      tree_unit_cost:          {cfg.cost.tree_unit_cost}   # $ per tree
      tree_spacing_m:          {cfg.cost.tree_spacing_m}   # metres between trees (2–3 m → 1200–1700 trees/ha)
      # cell_size_m is NOT set here — it must equal landscape_extent_m / N.
      # Examples for PEC (square extent = 664 415 m):
      #   N= 30 → cell_size_m: 22147.0
      #   N= 60 → cell_size_m: 11074.0
      #   N=100 → cell_size_m:  6644.0
      # Override via CLI: python foremost.py cost.cell_size_m=6644
      inaccessible_surcharge:  {cfg.cost.inaccessible_surcharge}
      elevation_base_m:        {cfg.cost.elevation_base_m}
      elevation_slope:         {cfg.cost.elevation_slope}  # extra cost / metre elevation
      noise_sigma:             {cfg.cost.noise_sigma}
 
    constraints:
      min_restore:    {cfg.constraints.min_restore}
      max_restore:    {cfg.constraints.max_restore}
      max_diameter:   {cfg.constraints.max_diameter}
      max_nb_cc:      {cfg.constraints.max_nb_cc}
      min_proportion: {cfg.constraints.min_proportion}
      max_cost:       .inf
 
    optimizer:
      # algo: GA | NSGA2 | NSGA3 | CTAEA | RNSGA3
      algo:         {cfg.optimizer.algo}
      # objective: MESH | IIC | COST | MESH_IIC | MESH_COST | IIC_COST | FULL
      objective:    {cfg.optimizer.objective}
      pop_size:     {cfg.optimizer.pop_size}
      n_gen:        {cfg.optimizer.n_gen}
      seed:         {cfg.optimizer.seed}
      verbose:      {str(cfg.optimizer.verbose).lower()}
      iic_max_dist: {cfg.optimizer.iic_max_dist}
      penalty:      {cfg.optimizer.penalty}
      pop_per_ref_point: {cfg.optimizer.pop_per_ref_point}
      mu:      {cfg.optimizer.mu}
 
    output:
      dir:       "{cfg.output.dir}"
      prefix:    "{cfg.output.prefix}"
      dpi:       {cfg.output.dpi}
      show:      {str(cfg.output.show).lower()}
      fig_saved: false
    """
    )
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "foremost.yaml"
    path.write_text(yaml_text)
    print(f"Default config written to {path}")
    return path


def _cfg_to_dataclass(cfg) -> ForemostConfig:
    """
    Convert a Hydra DictConfig (or plain dict) to a ForemostConfig dataclass.
    Works with or without Hydra installed.
    """

    def _get(d, key, default):
        try:
            return d[key]
        except (KeyError, TypeError):
            return getattr(d, key, default)

    def _sub(d, key, cls):
        try:
            sub = d[key]
        except (KeyError, TypeError):
            sub = getattr(d, key, None)
        if sub is None:
            return cls()
        if isinstance(sub, cls):
            return sub
        kw = (
            dict(sub)
            if hasattr(sub, "__iter__") and not isinstance(sub, str)
            else {k: getattr(sub, k) for k in vars(cls()).keys()}
        )
        try:
            return cls(**kw)
        except TypeError:
            # Filter to known fields — handles unknown CLI overrides gracefully
            known = set(vars(cls()).keys())
            filtered = {k: v for k, v in kw.items() if k in known}
            try:
                return cls(**filtered)
            except TypeError:
                return cls()

    return ForemostConfig(
        data=_sub(cfg, "data", DataConfig),
        cost=_sub(cfg, "cost", CostConfig),
        constraints=_sub(cfg, "constraints", ConstraintsConfig),
        optimizer=_sub(cfg, "optimizer", OptimizerConfig),
        output=_sub(cfg, "output", OutputConfig),
        mode=_get(cfg, "mode", 0),
    )


# =============================================================================
# 1.  DATA LAYER
# =============================================================================


@dataclass
class HabitatData:
    """
    Container for all input rasters required by foremost + cost extension.

    Parameters
    ----------
    habitat : 2-D int array
        Binary raster: 1 = existing habitat, 0 = non-habitat, -1 = excluded.
    restorable : 2-D float array
        Restorable area (ha or any surface unit) per cell.
        0 where no restoration is possible.
    accessible : 2-D int array
        Binary raster: 1 = cell is accessible (candidate for restoration).
    cost : 2-D float array
        Restoration cost per cell (any monetary / effort unit).
        Must be non-negative. Use 0 for existing habitat or locked-out cells.
    cell_area : float
        Area of one raster cell in the same unit as `restorable`.
    locked_out : 2-D bool array or None
        True where restoration is explicitly forbidden.
    nodata_value : int
        Value used for excluded cells in the `habitat` raster (default -1).
    elevation : 2-D float array or None
        Optional DEM raster (metres); used by compute_restoration_cost().
    bg_image : 2-D or 3-D array or None
        Optional RGB / grayscale satellite image for background visualisation.
    """

    habitat: np.ndarray
    restorable: np.ndarray
    accessible: np.ndarray
    cost: np.ndarray
    cell_area: float = 1.0
    locked_out: Optional[np.ndarray] = None
    nodata_value: int = -1
    elevation: Optional[np.ndarray] = None
    bg_image: Optional[np.ndarray] = None
    app_mask: Optional[np.ndarray] = None  # binary N×N; 1 = APP zone (Forest Code Art. 61-A)
    georef: Optional[dict] = None         # {"extent": [xmin,ymin,xmax,ymax], "crs": "EPSG:…"}

    def __post_init__(self):
        shapes = dict(
            habitat=self.habitat.shape,
            restorable=self.restorable.shape,
            accessible=self.accessible.shape,
            cost=self.cost.shape,
        )
        if len(set(shapes.values())) != 1:
            raise ValueError(f"All rasters must share the same shape. Got: {shapes}")
        if (self.cost < 0).any():
            raise ValueError("`cost` raster must be non-negative.")
        if self.locked_out is None:
            self.locked_out = np.zeros(self.habitat.shape, dtype=bool)

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def candidate_mask(self) -> np.ndarray:
        """Boolean mask of cells that may be restored."""
        return (
            (self.accessible == 1)
            & (self.habitat != 1)
            & (self.habitat != self.nodata_value)
            & (~self.locked_out)
        )

    @property
    def habitat_mask(self) -> np.ndarray:
        return self.habitat == 1

    @property
    def shape(self) -> tuple:
        return self.habitat.shape

    # ── synthetic factory ─────────────────────────────────────────────────────

    @classmethod
    def synthetic(
        cls,
        nrows: int = 30,
        ncols: int = 30,
        habitat_fraction: float = 0.30,
        restorable_fraction: float = 0.50,
        accessible_fraction: float = 1.0,
        cell_area: float = 1.0,
        seed: int = 42,
    ) -> "HabitatData":
        """
        Generate a synthetic landscape for testing.
        Includes a synthetic elevation DEM (dome-shaped hill).
        """
        cfg = ForemostConfig()
        rng = np.random.default_rng(cfg.data.seed_gen)
        shape = (cfg.data.nrows, cfg.data.ncols)

        raw = rng.random(shape)
        habitat = (raw < habitat_fraction).astype(int)
        print("********** Habitat Array Generated \n")  # , habitat)
        restorable = np.where(
            habitat == 0,
            rng.uniform(0.2, 1.0, shape) * cell_area,
            0.0,
        )
        restorable = (restorable < restorable_fraction).astype(float)
        print("********** Restorable Array Generated \n")  # , restorable)
        # print(np.sum(np.sum(restorable)))

        # accessible: all non-habitat cells accessible except 5% border cells
        # accessible = np.ones((nrows, ncols), dtype=int)
        # accessible[:2, :] = 0
        # accessible[-2:, :] = 0
        # accessible[:, :2] = 0
        # accessible[:, -2:] = 0

        # modified: all restorable cells that have random accessibility
        accessible_temp = np.random.choice(
            [0, 1],
            size=(nrows, ncols),
            p=[1.0 - accessible_fraction, accessible_fraction],
        )
        # accessible = np.where((restorable == 1) & (accessible_temp== 1),1,0.0)
        combined = restorable + accessible_temp
        accessible = np.where((combined == 2), 1.0, 0.0)
        print("********** Accessible & Restorable Array Generated\n")  # , accessible)

        locked_out = (rng.random(shape) < 0.10) & (habitat == 0)

        # Elevation: dome-shaped hill (max 800 m at centre, 0 at borders)
        ri, ci = np.mgrid[0:nrows, 0:ncols]
        elev = 800.0 * np.exp(
            -(
                ((ri - nrows / 2) / (nrows / 3)) ** 2
                + ((ci - ncols / 2) / (ncols / 3)) ** 2
            )
        )
        elevation = elev + rng.uniform(0, 30, shape)

        # Cost derived from elevation + distance
        dist = np.sqrt(
            ((ri - nrows / 2) / (nrows / 2)) ** 2
            + ((ci - ncols / 2) / (ncols / 2)) ** 2
        )
        dist /= dist.max()
        noise = rng.uniform(0, 0.3, shape)
        cost_raw = 10.0 * (0.4 + 0.6 * dist + noise)
        cost = np.where((habitat == 0) & (~locked_out), cost_raw, 0.0)

        # Synthetic RGB background image (colour-coded by habitat type)
        bg = np.zeros((nrows, ncols, 3), dtype=np.uint8)
        bg[habitat == 1] = [45, 106, 79]  # dark green
        bg[habitat == 0] = [240, 240, 230]  # light beige
        bg[locked_out] = [180, 160, 140]  # brown
        bg[habitat == -1] = [200, 200, 200]  # grey nodata
        # Add noise for realism
        noise_rgb = rng.integers(0, 25, (nrows, ncols, 3), dtype=np.uint8)
        bg = np.clip(bg.astype(int) + noise_rgb - 12, 0, 255).astype(np.uint8)

        # APP mask: cells within 2 planning units (~200 m) of valley floors
        # Proxy for riparian buffers mandated by Forest Code Art. 61-A
        elev_norm = elev / 800.0  # normalize to [0, 1]
        valley_mask = elev_norm < 0.15  # valley floor proxy (low elevation)
        dist_to_valley = distance_transform_edt(~valley_mask)
        app_mask = (dist_to_valley <= 2.0).astype(np.int32)

        return cls(
            habitat=habitat,
            restorable=restorable,
            accessible=accessible,
            cost=cost,
            cell_area=cell_area,
            locked_out=locked_out,
            elevation=elevation,
            bg_image=bg,
            app_mask=app_mask,
        )

    # ── raster factory ────────────────────────────────────────────────────────

    @classmethod
    def from_rasters(
        cls,
        habitat_path: str,
        restorable_path: str,
        accessible_path: str,
        cost_path: str,
        cell_area: float = 1.0,
        locked_out_path: Optional[str] = None,
        elevation_path: Optional[str] = None,
        accessible_value: int = 1,
        nodata_value: int = -1,
        aggregation_factor: int = 1,
        habitat_threshold: float = 0.5,
        bg_image_path: Optional[str] = None,
    ) -> "HabitatData":
        """Load habitat + cost data from GeoTIFF rasters (requires rasterio)."""
        try:
            import rasterio
        except ImportError:
            raise ImportError("Install rasterio: pip install rasterio")

        def _load(path):
            with rasterio.open(path) as src:
                return src.read(1).astype(float)

        habitat = _load(habitat_path)
        restorable = _load(restorable_path)
        accessible = (_load(accessible_path) == accessible_value).astype(int)
        cost = _load(cost_path)
        locked_out = (
            (_load(locked_out_path) == 1).astype(bool) if locked_out_path else None
        )
        elevation = _load(elevation_path) if elevation_path else None

        if aggregation_factor > 1:
            habitat, restorable, accessible, locked_out, cost, elevation = _aggregate(
                habitat,
                restorable,
                accessible,
                locked_out,
                cost,
                aggregation_factor,
                habitat_threshold,
                nodata_value,
                elevation,
            )

        # Optional RGB background image
        bg = None
        if bg_image_path:
            try:
                from PIL import Image

                bg = np.array(Image.open(bg_image_path).convert("RGB"))
                # Resize to match raster
                from PIL import Image as PILImage

                pil = PILImage.fromarray(bg)
                pil = pil.resize((habitat.shape[1], habitat.shape[0]), PILImage.LANCZOS)
                bg = np.array(pil)
            except Exception as exc:
                warnings.warn(f"Could not load bg_image: {exc}")

        return cls(
            habitat=habitat.astype(int),
            restorable=restorable,
            accessible=accessible,
            cost=np.clip(cost, 0, None),
            cell_area=cell_area,
            locked_out=locked_out,
            nodata_value=nodata_value,
            elevation=elevation,
            bg_image=bg,
        )

    @classmethod
    def from_config(cls, cfg: ForemostConfig) -> "HabitatData":
        """
        Build a HabitatData from a ForemostConfig.
        Uses synthetic data when paths are empty / not specified.
        """
        d = cfg.data
        has_paths = all(
            bool(getattr(d, k))
            for k in ("habitat_path", "restorable_path", "accessible_path")
        )
        if has_paths:
            cost_p = d.cost_path if d.cost_path else None
            data = cls.from_rasters(
                habitat_path=d.habitat_path,
                restorable_path=d.restorable_path,
                accessible_path=d.accessible_path,
                cost_path=cost_p or d.habitat_path,  # placeholder
                cell_area=d.cell_area,
                locked_out_path=d.locked_out_path or None,
                elevation_path=d.elevation_path or None,
            )
            # Recompute cost if path missing
            if not cost_p:
                data.cost = compute_restoration_cost(data, cfg.cost)
        else:
            data = cls.synthetic(
                nrows=d.nrows,
                ncols=d.ncols,
                habitat_fraction=d.habitat_fraction,
                cell_area=d.cell_area,
                seed=d.seed_gen,
            )
            # Always recompute cost from the ecological model
            data.cost = compute_restoration_cost(data, cfg.cost)

        return data


def _aggregate(
    habitat,
    restorable,
    accessible,
    locked_out,
    cost,
    factor,
    threshold,
    nodata_value,
    elevation=None,
):
    nrows, ncols = habitat.shape
    nr, nc = nrows // factor, ncols // factor
    h_a = np.full((nr, nc), nodata_value, dtype=float)
    r_a = np.zeros((nr, nc))
    a_a = np.zeros((nr, nc), dtype=int)
    lo_a = np.zeros((nr, nc), dtype=bool)
    c_a = np.zeros((nr, nc))
    e_a = None if elevation is None else np.zeros((nr, nc))
    for i in range(nr):
        for j in range(nc):
            sl = np.s_[i * factor : (i + 1) * factor, j * factor : (j + 1) * factor]
            bh = habitat[sl]
            valid = bh != nodata_value
            if not valid.any():
                continue
            h_a[i, j] = 1 if bh[valid].mean() >= threshold else 0
            r_a[i, j] = restorable[sl].sum()
            a_a[i, j] = 1 if accessible[sl].mean() >= 0.5 else 0
            c_a[i, j] = cost[sl].mean()
            if locked_out is not None:
                lo_a[i, j] = locked_out[sl].mean() >= 0.5
            if elevation is not None:
                e_a[i, j] = elevation[sl].mean()
    return (h_a, r_a, a_a, (lo_a if locked_out is not None else None), c_a, e_a)


# =============================================================================
# 1b.  ECOLOGICAL COST FUNCTION
# =============================================================================


def compute_restoration_cost(
    data: HabitatData,
    cfg: Optional[CostConfig] = None,
    *,
    tree_unit_cost: float = 15.0,
    tree_spacing_m: float = 2.5,
    cell_size_m: float = 100.0,
    inaccessible_surcharge: float = 0.40,
    elevation_base_m: float = 0.0,
    elevation_slope: float = 0.005,
    noise_sigma: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a restoration-cost raster (N×N) from ecological parameters.
    *** >>> TO DO: add the proximity to the closest watershed network in the cost model

    The cost of restoring a cell is modelled as:

        cost(i,j) = n_trees(i,j)              # trees needed
                  × tree_unit_cost             # $/tree
                  × accessibility_factor(i,j) # logistics surcharge
                  × elevation_factor(i,j)     # altitude penalty
                  × (1 + ε)                   # multiplicative noise

    where:
        n_trees         = restorable_area_m² / tree_spacing_m²
        restorable_area = data.restorable[i,j] × cell_size_m²
                         (if data.restorable is a fraction in [0,1])
                         OR data.restorable[i,j] directly (if in m²/ha)

        accessibility_factor = 1.0 if data.accessible[i,j] == 1
                               else  1 + inaccessible_surcharge

        elevation_factor = 1 + elevation_slope × max(0, elevation - elevation_base_m)
                         (ignored if no elevation data)

        ε ~ Normal(0, noise_sigma)   multiplicative noise

    Parameters
    ----------
    data    : HabitatData with restorable, accessible, and optionally elevation
    cfg     : CostConfig dataclass (takes precedence over keyword arguments)
    tree_unit_cost         : $ per planted tree
    tree_spacing_m         : planting density (m between trees)
    cell_size_m            : side length of one raster cell in metres
    inaccessible_surcharge : fractional cost increase for inaccessible cells (e.g. 0.4 = +40%)
    elevation_base_m       : reference elevation below which there is no altitude penalty
    elevation_slope        : fractional cost increase per metre of elevation above base
    noise_sigma            : standard deviation of multiplicative noise
    seed                   : random seed for the noise

    Returns
    -------
    cost : ndarray, shape = data.shape, dtype float64, non-negative.
           Zero on existing-habitat and nodata cells.
    """
    if cfg is not None:
        tree_unit_cost         = getattr(cfg, "tree_unit_cost",         tree_unit_cost)
        tree_spacing_m         = getattr(cfg, "tree_spacing_m",         tree_spacing_m)
        cell_size_m            = getattr(cfg, "cell_size_m",            cell_size_m)
        inaccessible_surcharge = getattr(cfg, "inaccessible_surcharge", inaccessible_surcharge)
        elevation_base_m       = getattr(cfg, "elevation_base_m",       elevation_base_m)
        elevation_slope        = getattr(cfg, "elevation_slope",        elevation_slope)
        noise_sigma            = getattr(cfg, "noise_sigma",            noise_sigma)

    nrows, ncols = data.shape
    rng = np.random.default_rng(seed)

    # ── 1. restorable area per cell (m²) ────────────────────────────────────
    # If restorable values look like fractions (≤ 1), multiply by cell area.
    rest = data.restorable.astype(float).copy()
    if rest.max() <= 1.01:
        rest = rest * (cell_size_m**2)  # convert fraction → m²
    else:
        pass  # assume already in m² or ha; use as-is

    # ── 2. number of trees ───────────────────────────────────────────────────
    trees_per_m2 = 1.0 / max(tree_spacing_m**2, 1e-9)
    n_trees = rest * trees_per_m2  # shape (N, N)

    # ── 3. base cost = n_trees × unit cost ───────────────────────────────────
    base_cost = n_trees * tree_unit_cost  # shape (N, N)

    # ── 4. accessibility factor ──────────────────────────────────────────────
    acc_factor = np.where(
        data.accessible == 1,
        1.0,
        1.0 + inaccessible_surcharge,
    )

    # ── 5. elevation factor ──────────────────────────────────────────────────
    if data.elevation is not None:
        elev_above = np.maximum(0.0, data.elevation - elevation_base_m)
        elev_factor = 1.0 + elevation_slope * elev_above
    else:
        elev_factor = np.ones((nrows, ncols), dtype=float)

    # ── 6. multiplicative noise ──────────────────────────────────────────────
    noise_factor = 1.0 + rng.normal(0.0, noise_sigma, (nrows, ncols))
    noise_factor = np.clip(noise_factor, 0.5, 2.5)  # cap at ±50 % of base

    # ── 7. final cost ────────────────────────────────────────────────────────
    cost = base_cost * acc_factor * elev_factor * noise_factor
    cost = np.clip(cost, 0.0, None)

    # Zero out non-candidate cells
    cost[data.habitat == 1] = 0.0
    cost[data.habitat == data.nodata_value] = 0.0
    if data.locked_out is not None:
        cost[data.locked_out] = 0.0

    return cost


def compute_slope(elevation: np.ndarray, cell_size_m: float = 100.0) -> np.ndarray:
    """
    Compute terrain slope in degrees from an elevation array.

    Parameters
    ----------
    elevation  : 2-D float array, elevation in metres (not normalized).
    cell_size_m: side length of one raster cell in metres (default 100 m).

    Returns
    -------
    slope_deg : ndarray, same shape as elevation, values in degrees [0, 90).
    """
    gy, gx = np.gradient(elevation, cell_size_m)
    slope_rad = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    return np.degrees(slope_rad)


# =============================================================================
# 2.  LANDSCAPE INDICES
# =============================================================================

_CONN8 = generate_binary_structure(2, 2)


def compute_patches(habitat_grid: np.ndarray) -> tuple:
    """8-connected patch labelling. Returns (labeled, n_patches, areas_cells)."""
    labeled, n = ndimage_label(habitat_grid == 1, structure=_CONN8)
    if n == 0:
        return labeled, n, np.array([], dtype=np.int64)
    _, areas = np.unique(labeled[labeled > 0], return_counts=True)
    return labeled, n, areas


def mesh(habitat_grid: np.ndarray, total_cells: int, cell_area: float = 1.0) -> float:
    """Effective Mesh Size — Jaeger (2000).
     MESH = (1 / A_total) * Σ a_i²

    where a_i is the area of patch i (in surface units) and A_total is the
    total landscape area (excluding nodata).

        Args:
            binary_map: Binary landscape (1=habitat, 0=non-habitat)
            cell_area: Area of each cell

        Returns:
            MESH value (larger = less fragmented)


    """
    _, _, areas = compute_patches(habitat_grid)
    if len(areas) == 0:
        return 0.0
    a = areas * cell_area
    return float(np.sum(a**2) / (total_cells * cell_area))


def iic(
    habitat_grid: np.ndarray,
    total_cells: int,
    cell_area: float = 1.0,
    max_dist: int = 10,
) -> float:
    """Integral Index of Connectivity — Pascual-Hortal & Saura (2006).

     IIC = [Σ_{i,j} (a_i * a_j) / (1 + nl_{ij})] / A_L²

    where nl_{ij} is the number of links (cells) in the shortest path between
    patch centroids i and j through the landscape graph (graph of patches
    connected when within `max_dist` cells of each other), and A_L is the
    total landscape area.

    For efficiency the landscape graph is built by checking whether any cell
    of patch i is within `max_dist` (Chebyshev distance) of any cell of
    patch j.
    """

    labeled, n, areas = compute_patches(habitat_grid)
    if n == 0:
        return 0.0
    A_L = total_cells * cell_area
    a = areas * cell_area

    # Build patch connectivity graph via KDTree (Chebyshev / L-inf metric).
    # Single query over all habitat cells — O(m log m) vs O(n × m²) dilation.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    all_coords = np.argwhere(labeled > 0)          # (m, 2) — all habitat cells
    all_labels = labeled[all_coords[:, 0], all_coords[:, 1]]  # patch label per cell
    tree = _cKDTree(all_coords)
    neighbor_lists = tree.query_ball_tree(tree, r=max_dist, p=np.inf)
    for idx, nbrs in enumerate(neighbor_lists):
        li = int(all_labels[idx])
        for ni in nbrs:
            lj = int(all_labels[ni])
            if lj > li:
                G.add_edge(li - 1, lj - 1)

    # All-pairs shortest path lengths in one BFS sweep, then vectorised sum.
    spl = dict(nx.all_pairs_shortest_path_length(G))
    NL = np.full((n, n), np.inf)
    for src, dsts in spl.items():
        for dst, length in dsts.items():
            NL[src, dst] = length

    finite = np.isfinite(NL)
    ai_aj = a[:, None] * a[None, :]
    num = float(np.sum(ai_aj[finite] / (1.0 + NL[finite])))
    return float(num / (A_L ** 2))


# =============================================================================
# 3.  SPATIAL CONSTRAINT HELPERS
# =============================================================================


def connected_components(sel_grid: np.ndarray) -> tuple:
    """Return (labeled, n_cc) for the set of selected (==1) cells."""
    labeled, n = ndimage_label(sel_grid == 1, structure=_CONN8)
    return labeled, n


def diameter(sel_grid: np.ndarray) -> int:
    """
    Diameter of the restored zone = longest shortest path in the grid graph
    restricted to selected cells (measured in cell hops, 8-connectivity).
    Uses BFS from each selected cell — suitable for small selections.
    """

    cells = list(zip(*np.where(sel_grid == 1)))
    if not cells:
        return 0
    if len(cells) == 1:
        return 1
    cell_set = set(cells)

    def bfs(start):
        dist = {start: 0}
        q = [start]
        mx = 0
        far = start
        while q:
            nq = []
            for r, c in q:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        nb = (r + dr, c + dc)
                        if nb in cell_set and nb not in dist:
                            dist[nb] = dist[(r, c)] + 1
                            if dist[nb] > mx:
                                mx, far = dist[nb], nb
                            nq.append(nb)
            q = nq
        return mx, far

    _, far1 = bfs(cells[0])
    mx, _ = bfs(far1)
    return mx + 1


# =============================================================================
# 4.  PROBLEM DEFINITION
# =============================================================================


class ObjectiveType(str, Enum):
    MESH = "MESH"
    IIC = "IIC"
    COST = "COST"
    MESH_IIC = "MESH_IIC"
    MESH_COST = "MESH_COST"
    IIC_COST = "IIC_COST"
    FULL = "FULL"

    @property
    def is_multi(self) -> bool:
        return self in {self.MESH_IIC, self.MESH_COST, self.IIC_COST, self.FULL}

    @property
    def n_obj(self) -> int:
        return {
            self.MESH: 1,
            self.IIC: 1,
            self.COST: 1,
            self.MESH_IIC: 2,
            self.MESH_COST: 2,
            self.IIC_COST: 2,
            self.FULL: 3,
        }[self]

    @property
    def needs_mesh(self) -> bool:
        return self in {self.MESH, self.MESH_IIC, self.MESH_COST, self.FULL}

    @property
    def needs_iic(self) -> bool:
        return self in {self.IIC, self.MESH_IIC, self.IIC_COST, self.FULL}

    @property
    def needs_cost(self) -> bool:
        return self in {self.COST, self.MESH_COST, self.IIC_COST, self.FULL}


@dataclass
class RestorationConstraints:
    """
    Spatial and budgetary constraints on the restoration selection.

    Parameters
    ----------
    min_restore    : minimum total restorable area to select
    max_restore    : maximum total restorable area to select
    max_diameter   : maximum spatial diameter of the restored zone (cells)
    max_nb_cc      : maximum number of connected components
    min_proportion : minimum habitat proportion after restoration (0-1)
    max_cost       : maximum total restoration budget
    """

    min_restore: float = 2.0
    max_restore: float = float("inf")
    max_diameter: int = 10
    max_nb_cc: int = 1
    min_proportion: float = 0.0
    max_cost: float = float("inf")
    min_app_compliance: float = 0.0
    max_slope_deg: float = float("inf")
    min_patch_size: int = 0
    max_edge_ratio: float = 1.0
    min_iic_delta: float = 0.0
    max_distance_to_habitat: int = 0
    min_carbon_stock: float = 0.0
    carbon_rate_tco2_ha_yr: float = 3.0
    carbon_horizon_yr: int = 20
    min_corridor_width: int = 0

    @classmethod
    def from_config(cls, cfg: ConstraintsConfig) -> "RestorationConstraints":
        return cls(**asdict(cfg))


class RestorationProblem(ElementwiseProblem):
    """
    pymoo ElementwiseProblem for the restopt + cost optimisation problem.

    IMPORTANT: attributes are set BEFORE super().__init__() to prevent pymoo
    from overwriting self.data (which pymoo uses internally as a cache dict).
    We use self.habitat_data to hold our HabitatData instance.

    Decision variables
    ------------------
    x : binary vector of length n_candidates
        x[k] = 1  →  restore candidate cell k
        x[k] = 0  →  do not restore

    Objectives  (minimized internally; negated MESH / IIC scores)
    ----------
    Single-objective (MESH or IIC or COST): 1 objective, 0 inequality constraints
    Multi-objective (MULTI)        : 3 objectives, 0 inequality constraints
    Constraint violations are handled as penalty terms inside the objectives.

    Parameters
    ----------
    data        : HabitatData
    constraints : RestorationConstraints
    objective   : ObjectiveType
    penalty     : coefficient for constraint violation penalty
    iic_max_dist: max patch distance (cells) considered for IIC graph


    """

    def __init__(
        self,
        data: HabitatData,
        constraints: RestorationConstraints,
        objective: ObjectiveType = ObjectiveType.MESH,
        penalty: float = 1e6,
        iic_max_dist: int = 10,
        cell_size_m: float = 100.0,
    ):
        # Assign before super().__init__() — see class docstring
        self.habitat_data = data
        self.constraints = constraints
        self.objective = objective
        self.penalty = penalty
        self.iic_max_dist = iic_max_dist
        self.cell_size_m = cell_size_m

        cr, cc = np.where(data.candidate_mask)
        self._candidate_rows = cr
        self._candidate_cols = cc
        self.n_candidates = len(cr)
        if self.n_candidates == 0:
            raise ValueError("No candidate cells found — check input rasters.")

        self.total_area_cells = int((data.habitat != data.nodata_value).sum())
        cand_costs = data.cost[cr, cc]
        self._cost_scale = float(cand_costs.sum()) or 1.0

        # ---- NC3: precompute baseline IIC --------------------------------
        if constraints.min_iic_delta > 0.0:
            self._baseline_iic = iic(
                data.habitat,
                total_cells=self.total_area_cells,
                cell_area=data.cell_area,
                max_dist=iic_max_dist,
            )
        else:
            self._baseline_iic = 0.0

        # ---- NC4: precompute per-candidate Chebyshev distance to habitat -
        if constraints.max_distance_to_habitat > 0:
            hab_rows, hab_cols = np.where(data.habitat == 1)
            if len(hab_rows) == 0:
                self._cand_dist_to_hab = np.full(self.n_candidates, np.inf)
            else:
                dr = np.abs(cr[:, None] - hab_rows[None, :])
                dc = np.abs(cc[:, None] - hab_cols[None, :])
                self._cand_dist_to_hab = np.maximum(dr, dc).min(axis=1)
        else:
            self._cand_dist_to_hab = None

        # ---- NC5: precompute per-candidate slope mask --------------------
        if np.isfinite(constraints.max_slope_deg) and data.elevation is not None:
            slope_grid = compute_slope(data.elevation, cell_size_m=cell_size_m)
            self._cand_slope = slope_grid[cr, cc]
        else:
            self._cand_slope = None

        super().__init__(
            n_var=self.n_candidates,
            n_obj=objective.n_obj,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _build_habitat_grid(self, x):
        g = self.habitat_data.habitat.copy().astype(int)
        sel = x.astype(bool)
        g[self._candidate_rows[sel], self._candidate_cols[sel]] = 1
        return g

    def _build_selection_grid(self, x):
        g = np.zeros(self.habitat_data.shape, dtype=int)
        sel = x.astype(bool)
        g[self._candidate_rows[sel], self._candidate_cols[sel]] = 1
        return g

    def _violation(self, x):
        c = self.constraints
        sel = x.astype(bool)
        v = 0.0

        # ---- 1. Restorable area bounds --------------------------------

        area = float(
            self.habitat_data.restorable[
                self._candidate_rows[sel], self._candidate_cols[sel]
            ].sum()
        )
        if area < c.min_restore:
            v += (c.min_restore - area) ** 2
        if area > c.max_restore:
            v += (area - c.max_restore) ** 2
        tc = float(
            self.habitat_data.cost[
                self._candidate_rows[sel], self._candidate_cols[sel]
            ].sum()
        )
        if tc > c.max_cost:
            v += ((tc - c.max_cost) / self._cost_scale) ** 2

        # ---- 2. Compactness (max diameter) ----------------------------

        if sel.any():
            sg = self._build_selection_grid(x)
            d = diameter(sg)
            if d > c.max_diameter:
                v += (d - c.max_diameter) ** 2
            # ---- 3. Connected components ------------------------------
            _, n_cc = connected_components(sg)
            if n_cc > c.max_nb_cc:
                v += (n_cc - c.max_nb_cc) ** 2

        # ---- 4. Minimum habitat proportion ---------------------------

        if c.min_proportion > 0.0:
            hg = self._build_habitat_grid(x)
            valid = self.habitat_data.habitat != self.habitat_data.nodata_value
            prop = hg[valid].mean()
            if prop < c.min_proportion:
                v += (c.min_proportion - prop) ** 2

        # ---- 5. APP compliance (Brazilian Forest Code Art. 61-A) -----
        if c.min_app_compliance > 0.0 and self.habitat_data.app_mask is not None:
            if sel.any():
                app_vals = self.habitat_data.app_mask[
                    self._candidate_rows[sel], self._candidate_cols[sel]
                ]
                app_fraction = float(app_vals.mean())
                if app_fraction < c.min_app_compliance:
                    v += (c.min_app_compliance - app_fraction) ** 2

        # ---- 6. Slope (NC5) — uses precomputed per-candidate slope ----
        if self._cand_slope is not None and sel.any():
            excess = np.maximum(0.0, self._cand_slope[sel] - c.max_slope_deg)
            if excess.any():
                v += float((excess ** 2).sum())

        # ---- NC1: Minimum viable patch size ---------------------------
        if c.min_patch_size > 0 and sel.any():
            sg = self._build_selection_grid(x)
            labeled_sel, n_sel = ndimage_label(sg == 1, structure=_CONN8)
            for lbl in range(1, n_sel + 1):
                cc_size = int((labeled_sel == lbl).sum())
                if cc_size < c.min_patch_size:
                    v += (c.min_patch_size - cc_size) ** 2

        # ---- NC2: Maximum edge-to-area ratio --------------------------
        if c.max_edge_ratio < 0.999 and sel.any():
            sg = self._build_selection_grid(x)
            n_selected = int(sel.sum())
            if n_selected > 0:
                from scipy.ndimage import binary_erosion as _bin_erosion
                interior = _bin_erosion(
                    sg == 1, structure=np.ones((3, 3), dtype=int), border_value=0
                )
                n_edge = n_selected - int(interior.sum())
                ratio = n_edge / n_selected
                if ratio > c.max_edge_ratio:
                    v += (ratio - c.max_edge_ratio) ** 2 * n_selected

        # ---- NC3: Minimum IIC delta -----------------------------------
        if c.min_iic_delta > 0.0 and sel.any():
            hg = self._build_habitat_grid(x)
            current_iic = iic(
                hg,
                total_cells=self.total_area_cells,
                cell_area=self.habitat_data.cell_area,
                max_dist=self.iic_max_dist,
            )
            delta = current_iic - self._baseline_iic
            if delta < c.min_iic_delta:
                v += (c.min_iic_delta - delta) ** 2 * 1e4

        # ---- NC4: Maximum distance to existing habitat ----------------
        if self._cand_dist_to_hab is not None and sel.any():
            excess_dist = self._cand_dist_to_hab[sel] - c.max_distance_to_habitat
            excess_dist = excess_dist[excess_dist > 0]
            if len(excess_dist) > 0:
                v += float((excess_dist ** 2).sum())

        # ---- NC7: Minimum carbon stock --------------------------------
        if c.min_carbon_stock > 0.0 and sel.any():
            n_selected = int(sel.sum())
            # Assume 100 m × 100 m cells → 1 ha per cell
            total_area_ha = float(n_selected)
            carbon_total = total_area_ha * c.carbon_rate_tco2_ha_yr * c.carbon_horizon_yr
            if carbon_total < c.min_carbon_stock:
                v += ((c.min_carbon_stock - carbon_total) / max(c.min_carbon_stock, 1.0)) ** 2

        # ---- NC8: Minimum corridor width (morphological erosion) ------
        if c.min_corridor_width > 0 and sel.any():
            from scipy.ndimage import binary_erosion as _bin_erosion2
            sg = self._build_selection_grid(x)
            eroded = sg.astype(bool)
            _, n_initial = ndimage_label(eroded, structure=_CONN8)
            if n_initial > 0:
                struct3 = np.ones((3, 3), dtype=int)
                for step in range(1, c.min_corridor_width + 1):
                    eroded = _bin_erosion2(eroded, structure=struct3, border_value=0)
                    _, n_eroded = ndimage_label(eroded, structure=_CONN8)
                    if n_eroded != n_initial or not eroded.any():
                        v += (c.min_corridor_width - step + 1) ** 2
                        break

        return v

    # ── pymoo evaluation ──────────────────────────────────────────────────────

    def _evaluate(self, x, out, *args, **kwargs):
        obj = self.objective
        P = self.penalty * self._violation(x)
        sel = x.astype(bool)

        mesh_v = iic_v = cost_n = None
        if obj.needs_mesh or obj.needs_iic:
            hg = self._build_habitat_grid(x)
        if obj.needs_mesh:
            mesh_v = mesh(hg, self.total_area_cells, self.habitat_data.cell_area)
        if obj.needs_iic:
            iic_v = iic(
                hg,
                self.total_area_cells,
                self.habitat_data.cell_area,
                self.iic_max_dist,
            )
        if obj.needs_cost:
            cost_v = float(
                self.habitat_data.cost[
                    self._candidate_rows[sel], self._candidate_cols[sel]
                ].sum()
            )
            cost_n = cost_v / self._cost_scale

        if obj == ObjectiveType.MESH:
            out["F"] = [-mesh_v + P]
        elif obj == ObjectiveType.IIC:
            out["F"] = [-iic_v + P]
        elif obj == ObjectiveType.COST:
            out["F"] = [cost_n + P]
        elif obj == ObjectiveType.MESH_IIC:
            out["F"] = [-mesh_v + P, -iic_v + P]
        elif obj == ObjectiveType.MESH_COST:
            out["F"] = [-mesh_v + P, cost_n + P]
        elif obj == ObjectiveType.IIC_COST:
            out["F"] = [-iic_v + P, cost_n + P]
        elif obj == ObjectiveType.FULL:
            out["F"] = [-mesh_v + P, -iic_v + P, cost_n + P]

    def decode_solution(self, x: np.ndarray) -> dict:
        sel = x.astype(bool)
        sg = self._build_selection_grid(x)
        hg = self._build_habitat_grid(x)
        rs, cs = self._candidate_rows[sel], self._candidate_cols[sel]
        area = float(self.habitat_data.restorable[rs, cs].sum())
        tc = float(self.habitat_data.cost[rs, cs].sum())
        _, n_cc = connected_components(sg)
        diam = diameter(sg) if sel.any() else 0
        mesh_v = mesh(hg, self.total_area_cells, self.habitat_data.cell_area)
        iic_v = iic(
            hg, self.total_area_cells, self.habitat_data.cell_area, self.iic_max_dist
        )
        _, n_pat, _ = compute_patches(hg)
        return dict(
            n_restored_cells=int(sel.sum()),
            total_restored_area=area,
            total_cost=tc,
            n_connected_components=n_cc,
            diameter_cells=diam,
            n_patches=n_pat,
            mesh=mesh_v,
            iic=iic_v,
            habitat_grid=hg,
            selection_grid=sg,
        )


# =============================================================================
# 5.  SOLVER  —  now with NSGA3 / CTAEA / RNSGA3
# =============================================================================


class ProgressCallback(Callback):
    def __init__(self, every: int = 10):
        super().__init__()
        self.every = every

    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % self.every == 0 or gen == 1:
            try:
                F = algorithm.pop.get("F")
                best = F.min(axis=0) if F is not None and len(F) > 0 else None
                if best is not None:
                    fmt = "  ".join(f"{v:+.4f}" for v in best)
                    print(f"  Gen {gen:4d} | best F = [{fmt}]")
            except Exception:
                print(f"  Gen {gen:4d}")


def _build_algorithm(
    algo_name: str,
    n_obj: int,
    pop_size: int,
    sampling,
    crossover,
    mutation,
) -> object:
    """
    Instantiate a pymoo algorithm by name.

    Supported names
    ---------------
    GA       : single-objective genetic algorithm
    NSGA2    : Non-dominated Sorting GA II (Deb et al. 2002)
    NSGA3    : Non-dominated Sorting GA III (Deb & Jain 2014)
    CTAEA    : Constrained Two-Archive Evolutionary Algorithm (Li et al. 2019)
    RNSGA3   : Reference-point-based NSGA3 with user-supplied aspiration points
               (Blank et al. 2019)
    """
    name = algo_name.upper()
    common = dict(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    if name == "GA":
        if n_obj > 1:
            warnings.warn(
                f"GA is a single-objective algorithm; "
                f"falling back to NSGA2 for {n_obj}-objective problem."
            )
            return NSGA2(**common)
        return GA(**common)

    if name == "NSGA2":
        return NSGA2(**common)

    if name == "NSGA3":
        if not _HAS_NSGA3:
            warnings.warn("NSGA3 not available; falling back to NSGA2.")
            return NSGA2(**common)
        n_part = max(4, pop_size // (n_obj * 3))
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)
        return NSGA3(ref_dirs=ref_dirs, **common)

    if name == "CTAEA":
        if not _HAS_CTAEA:
            warnings.warn(
                "CTAEA not available (pymoo ≥ 0.6 required); " "falling back to NSGA2."
            )
            return NSGA2(**common)
        if n_obj < 2:
            warnings.warn("CTAEA is multi-objective; using NSGA2 for Single-objective.")
            return NSGA2(**common)
        n_part = max(4, min(pop_size // (n_obj * 3), 12))
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)
        # CTAEA sets pop_size from ref_dirs — pass only operators, not pop_size
        ctaea_kwargs = {k: v for k, v in common.items()
                        if k in ("sampling", "crossover", "mutation")}
        return CTAEA(ref_dirs=ref_dirs, **ctaea_kwargs)

    if name == "RNSGA3":
        if not _HAS_RNSGA3:
            warnings.warn("RNSGA3 not available; falling back to NSGA2.")
            return NSGA2(**common)
        if n_obj < 2:
            warnings.warn(
                "RNSGA3 is multi-objective; using NSGA2 for Single-objective."
            )
            return NSGA2(**common)
        # Aspiration points: identity simplex corners + centroid
        corners = np.eye(n_obj)
        centroid = np.full((1, n_obj), 1.0 / n_obj)
        ref_points = np.vstack([corners, centroid])
        # pop_per_ref_point must satisfy Das-Dennis triangular constraint
        # Nearest triangular number >= pop_size // n_ref
        n_ref = len(ref_points)
        target_ppr = max(10, pop_size // n_ref)
        _n = int((-1 + (1 + 8 * target_ppr) ** 0.5) / 2)
        ppr = _n * (_n + 1) // 2
        if ppr < target_ppr:
            ppr = (_n + 1) * (_n + 2) // 2
        rnsga3_kwargs = {k: v for k, v in common.items()
                         if k in ("sampling", "crossover", "mutation")}
        return RNSGA3(ref_points=ref_points, pop_per_ref_point=ppr, mu=0.1,
                      **rnsga3_kwargs)

    raise ValueError(
        f"Unknown algorithm '{algo_name}'. " f"Choose from: {SUPPORTED_ALGOS}"
    )


def solve(
    problem: RestorationProblem,
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
    verbose: bool = True,
    algo_name: str = "NSGA2",
) -> dict:
    """
    Run the evolutionary optimisation.

    Parameters
    ----------
    problem   : RestorationProblem instance
    pop_size  : population size
    n_gen     : number of generations
    seed      : random seed
    verbose   : print progress
    algo_name : one of "GA", "NSGA2", "NSGA3", "CTAEA", "RNSGA3"

    Returns
    -------
    dict with keys:
        "result"    : raw pymoo Result
        "solutions" : list of decoded solution dicts
        "algo_name" : name of the algorithm used (str)
    """
    # Auto-select algorithm for Single-objective
    if not problem.objective.is_multi and algo_name.upper() not in ("GA", "NSGA2"):
        warnings.warn(
            f"{algo_name} is multi-objective; switching to GA for "
            f"single-objective problem."
        )
        algo_name = "GA"

    sampling = BinaryRandomSampling()
    crossover = TwoPointCrossover()
    mutation = BitflipMutation(prob=max(1.0 / problem.n_var, 0.01))
    termination = get_termination("n_gen", n_gen)

    algorithm = _build_algorithm(
        algo_name,
        problem.objective.n_obj,
        pop_size,
        sampling,
        crossover,
        mutation,
    )
    actual_algo = type(algorithm).__name__

    if verbose:
        w = 62
        print(f"\n{'─'*w}")
        print(
            f"  FOREMOST  |  algo : {actual_algo:<10} "
            f"|  obj : {problem.objective.value}"
        )
        print(
            f"  candidates : {problem.n_candidates:>4d}  "
            f"|  pop : {pop_size}  |  gen : {n_gen}"
        )
        print(f"{'─'*w}")

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False,
        **({} if not verbose else {"callback": ProgressCallback(every=max(1, n_gen // 10))}),
    )

    # ── population-level feasibility (empirical, not tautological) ───────────
    # FOREMOST uses penalty-based constraint handling (penalty λ=10⁶ added to F).
    # We do NOT set out["G"], so ind.G is always None.  Instead we detect
    # infeasibility by checking whether any F value exceeds a penalty threshold
    # (10.0 covers all normal objective ranges; penalty adds ~10⁶).
    PENALTY_THRESHOLD = 10.0
    try:
        pop_F = result.pop.get("F")
        if pop_F is not None and len(pop_F) > 0:
            n_pop = len(pop_F)
            n_infeasible = int((pop_F > PENALTY_THRESHOLD).any(axis=1).sum())
        else:
            n_pop = len(result.pop)
            n_infeasible = 0
        pop_feasibility_rate = 1.0 - n_infeasible / n_pop if n_pop > 0 else 1.0
        if verbose:
            print(
                f"[feasibility] Population: {n_pop - n_infeasible}/{n_pop} feasible "
                f"({pop_feasibility_rate:.1%})"
            )
    except Exception:
        pop_feasibility_rate = float("nan")

    if problem.objective.is_multi:
        solutions = [problem.decode_solution(result.X[i]) for i in range(len(result.X))]
    else:
        solutions = [problem.decode_solution(result.X)]

    # ── Hypervolume indicator (multi-objective only) ──────────────────────────
    # Compute HV relative to a fixed reference point for reproducible reporting.
    # Reference: objectives are negated MESH/IIC (so negatives) + normalised cost.
    # We use the worst possible reference: 0 for negated objectives, 1.5 for cost.
    hv_value = float("nan")
    if problem.objective.is_multi and result.F is not None and len(result.F) > 0:
        try:
            from pymoo.indicators.hv import HV
            n_obj = result.F.shape[1]
            ref_point = np.ones(n_obj) * 1.5
            hv_indicator = HV(ref_point=ref_point)
            hv_value = float(hv_indicator.do(result.F))
            if verbose:
                print(f"[hypervolume] HV = {hv_value:.6f}  "
                      f"(ref_point = {ref_point.tolist()}, n_obj = {n_obj})")
        except Exception as e:
            if verbose:
                print(f"[hypervolume] Could not compute HV: {e}")

    if verbose:
        print(f"\n  Optimisation complete  [{actual_algo}]")
        if not problem.objective.is_multi:
            s = solutions[0]
            print(f"  Restored cells  : {s['n_restored_cells']}")
            print(f"  Restored area   : {s['total_restored_area']:.3f}")
            print(f"  Total cost      : {s['total_cost']:.3f}")
            print(f"  MESH            : {s['mesh']:.5f}")
            print(f"  IIC             : {s['iic']:.7f}")
            print(f"  Diameter        : {s['diameter_cells']} cells")
            print(f"  #Connected components   : {s['n_connected_components']}")
        else:
            print(f"  Pareto front    : {len(solutions)} non-dominated solution(s)")
            if not (hv_value != hv_value):  # not NaN
                print(f"  Hypervolume     : {hv_value:.6f}")

    return dict(
        result=result,
        solutions=solutions,
        algo_name=actual_algo,
        pop_feasibility_rate=pop_feasibility_rate,
        hypervolume=hv_value,
    )


# =============================================================================
# 6.  VISUALISATION
# =============================================================================

# ── helpers ───────────────────────────────────────────────────────────────────


def _hab_cmap_norm():
    """(cmap, norm) for a 4-class habitat/solution raster."""
    cmap = mcolors.ListedColormap(
        [_C["nodata"], _C["other"], _C["habitat"], _C["restored"]]
    )
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    return cmap, norm


def _encode_grid(habitat_orig: np.ndarray, selection: np.ndarray) -> np.ndarray:
    """Encode combined grid: -1=nodata, 0=other, 1=habitat, 2=restored."""
    out = habitat_orig.astype(float).copy()
    out[selection == 1] = 2
    return out


def _hab_legend(ax):
    patches = [
        mpatches.Patch(color=_C["habitat"], label="Existing habitat"),
        mpatches.Patch(color=_C["restored"], label="Restored cell"),
        mpatches.Patch(color=_C["other"], label="Non-habitat"),
        mpatches.Patch(color=_C["nodata"], label="No data"),
    ]
    ax.legend(
        handles=patches,
        loc="lower right",
        fontsize=7,
        framealpha=0.85,
        edgecolor="none",
    )


def _overlay_bg(ax, bg_image: Optional[np.ndarray], alpha: float = 0.35):
    """
    Draw bg_image as a translucent background on ax.
    bg_image can be (H, W) grayscale or (H, W, 3) RGB uint8 / float.
    """
    if bg_image is None:
        return
    img = bg_image.astype(float)
    if img.ndim == 2:
        # grayscale → stretch to [0,1] and display
        lo, hi = img.min(), img.max()
        img = (img - lo) / (hi - lo + 1e-9)
        ax.imshow(
            img,
            cmap="gray",
            aspect="auto",
            vmin=0,
            vmax=1,
            alpha=alpha,
            zorder=0,
            interpolation="nearest",
        )
    else:
        # RGB — normalize if needed
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0, 1)
        ax.imshow(img, aspect="auto", alpha=alpha, zorder=0, interpolation="nearest")


def _algo_subtitle(algo_name: Optional[str], objective: Optional[str] = None) -> str:
    parts = []
    if algo_name:
        parts.append(f"algo : {algo_name}")
    if objective:
        parts.append(f"obj : {objective}")
    return "  |  ".join(parts) if parts else ""


# ── plot_solution ─────────────────────────────────────────────────────────────


def plot_solution(
    data: HabitatData,
    solution: dict,
    title: str = "Restoration solution",
    algo_name: Optional[str] = None,
    figsize: tuple = (16, 9),
    save_path: Optional[str] = None,
    fig_saved: bool = False,
    bg_alpha: float = 0.30,
    show: bool = False,
):
    """
    Rich 2×3 layout with optional satellite background in transparency.

    ┌──────────────┬──────────────────┬─────────────┐
    │ Original     │ Post-restoration │ Cost heatmap│
    │ habitat      │ habitat          │ + selection │
    ├──────────────┼──────────────────┼─────────────┤
    │ Selection    │ Restorable area  │ Metrics     │
    │ map          │ heatmap          │ panel       │
    └──────────────┴──────────────────┴─────────────┘

    Parameters
    ----------
    data      : HabitatData (data.bg_image used as background if available)
    solution  : dict returned by RestorationProblem.decode_solution()
    title     : main figure title
    algo_name : algorithm name, displayed in every subplot and suptitle
    figsize   : (width, height) in inches
    save_path : if given, save figure to this path
    bg_alpha  : transparency of the satellite background (0=invisible, 1=opaque)
    show      : call plt.show() if True
    """
    hc, hn = _hab_cmap_norm()
    combined = _encode_grid(data.habitat, solution["selection_grid"])
    bg = data.bg_image  # may be None

    subtitle = _algo_subtitle(algo_name)
    full_title = f"{title}\n{subtitle}" if subtitle else title

    fig = plt.figure(figsize=figsize, facecolor=_C["bg"])
    fig.suptitle(full_title, fontsize=13, fontweight="bold", y=0.99, color="#111111")
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.40,
        wspace=0.30,
        left=0.05,
        right=0.96,
        top=0.92,
        bottom=0.05,
    )

    sel_contour_kw = dict(levels=[0.5], colors=["#0525f5"], linewidths=1.6)
    algo_tag = f"[{algo_name}]" if algo_name else ""

    # ── (0,0) original habitat ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _overlay_bg(ax, bg, alpha=bg_alpha)
    ax.imshow(
        data.habitat, cmap=hc, norm=hn, interpolation="nearest", alpha=0.85, zorder=1
    )
    ax.set_title(f"Original habitat  {algo_tag}", fontsize=10, fontweight="bold", pad=5)
    _hab_legend(ax)
    ax.axis("off")

    # ── (0,1) post-restoration ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _overlay_bg(ax, bg, alpha=bg_alpha)
    ax.imshow(combined, cmap=hc, norm=hn, interpolation="nearest", alpha=0.85, zorder=1)
    ax.set_title(f"Post-restoration  {algo_tag}", fontsize=10, fontweight="bold", pad=5)
    _hab_legend(ax)
    ax.axis("off")

    # ── (0,2) cost heatmap ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    _overlay_bg(ax, bg, alpha=bg_alpha)
    cost_disp = data.cost.astype(float).copy()
    cost_disp[data.habitat == data.nodata_value] = np.nan
    cost_disp[data.habitat == 1] = np.nan
    im = ax.imshow(
        cost_disp, cmap="YlOrRd", interpolation="nearest", alpha=0.85, zorder=1
    )
    if solution["selection_grid"].sum() > 0:
        ax.contour(solution["selection_grid"], zorder=2, **sel_contour_kw)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cost / cell")
    ax.set_title(f"Restoration cost  {algo_tag}", fontsize=10, fontweight="bold", pad=5)
    ax.axis("off")

    # ── (1,0) selection map ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _overlay_bg(ax, bg, alpha=bg_alpha)
    sc_cm = mcolors.ListedColormap([_C["other"], _C["restored"]])
    sc_nm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], sc_cm.N)
    ax.imshow(
        solution["selection_grid"],
        cmap=sc_cm,
        norm=sc_nm,
        interpolation="nearest",
        alpha=0.85,
        zorder=1,
    )
    ax.set_title(f"Selected cells  {algo_tag}", fontsize=10, fontweight="bold", pad=5)
    patches = [
        mpatches.Patch(color=_C["restored"], label="Selected"),
        mpatches.Patch(color=_C["other"], label="Not selected"),
    ]
    ax.legend(
        handles=patches,
        loc="lower right",
        fontsize=7,
        framealpha=0.85,
        edgecolor="none",
    )
    ax.axis("off")

    # ── (1,1) restorable area heatmap ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _overlay_bg(ax, bg, alpha=bg_alpha)
    rest_disp = data.restorable.astype(float).copy()
    rest_disp[data.habitat == 1] = np.nan
    rest_disp[data.habitat == data.nodata_value] = np.nan
    hr = mcolors.ListedColormap([_C["other"], _C["restored"]])
    im = ax.imshow(rest_disp, cmap=hr, interpolation="nearest", alpha=0.85, zorder=1)
    if solution["selection_grid"].sum() > 0:
        ax.contour(solution["selection_grid"], zorder=2, **sel_contour_kw)
    # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Restorable area")
    ax.set_title(
        f"Restorable area \n(blue = selected)  {algo_tag}",
        fontsize=10,
        fontweight="bold",
        pad=5,
    )
    ax.axis("off")

    # ── (1,2) metrics panel ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    metrics = [
        ("Restored cells", f"{solution['n_restored_cells']}", _C["restored"]),
        ("Restored area", f"{solution['total_restored_area']:.3f}", "#52b788"),
        ("Total cost", _fmt_cost(solution["total_cost"]), _C["accent"]),
        ("MESH", f"{solution['mesh']:.5f}", _C["habitat"]),
        ("IIC", f"{solution['iic']:.7f}", "#1d6a96"),
        ("Diameter (cells)", f"{solution['diameter_cells']}", "#6d6875"),
        ("# conn. comp.", f"{solution['n_connected_components']}", "#8d99ae"),
    ]
    if algo_name:
        metrics.insert(0, ("Algorithm", algo_name, "#111111"))

    y_pos = np.linspace(0.95, 0.04, len(metrics))
    for yi, (label, fmt, color) in zip(y_pos, metrics):
        ax.text(
            0.03,
            yi,
            label,
            va="center",
            fontsize=8,
            color="#444444",
            transform=ax.transAxes,
        )
        ax.text(
            0.97,
            yi,
            fmt,
            va="center",
            ha="right",
            fontsize=8,
            fontweight="bold",
            color=color,
            transform=ax.transAxes,
        )

    for yi in y_pos[:-1]:
        ax.axhline(
            yi - (y_pos[0] - y_pos[1]) * 0.4,
            color="#dddddd",
            lw=0.7,
            xmin=0.01,
            xmax=0.99,
        )
    ax.set_title(f"Solution metrics  {algo_tag}", fontsize=10, fontweight="bold", pad=5)
    ax.set_facecolor("#f8f9fa")

    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    if fig_saved and save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── plot_pareto_front (2-obj) ─────────────────────────────────────────────────


def plot_pareto_front(
    solutions: List[dict],
    objective: ObjectiveType,
    algo_name: Optional[str] = None,
    save_path: Optional[str] = None,
    fig_saved: bool = False,
    figsize: tuple = (9, 6),
    show: bool = False,
):
    """Scatter plot of a 2-objective Pareto front, coloured by total cost."""
    if not objective.is_multi or objective.n_obj != 2:
        print("[plot_pareto_front] Only supported for 2-objective problems.")
        return

    _axes = {
        ObjectiveType.MESH_IIC: ("MESH", "IIC", True, True),
        ObjectiveType.MESH_COST: ("MESH", "Total cost", True, False),
        ObjectiveType.IIC_COST: ("IIC", "Total cost", True, False),
    }
    xl, yl, inv_x, inv_y = _axes.get(objective, ("Obj 1", "Obj 2", False, False))

    def _val(s, label):
        if label == "MESH":
            return s["mesh"]
        if label == "IIC":
            return s["iic"]
        if label == "Total cost":
            return s["total_cost"]
        return 0.0

    xv = [_val(s, xl) for s in solutions]
    yv = [_val(s, yl) for s in solutions]
    costs = [s["total_cost"] for s in solutions]
    n_cells = [s["n_restored_cells"] for s in solutions]

    algo_tag = f" [{algo_name}]" if algo_name else ""
    subtitle = _algo_subtitle(algo_name, objective.value)

    fig, ax = plt.subplots(figsize=figsize, facecolor=_C["bg"])
    n_min, n_max = min(n_cells), max(n_cells)
    n_range = max(n_max - n_min, 1)
    sizes = [20 + 180 * (n - n_min) / n_range for n in n_cells]
    sc = ax.scatter(
        xv,
        yv,
        c=costs,
        cmap="YlOrRd",
        s=sizes,
        edgecolors="#555",
        linewidths=0.4,
        alpha=0.85,
        zorder=3,
    )
    _cfmt, _csuffix = _cost_scale_formatter(costs)
    cb = plt.colorbar(sc, ax=ax, label=f"Total restoration cost{_csuffix}")
    cb.ax.yaxis.label.set_fontsize(10)
    cb.ax.yaxis.set_major_formatter(_cfmt)
    if "cost" in xl.lower():
        ax.xaxis.set_major_formatter(_cfmt)
        xl = f"Total cost{_csuffix}"
    if "cost" in yl.lower():
        ax.yaxis.set_major_formatter(_cfmt)
        yl = f"Total cost{_csuffix}"
    ax.set_xlabel(xl, fontsize=12)
    ax.set_ylabel(yl, fontsize=12)
    ax.set_title(
        f"Pareto front — {objective.value}{algo_tag}\n"
        f"({len(solutions)} non-dominated solutions, "
        f"size ∝ restored cells)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#fafafa")

    # Annotate extremes — alternate quadrants so labels never overlap
    _short = lambda s: s.split("(")[0].strip()   # drop "(× 10^N B$)" from axis label
    _extremes = [
        (np.argmin(xv) if not inv_x else np.argmax(xv), f"Best {_short(xl)}", ( 50,  28)),
        (np.argmax(yv) if inv_y else np.argmin(yv),      f"Best {_short(yl)}", (-60, -36)),
    ]
    _seen = set()
    for idx, lbl, xytext in _extremes:
        pt = (round(xv[idx], 6), round(yv[idx], 6))
        if pt in _seen:
            continue          # skip duplicate when both extremes are the same solution
        _seen.add(pt)
        ax.annotate(
            lbl,
            xy=(xv[idx], yv[idx]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8,
            color=_C["accent"],
            arrowprops=dict(arrowstyle="->", color=_C["accent"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    plt.tight_layout()
    if fig_saved and save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── plot_pareto_front_3d ──────────────────────────────────────────────────────


def plot_pareto_front_3d(
    solutions: List[dict],
    algo_name: Optional[str] = None,
    save_path: Optional[str] = None,
    fig_saved: bool = False,
    figsize: tuple = (10, 7),
    show: bool = False,
):
    """3-D scatter for the FULL (MESH x IIC x cost) Pareto front."""
    mesh_v = [s["mesh"] for s in solutions]
    iic_v = [s["iic"] for s in solutions]
    cost_v = [s["total_cost"] for s in solutions]
    algo_tag = f" [{algo_name}]" if algo_name else ""

    fig = plt.figure(figsize=figsize, facecolor=_C["bg"])
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        mesh_v,
        iic_v,
        cost_v,
        c=cost_v,
        cmap="YlOrRd",
        s=40,
        edgecolors="#444",
        linewidths=0.3,
        alpha=0.8,
    )
    _cfmt3d, _csuffix3d = _cost_scale_formatter(cost_v)
    cb3d = plt.colorbar(sc, ax=ax, label=f"Total cost{_csuffix3d}", shrink=0.6, pad=0.1)
    cb3d.ax.yaxis.set_major_formatter(_cfmt3d)
    ax.set_xlabel("MESH", fontsize=10, labelpad=8)
    ax.set_ylabel("IIC", fontsize=10, labelpad=8)
    ax.set_zlabel(f"Total cost{_csuffix3d}", fontsize=10, labelpad=8)
    ax.zaxis.set_major_formatter(_cfmt3d)
    ax.set_title(
        f"3-D Pareto front — FULL{algo_tag}\n"
        f"({len(solutions)} non-dominated solutions)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if fig_saved and save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── plot_solutions_comparison ─────────────────────────────────────────────────


def plot_solutions_comparison(
    data: HabitatData,
    solutions: List[dict],
    labels: List[str],
    algo_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    fig_saved: bool = False,
    figsize: Optional[tuple] = None,
    bg_alpha: float = 0.30,
    show: bool = False,
):
    """
    Side-by-side selection maps + metric bar charts for N solutions.
    Useful to compare single-objective optima (MESH vs IIC vs COST).
    """
    n = len(solutions)
    fig_w = max(14, 4 * n)
    fig, axes = plt.subplots(2, n, figsize=figsize or (fig_w, 8), facecolor=_C["bg"])
    if n == 1:
        axes = axes.reshape(2, 1)

    sc = mcolors.ListedColormap([_C["other"], _C["restored"]])
    sn = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], sc.N)
    fig.suptitle("Solution comparison", fontsize=13, fontweight="bold", y=1.01)

    for col, (sol, lab) in enumerate(zip(solutions, labels)):
        an = algo_names[col] if algo_names else None
        algo_tag = f"\n[{an}]" if an else ""

        at = axes[0][col]
        _overlay_bg(at, data.bg_image, alpha=bg_alpha)
        at.imshow(
            sol["selection_grid"],
            cmap=sc,
            norm=sn,
            interpolation="nearest",
            alpha=0.85,
            zorder=1,
        )
        at.contour(
            data.habitat_mask.astype(int),
            levels=[0.5],
            colors=[_C["habitat"]],
            linewidths=0.8,
            alpha=0.6,
            zorder=2,
        )
        at.set_title(f"{lab}{algo_tag}", fontsize=10, fontweight="bold")
        at.axis("off")

        # bottom: horizontal bar chart
        ab = axes[1][col]
        ab.axis("off")
        metric_rows = [
            ("MESH", sol["mesh"], "#2d6a4f"),
            ("IIC x1000", sol["iic"] * 1000, "#1d6a96"),
            ("Cost", sol["total_cost"], _C["accent"]),
            ("Restored area", sol["total_restored_area"], "#52b788"),
            ("Diameter", sol["diameter_cells"], "#6d6875"),
        ]
        y_pos = np.linspace(0.85, 0.10, len(metric_rows))
        max_v = max(abs(r[1]) for r in metric_rows) or 1.0
        for yi, (mlabel, mval, mcolor) in zip(y_pos, metric_rows):
            norm_v = abs(mval) / max_v
            ab.barh(
                yi,
                norm_v,
                height=0.10,
                color=mcolor,
                alpha=0.75,
                transform=ab.transAxes,
            )
            ab.text(
                -0.02,
                yi,
                mlabel,
                va="center",
                ha="right",
                fontsize=7.5,
                transform=ab.transAxes,
                color="#333",
            )
            ab.text(
                norm_v + 0.02,
                yi,
                _fmt_cost(mval) if mlabel == "Cost" else f"{mval:.3f}",
                va="center",
                ha="left",
                fontsize=7.5,
                fontweight="bold",
                color=mcolor,
                transform=ab.transAxes,
            )
        ab.set_xlim(0, 1.45)
        if col == 0:
            ab.set_title("Metrics (normalised)", fontsize=9)

    plt.tight_layout()
    if fig_saved and save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)



# ── _knee_point_3d ────────────────────────────────────────────────────────────


def _knee_point_3d(solutions: List[dict]) -> dict:
    """
    Select the balanced ("knee") solution from a 3-objective MESH × IIC × Cost
    Pareto front using the closest-to-ideal heuristic.

    Each objective is normalised to [0, 1] (0 = best, 1 = worst).
    The solution with minimum Euclidean distance to the ideal point (0, 0, 0)
    is returned.  Falls back to the first solution when the front is trivial.
    """
    if len(solutions) == 1:
        return solutions[0]

    mesh = np.array([s["mesh"]       for s in solutions], dtype=float)
    iic  = np.array([s["iic"]        for s in solutions], dtype=float)
    cost = np.array([s["total_cost"] for s in solutions], dtype=float)

    def _norm(v: np.ndarray, hi_is_good: bool) -> np.ndarray:
        lo, hi = v.min(), v.max()
        if hi == lo:
            return np.zeros_like(v)
        n = (v - lo) / (hi - lo)
        return (1.0 - n) if hi_is_good else n

    dist = np.sqrt(
        _norm(mesh, hi_is_good=True)  ** 2 +
        _norm(iic,  hi_is_good=True)  ** 2 +
        _norm(cost, hi_is_good=False) ** 2
    )
    idx = int(np.argmin(dist))
    print(f"  [knee] Selected solution {idx + 1}/{len(solutions)}: "
          f"MESH={mesh[idx]:.3f}  IIC={iic[idx]:.4f}  "
          f"Cost={_fmt_cost(cost[idx])}")
    return solutions[idx]


# ── plot_restoration_map ──────────────────────────────────────────────────────


def plot_restoration_map(
    data: HabitatData,
    solution: dict,
    out_path: str,
    label: str = "",
    algo_name: Optional[str] = None,
    bg_alpha: float = 0.30,
    dpi: int = 150,
    show: bool = False,
) -> None:
    """
    Publication-quality single-panel restoration map for the knee / best
    3-objective Pareto solution.

    Cell colour key
    ---------------
    Dark green  : existing habitat
    Orange      : selected for restoration
    Light amber : restorable but not selected
    Light grey  : non-restorable / other
    """
    nrows, ncols = data.habitat.shape
    sel_grid = solution["selection_grid"]  # 0/1 N×N

    # ── categorical grid: 0=other, 1=habitat, 2=restorable(no sel), 3=selected ─
    cat = np.zeros((nrows, ncols), dtype=np.uint8)
    cat[data.restorable > 0] = 2
    cat[data.habitat == 1]   = 1
    cat[sel_grid == 1]       = 3

    nodata_mask = data.habitat == data.nodata_value

    cmap = mcolors.ListedColormap(["#f0f0f0", "#2d6a4f", "#ffe0a0", "#f4a261"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor="white")

    if data.bg_image is not None:
        _overlay_bg(ax, data.bg_image, alpha=bg_alpha)

    ax.imshow(cat, cmap=cmap, norm=norm, interpolation="nearest",
              origin="upper", alpha=0.85, zorder=2)

    # mask nodata cells with opaque white
    if nodata_mask.any():
        nodata_rgba = np.ones((nrows, ncols, 4), dtype=np.float32)
        nodata_rgba[~nodata_mask, 3] = 0.0
        ax.imshow(nodata_rgba, origin="upper", interpolation="nearest", zorder=3)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor="#2d6a4f", label="Existing habitat"),
        mpatches.Patch(facecolor="#f4a261", label="Selected for restoration"),
        mpatches.Patch(facecolor="#ffe0a0", edgecolor="#bbb",
                       label="Restorable (not selected)"),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="#aaa", label="Other"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=10,
              frameon=True, framealpha=0.9)

    # ── metrics inset ─────────────────────────────────────────────────────────
    mesh_v  = solution.get("mesh", float("nan"))
    iic_v   = solution.get("iic",  float("nan"))
    cost_v  = solution.get("total_cost", float("nan"))
    n_cells = solution.get("n_restored_cells", 0)
    n_cc    = solution.get("n_connected_components", "—")
    diam    = solution.get("diameter_cells", "—")

    metrics_text = (
        f"MESH  : {mesh_v:.3f}\n"
        f"IIC   : {iic_v:.4f}\n"
        f"Cost  : {_fmt_cost(cost_v)}\n"
        f"Cells : {n_cells}  |  CC: {n_cc}  |  Ø: {diam}"
    )
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.88),
            zorder=10)

    # ── title and grid ────────────────────────────────────────────────────────
    algo_tag = f"  [{algo_name}]" if algo_name else ""
    title_lines = [f"Best 3-objective Pareto solution{algo_tag}"]
    if label:
        title_lines.append(label)
    ax.set_title("\n".join(title_lines), fontsize=12, fontweight="bold", pad=8)

    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.3, zorder=1)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    print(f"  Saved \u2192 {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── export_raster_mask ────────────────────────────────────────────────────────


def _save_pareto_selections(solutions: list, npz_path: str) -> None:
    """Save selection_grid for every Pareto solution into a compressed .npz (sel_0…sel_N)."""
    arrays = {}
    for i, s in enumerate(solutions):
        sg = s.get("selection_grid")
        if sg is not None:
            arrays[f"sel_{i}"] = sg.astype(np.uint8)
    if arrays:
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
        print(f"[npz] Pareto selections saved: {npz_path}  ({len(arrays)} solutions)")


def export_raster_mask(
    data: HabitatData,
    solution: dict,
    out_dir: str,
    pfx: str,
    cfg=None,
    suffix: str = "best_pareto3d_mask",
) -> None:
    """
    Export the binary restoration mask of a Pareto solution.

    Always saves  <out_dir>/<pfx>_{suffix}.npy  (uint8, 0/1).
    Also saves a georeferenced GeoTIFF when rasterio is importable.

    Spatial extent priority:
      1. data.georef  — set from the QGIS plugin session JSON (exact grid extent)
      2. cfg.data.elevation_path — bounding box + CRS read from the raster
    Selected cells (value=1) are written; unselected (value=0) are nodata → transparent.
    """
    sel_grid = solution.get("selection_grid")
    if sel_grid is None:
        print("[export_raster_mask] selection_grid missing — skipping.")
        return

    mask = sel_grid.astype(np.uint8)
    os.makedirs(out_dir, exist_ok=True)

    npy_path = f"{out_dir}/{pfx}_{suffix}.npy"
    np.save(npy_path, mask)
    print(f"  Saved \u2192 {npy_path}")

    # ── GeoTIFF (optional) ────────────────────────────────────────────────────
    try:
        import rasterio as _rio
        from rasterio.transform import from_bounds as _from_bounds
        from rasterio.crs import CRS as _CRS
    except ImportError:
        print("  [export_raster_mask] rasterio not available — GeoTIFF skipped.")
        return

    sq_bounds = None
    working_crs = "EPSG:3857"

    # Priority 1: georef from QGIS plugin session JSON (exact planning-grid extent)
    if data.georef is not None:
        ext = data.georef.get("extent", [])
        crs_auth = data.georef.get("crs", "")
        if len(ext) == 4:
            xmin, ymin, xmax, ymax = ext
            span = max(xmax - xmin, ymax - ymin)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            sq_bounds = (cx - span / 2, cy - span / 2,
                         cx + span / 2, cy + span / 2)
            if crs_auth:
                working_crs = crs_auth
            print(f"  [export_raster_mask] georef from session JSON  "
                  f"extent={ext}  crs={working_crs}")

    # Priority 2: derive extent + CRS from elevation raster
    if sq_bounds is None and cfg is not None:
        working_crs = getattr(getattr(cfg, "data", None), "working_crs", working_crs)
        elev_path = getattr(getattr(cfg, "data", None), "elevation_path", "")
        if elev_path and os.path.isfile(elev_path):
            try:
                with _rio.open(elev_path) as src:
                    b = src.bounds
                    span = max(b.right - b.left, b.top - b.bottom)
                    cx = (b.left + b.right) / 2
                    cy = (b.bottom + b.top) / 2
                    sq_bounds = (cx - span / 2, cy - span / 2,
                                 cx + span / 2, cy + span / 2)
                    if src.crs is not None:
                        working_crs = src.crs.to_string()
                    print(f"  [export_raster_mask] sq_bounds from "
                          f"{os.path.basename(elev_path)}  crs={working_crs}")
            except Exception as exc:
                print(f"  [export_raster_mask] Could not read elevation raster: {exc}")

    if sq_bounds is None:
        print("  [export_raster_mask] No spatial extent available — GeoTIFF skipped.")
        return

    nrows, ncols = mask.shape
    west, south, east, north = sq_bounds
    transform = _from_bounds(west, south, east, north, ncols, nrows)

    tif_path = f"{out_dir}/{pfx}_{suffix}.tif"
    with _rio.open(
        tif_path, "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=_rio.uint8,
        crs=_CRS.from_string(working_crs),
        transform=transform,
        nodata=0,
        compress="lzw",
    ) as dst:
        dst.write(mask, 1)
    print(f'  Saved \u2192 {tif_path}')


# ── plot_cost_surface ─────────────────────────────────────────────────────────


def plot_cost_surface(
    data: HabitatData,
    algo_name: Optional[str] = None,
    save_path: Optional[str] = None,
    fig_saved: bool = False,
    figsize: tuple = (14, 5),
    show: bool = False,
):
    """
    3-panel figure illustrating the ecological cost function inputs and output.
    Panel 1 : Elevation DEM (if available) or accessibility
    Panel 2 : Restorable area fraction
    Panel 3 : Computed restoration cost
    """
    algo_tag = f" [{algo_name}]" if algo_name else ""
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=_C["bg"])
    fig.suptitle(
        f"Ecological cost function components{algo_tag}", fontsize=13, fontweight="bold"
    )

    # Panel 1: elevation or accessibility
    ax = axes[0]
    _overlay_bg(ax, data.bg_image, alpha=0.25)
    if data.elevation is not None:
        elev_disp = data.elevation.astype(float).copy()
        elev_disp[data.habitat == 1] = np.nan
        im = ax.imshow(
            elev_disp, cmap="terrain", interpolation="nearest", alpha=0.9, zorder=1
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")
        ax.set_title("Digital elevation model", fontsize=11, fontweight="bold")
    else:
        acc_disp = data.accessible.astype(float).copy()
        im = ax.imshow(
            acc_disp, cmap="RdYlGn", interpolation="nearest", alpha=0.9, zorder=1
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accessible")
        ax.set_title("Accessibility", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Panel 2: restorable area
    ax = axes[1]
    _overlay_bg(ax, data.bg_image, alpha=0.25)
    rest_disp = data.restorable.astype(float).copy()
    rest_disp[data.habitat == 1] = np.nan
    rest_disp[data.habitat == data.nodata_value] = np.nan
    im = ax.imshow(rest_disp, cmap="YlGn", interpolation="nearest", alpha=0.9, zorder=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Restorable area")
    ax.set_title("Restorable area fraction", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Panel 3: cost surface
    ax = axes[2]
    _overlay_bg(ax, data.bg_image, alpha=0.25)
    cost_disp = data.cost.astype(float).copy()
    cost_disp[data.habitat == 1] = np.nan
    cost_disp[data.habitat == data.nodata_value] = np.nan
    im = ax.imshow(
        cost_disp, cmap="YlOrRd", interpolation="nearest", alpha=0.9, zorder=1
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$ / cell")
    ax.set_title(
        "Restoration cost\n(area × accessibility × elevation × tree cost)",
        fontsize=11,
        fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()
    if fig_saved and save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# 7.  HIGH-LEVEL BUILDER API
# =============================================================================


class ForemostProblemBuilder:
    """
    Fluent builder that mirrors the Foremost R-package interface.
    Supports all 5 algorithms via the algo= parameter of .solve().

    Example
    -------
    >>> result = (
    ...     ForemostProblemBuilder(data)
    ...     .set_full_objective()
    ...     .add_restorable_constraint(min_restore=20.0, max_restore=100.0)
    ...     .add_compactness_constraint(max_diameter=9)
    ...     .add_connected_constraint(max_nb_cc=10)
    ...     .add_budget_constraint(max_cost=5_000_000.0)
    ...     .solve(pop_size=100, n_gen=100, algo="CTAEA")
    ... )
    """

    def __init__(self, data: HabitatData, cell_size_m: float = 100.0):
        self._data = data
        self._objective = ObjectiveType.MESH
        self._constraints = RestorationConstraints()
        self._penalty = 1e6
        self._iic_max_dist = 10
        self._cell_size_m = cell_size_m

    # ── objectives ────────────────────────────────────────────────────────────
    def set_max_mesh_objective(self) -> "ForemostProblemBuilder":
        self._objective = ObjectiveType.MESH
        return self

    def set_max_iic_objective(self) -> "ForemostProblemBuilder":
        self._objective = ObjectiveType.IIC
        return self

    def set_min_cost_objective(self) -> "ForemostProblemBuilder":
        self._objective = ObjectiveType.COST
        return self

    def set_mesh_iic_objective(self) -> "ForemostProblemBuilder":
        self._objective = ObjectiveType.MESH_IIC
        return self

    def set_mesh_cost_objective(self) -> "ForemostProblemBuilder":
        self._objective = ObjectiveType.MESH_COST
        return self

    def set_iic_cost_objective(self) -> "ForemostProblemBuilder":
        self._objective = ObjectiveType.IIC_COST
        return self

    def set_full_objective(self) -> "ForemostProblemBuilder":
        self._objective = ObjectiveType.FULL
        return self

    # ── constraints ───────────────────────────────────────────────────────────
    def add_restorable_constraint(
        self,
        min_restore: float = 100.0,
        max_restore: float = 400.0,
        max_diameter: float = 10.0,
        max_nb_cc: int = 10,
        min_proportion: float = 0.0,
        max_cost: float = float("inf"),
    ) -> "ForemostProblemBuilder":

        self._constraints.min_restore = min_restore
        self._constraints.max_restore = max_restore
        self._constraints.max_diameter = max_diameter
        self._constraints.max_nb_cc = max_nb_cc
        self._constraints.min_proportion = min_proportion
        self._constraints.max_cost = max_cost

        return self

    def add_compactness_constraint(
        self, max_diameter: int = 5
    ) -> "ForemostProblemBuilder":
        self._constraints.max_diameter = max_diameter
        return self

    def add_connected_constraint(self, max_nb_cc: int = 1) -> "ForemostProblemBuilder":
        self._constraints.max_nb_cc = max_nb_cc
        return self

    def add_locked_out_constraint(self) -> "ForemostProblemBuilder":
        return self  # embedded in HabitatData.candidate_mask

    def add_min_proportion_constraint(
        self, min_proportion: float
    ) -> "ForemostProblemBuilder":
        self._constraints.min_proportion = min_proportion
        return self

    def add_budget_constraint(self, max_cost: float) -> "ForemostProblemBuilder":
        self._constraints.max_cost = max_cost
        return self

    # ── options ───────────────────────────────────────────────────────────────
    def set_penalty(self, p: float) -> "ForemostProblemBuilder":
        self._penalty = p
        return self

    def set_iic_max_dist(self, d: int) -> "ForemostProblemBuilder":
        self._iic_max_dist = d
        return self

    # ── build / solve ─────────────────────────────────────────────────────────
    def build(self) -> RestorationProblem:
        return RestorationProblem(
            data=self._data,
            constraints=self._constraints,
            objective=self._objective,
            penalty=self._penalty,
            iic_max_dist=self._iic_max_dist,
            cell_size_m=self._cell_size_m,
        )

    def solve(
        self,
        pop_size: int = 100,
        n_gen: int = 100,
        seed: int = 42,
        verbose: bool = True,
        algo: str = "NSGA2",
    ) -> dict:
        return solve(
            self.build(),
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            verbose=verbose,
            algo_name=algo,
        )


# =============================================================================
# 8.  DEMO / HYDRA ENTRY POINT  —  THREE MODES
# =============================================================================

# ── helpers ───────────────────────────────────────────────────────────────────


def _ask_folder(prompt: str = "Select folder") -> str:
    """Open a tkinter folder-picker dialog; return the path or empty string."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title=prompt)
        root.destroy()
        return folder or ""
    except Exception:
        return ""


def _ask_file(prompt: str = "Select file", filetypes: Optional[list] = None) -> str:
    """Open a tkinter file-picker dialog; return the path or empty string."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title=prompt,
            filetypes=filetypes or [("All files", "*.*")],
        )
        root.destroy()
        return path or ""
    except Exception:
        return ""


def load_npy_arrays(folder: str) -> dict:
    """
    Discover and load the four FOREMOST ``.npy`` arrays from *folder*.

    File matching is keyword-based (case-insensitive):
        ``habitat``, ``restorable``, ``accessible``, ``cost`` (or ``cout``).

    Returns a dict with those four keys.  Raises FileNotFoundError / KeyError
    if anything is missing.
    """
    folder_p = Path(folder)
    if not folder_p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    npy_files = sorted(folder_p.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {folder}")

    matched: dict[str, Path] = {}

    def _find(kw: str) -> np.ndarray:
        hits = [f for f in npy_files if kw.lower() in f.name.lower()]
        if not hits:
            if kw == "cost":
                hits = [f for f in npy_files if "cout" in f.name.lower()]
            if not hits:
                raise KeyError(
                    f"No .npy file matching '{kw}' in {folder}\n"
                    f"  Found: {[f.name for f in npy_files]}"
                )
        matched[kw] = hits[0]
        print(f"  [{kw:>12s}]  ← {hits[0].name}")
        return np.load(str(hits[0]))

    arrays = {k: _find(k) for k in ("habitat", "restorable", "accessible", "cost")}
    arrays["_matched_files"] = matched   # carry file metadata for prefix derivation
    return arrays


def _arrays_to_habitatdata(arrays: dict, cell_area: float = 1.0) -> "HabitatData":
    """Convert a dict of numpy arrays to a HabitatData instance."""
    hab = np.asarray(arrays["habitat"]).astype(int)
    rest = np.asarray(arrays["restorable"]).astype(float)
    acc = np.asarray(arrays["accessible"]).astype(int)
    cost = np.asarray(
        arrays.get("cost", arrays.get("cout", np.ones_like(hab, dtype=float)))
    ).astype(float)

    shapes = {hab.shape, rest.shape, acc.shape, cost.shape}
    if len(shapes) > 1:
        raise ValueError(
            f"Shape mismatch: habitat={hab.shape}, restorable={rest.shape}, "
            f"accessible={acc.shape}, cost={cost.shape}"
        )

    locked_out = ((hab == 0) & (rest == 0) & (acc == 0)).astype(bool)

    return HabitatData(
        habitat=hab,
        restorable=rest,
        accessible=acc,
        cost=np.clip(cost, 0, None),
        cell_area=cell_area,
        locked_out=locked_out,
    )


def load_habitatdata_from_npy(npy_dir: str = "outputs/", cell_area: float = 1.0) -> "HabitatData":
    """
    Load a HabitatData from .npy array files exported by satellite_annotator.py.

    This is the mode=1 equivalent for use inside Python scripts that need to build
    a HabitatData without calling run_demo().  All experiment scripts should use
    this function instead of HabitatData.from_config() with synthetic DataConfig.

    Parameters
    ----------
    npy_dir : str
        Folder containing land_use_classify_{habitat,restorable,accessible,cost}_N*.npy
        (default: "outputs/")
    cell_area : float
        Area of one raster cell (default: 1.0)

    Returns
    -------
    HabitatData
    """
    arrays = load_npy_arrays(npy_dir)
    return _arrays_to_habitatdata(arrays, cell_area=cell_area)


def _save_pareto_csv(solutions: list, csv_path: str,
                     hypervolume: float = float("nan"),
                     algo: str = "",
                     knee_solution: Optional[dict] = None,
                     cell_size_m: float = 0.0) -> None:
    """Save a Pareto front to CSV with full per-solution metrics.

    Columns: rank, mesh, iic, cost, cost_per_ha, n_cells, total_area,
             n_cc, diameter_cells, n_patches, is_knee
    cost_per_ha = total_cost / (n_cells × cell_size_m² / 10 000) — resolution-normalized cost.
    First row is a metadata comment (algo, HV).
    """
    import csv as _csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    knee_id = id(knee_solution) if knee_solution is not None else None
    with open(csv_path, "w", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow([f"# algo={algo}  hypervolume={hypervolume:.6f}"
                         f"  n_solutions={len(solutions)}"
                         f"  cell_size_m={cell_size_m:.2f}"])
        writer.writerow([
            "rank", "mesh", "iic", "cost", "cost_per_ha",
            "n_cells", "total_area",
            "n_cc", "diameter_cells", "n_patches",
            "is_knee",
        ])
        for rank, s in enumerate(solutions, start=1):
            n_cells = int(s.get("n_restored_cells", 0))
            cost    = s.get("total_cost", 0.0)
            phys_ha = n_cells * cell_size_m ** 2 / 1e4 if cell_size_m > 0 else 0.0
            cost_per_ha = cost / phys_ha if phys_ha > 0 else float("nan")
            writer.writerow([
                rank,
                f"{s.get('mesh', 0.0):.6f}",
                f"{s.get('iic', 0.0):.8f}",
                f"{cost:.2f}",
                f"{cost_per_ha:.4f}" if cost_per_ha == cost_per_ha else "nan",
                n_cells,
                f"{s.get('total_restored_area', 0.0):.4f}",
                int(s.get("n_connected_components", 0)),
                int(s.get("diameter_cells", 0)),
                int(s.get("n_patches", 0)),
                1 if (knee_id is not None and id(s) == knee_id) else 0,
            ])
    print(f"[csv] Pareto front saved: {csv_path}  ({len(solutions)} solutions)")


def _save_results_summary_csv(
    runs: list,          # list of (objective_label, result_dict)
    csv_path: str,
    cell_size_m: float = 0.0,
) -> None:
    """Save one-row-per-run summary table.

    Columns: objective, algo, n_solutions, hypervolume, pop_feasibility_rate,
             mesh_min, mesh_max, iic_min, iic_max,
             cost_min, cost_max, cost_per_ha_min, cost_per_ha_max,
             knee_mesh, knee_iic, knee_cost, knee_cost_per_ha, knee_n_cells
    cost_per_ha = total_cost / (n_cells × cell_size_m² / 10 000).
    """
    import csv as _csv

    def _cph(cost, n_cells):
        phys_ha = n_cells * cell_size_m ** 2 / 1e4 if cell_size_m > 0 else 0.0
        return cost / phys_ha if phys_ha > 0 else float("nan")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    header = [
        "objective", "algo", "n_solutions", "hypervolume", "pop_feasibility_rate",
        "mesh_min", "mesh_max",
        "iic_min", "iic_max",
        "cost_min", "cost_max",
        "cost_per_ha_min", "cost_per_ha_max",
        "knee_mesh", "knee_iic", "knee_cost", "knee_cost_per_ha", "knee_n_cells",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow(header)
        for obj_label, res in runs:
            sols = res.get("solutions", [])
            if not sols:
                continue
            mesh_vals  = [s.get("mesh",       float("nan")) for s in sols]
            iic_vals   = [s.get("iic",        float("nan")) for s in sols]
            cost_vals  = [s.get("total_cost", float("nan")) for s in sols]
            cph_vals   = [_cph(s.get("total_cost", 0.0), s.get("n_restored_cells", 0))
                          for s in sols]
            # knee: for 3-obj fronts use _knee_point_3d; for 2-obj or 1-obj use first
            if len(mesh_vals) > 1 and not all(v == mesh_vals[0] for v in iic_vals):
                knee = _knee_point_3d(sols)
            else:
                knee = sols[0]
            knee_cph = _cph(knee.get("total_cost", 0.0), knee.get("n_restored_cells", 0))
            writer.writerow([
                obj_label,
                res.get("algo_name", ""),
                len(sols),
                f"{res.get('hypervolume', float('nan')):.6f}",
                f"{res.get('pop_feasibility_rate', float('nan')):.4f}",
                f"{min(mesh_vals):.6f}",  f"{max(mesh_vals):.6f}",
                f"{min(iic_vals):.8f}",   f"{max(iic_vals):.8f}",
                f"{min(cost_vals):.2f}",  f"{max(cost_vals):.2f}",
                f"{min(cph_vals):.4f}",   f"{max(cph_vals):.4f}",
                f"{knee.get('mesh', float('nan')):.6f}",
                f"{knee.get('iic',  float('nan')):.8f}",
                f"{knee.get('total_cost', float('nan')):.2f}",
                f"{knee_cph:.4f}" if knee_cph == knee_cph else "nan",
                int(knee.get("n_restored_cells", 0)),
            ])
    print(f"[csv] Results summary saved: {csv_path}  ({len(runs)} runs)")


def _save_knee_csv(solution: dict, csv_path: str, algo: str = "",
                   objective: str = "FULL", cell_size_m: float = 0.0) -> None:
    """Save the knee (best balanced) 3-D Pareto solution as a single-row CSV."""
    import csv as _csv
    n_cells  = int(solution.get("n_restored_cells", 0))
    cost     = solution.get("total_cost", float("nan"))
    phys_ha  = n_cells * cell_size_m ** 2 / 1e4 if cell_size_m > 0 else 0.0
    cost_per_ha = cost / phys_ha if phys_ha > 0 else float("nan")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fields = [
        ("objective",        objective),
        ("algo",             algo),
        ("mesh",             f"{solution.get('mesh', float('nan')):.6f}"),
        ("iic",              f"{solution.get('iic',  float('nan')):.8f}"),
        ("cost",             f"{cost:.2f}"),
        ("cost_per_ha",      f"{cost_per_ha:.4f}" if cost_per_ha == cost_per_ha else "nan"),
        ("n_cells",          n_cells),
        ("total_area",       f"{solution.get('total_restored_area', 0.0):.4f}"),
        ("n_cc",             int(solution.get("n_connected_components", 0))),
        ("diameter_cells",   int(solution.get("diameter_cells", 0))),
        ("n_patches",        int(solution.get("n_patches", 0))),
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow([k for k, _ in fields])
        writer.writerow([v for _, v in fields])
    print(f"[csv] Knee solution saved: {csv_path}")


def _run_all_objectives(
    data: "HabitatData", cfg: "RestoptConfig", out_dir: str, pfx: str, label: str = ""
) -> tuple:
    """
    Run all 7 restopt objectives against *data* and save figures.

    Returns (r_mesh, r_iic, r_cost, r_mc, r_ic, r_full).
    """
    cc = dict(
        min_restore=cfg.constraints.min_restore,
        max_restore=cfg.constraints.max_restore,
        max_diameter=cfg.constraints.max_diameter,
        max_nb_cc=cfg.constraints.max_nb_cc,
    )
    algo = cfg.optimizer.algo
    ps = cfg.optimizer.pop_size
    ng = cfg.optimizer.n_gen
    sd = cfg.optimizer.seed
    vb = cfg.optimizer.verbose
    show = cfg.output.show
    tag = f"[{label}]" if label else ""

    plot_cost_surface(
        data,
        algo_name=cfg.optimizer.algo,
        save_path=f"{out_dir}/{pfx}_{algo}_0_cost_surface.png",
        fig_saved=cfg.output.fig_saved,
        show=cfg.output.show,
    )

    def _solve(builder, single: bool):
        a = "GA" if single else algo
        return (builder
                .add_restorable_constraint(**cc)
                .add_budget_constraint(max_cost=cfg.constraints.max_cost)
                .solve(pop_size=ps, n_gen=ng, seed=sd, verbose=vb, algo=a))

    cell_size_m = cfg.cost.cell_size_m  # propagate from config to problem

    # ── 1. Maximise MESH ──────────────────────────────────────────────────────

    print(f"\n{tag} --- 1/7  Maximise MESH ---")
    r_mesh = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_max_mesh_objective(), True)
    plot_solution(
        data,
        r_mesh["solutions"][0],
        title=f"Single-objective — Max MESH  {label}",
        algo_name=r_mesh["algo_name"],
        save_path=f"{out_dir}/{pfx}_1_mesh.png",
        fig_saved=cfg.output.fig_saved,
        show=show,
    )

    # ── 2. Maximise IIC ──────────────────────────────────────────────────────
    print(f"\n{tag} --- 2/7  Maximise IIC ---")
    r_iic = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_max_iic_objective(), True)
    plot_solution(
        data,
        r_iic["solutions"][0],
        title=f"Single-objective — Max IIC  {label}",
        algo_name=r_iic["algo_name"],
        save_path=f"{out_dir}/{pfx}_2_iic.png",
        fig_saved=cfg.output.fig_saved,
        show=show,
    )

    # ── Export best-MESH and best-IIC grids + compute Jaccard similarity ──────
    _mesh_grid = r_mesh["solutions"][0].get("selection_grid")
    _iic_grid  = r_iic["solutions"][0].get("selection_grid")
    if _mesh_grid is not None and _iic_grid is not None:
        np.save(f"{out_dir}/{pfx}_best_mesh_grid.npy", _mesh_grid.astype(np.uint8))
        np.save(f"{out_dir}/{pfx}_best_iic_grid.npy",  _iic_grid.astype(np.uint8))
        _mesh_cells = set(zip(*np.where(_mesh_grid == 1)))
        _iic_cells  = set(zip(*np.where(_iic_grid  == 1)))
        _union = len(_mesh_cells | _iic_cells)
        _intersection = len(_mesh_cells & _iic_cells)
        jaccard = _intersection / _union if _union > 0 else 0.0
        print(f"[jaccard] Best-MESH ∩ Best-IIC Jaccard similarity = {jaccard:.4f}  "
              f"(intersection={_intersection}, union={_union})")

    # ── 3. Minimize Cost ──────────────────────────────────────────────────────
    #
    print(f"\n{tag} --- 3/7  Minimise Cost ---")
    r_cost = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_min_cost_objective(), True)
    plot_solution(
        data,
        r_cost["solutions"][0],
        title=f"Single-objective — Min Cost  {label}",
        algo_name=r_cost["algo_name"],
        save_path=f"{out_dir}/{pfx}_3_cost.png",
        fig_saved=cfg.output.fig_saved,
        show=show,
    )

    # print(f"\n--- 2/7  Maximise IIC  [{algo}] ---")
    # r_iic = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_max_iic_objective())
    #          # .add_restorable_constraint(min_restore=min_restore, max_restore=max_restore)
    #          # .add_compactness_constraint(max_diameter=max_diameter)
    #          # .add_connected_constraint(max_nb_cc=max_nb_cc)
    #          # .add_budget_constraint(max_cost=max_cost)
    #          # .solve(pop_size=ps, n_gen=ng, seed=sd, verbose=vb, algo="GA"))
    # plot_solution(data, r_iic["solutions"][0], title="Single-objective — Maximise IIC",algo_name=r_iic["algo_name"],save_path=f"{out_dir}/{pfx}_{algo}_2_iic.png",show=show)
    #
    # # ── 3. Minimise COST ─────────────────────────────────────────────────────
    # print(f"\n--- 3/7  Minimise Cost  [{algo}] ---")
    # r_cost = (ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_min_cost_objective()
    #           .add_restorable_constraint(min_restore=min_restore, max_restore=max_restore)
    #           .add_compactness_constraint(max_diameter=max_diameter)
    #           .add_connected_constraint(max_nb_cc=max_nb_cc)
    #           .add_budget_constraint(max_cost=max_cost)
    #           .solve(pop_size=ps, n_gen=ng, seed=sd, verbose=vb, algo="GA"))
    # plot_solution(data, r_cost["solutions"][0],
    #               title="Single-objective — Minimise Cost",
    #               algo_name=r_cost["algo_name"],
    #               save_path=f"{out_dir}/{pfx}_{algo}_3_cost.png",
    #               show=show)

    # ── 4. Comparison Single-objective ──────────────────────────────────────────────
    plot_solutions_comparison(
        data,
        [r_mesh["solutions"][0], r_iic["solutions"][0], r_cost["solutions"][0]],
        ["Max MESH", "Max IIC", "Min Cost"],
        algo_names=[r_mesh["algo_name"], r_iic["algo_name"], r_cost["algo_name"]],
        save_path=f"{out_dir}/{pfx}_{algo}_4_comparison.png",
        fig_saved=cfg.output.fig_saved,
        show=show,
    )

    # 5. MESH × Cost
    print(f"\n{tag} --- 5/7  Pareto: MESH × Cost  [{algo}] ---")
    r_mc = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_mesh_cost_objective(), False)
    plot_pareto_front(
        r_mc["solutions"],
        ObjectiveType.MESH_COST,
        algo_name=r_mc["algo_name"],
        save_path=f"{out_dir}/{pfx}_5_pareto_mesh_cost.png",
        fig_saved=cfg.output.fig_saved,
        show=show,
    )

    # 6. IIC × Cost
    print(f"\n{tag} --- 6/7  Pareto: IIC × Cost  [{algo}] ---")
    r_ic = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_iic_cost_objective(), False)
    plot_pareto_front(
        r_ic["solutions"],
        ObjectiveType.IIC_COST,
        algo_name=r_ic["algo_name"],
        save_path=f"{out_dir}/{pfx}_6_pareto_iic_cost.png",
        fig_saved=cfg.output.fig_saved,
        show=show,
    )

    # 7. FULL
    print(f"\n{tag} --- 7/7  Pareto FULL: MESH × IIC × Cost  [{algo}] ---")
    r_full = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_full_objective(), False)
    plot_pareto_front_3d(
        r_full["solutions"],
        algo_name=r_full["algo_name"],
        save_path=f"{out_dir}/{pfx}_7_pareto_3d.png",
        fig_saved=cfg.output.fig_saved,
        show=show,
    )

    # ── CSV exports ───────────────────────────────────────────────────────────
    _csm = cfg.cost.cell_size_m   # metres per cell side — needed for cost_per_ha

    # 2-objective Pareto fronts
    _save_pareto_csv(
        r_mc["solutions"], f"{out_dir}/{pfx}_pareto_mesh_cost.csv",
        hypervolume=r_mc.get("hypervolume", float("nan")),
        algo=r_mc["algo_name"], cell_size_m=_csm,
    )
    _save_pareto_csv(
        r_ic["solutions"], f"{out_dir}/{pfx}_pareto_iic_cost.csv",
        hypervolume=r_ic.get("hypervolume", float("nan")),
        algo=r_ic["algo_name"], cell_size_m=_csm,
    )

    # 3-objective FULL Pareto front (with knee flag)
    _best = _knee_point_3d(r_full["solutions"]) if r_full["solutions"] else None
    _save_pareto_csv(
        r_full["solutions"], f"{out_dir}/{pfx}_pareto_full.csv",
        hypervolume=r_full.get("hypervolume", float("nan")),
        algo=r_full["algo_name"],
        knee_solution=_best, cell_size_m=_csm,
    )

    # Knee solution single-row record
    if _best is not None:
        _save_knee_csv(
            _best, f"{out_dir}/{pfx}_knee_solution.csv",
            algo=r_full["algo_name"], objective="FULL", cell_size_m=_csm,
        )

    # Per-run summary (one row per objective)
    _save_results_summary_csv(
        [
            ("MESH",      r_mesh),
            ("IIC",       r_iic),
            ("COST",      r_cost),
            ("MESH_COST", r_mc),
            ("IIC_COST",  r_ic),
            ("FULL",      r_full),
        ],
        f"{out_dir}/{pfx}_summary.csv",
        cell_size_m=_csm,
    )

    # ── Best 3-D Pareto solution: restoration map + raster mask ─────────────
    if _best is not None:
        plot_restoration_map(
            data, _best,
            out_path=f"{out_dir}/{pfx}_best_pareto3d_map.png",
            label=label,
            algo_name=r_full["algo_name"],
            bg_alpha=0.30,
            dpi=cfg.output.dpi,
            show=show,
        )
        export_raster_mask(data, _best, out_dir=out_dir, pfx=pfx, cfg=cfg)

    print(f"\n  Demo complete — outputs saved to {out_dir}/")
    return r_mesh, r_iic, r_cost, r_mc, r_ic, r_full


# ── MODE 0: synthetic landscape from YAML config ──────────────────────────────


def _run_mode0(cfg: "ForemostConfig") -> tuple:
    """
    Mode 0 — Generate a synthetic landscape entirely from the YAML / dataclass
    configuration and run all 7 objectives.

    No files are required.  Every parameter (grid size, habitat fraction,
    seed, cost model, constraints, algorithm) is read from *cfg*.

    Landscape pipeline
    ------------------
    HabitatData.from_config(cfg) →
        • random binary habitat raster  (nrows × ncols, habitat_fraction, seed)
        • derived restorable / accessible / locked_out masks
        • synthetic Gaussian elevation DEM
        • compute_restoration_cost() with cost.* parameters
        • synthetic RGB background image
    """
    print("\n" + "=" * 65)
    print("  FOREMOST  —  MODE 0: Synthetic landscape")
    print(
        f"  Grid : {cfg.data.nrows}×{cfg.data.ncols}  "
        f"Data generation seed={cfg.data.seed_gen}  "
        f"Optimization algorithm={cfg.optimizer.algo}"
        f" Size N={cfg.data.ncols}"
    )
    print("=" * 65)

    data = HabitatData.from_config(cfg)
    cands = data.candidate_mask

    print(f"\n  Habitat cells   : {data.habitat_mask.sum()}")
    print(f"  Candidate cells : {cands.sum()}")
    if cands.any():
        print(
            f"  Cost range ($)  : "
            f"{data.cost[cands].min():.1f} – {data.cost[cands].max():.1f}"
        )
        print(f"  Cost total ($)  : {data.cost[cands].sum():.0f}")

    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)
    pfx = cfg.output.prefix

    results = _run_all_objectives(data, cfg, out_dir, pfx, label="mode=0")
    print(f"\n  Mode 0 complete — outputs saved to {out_dir}/")
    return results


# ── Single-objective dispatcher (used by MODE 1 and MODE 2) ──────────────────


def _run_selected_objective(
    data: "HabitatData",
    cfg:  "RestoptConfig",
    out_dir: str,
    pfx: str,
    label: str = "",
) -> tuple:
    """
    Run only the objective named in ``cfg.optimizer.objective``.

    Returns a 1-tuple containing the result dict of the chosen run,
    so callers can always do ``results[0]`` to get the result.
    Saves exactly the figures / CSVs relevant to that objective.
    """
    cc = dict(
        min_restore=cfg.constraints.min_restore,
        max_restore=cfg.constraints.max_restore,
        max_diameter=cfg.constraints.max_diameter,
        max_nb_cc=cfg.constraints.max_nb_cc,
    )
    algo        = cfg.optimizer.algo
    ps          = cfg.optimizer.pop_size
    ng          = cfg.optimizer.n_gen
    sd          = cfg.optimizer.seed
    vb          = cfg.optimizer.verbose
    show        = cfg.output.show
    cell_size_m = cfg.cost.cell_size_m
    tag         = f"[{label}]" if label else ""
    _csm        = cell_size_m

    obj_str = cfg.optimizer.objective.upper() if cfg.optimizer.objective else "FULL"
    try:
        obj = ObjectiveType(obj_str)
    except ValueError:
        print(f"[optimizer] Unknown objective '{obj_str}' — defaulting to FULL.")
        obj = ObjectiveType.FULL

    def _solve(builder, single: bool):
        a = "GA" if single else algo
        return (builder
                .add_restorable_constraint(**cc)
                .add_budget_constraint(max_cost=cfg.constraints.max_cost)
                .solve(pop_size=ps, n_gen=ng, seed=sd, verbose=vb, algo=a))

    print(f"\n{tag} --- Objective: {obj.value}  |  Algorithm: {algo} ---")

    plot_cost_surface(
        data,
        algo_name=algo,
        save_path=f"{out_dir}/{pfx}_{algo}_0_cost_surface.png",
        fig_saved=cfg.output.fig_saved,
        show=show,
    )

    def _knee(sols):
        return _knee_point_3d(sols) if len(sols) > 1 else sols[0]

    if obj == ObjectiveType.MESH:
        r = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_max_mesh_objective(), True)
        plot_solution(data, r["solutions"][0],
                      title=f"Single-objective — Max MESH  {label}",
                      algo_name=r["algo_name"],
                      save_path=f"{out_dir}/{pfx}_1_mesh.png",
                      fig_saved=cfg.output.fig_saved, show=show)
        _save_pareto_csv(r["solutions"], f"{out_dir}/{pfx}_pareto_mesh.csv",
                         algo=r["algo_name"], cell_size_m=_csm)
        _save_pareto_selections(r["solutions"], f"{out_dir}/{pfx}_pareto_mesh_selections.npz")
        export_raster_mask(data, r["solutions"][0], out_dir=out_dir, pfx=pfx, cfg=cfg,
                           suffix="best_mesh_mask")
        _save_results_summary_csv([("MESH", r)], f"{out_dir}/{pfx}_summary.csv", cell_size_m=_csm)
        return (r,)

    elif obj == ObjectiveType.IIC:
        r = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_max_iic_objective(), True)
        plot_solution(data, r["solutions"][0],
                      title=f"Single-objective — Max IIC  {label}",
                      algo_name=r["algo_name"],
                      save_path=f"{out_dir}/{pfx}_2_iic.png",
                      fig_saved=cfg.output.fig_saved, show=show)
        _save_pareto_csv(r["solutions"], f"{out_dir}/{pfx}_pareto_iic.csv",
                         algo=r["algo_name"], cell_size_m=_csm)
        _save_pareto_selections(r["solutions"], f"{out_dir}/{pfx}_pareto_iic_selections.npz")
        export_raster_mask(data, r["solutions"][0], out_dir=out_dir, pfx=pfx, cfg=cfg,
                           suffix="best_iic_mask")
        _save_results_summary_csv([("IIC", r)], f"{out_dir}/{pfx}_summary.csv", cell_size_m=_csm)
        return (r,)

    elif obj == ObjectiveType.COST:
        r = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_min_cost_objective(), True)
        plot_solution(data, r["solutions"][0],
                      title=f"Single-objective — Min Cost  {label}",
                      algo_name=r["algo_name"],
                      save_path=f"{out_dir}/{pfx}_3_cost.png",
                      fig_saved=cfg.output.fig_saved, show=show)
        _save_pareto_csv(r["solutions"], f"{out_dir}/{pfx}_pareto_cost.csv",
                         algo=r["algo_name"], cell_size_m=_csm)
        _save_pareto_selections(r["solutions"], f"{out_dir}/{pfx}_pareto_cost_selections.npz")
        export_raster_mask(data, r["solutions"][0], out_dir=out_dir, pfx=pfx, cfg=cfg,
                           suffix="best_cost_mask")
        _save_results_summary_csv([("COST", r)], f"{out_dir}/{pfx}_summary.csv", cell_size_m=_csm)
        return (r,)

    elif obj == ObjectiveType.MESH_IIC:
        r = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_mesh_iic_objective(), False)
        plot_pareto_front(r["solutions"], ObjectiveType.MESH_IIC,
                          algo_name=r["algo_name"],
                          save_path=f"{out_dir}/{pfx}_pareto_mesh_iic.png",
                          fig_saved=cfg.output.fig_saved, show=show)
        _knee_mi = _knee(r["solutions"])
        _save_pareto_csv(r["solutions"], f"{out_dir}/{pfx}_pareto_mesh_iic.csv",
                         hypervolume=r.get("hypervolume", float("nan")),
                         algo=r["algo_name"], knee_solution=_knee_mi, cell_size_m=_csm)
        _save_pareto_selections(r["solutions"], f"{out_dir}/{pfx}_pareto_mesh_iic_selections.npz")
        export_raster_mask(data, _knee_mi, out_dir=out_dir, pfx=pfx, cfg=cfg,
                           suffix="best_mesh_iic_mask")
        _save_results_summary_csv([("MESH_IIC", r)], f"{out_dir}/{pfx}_summary.csv", cell_size_m=_csm)
        return (r,)

    elif obj == ObjectiveType.MESH_COST:
        r = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_mesh_cost_objective(), False)
        plot_pareto_front(r["solutions"], ObjectiveType.MESH_COST,
                          algo_name=r["algo_name"],
                          save_path=f"{out_dir}/{pfx}_5_pareto_mesh_cost.png",
                          fig_saved=cfg.output.fig_saved, show=show)
        _knee_mc = _knee(r["solutions"])
        _save_pareto_csv(r["solutions"], f"{out_dir}/{pfx}_pareto_mesh_cost.csv",
                         hypervolume=r.get("hypervolume", float("nan")),
                         algo=r["algo_name"], knee_solution=_knee_mc, cell_size_m=_csm)
        _save_pareto_selections(r["solutions"], f"{out_dir}/{pfx}_pareto_mesh_cost_selections.npz")
        export_raster_mask(data, _knee_mc, out_dir=out_dir, pfx=pfx, cfg=cfg,
                           suffix="best_mesh_cost_mask")
        _save_results_summary_csv([("MESH_COST", r)], f"{out_dir}/{pfx}_summary.csv", cell_size_m=_csm)
        return (r,)

    elif obj == ObjectiveType.IIC_COST:
        r = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_iic_cost_objective(), False)
        plot_pareto_front(r["solutions"], ObjectiveType.IIC_COST,
                          algo_name=r["algo_name"],
                          save_path=f"{out_dir}/{pfx}_6_pareto_iic_cost.png",
                          fig_saved=cfg.output.fig_saved, show=show)
        _knee_ic = _knee(r["solutions"])
        _save_pareto_csv(r["solutions"], f"{out_dir}/{pfx}_pareto_iic_cost.csv",
                         hypervolume=r.get("hypervolume", float("nan")),
                         algo=r["algo_name"], knee_solution=_knee_ic, cell_size_m=_csm)
        _save_pareto_selections(r["solutions"], f"{out_dir}/{pfx}_pareto_iic_cost_selections.npz")
        export_raster_mask(data, _knee_ic, out_dir=out_dir, pfx=pfx, cfg=cfg,
                           suffix="best_iic_cost_mask")
        _save_results_summary_csv([("IIC_COST", r)], f"{out_dir}/{pfx}_summary.csv", cell_size_m=_csm)
        return (r,)

    else:  # FULL (default)
        r = _solve(ForemostProblemBuilder(data, cell_size_m=cell_size_m).set_full_objective(), False)
        plot_pareto_front_3d(r["solutions"],
                             algo_name=r["algo_name"],
                             save_path=f"{out_dir}/{pfx}_7_pareto_3d.png",
                             fig_saved=cfg.output.fig_saved, show=show)
        _best = _knee_point_3d(r["solutions"]) if r["solutions"] else None
        _save_pareto_csv(r["solutions"], f"{out_dir}/{pfx}_pareto_full.csv",
                         hypervolume=r.get("hypervolume", float("nan")),
                         algo=r["algo_name"], knee_solution=_best, cell_size_m=_csm)
        _save_pareto_selections(r["solutions"], f"{out_dir}/{pfx}_pareto_full_selections.npz")
        if _best is not None:
            _save_knee_csv(_best, f"{out_dir}/{pfx}_knee_solution.csv",
                           algo=r["algo_name"], objective="FULL", cell_size_m=_csm)
            plot_restoration_map(data, _best,
                                 out_path=f"{out_dir}/{pfx}_best_pareto3d_map.png",
                                 label=label, algo_name=r["algo_name"],
                                 bg_alpha=0.30, dpi=cfg.output.dpi, show=show)
            export_raster_mask(data, _best, out_dir=out_dir, pfx=pfx, cfg=cfg)
        _save_results_summary_csv([("FULL", r)], f"{out_dir}/{pfx}_summary.csv", cell_size_m=_csm)
        return (r,)


# ── MODE 1: load .npy arrays from folder ──────────────────────────────────────


def _run_mode1(cfg: "ForemostConfig") -> tuple:
    """
    Mode 1 — Load pre-exported ``.npy`` arrays from a folder and run the
    objective selected in ``cfg.optimizer.objective``.

    Array discovery
    ---------------
    Files are matched by keyword (case-insensitive):
        ``habitat``, ``restorable``, ``accessible``, ``cost`` / ``cout``.

    This matches the naming produced by annotator.py on export:
        ``{stem}_habitat_N{N}.npy``
        ``{stem}_restorable_N{N}.npy``
        ``{stem}_accessible_N{N}.npy``
        ``{stem}_cost_N{N}.npy``

    An optional ``elevation`` array (also exported by annotator) is
    loaded when present and attached to ``HabitatData.elevation`` so the cost
    model can use it.

    Folder resolution
    -----------------
    1. ``cfg.data.npy_folder``  (YAML key ``data.npy_folder`` or CLI flag
       ``--npy-folder``).
    2. Tkinter folder-picker dialog when the above is empty.
    3. Abort if the dialog is cancelled.
    """
    print("\n" + "=" * 65)
    print("    FORMEOST  —  MODE 1: Load .npy arrays from folder")
    print(f"  Algorithm : {cfg.optimizer.algo}")
    print("=" * 65)

    # ── Explicit per-array paths (data.*_path) take priority ─────────────────
    _explicit: dict[str, np.ndarray] = {}
    _explicit_dirs: list[str] = []
    for _key, _attr in [("habitat",    "habitat_path"),
                         ("restorable", "restorable_path"),
                         ("accessible", "accessible_path"),
                         ("cost",       "cost_path")]:
        _p = getattr(cfg.data, _attr, "").strip()
        if _p and os.path.isfile(_p):
            _explicit[_key] = np.load(_p)
            _explicit_dirs.append(os.path.dirname(_p))
            print(f"  [{_key:>12s}]  ← {os.path.basename(_p)}  (explicit path)")

    # ── Folder-based discovery for any array not given explicitly ─────────────
    _missing = [k for k in ("habitat", "restorable", "accessible", "cost")
                if k not in _explicit]

    folder = (cfg.data.npy_folder or "").strip()
    if _missing:
        if not folder:
            print("\n  No npy_folder configured — opening folder picker …")
            folder = _ask_folder("Select folder containing .npy arrays")
        if not folder:
            print("  No folder selected.  Aborting.")
            return ()
        print(f"\n  Loading arrays from: {folder}")
        _folder_arrays = load_npy_arrays(folder)
        for _k in _missing:
            _explicit[_k] = _folder_arrays[_k]
    arrays = _explicit

    # ── Elevation: explicit path first, then scan folder ─────────────────────
    # TIFs are deferred — they require georef for proper resampling to N×N grid.
    elev_arr = None
    _elev_tif_path = ""  # deferred TIF path (resolved after georef is loaded)
    _elev_path = getattr(cfg.data, "elevation_path", "").strip()
    if _elev_path and os.path.isfile(_elev_path):
        _ext = os.path.splitext(_elev_path)[1].lower()
        if _ext in (".tif", ".tiff"):
            _elev_tif_path = _elev_path   # defer until georef is known
            print(f"  [   elevation]  GeoTIFF queued for resampling: {os.path.basename(_elev_path)}")
        else:
            elev_arr = np.load(_elev_path).astype(np.float64)
            print(f"  [   elevation]  ← {os.path.basename(_elev_path)}  (explicit path)")
    elif folder:
        for f in sorted(Path(folder).glob("*.npy")):
            if "elevation" in f.name.lower():
                elev_arr = np.load(str(f)).astype(np.float64)
                print(f"  [   elevation]  ← {f.name}")
                break

    for k, v in arrays.items():
        if not isinstance(v, np.ndarray):
            continue
        print(
            f"    {k:>12s} : shape={v.shape}  dtype={v.dtype}  "
            f"min={float(v.min()):.3g}  max={float(v.max()):.3g}"
        )

    # Load session JSON for exact georef (exported by QGIS plugin alongside .npy)
    import json as _json
    georef = None
    _search_dirs = ([Path(folder)] if folder else []) + [Path(d) for d in _explicit_dirs]
    _seen_dirs: set[str] = set()
    for _sf in (f for d in _search_dirs
                if str(d) not in _seen_dirs and not _seen_dirs.add(str(d))
                for f in sorted(d.glob("*session*N*.json"))):
        try:
            _gref = _json.loads(_sf.read_text(encoding="utf-8")).get("georef")
            if _gref and len(_gref.get("extent", [])) == 4:
                georef = _gref
                print(f"  [georef] {_sf.name}  →  extent={_gref['extent']}  crs={_gref.get('crs','?')}")
                break
        except Exception as _e:
            print(f"  [georef] Could not read {_sf.name}: {_e}")
    if georef is None:
        print("  [georef] No session JSON found — GeoTIFF will fall back to elevation raster bounds.")

    data = _arrays_to_habitatdata(arrays, cell_area=cfg.data.cell_area)
    if georef is not None:
        data.georef = georef
    # ── Attach elevation (npy: direct; TIF: reproject to grid) ──────────────
    if _elev_tif_path:
        try:
            import rasterio as _rio
            import rasterio.warp as _warp
            from rasterio.enums import Resampling as _Resamp
            from rasterio.transform import from_bounds as _from_bounds
            _nrows, _ncols = data.shape
            with _rio.open(_elev_tif_path) as _src:
                if georef and len(georef.get("extent", [])) == 4:
                    _minx, _miny, _maxx, _maxy = georef["extent"]
                    _grid_crs = georef.get("crs", "EPSG:3857")
                    _dst_transform = _from_bounds(_minx, _miny, _maxx, _maxy, _ncols, _nrows)
                    _elev_grid = np.zeros((_nrows, _ncols), dtype=np.float64)
                    _warp.reproject(
                        source=_rio.band(_src, 1),
                        destination=_elev_grid,
                        src_transform=_src.transform,
                        src_crs=_src.crs,
                        dst_transform=_dst_transform,
                        dst_crs=_grid_crs,
                        resampling=_Resamp.average,
                        src_nodata=float("nan"),
                        dst_nodata=0.0,
                    )
                    np.nan_to_num(_elev_grid, copy=False)
                else:
                    # No georef: block-average the full raster down to N×N
                    _raw = _src.read(1, masked=True).filled(0.0).astype(np.float64)
                    from scipy.ndimage import zoom as _zoom
                    _elev_grid = _zoom(_raw, (_nrows / _raw.shape[0], _ncols / _raw.shape[1]))
            data.elevation = _elev_grid
            print(
                f"  Elevation TIF resampled → {_nrows}×{_ncols}  "
                f"min={_elev_grid.min():.1f}m  max={_elev_grid.max():.1f}m"
            )
        except Exception as _e:
            print(f"  [elevation TIF] reproject failed: {_e} — skipped")

    elif elev_arr is not None:
        try:
            if elev_arr.shape == data.shape:
                data.elevation = elev_arr
                print(f"  Elevation array attached  shape={elev_arr.shape}")
            else:
                print(
                    f"  [elevation] Shape {elev_arr.shape} ≠ grid {data.shape} "
                    f"— skipped"
                )
        except Exception as exc:
            print(f"  [elevation] Could not attach: {exc}")

    cands = data.candidate_mask
    print(f"\n  Habitat cells   : {data.habitat_mask.sum()}")
    print(f"  Candidate cells : {cands.sum()}")
    if cands.any():
        print(
            f"  Cost range ($)  : "
            f"{data.cost[cands].min():.1f} – {data.cost[cands].max():.1f}"
        )

    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)

    # Derive prefix and label from the cost npy filename stem
    matched = arrays.get("_matched_files", {})
    if "cost" in matched:
        pfx = matched["cost"].stem   # e.g. LANDUSE_corrected_APP_rivers_2_cost_N100
        # Extract "N100" grid-size tag and base dataset name for figure titles
        _m = re.search(r"_N(\d+)$", pfx)
        if _m:
            _n_tag   = f"N={_m.group(1)}"
            _base    = re.sub(r"_cost_N\d+$|_N\d+$", "", pfx)
            label    = f"{_base}  ·  {_n_tag}"
        else:
            label = pfx
    else:
        pfx   = cfg.output.prefix
        label = "mode=1"

    # Support running multiple objectives in sequence.
    # cfg.optimizer.objectives (list) takes priority over cfg.optimizer.objective.
    objectives = list(getattr(cfg.optimizer, "objectives", None) or [])
    if not objectives:
        objectives = [cfg.optimizer.objective or "FULL"]

    all_results = []
    for obj_str in objectives:
        cfg.optimizer.objective = obj_str.upper()
        print(f"\n{'='*60}\n  Objective: {cfg.optimizer.objective}\n{'='*60}")
        all_results.append(
            _run_selected_objective(data, cfg, out_dir, pfx, label=label)
        )

    print(f"\n  Mode 1 complete ({len(objectives)} objective(s)) — outputs saved to {out_dir}/")
    results = tuple(r for group in all_results for r in group)
    return results


# ── MODE 2: launch annotator GUI, then optimise ─────────────────────


def _run_mode2(cfg: "ForemostConfig") -> tuple:
    """
    Mode 2 — Launch the :mod:`annotator` interactive GUI, collect
    the annotated N×N arrays (including the elevation array when available),
    and run all 7 foremost objectives.

    Workflow
    --------
    1. Import ``annotation`` from the same directory as this file
       (falling back to ``sys.path``).
    2. Determine the source file:
         - ``cfg.data.image_path`` → raster image
         - ``cfg.data.gpkg_path``  → GeoPackage layer
         - Otherwise a file-picker dialog is opened.
    3. Call ``annotate()`` or ``annotate_gpkg()`` — both block until
       the user clicks **Export NumPy Arrays** in the GUI.
    4. The returned dict contains at minimum:
         ``habitat``, ``restorable``, ``accessible``, ``cost``
       and optionally ``elevation`` (N×N float64, values in [0, 1]).
    5. All 7 objectives are run with the optimizer settings from *cfg*.

    Returns
    -------
    tuple of 6 result dicts, or empty tuple () if the user cancels.
    """
    print("\n" + "=" * 65)
    print("  FOREMOST —  MODE 2: Annotator GUI")
    print(f"  Algorithm : {cfg.optimizer.algo}")
    print("=" * 65)

    # ── 1. Import annotator ─────────────────────────────────────────
    try:
        _sa_dir = Path(__file__).resolve().parent
        _sa_file = _sa_dir / "annotation.py"
        if _sa_file.exists():
            import importlib.util as _ilu

            _spec = _ilu.spec_from_file_location("annotation", str(_sa_file))
            sa = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(sa)
        else:
            import annotation as sa  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Cannot import annotation.py.\n"
            f"  Expected: {Path(__file__).resolve().parent / 'annotation.py'}\n"
            f"  Detail  : {exc}"
        )

    # ── 2. Source file ────────────────────────────────────────────────────────
    gpkg_path = (cfg.data.gpkg_path or "").strip()

    image_path = (cfg.data.image_path or "").strip()

    if not image_path and not gpkg_path:
        print("\n  No source file in config — opening file picker …")
        chosen = _ask_file(
            "Select an image or GPKG for annotation",
            filetypes=[
                (
                    "Images & vectors",
                    "*.tif *.tiff *.png *.jpg *.jpeg *.gpkg *.shp *.geojson",
                ),
                ("GeoTIFF", "*.tif *.tiff"),
                ("Images", "*.png *.jpg *.jpeg"),
                ("GeoPackage", "*.gpkg"),
                ("All", "*.*"),
            ],
        )
        if not chosen:
            print("  No file selected.  Aborting.")
            return ()
        ext = Path(chosen).suffix.lower()
        if ext in (".gpkg", ".shp", ".geojson", ".json"):
            gpkg_path = chosen
        else:
            image_path = chosen

    N = cfg.data.nrows

    # ── 3. Launch GUI ─────────────────────────────────────────────────────────
    if gpkg_path:
        print(f"\n  Annotating GPKG: {Path(gpkg_path).name}  (N={N}×{N}) …")
        arrays = sa.annotate_gpkg(gpkg_path=gpkg_path, N=N, ask_N=False)
        # annotate_gpkg returns a GeoDataFrame; extract the arrays from the app
        # by calling annotate-style instead
        if arrays is None:
            raise RuntimeError("Annotator GUI closed without exporting arrays.")

        # Build arrays dict from GeoDataFrame columns
        import numpy as _np

        N2 = N * N

        def _col(col, default):
            if hasattr(arrays, "columns") and col in arrays.columns:
                return _np.asarray(arrays[col].values[:N2], dtype=float)
            return _np.full(N2, default, dtype=float)

        arrays = {
            "habitat": (_col("class", 0) == 1).astype(int).reshape(N, N),
            "restorable": _col("restorable", 0).astype(int).reshape(N, N),
            "accessible": _col("accessible", 0).astype(int).reshape(N, N),
            "cost": _col("cost", 0.0).reshape(N, N),
        }
    else:
        print(f"\n  Annotating image: {Path(image_path).name}  (N={N}×{N}) …")
        arrays = sa.annotate(image_path=image_path, N=N, ask_N=False)
        if not arrays:
            raise RuntimeError("Annotator GUI closed without exporting arrays.")

    # ── 4. Persist arrays to disk ─────────────────────────────────────────────
    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)
    pfx = cfg.output.prefix
    N_str = f"N{N}"

    # print("\n  Saving arrays from Annotator:")
    # for key in ("habitat", "restorable", "accessible", "cost"):
    #     arr = np.asarray(arrays[key])
    #     path = os.path.join(out_dir, f"{pfx}_{key}_{N_str}.npy")
    #     np.save(path, arr)
    #     print(f"  Saved  {key:>12s}  →  {path}")
    #
    # elev = arrays.get("elevation")
    # if elev is not None:
    #     arr = np.asarray(elev, dtype=np.float64)
    #     path = os.path.join(out_dir, f"{pfx}_elevation_{N_str}.npy")
    #     np.save(path, arr)
    #     print(f"  Saved  {'elevation':>12s}  →  {path}")

    # ── 5. Delegate to mode 1 ─────────────────────────────────────────────────
    print(f"\n  Delegating to mode 1 — loading from: {out_dir}")
    cfg.data.npy_folder = out_dir
    return _run_mode1(cfg)


# ── public dispatcher ─────────────────────────────────────────────────────────


def run_demo(
    mode: int = 2,
    cfg: Optional["ForemostConfig"] = None,
) -> tuple:
    """
    Run the foremost pipeline in one of three modes.

    Parameters
    ----------
    mode : int
        **0** — Synthetic landscape from YAML / dataclass config (no files).
        **1** — Load pre-exported ``.npy`` arrays from a folder on disk.
        **2** — Launch the annotator GUI and use the returned arrays.

        All three modes run the same 7 objectives and save the same figures.

    cfg : ForemostConfig or None
        Configuration object.  If None, defaults are used.

    Returns
    -------
    tuple of 6 result dicts (r_mesh, r_iic, r_cost, r_mc, r_ic, r_full),
    or ``()`` if the user cancels in modes 1 / 2.

    Examples
    --------
    >>> run_demo(0)                            # synthetic, no files
    >>> run_demo(1)                            # folder picker dialog
    >>> cfg = ForemostConfig(); cfg.data.npy_folder = "outputs/"
    >>> run_demo(1, cfg)                       # configured folder
    >>> run_demo(2)                            # file picker + GUI
    >>> cfg.data.image_path = "zone.tif"
    >>> run_demo(2, cfg)                       # pre-configured image
    """
    if cfg is None:
        cfg = ForemostConfig()
    cfg.data.mode = mode  # keep cfg consistent with the argument

    if mode == 0:
        return _run_mode0(cfg)
    elif mode == 1:
        return _run_mode1(cfg)
    elif mode == 2:
        return _run_mode2(cfg)
    else:
        raise ValueError(
            f"Unknown mode={mode}.  Use 0 (synthetic), 1 (npy folder), "
            "2 (Annotator GUI)."
        )


# ── Mode picker GUI ───────────────────────────────────────────────────────────


def _pick_mode_gui() -> int:
    """Show a small tkinter dialog to pick the run mode. Returns 0, 1, or 2."""
    import tkinter as tk

    chosen = [2]  # mutable container so _ok can write to it

    root = tk.Tk()
    root.title("FOREMOST — Select Mode")
    root.resizable(False, False)

    tk.Label(
        root, text="Select run mode:", font=("Helvetica", 12, "bold"), pady=10
    ).pack()

    _MODES = [
        (
            0,
            "Mode 0 — Synthetic landscape",
            "Generate a landscape from YAML / default config (no files needed)",
        ),
        (
            1,
            "Mode 1 — Load .npy arrays",
            "Load pre-exported NumPy arrays from a folder on disk",
        ),
        (
            2,
            "Mode 2 — Annotator GUI",
            "Launch the interactive annotation tool on a raster or GPKG file",
        ),
    ]

    var = tk.IntVar(value=2)
    for val, label, desc in _MODES:
        frame = tk.Frame(root)
        frame.pack(anchor="w", padx=20, pady=4)
        tk.Radiobutton(
            frame, text=label, variable=var, value=val, font=("Helvetica", 10)
        ).pack(anchor="w")
        tk.Label(frame, text=desc, fg="gray", font=("Helvetica", 9), padx=24).pack(
            anchor="w"
        )

    def _ok():
        chosen[0] = var.get()
        root.destroy()

    tk.Button(
        root,
        text="Run",
        command=_ok,
        width=12,
        bg="#2e7d32",
        fg="black",
        font=("Helvetica", 10),
    ).pack(pady=12)

    root.mainloop()

    return chosen[0]


# ── Hydra entry point ─────────────────────────────────────────────────────────

if _HAS_HYDRA:

    @hydra.main(config_path="conf", config_name="foremost", version_base=None)
    def _hydra_main(hydra_cfg: DictConfig) -> None:
        """Hydra-decorated main — called when hydra-core is installed."""
        cfg = _cfg_to_dataclass(hydra_cfg)
        run_demo(mode=cfg.mode, cfg=cfg)


def main():
    """
    CLI entry point.

    ::

        python foremost.py                         # mode 0 (synthetic)
        python foremost.py --mode 1                # load .npy folder
        python foremost.py --mode 1 --npy-folder (to be selected manually with the folfer picker ; the flder should
                                                    contain npy narrays )
        python foremost.py --mode 2                # annotator GUI
        python foremost.py --mode 2 --image zone.tif
        python foremost.py --mode 2 --gpkg zones.gpkg
        python foremost.py --write-config
        python foremost.py --algo CTAEA --mode 0
        # Hydra overrides:
        python foremost.py mode=1 data.npy_folder=outputs/
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="FOREMOST — Ecological Restoration Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
        Modes
        -----
          0  Synthetic landscape from YAML / default config  (default)
          1  Load .npy arrays from a folder on disk
          2  Launch annotator GUI then optimize
        """
        ),
    )
    parser.add_argument(
        "--write-config", action="store_true", help="Write conf/foremost.yaml and exit"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Run mode (0/1/2); omit to show a GUI picker",
    )
    parser.add_argument(
        "--npy-folder",
        type=str,
        default=None,
        metavar="DIR",
        help="[mode 1] Folder containing .npy arrays",
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None, help="[mode 2] image path"
    )
    parser.add_argument(
        "--gpkg", "-g", type=str, default=None, help="[mode 2] GeoPackage path"
    )
    parser.add_argument("--algo", type=str, default=None, choices=list(SUPPORTED_ALGOS))
    parser.add_argument(
        "--objective", type=str, default=None, choices=[o.value for o in ObjectiveType]
    )
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--n-gen", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    # Pass remaining args to Hydra if installed
    args, remaining = parser.parse_known_args()

    if args.write_config:
        write_default_yaml("conf")
        return

    if _HAS_HYDRA and (remaining or Path("conf/foremost.yaml").exists()):
        sys.argv = [sys.argv[0]] + remaining
        _hydra_main()
        return

    cfg = ForemostConfig()
    if args.algo:
        cfg.optimizer.algo = args.algo
    if args.objective:
        cfg.optimizer.objective = args.objective
    if args.pop_size:
        cfg.optimizer.pop_size = args.pop_size
    if args.n_gen:
        cfg.optimizer.n_gen = args.n_gen
    if args.seed:
        cfg.optimizer.seed = args.seed
    if args.out_dir:
        cfg.output.dir = args.out_dir
    if args.npy_folder:
        cfg.data.npy_folder = args.npy_folder
    if args.image:
        cfg.data.image_path = args.image
    if args.gpkg:
        cfg.data.gpkg_path = args.gpkg

    # mode = args.mode if args.mode is not None else
    mode = 2
    cfg.data.mode = mode


if __name__ == "__main__":
    main()
