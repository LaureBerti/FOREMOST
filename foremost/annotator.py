"""
Annotator.py
======================
Interactive annotation tool for FOREMOST ecological restoration planning.
Supports raster images (PNG, JPEG, GeoTIFF) and vector files (GPKG / GeoJSON).
 

Features
--------
• Hydra / YAML configuration  (conf/annotator.yaml)
• N×N grid annotation — identical for raster images and GPKG files
• Zoom + / − · Arrow-key pan · Right-click drag pan
• Multi-cell paint (hold left-click and drag)
• Inline cost entry on Restorable Accessible cells
• Default restoration cost for CLASS_RA cells with no cost set
• Overlay layers loaded from local files (GPKG / SHP / GeoTIFF):
    Elevation  (DEM or GPKG with DN field)
    Roads      (GPKG / SHP — also used for accessibility matrix)
    Cadastral  (GPKG / SHP)
    Hydrology  (GPKG / SHP)
• Layer control panel: toggle · opacity · Load · Change (↺)
• All layers reprojected to a common working CRS with CRS trace in console
• Road-based accessibility matrix  (networkx graph from roads GPKG)
• GPKG/OSM-XML layer extraction  (extract_layers_from_osm)
• Auto-annotation: label cells as Habitat from image pixel brightness
• Session save / load (JSON)
• Export: NumPy arrays + annotated PNG image + annotated GPKG (if applicable)
• Road-based accessibility matrix: compute_accessibility_from_roads()
• Hydra / YAML configuration for all parameters  (conf/annotator.yaml)
• Elevation data layer overlay  (DEM raster or SRTM via elevation package)
• Road network layer  (OSM via osmnx or local shapefile)
• Cadastral data layer  (local GPKG/SHP, or French IGN open data)
• Hydrology network layer  (OSM waterways or local shapefile)
• Layer control panel: toggle visibility + transparency per layer
• Default restoration cost filled automatically into CLASS_RA cells with no cost
 
All 7 layers (background + 4 overlays + annotation + grid) are spatially aligned
when the source has georeference information (GeoTIFF via rasterio, or GPKG).
 
Keyboard shortcuts
------------------
  h / r / n   Select annotation class
  Arrow keys   Pan inside zoomed canvas
  = / +        Zoom in  ×1.5   |   -  Zoom out   |   0  Reset zoom
  Scroll       Zoom centred on cursor
  Right-drag   Pan
  Del          Clear selected cell
  Ctrl+Z       Undo   |   Ctrl+S  Export
 
Usage
-----
  python annotation.py --write-config    # write conf/annotator.yaml
  python annotation.py                   # file picker (uses YAML if present)
  python annotation.py --image zone.tif
  python annotation.py --gpkg zones.gpkg --N 30
  python annotation.py layers.roads.enabled=true   # Hydra override
 
Dependencies (core)
-------------------
  pip install pillow numpy

 
Optional extras
---------------
  pip install rasterio                     # GeoTIFF multi-band, DEM loading
  pip install geopandas matplotlib shapely # GPKG annotation, vector layers
  pip install networkx scipy               # road graph, accessibility matrix
  pip install hydra-core omegaconf         # YAML config system
  pip install pyproj                       # CRS transformation
"""
 

 
import argparse
import json
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import geopandas as gpd
from shapely.geometry import Polygon, mapping

import numpy as np
 
# ── PIL ────────────────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    print("ERROR: Pillow is required.  pip install pillow")
    sys.exit(1)
 
# ── tkinter ────────────────────────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except ImportError:
    print("ERROR: tkinter is required (included in standard Python).")
    sys.exit(1)
 
# ── Hydra / OmegaConf (optional) ─────────────────────────────────────────────
try:
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf
    _HAS_HYDRA = True
except ImportError:
    _HAS_HYDRA = False
    DictConfig = dict
 
 
# =============================================================================
# 0.  HYDRA CONFIGURATION DATACLASSES
# =============================================================================
 
@dataclass
class LayerItemConfig:
    """Configuration for a single overlay layer."""
    enabled:   bool  = False
    path:      str   = ""         # local file path; empty = fetch from OSM (roads/hydrology)
    alpha:     float = 0.45
    color:     str   = "#ffffff"  # line/edge colour for vector layers
    facecolor: str   = ""         # fill colour for polygon layers (empty = transparent)
    linewidth: float = 1.2
    colormap:  str   = "terrain"  # for raster layers (elevation)
 
@dataclass
class LayersConfig:
    elevation: LayerItemConfig = field(
        default_factory=lambda: LayerItemConfig(
            colormap="terrain", alpha=0.35, color="#888888"))
    roads: LayerItemConfig = field(
        default_factory=lambda: LayerItemConfig(
            color="#e63946", alpha=0.65, linewidth=1.6))
    cadastral: LayerItemConfig = field(
        default_factory=lambda: LayerItemConfig(
            color="#f4a261", facecolor="#f4a26118", alpha=0.50, linewidth=0.8))
    hydrology: LayerItemConfig = field(
        default_factory=lambda: LayerItemConfig(
            color="#4cc9f0", alpha=0.65, linewidth=1.4))
 
@dataclass
class DataConfig:
    image_path: str = ""
    gpkg_path:  str = ""
    layer:      str = ""
    N:          int = 30
 
@dataclass
class CostConfig:
    default_cost:  float = 100.0    # $ applied to CLASS_RA cells with no cost
    auto_fill:     bool  = True     # auto-fill on export if True
 
@dataclass
class UIConfig:
    canvas_size:        int   = 800
    max_zoom:           float = 16.0
    pan_key_step:       int   = 40
    auto_hab_threshold: float = 0.95
    auto_label_threshold: int = 50
    panel_width:        int   = 310
 
@dataclass
class OutputConfig:
    folder: str = ""
    prefix: str = "annotation"
    dpi:    int = 150
 
@dataclass
class AnnotatorConfig:
    data:    DataConfig    = field(default_factory=DataConfig)
    layers:  LayersConfig  = field(default_factory=LayersConfig)
    cost:    CostConfig    = field(default_factory=CostConfig)
    ui:      UIConfig      = field(default_factory=UIConfig)
    output:  OutputConfig  = field(default_factory=OutputConfig)
 
 
def write_default_yaml(config_dir: str = "conf") -> Path:
    """Write conf/annotator.yaml with default settings.  Call once to bootstrap."""
    cfg = AnnotatorConfig()
    yaml_text = textwrap.dedent(f"""\
    # annotation.py — Hydra configuration
    # Override from CLI: python annotation.py layers.roads.enabled=true
    # ──────────────────────────────────────────────────────────────────────────
 
    data:
      image_path: ""       # path to PNG/JPG/GeoTIFF, or leave empty for file picker
      gpkg_path:  ""       # path to GeoPackage
      layer:      ""       # GPKG layer name (empty = first layer)
      N:          {cfg.data.N}             # grid size N×N
 
    layers:
      elevation:
        enabled:   false
        path:      ""      # path to DEM GeoTIFF (empty = try SRTM via elevation pkg)
        alpha:     {cfg.layers.elevation.alpha}
        colormap:  "{cfg.layers.elevation.colormap}"
      roads:
        enabled:   false
        path:      "" # {cfg.layers.roads.path}  path to shapefile/GPKG (empty = fetch from OSM)
        alpha:     {cfg.layers.roads.alpha}
        color:     "{cfg.layers.roads.color}"
        linewidth: {cfg.layers.roads.linewidth}
      cadastral:
        enabled:   false
        path:      ""      # path to cadastral shapefile/GPKG
        alpha:     {cfg.layers.cadastral.alpha}
        color:     "{cfg.layers.cadastral.color}"
        facecolor: "{cfg.layers.cadastral.facecolor}"
        linewidth: {cfg.layers.cadastral.linewidth}
      hydrology:
        enabled:   false
        path:      ""      # path to shapefile/GPKG (empty = fetch waterways from OSM)
        alpha:     {cfg.layers.hydrology.alpha}
        color:     "{cfg.layers.hydrology.color}"
        linewidth: {cfg.layers.hydrology.linewidth}
 
    cost:
      default_cost: {cfg.cost.default_cost}    # $ per cell applied to CLASS_RA cells with no cost
      auto_fill:    {str(cfg.cost.auto_fill).lower()}   # fill missing costs automatically on export
 
    ui:
      canvas_size:        {cfg.ui.canvas_size}
      max_zoom:           {cfg.ui.max_zoom}
      pan_key_step:       {cfg.ui.pan_key_step}
      auto_hab_threshold: {cfg.ui.auto_hab_threshold}
      auto_label_threshold: {cfg.ui.auto_label_threshold}
      panel_width:        {cfg.ui.panel_width}
 
    output:
      folder: ""
      prefix: "{cfg.output.prefix}"
      dpi:    {cfg.output.dpi}
    """)
    p = Path(config_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / "annotator.yaml"
    out.write_text(yaml_text)
    print(f"Default config written to {out}")
    return out
 
 
def _cfg_from_dict(d) -> AnnotatorConfig:
    """Build AnnotatorConfig from a dict-like object (OmegaConf or plain dict)."""
    def _get(obj, key, default):
        try:
            return obj[key]
        except (KeyError, TypeError):
            return getattr(obj, key, default)
 
    def _sub(obj, key, cls):
        raw = _get(obj, key, None)
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        try:
            items = dict(raw) if hasattr(raw, "items") else {k: getattr(raw, k) for k in vars(cls()).keys()}
            return cls(**{k: v for k, v in items.items() if k in vars(cls()).keys()})
        except Exception:
            return cls()
 
    layers_raw = _get(d, "layers", None)
    layers_cfg = LayersConfig()
    if layers_raw is not None:
        for lname in ("elevation", "roads", "cadastral", "hydrology"):
            raw_l = _get(layers_raw, lname, None)
            if raw_l is not None:
                kw = {}
                for f in vars(LayerItemConfig()).keys():
                    v = _get(raw_l, f, None)
                    if v is not None:
                        kw[f] = v
                setattr(layers_cfg, lname, LayerItemConfig(**kw))
 
    return AnnotatorConfig(
        data    = _sub(d, "data",   DataConfig),
        layers  = layers_cfg,
        cost    = _sub(d, "cost",   CostConfig),
        ui      = _sub(d, "ui",     UIConfig),
        output  = _sub(d, "output", OutputConfig),
    )
 
 
# =============================================================================
# 1.  CONSTANTS
# =============================================================================
 
CLASS_NONE = 0
CLASS_HAB  = 1
CLASS_RA   = 2
CLASS_NR   = 3
N_CLASSES  = 3
 
CLASS_META = {
    CLASS_NONE: dict(label="Not annotated",           rgb=(220, 220, 220), alpha=0,   key="—"),
    CLASS_HAB:  dict(label="Habitat",               rgb=( 45, 106,  79), alpha=140, key="h"),
    CLASS_RA:   dict(label="Restorable", rgb=(244, 162,  97), alpha=140, key="r"),
    CLASS_NR:   dict(label="Non-Restorable",        rgb=(141, 153, 174), alpha=140, key="n"),
}
 
LAYER_NAMES  = ("elevation", "roads", "cadastral", "hydrology")
LAYER_LABELS = {
    "elevation": "🏔  Elevation",
    "roads":     "🛣  Roads",
    "cadastral": "🏛  Cadastre",
    "hydrology": "💧  Hydrology",
}
 
def _rgb_hex(cls: int) -> str:
    r, g, b = CLASS_META[cls]["rgb"]
    return f"#{r:02x}{g:02x}{b:02x}"
 
# Display constants (overridable via UIConfig)
CANVAS_SIZE  = 800
MAX_ZOOM     = 16.0
MIN_ZOOM     = 1.0
ZOOM_STEP    = 1.5
PAN_KEY_STEP = 40
GRID_ALPHA   = 80
SEL_WIDTH    = 3
PANEL_WIDTH  = 310
FONT_LABEL   = ("Helvetica", 11)
FONT_TITLE   = ("Helvetica", 13, "bold")
FONT_SMALL   = ("Helvetica",  9)
FONT_MONO    = ("Courier",   10)
BG_DARK      = "#1a1a2e"
BG_MID       = "#16213e"
BG_PANEL     = "#0f3460"
FG_LIGHT     = "#e0e0e0"
FG_DIM       = "#a0a0b0"
ACCENT       = "#e94560"
AUTO_HAB_THRESHOLD = 0.95
AUTO_LABEL_THRESHOLD = 50
 
 
# =============================================================================
# 2.  DATA MODEL
# =============================================================================
 
class AnnotationGrid:
    """N×N grid storing class labels and restoration costs."""
 
    def __init__(self, N: int):
        self.N       = N
        self.labels  = np.zeros((N, N), dtype=int)
        self.costs   = np.zeros((N, N), dtype=float)
        self._history: list[tuple] = []
 
    # ── mutation ──────────────────────────────────────────────────────────────

    def set_cell(self, row: int, col: int, cls: int, cost: float = 0.0):
        self._history.append((row, col, int(self.labels[row, col]),
                               float(self.costs[row, col])))
        self.labels[row, col] = cls
        self.costs[row, col]  = cost if cls == CLASS_RA else 0.0
 
    def clear_cell(self, row: int, col: int):
        self.set_cell(row, col, CLASS_NONE, 0.0)
 
    def undo(self) -> Optional[tuple]:
        if not self._history:
            return None
        row, col, cls, cost = self._history.pop()
        self.labels[row, col] = cls
        self.costs[row, col]  = cost
        return row, col
 
    def bulk_set(self, cells: list[tuple[int, int]], cls: int, cost: float = 0.0):
        """Set multiple cells at once (single undo entry per call)."""
        for row, col in cells:
            self._history.append((row, col, int(self.labels[row, col]),
                                   float(self.costs[row, col])))
            self.labels[row, col] = cls
            self.costs[row, col]  = cost if cls == CLASS_RA else 0.0
 
    def apply_default_cost(self, default: float) -> int:
        """
        Fill restoration cost for every CLASS_RA cell where cost == 0.
        Returns the number of cells updated.
        """
        mask = (self.labels == CLASS_RA) & (self.costs == 0.0)
        self.costs[mask] = default
        return int(mask.sum())
 
    def counts(self) -> dict:
        return {cls: int((self.labels == cls).sum())
                for cls in range(N_CLASSES + 1)}
 
    def coverage(self) -> float:
        return float((self.labels != CLASS_NONE).sum()) / (self.N * self.N) * 100.0
 
    # ── export arrays ─────────────────────────────────────────────────────────

    def to_arrays(self) -> dict[str, np.ndarray]:
        """
        Convert annotations to FOREMOST-compatible arrays.
        Parameters
        ----------
        elevation_img : np.ndarray (H×W) or (H×W×C), dtype uint8, optional
            Elevation layer image (the PIL RGBA array stored in the renderer).
            When provided, it is converted to a normalised (N×N) float64 array
            with values in [0, 1].  Each cell receives the mean luminance of
            the pixels that fall within it.  The result is saved alongside
            the other arrays as ``elevation``.
        N : int, optional
            Grid size.  Inferred from ``self.N`` when None.

        Returns
        -------
        dict with keys:
            ``habitat``    (N, N) int    — 1 = habitat present
            ``restorable`` (N, N) int    — 1 = CLASS_RA cell
            ``accessible`` (N, N) int    — same as restorable
            ``cost``       (N, N) float  — restoration cost (0 where not RA)
            ``elevation``  (N, N) float  — normalised elevation [0, 1];
                                           included only when *elevation_img*
                                           is provided and non-None.
        """
        habitat    = (self.labels == CLASS_HAB).astype(int)
        restorable = (self.labels == CLASS_RA ).astype(int)
        print("Restorable Matrix:\n", restorable)

        #accessible = restorable.copy() if manually defined
        cfg= AnnotatorConfig()
        cfg.layers.roads.path = "./input/ROADS_acorda.gpkg"
        pil_bg, sq_w, sq4, gdf =_load_gpkg_with_meta(path=cfg.layers.roads.path, layer=None,size=800,working_crs= CRS_WEB_MERCATOR)

        accessible = compute_accessibility_from_roads(cfg=cfg,sq_bounds=sq_w,N=30,max_distance_m=100,working_crs=CRS_WEB_MERCATOR)
        print("Accessible Matrix (max. 500 m from roads):\n", accessible)
        rest_acc=np.add(restorable,accessible)
        rest_acc[rest_acc<2]=0
        print("Accessible & Restorable Matrix:\n", rest_acc/2)

        cost       = self.costs.copy()
        return dict(habitat=habitat, restorable=restorable,
                    accessible=rest_acc/2, cost=cost)

    # ── persistence ───────────────────────────────────────────────────────────
 
    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump({"N": self.N, "labels": self.labels.tolist(),
                       "costs": self.costs.tolist()}, f)
 
    @classmethod
    def load_json(cls, path: str) -> "AnnotationGrid":
        with open(path) as f:
            d = json.load(f)
        g = cls(d["N"])
        g.labels = np.array(d["labels"], dtype=int)
        g.costs  = np.array(d["costs"],  dtype=float)
        return g
 
 
# =============================================================================
# 3.  SPATIAL HELPERS  (CRS-agnostic + SIRGAS 2000 / UTM 25S support)
# =============================================================================

def _elevation_img_to_array(img: np.ndarray, N: int) -> np.ndarray:
    """
    Convert an elevation overlay image to a normalised N×N float64 array.

    The elevation layer stored in :class:`GridRenderer` is a PIL RGBA image
    painted with a terrain colormap.  This function inverts that process:
    it extracts the **luminance** (perceived brightness) of each pixel and
    resamples the result onto the N×N annotation grid by block-averaging.

    Steps
    -----
    1. Accept a PIL Image, a numpy uint8 array (H×W×C), or a float array.
    2. Convert to greyscale luminance: ``L = 0.299R + 0.587G + 0.114B``.
    3. Reshape into N×N blocks by nearest-neighbour pooling (block-average).
    4. Normalise to [0, 1] using the 2nd–98th percentile of the N×N result
       (removes influence of pure-black / pure-white border pixels).

    Parameters
    ----------
    img : numpy ndarray (H, W) or (H, W, 3) or (H, W, 4), dtype uint8 or float,
          OR PIL Image
        Elevation layer rendered as a coloured image.  Greyscale, RGB, and
        RGBA formats are all accepted.  Values are expected in [0, 255] for
        integer arrays or [0, 1] for float arrays.
    N : int
        Target grid size (output will be N×N).

    Returns
    -------
    elev_norm : np.ndarray, shape (N, N), dtype float64, values in [0, 1].
        Row 0 corresponds to the top of the bounding box.

    Notes
    -----
    * The function is intentionally simple so it works even when scipy or
      rasterio are absent.
    * If the image is entirely uniform (no elevation variation) the returned
      array is all-zeros.
    * The output can be saved directly as ``{stem}_elevation_N{N}.npy``.
    """
    # ── 1. Convert PIL → numpy if needed ─────────────────────────────────────
    if hasattr(img, "convert"):  # PIL Image
        img_arr = np.array(img.convert("RGB"), dtype=np.float64)
    else:
        img_arr = np.asarray(img, dtype=np.float64)

    # Normalise float range to [0, 255]
    if img_arr.max() <= 1.01:
        img_arr = img_arr * 255.0

    # ── 2. Greyscale luminance ────────────────────────────────────────────────
    if img_arr.ndim == 2:
        gray = img_arr
    elif img_arr.shape[2] >= 3:
        gray = (0.299 * img_arr[:, :, 0]
                + 0.587 * img_arr[:, :, 1]
                + 0.114 * img_arr[:, :, 2])
    else:
        gray = img_arr[:, :, 0]

    H, W = gray.shape

    # ── 3. Block-average onto N×N grid ────────────────────────────────────────
    elev_grid = np.zeros((N, N), dtype=np.float64)
    cell_h = H / N
    cell_w = W / N

    for ri in range(N):
        r0 = int(ri * cell_h)
        r1 = max(r0 + 1, int((ri + 1) * cell_h))
        r1 = min(r1, H)
        for ci in range(N):
            c0 = int(ci * cell_w)
            c1 = max(c0 + 1, int((ci + 1) * cell_w))
            c1 = min(c1, W)
            block = gray[r0:r1, c0:c1]
            elev_grid[ri, ci] = float(block.mean()) if block.size else 0.0

    # ── 4. Percentile normalisation ───────────────────────────────────────────
    lo = float(np.percentile(elev_grid, 2))
    hi = float(np.percentile(elev_grid, 98))
    if hi - lo < 1e-6:
        return np.zeros((N, N), dtype=np.float64)

    elev_norm = np.clip((elev_grid - lo) / (hi - lo), 0.0, 1.0)
    return elev_norm.astype(np.float64)


# =============================================================================

# ── canonical CRS identifiers used throughout this module ────────────────────
CRS_WEB_MERCATOR = "EPSG:3857"    # display / tile layers (metre-based)
CRS_WGS84        = "EPSG:4326"    # geographic lon/lat
CRS_SIRGAS_UTM25S = "EPSG:31985"  # SIRGAS 2000 / UTM Zone 25S  (Brazil SE)


def _square_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop to largest square then resize to (size × size) RGBA."""
    w, h = img.size
    sq   = min(w, h)
    img  = img.crop(((w - sq) // 2, (h - sq) // 2,
                      (w - sq) // 2 + sq, (h - sq) // 2 + sq))
    return img.resize((size, size), Image.LANCZOS).convert("RGBA")
 
 
# ── generic bounds utilities ──────────────────────────────────────────────────

def make_square_bounds(
    minx: float, miny: float, maxx: float, maxy: float
) -> Tuple[float, float, float, float]:
    """
    Expand a bounding box to a square by extending the shorter axis symmetrically.
    Works in any metre-based or degree-based CRS (pure arithmetic, CRS-agnostic).

    Returns
    -------
    (minx, miny, maxx, maxy)  —  square bounding box, same CRS as input.
    """
    span = max(maxx - minx, maxy - miny)
    cx   = (minx + maxx) / 2
    cy   = (miny + maxy) / 2
    return cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2
 
 
# Keep the old name as an alias so existing call-sites keep working.
_make_square_bounds_3857 = make_square_bounds


def reproject_bounds(
    bounds:     Tuple[float, float, float, float],
    from_crs:   str,
    to_crs:     str,
) -> Tuple[float, float, float, float]:
    """
    Reproject a bounding box (minx, miny, maxx, maxy) from *from_crs* to
    *to_crs*.

    The four corners of the box are transformed individually and the result
    is the tight axis-aligned bounding box of those four projected points.
    This correctly handles CRS with non-linear distortion (e.g. UTM near
    the zone boundary).

    Parameters
    ----------
    bounds   : (minx, miny, maxx, maxy) in *from_crs*
    from_crs : source CRS as an EPSG string, e.g. ``"EPSG:31985"``
    to_crs   : target CRS as an EPSG string, e.g. ``"EPSG:3857"``

    Returns
    -------
    (minx, miny, maxx, maxy) in *to_crs*

    Requires
    --------
    pyproj  (``pip install pyproj``)

    Raises
    ------
    ImportError  if pyproj is not installed.
    pyproj.exceptions.CRSError  if either CRS string is invalid.

    Examples
    --------
    >>> # Convert SIRGAS UTM 25S bbox to Web Mercator
    >>> sirgas_bounds = (680_000, 7_350_000, 720_000, 7_390_000)
    >>> merc = reproject_bounds(sirgas_bounds, "EPSG:31985", "EPSG:3857")

    >>> # Convert WGS84 bbox to SIRGAS UTM 25S
    >>> wgs84_bounds = (-43.5, -23.1, -43.0, -22.8)
    >>> sirgas = reproject_bounds(wgs84_bounds, "EPSG:4326", "EPSG:31985")
    """
    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError(
            "pyproj is required for CRS reprojection.\n"
            "  pip install pyproj"
        )

    t = Transformer.from_crs(from_crs, to_crs, always_xy=True)

    minx, miny, maxx, maxy = bounds
    # Transform all four corners to handle non-linear distortion
    corners = [
        t.transform(minx, miny),
        t.transform(maxx, miny),
        t.transform(maxx, maxy),
        t.transform(minx, maxy),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def transform_bounds_to_sirgas(
    bounds:   Tuple[float, float, float, float],
    from_crs: str = CRS_WEB_MERCATOR,
) -> Tuple[float, float, float, float]:
    """
    Transform a bounding box to **SIRGAS 2000 / UTM Zone 25S  (EPSG:31985)**.

    This is the official geodetic reference system for Brazil (adopted 2005)
    and the preferred working CRS for projects in south-eastern Brazil
    (roughly longitude 30°W–36°W, all latitudes).  The metre-based UTM
    projection makes it well-suited for distance/area calculations and for
    loading local GeoTIFF and GeoPackage files without re-projection
    artefacts.

    Parameters
    ----------
    bounds   : (minx, miny, maxx, maxy) in *from_crs*
    from_crs : CRS of the input bounds.
               Common values:
               - ``"EPSG:3857"``  Web Mercator (default — pipeline CRS)
               - ``"EPSG:4326"``  WGS84 geographic lon/lat
               - ``"EPSG:31985"`` SIRGAS UTM 25S (identity transform)
               - Any EPSG code understood by pyproj

    Returns
    -------
    (minx, miny, maxx, maxy) in EPSG:31985 (SIRGAS 2000 / UTM Zone 25S).
    Units: metres.  Easting ≈ 600 000–800 000 m, Northing ≈ 7 000 000–9 000 000 m.

    Requires
    --------
    pyproj  (``pip install pyproj``)

    Notes
    -----
    * The returned bounds are **not** necessarily square.  Call
      ``make_square_bounds()`` afterwards if a square extent is needed.
    * For data outside UTM Zone 25S coverage (roughly 30°W–36°W) consider
      adjacent zones: EPSG:31984 (24S) or EPSG:31986 (26S).

    Examples
    --------
    >>> # From Web Mercator bounds (default)
    >>> merc = (-4_880_000, -2_700_000, -4_820_000, -2_640_000)
    >>> sirgas = transform_bounds_to_sirgas(merc)
    >>> print([f"{v:.0f}" for v in sirgas])

    >>> # From WGS84 geographic coordinates
    >>> wgs84 = (-43.5, -23.1, -43.0, -22.8)
    >>> sirgas2 = transform_bounds_to_sirgas(wgs84, from_crs="EPSG:4326")

    >>> # Use result to load a local GeoTIFF clipped to SIRGAS extent
    >>> sq_sirgas = make_square_bounds(*sirgas2)
    >>> img, _, _, _ = _load_tiff_with_meta("dem.tif", size=800,
    ...                                      working_crs="EPSG:31985")
    """
    if from_crs.upper() in (CRS_SIRGAS_UTM25S, "EPSG:31985"):
        return bounds   # already in target CRS — no transform needed

    return reproject_bounds(bounds, from_crs, CRS_SIRGAS_UTM25S)


def transform_bounds_from_sirgas(
    bounds:  Tuple[float, float, float, float],
    to_crs:  str = CRS_WEB_MERCATOR,
) -> Tuple[float, float, float, float]:
    """
    Transform a bounding box **from SIRGAS 2000 / UTM Zone 25S (EPSG:31985)**
    to *to_crs*.

    Inverse of :func:`transform_bounds_to_sirgas`.

    Parameters
    ----------
    bounds : (minx, miny, maxx, maxy) in EPSG:31985
    to_crs : target CRS  (default: ``"EPSG:3857"`` Web Mercator)

    Returns
    -------
    (minx, miny, maxx, maxy) in *to_crs*

    Examples
    --------
    >>> sq_sirgas = (680_000.0, 7_350_000.0, 720_000.0, 7_390_000.0)
    >>> sq_merc = transform_bounds_from_sirgas(sq_sirgas)          # → EPSG:3857
    >>> sq_wgs  = transform_bounds_from_sirgas(sq_sirgas, "EPSG:4326")  # → lon/lat
    """
    if to_crs.upper() in (CRS_SIRGAS_UTM25S, "EPSG:31985"):
        return bounds

    return reproject_bounds(bounds, CRS_SIRGAS_UTM25S, to_crs)


def _bounds_3857_to_4326(bounds: Tuple) -> Tuple:
    """
    Convert (minx, miny, maxx, maxy) from EPSG:3857 to EPSG:4326.
    Thin wrapper around :func:`reproject_bounds` kept for backwards
    compatibility.  Falls back to a linear approximation when pyproj is
    not installed.
    """
    try:
        return reproject_bounds(bounds, CRS_WEB_MERCATOR, CRS_WGS84)
    except ImportError:
        # Linear approximation  (±5 km error at mid-latitudes — acceptable
        # for display only, do not use for metric calculations)
        k = 1.0 / 111_320.0
        return (bounds[0] * k, bounds[1] * k,
                bounds[2] * k, bounds[3] * k)
 
 
def bounds_to_crs(
    bounds:   Tuple[float, float, float, float],
    from_crs: str,
    to_crs:   str,
    square:   bool = False,
) -> Tuple[float, float, float, float]:
    """
    High-level convenience: reproject bounds and optionally square them.

    Combines :func:`reproject_bounds` + :func:`make_square_bounds` in a
    single call.  Handles the identity case (``from_crs == to_crs``)
    without invoking pyproj.

    Parameters
    ----------
    bounds   : input bounding box (minx, miny, maxx, maxy)
    from_crs : source CRS string
    to_crs   : target CRS string
    square   : if True, expand the result to a square before returning

    Returns
    -------
    (minx, miny, maxx, maxy) in *to_crs*, optionally squared.

    Examples
    --------
    >>> wgs84 = (-43.5, -23.1, -43.0, -22.8)
    >>> sq_s   = bounds_to_crs(wgs84, "EPSG:4326", "EPSG:31985", square=True)
    >>> sq_merc = bounds_to_crs(wgs84, "EPSG:4326", "EPSG:3857",  square=True)
    """
    if from_crs.upper() == to_crs.upper():
        result = bounds
    else:
        result = reproject_bounds(bounds, from_crs, to_crs)

    return make_square_bounds(*result) if square else result


def _render_matplotlib_to_pil(
    plot_fn,          # callable(ax) → None
    sq_bounds: Tuple,
    size:      int,
    bg_color:  str = "#00000000",
) -> Image.Image:
    """
    Render a matplotlib plot function onto a (size × size) PIL RGBA image,
    with axes locked to *sq_bounds* = (minx, miny, maxx, maxy).
 
    The function is CRS-agnostic: it accepts bounds in any metric CRS
    (EPSG:3857, EPSG:31985, …).  All layer rendering uses this helper so
    every layer is pixel-aligned regardless of the working CRS.

    ``sq_bounds_3857`` parameters elsewhere in the module may now carry
    SIRGAS bounds when ``working_crs="EPSG:31985"`` is active.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
 
    dpi = 100
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111)
    ax.set_xlim(sq_bounds[0], sq_bounds[2])
    ax.set_ylim(sq_bounds[1], sq_bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")
 
    # Parse background colour to RGBA
    if bg_color.startswith("#") and len(bg_color) == 9:
        # "#RRGGBBAA"
        r = int(bg_color[1:3], 16) / 255
        g = int(bg_color[3:5], 16) / 255
        b = int(bg_color[5:7], 16) / 255
        a = int(bg_color[7:9], 16) / 255
        fig.patch.set_facecolor((r, g, b, a))
        ax.set_facecolor((r, g, b, a))
    else:
        fig.patch.set_facecolor(bg_color if bg_color else (0, 0, 0, 0))
        ax.set_facecolor(bg_color if bg_color else (0, 0, 0, 0))
 
    plot_fn(ax)
 
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", (size, size), buf, "raw", "RGBA", 0, 1)
    return img.copy()   # detach from buffer
 
 
# =============================================================================
# 4.  IMAGE / GPKG LOADING  (returns bounds in configurable working CRS)
# =============================================================================
 
# Return type: (PIL RGBA, sq_bounds_working, sq_bounds_4326, GDF or None)
# sq_bounds_working is in whatever working_crs was requested (default EPSG:3857,
# or EPSG:31985 for SIRGAS projects).
LoadResult = Tuple[Image.Image,
                   Optional[Tuple[float,float,float,float]],
                   Optional[Tuple[float,float,float,float]],
                   Optional["gpd.GeoDataFrame"]]
 
 
def _load_tiff_with_meta(
    path:        str,
    size:        int,
    working_crs: str = CRS_WEB_MERCATOR,
) -> LoadResult:
    """
    Load a GeoTIFF, extract a square bounding box in *working_crs*, and
    return a PIL RGBA thumbnail + spatial metadata.

    Parameters
    ----------
    path        : path to the GeoTIFF file
    size        : side length of the output PIL image (pixels)
    working_crs : CRS used for the returned ``sq_bounds`` tuple.
                  Typical values:

                  ``"EPSG:3857"``  — Web Mercator (default; global coverage)
                  ``"EPSG:31985"`` — SIRGAS 2000 / UTM Zone 25S (Brazil)

                  Any EPSG code understood by pyproj is accepted.
                  When the file's native CRS matches *working_crs*, no
                  reprojection is performed.

    Returns
    -------
    (pil_image, sq_bounds_working, sq_bounds_4326, None)
        - ``pil_image``        : RGBA thumbnail, *size* × *size* px
        - ``sq_bounds_working``: square bbox in *working_crs* (or None if
                                 the file has no georeference)
        - ``sq_bounds_4326``   : same extent in WGS84 lon/lat (or None)
        - Fourth element       : always None for raster files (reserved for
                                 the GeoDataFrame returned by GPKG loaders)
    """
    sq_working = sq4 = None
    try:
        import rasterio
        from rasterio.warp import transform_bounds
 
        with rasterio.open(path) as ds:
            nb, nodata = ds.count, ds.nodata
 
            def _norm(b):
                arr = ds.read(b).astype(np.float64)
                if nodata is not None:
                    arr = np.where(arr == nodata, np.nan, arr)
                lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
                if hi - lo < 1e-9:
                    return np.zeros_like(arr, dtype=np.uint8)
                return np.nan_to_num(
                    np.clip((arr - lo) / (hi - lo) * 255, 0, 255), nan=0
                ).astype(np.uint8)
 
            if nb >= 3:
                r, g, b = _norm(1), _norm(2), _norm(3)
            elif nb == 2:
                r, g = _norm(1), _norm(2); b = r
            else:
                r = _norm(1); g = b = r
 
            # Extract bounds in the requested working CRS
            try:
                raw = ds.bounds
                native_crs = str(ds.crs)
                native_name = _crs_display_name(ds.crs)
                working_name = _crs_display_name(working_crs)

                print(f"  [tiff] File CRS     : {native_name}")
                if native_crs != working_crs:
                    print(f"  [tiff] Converting   : {native_name}  →  {working_name}")

                # Transform native bounds → working CRS
                b_working = transform_bounds(
                    ds.crs, working_crs,
                    raw.left, raw.bottom, raw.right, raw.top
                )
                sq_working = make_square_bounds(*b_working)

                if native_crs != working_crs:
                    print(f"  [tiff] ✓ Bounds converted to {working_name}")
                else:
                    print(f"  [tiff] ✓ Already in working CRS — no conversion needed")

                # Also compute WGS84 bounds for OSM / external API queries
                b4 = transform_bounds(
                    ds.crs, CRS_WGS84,
                    raw.left, raw.bottom, raw.right, raw.top
                )
                sq4 = make_square_bounds(*b4)

                print(f"  [tiff] CRS: {native_crs} → working CRS: {working_crs}")
            except Exception as exc:
                print(f"  [tiff] Bounds extraction failed: {exc}")
 
        rgb = np.stack([r, g, b], axis=-1)
        return _square_crop_resize(Image.fromarray(rgb, "RGB"), size), \
               sq_working, sq4, None
 
    except ImportError:
        pass
    except Exception as exc:
        print(f"  [rasterio] Warning: {exc} — PIL fallback")
 
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    return _square_crop_resize(img, size), None, None, None
 
 
def _load_gpkg_with_meta(
    path:        str,
    layer:       Optional[str],
    size:        int,
    working_crs: str = CRS_WEB_MERCATOR,
) -> LoadResult:
    """
    Rasterise a GeoPackage layer as a background PIL image and extract a
    square bounding box in *working_crs*.

    Parameters
    ----------
    path        : path to the GeoPackage (or Shapefile / GeoJSON)
    layer       : layer name to load; if None the first layer is used
    size        : side length of the output PIL image (pixels)
    working_crs : CRS used for the returned ``sq_bounds`` tuple and for
                  internal rendering alignment.

                  ``"EPSG:3857"``  — Web Mercator (default)
                  ``"EPSG:31985"`` — SIRGAS 2000 / UTM Zone 25S

                  The GeoDataFrame is reprojected to *working_crs* so that
                  matplotlib axes coordinates match the returned bounds.

    Returns
    -------
    (pil_image, sq_bounds_working, sq_bounds_4326, gdf)
        - ``pil_image``        : RGBA thumbnail, *size* × *size* px
        - ``sq_bounds_working``: square bbox in *working_crs*
        - ``sq_bounds_4326``   : same extent in WGS84 lon/lat
        - ``gdf``              : GeoDataFrame reprojected to *working_crs*
    """
    import geopandas as gpd
 
    avail = gpd.list_layers(path)["name"].tolist()
    if layer is None:
        layer = avail[0]
    elif layer not in avail:
        raise ValueError(f"Layer '{layer}' not found.  Available: {avail}")
 
    gdf = gpd.read_file(path, layer=layer).reset_index(drop=True)
    if len(gdf) == 0:
        raise ValueError("GPKG layer is empty.")

    # Reproject to working CRS — with console trace
    src_crs  = gdf.crs
    if src_crs is None:
        print(f"  [gpkg] ⚠  Layer '{layer}' has NO CRS — assuming EPSG:4326")
        gdf      = gdf.set_crs(CRS_WGS84)
        src_crs  = gdf.crs

    src_name     = _crs_display_name(src_crs)
    working_name = _crs_display_name(working_crs)
    print(f"  [gpkg] File CRS     : {src_name}")

    if str(src_crs) != working_crs:
        print(f"  [gpkg] Converting   : {src_name}  →  {working_name}")
        gdf = gdf.to_crs(working_crs)
        print(f"  [gpkg] ✓ Conversion complete  ({len(gdf):,} features)")
    else:
        print(f"  [gpkg] ✓ Already in working CRS — no conversion needed")
 
    sq_working = make_square_bounds(*gdf.total_bounds)

    # WGS84 bounds (for OSM queries etc.)
    try:
        sq4 = make_square_bounds(
            *reproject_bounds(sq_working, working_crs, CRS_WGS84))
    except Exception:
        sq4 = None
 
    geom_type = gdf.geometry.geom_type.mode()[0] if len(gdf) else "Polygon"
 
    def _plot(ax):
        ax.set_facecolor("#0f3460")
        if "Point" in geom_type:
            ax.scatter(gdf.geometry.x.values, gdf.geometry.y.values,
                       s=18, color="#52b788", edgecolors="#ffffff",
                       linewidths=0.5, zorder=3)
        elif "Line" in geom_type:
            gdf.plot(ax=ax, color="#52b788", linewidth=1.0, zorder=3)
        else:
            gdf.plot(ax=ax, facecolor="#2d6a4f", edgecolor="#95d5b2",
                     linewidth=0.6, alpha=0.75, zorder=3)
        extent = max(sq_working[2] - sq_working[0],
                     sq_working[3] - sq_working[1])
        if extent > 0 and (size / extent) * (extent / len(gdf) ** 0.5) > 20:
            for idx, row in gdf.iterrows():
                try:
                    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
                    ax.text(cx, cy, str(idx), fontsize=5,
                            ha="center", va="center", color="#ffffff", zorder=4)
                except Exception:
                    pass
 
    img = _render_matplotlib_to_pil(_plot, sq_working, size, bg_color="#1a1a2e")
    return img, sq_working, sq4, gdf
 
 
def _load_source_with_meta(
    source:      str,
    layer:       Optional[str],
    size:        int,
    working_crs: str = CRS_WEB_MERCATOR,
) -> LoadResult:
    """
    Dispatch loading by file extension.  Returns (PIL RGBA, sq_bounds, sq4, gdf|None).

    Parameters
    ----------
    source      : path to image or vector file
    layer       : GPKG layer name (ignored for rasters)
    size        : output image side length in pixels
    working_crs : primary CRS for spatial metadata
                  (``"EPSG:3857"`` or ``"EPSG:31985"`` or any EPSG code)
    """
    ext = Path(source).suffix.lower()
    if ext in (".tif", ".tiff"):
        return _load_tiff_with_meta(source, size, working_crs=working_crs)
    elif ext in (".gpkg", ".shp", ".geojson", ".json"):
        return _load_gpkg_with_meta(source, layer, size, working_crs=working_crs)
    else:
        img = Image.open(source)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        return _square_crop_resize(img, size), None, None, None
 
 
# =============================================================================
# 5.  LAYER RENDERING
# =============================================================================
 
def _render_elevation_layer(
    cfg:            LayerItemConfig,
    sq_bounds_3857: Tuple,
    size:           int,
    dn_field:       str = "DN",
    working_crs:    str = CRS_WEB_MERCATOR,
) -> Optional[Image.Image]:
    """
    Render an elevation layer as a terrain-coloured RGBA overlay.

    Accepted sources (auto-detected from ``cfg.path`` extension)
    ------------------------------------------------------------
    **GeoPackage / Shapefile / GeoJSON** (``.gpkg``, ``.shp``, ``.geojson``)
        Vector file containing polygon or point geometries with elevation
        values stored in a numeric attribute column.  The column name is
        given by *dn_field* (default ``"DN"``).  Common origins:

        * SRTM tiles exported by QGIS / GDAL as polygon grids (each
          polygon = one DEM pixel, ``DN`` = elevation in metres).
        * Contour-line shapefiles (each feature = one elevation band).
        * Point-elevation surveys with a height attribute.

        Algorithm:
          1. Read the GeoDataFrame and reproject to EPSG:3857.
          2. Clip to *sq_bounds_3857*.
          3. Rasterise the ``DN`` field onto a (*size* × *size*) grid
             via ``rasterio.features.rasterize`` (for polygons/points)
             or weighted scatter (for points).
          4. Interpolate any no-data gaps with nearest-neighbour fill.
          5. Apply 2nd–98th percentile normalisation + colormap.

    **GeoTIFF / raster** (``.tif``, ``.tiff``)
        Single-band raster (e.g. SRTM, Copernicus DEM).  The ``DN``
        parameter is ignored; band 1 is used directly.

    **No file / file not found**
        A synthetic Gaussian hill placeholder is rendered so the rest
        of the visualisation pipeline is never blocked.

    Parameters
    ----------
    cfg : LayerItemConfig
        ``cfg.path``     — path to the elevation file (empty → placeholder).
        ``cfg.colormap`` — matplotlib colormap name (default ``"plasma"``).
        ``cfg.alpha``    — overall layer opacity (applied later by the renderer).
    sq_bounds_3857 : tuple (minx, miny, maxx, maxy)
        Square bounding box in EPSG:3857.  All data outside this box is
        discarded; the rendered image covers exactly this spatial extent.
    size : int
        Side length of the output image in pixels (e.g. 800).
    dn_field : str
        Name of the elevation attribute column in vector files.
        Ignored for raster inputs.  Default: ``"DN"``.

    Returns
    -------
    PIL.Image.Image  —  RGBA image, *size* × *size* pixels.

    Raises
    ------
    Does **not** raise.  Any loading error is caught, logged, and the
    function falls back to the synthetic placeholder.

    Examples
    --------
    >>> cfg = LayerItemConfig(path="srtm_dem.gpkg", colormap="terrain")
    >>> img = _render_elevation_layer(cfg, sq_bounds_3857, size=800,
    ...                               dn_field="DN")
    >>> renderer.set_layer("elevation", img)

    >>> # Same API for a plain GeoTIFF:
    >>> cfg2 = LayerItemConfig(path="dem_30m.tif", colormap="plasma")
    >>> img2 = _render_elevation_layer(cfg2, sq_bounds_3857, size=800)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image as _PIL

    norm_arr: Optional[np.ndarray] = None

    if cfg.path and Path(cfg.path).exists():
        ext = Path(cfg.path).suffix.lower()
 
        # ── branch A: vector file (GPKG / SHP / GeoJSON) ─────────────────────
        if ext in (".gpkg", ".shp", ".geojson", ".json"):
            norm_arr = _elevation_from_vector(
                cfg.path, sq_bounds_3857, size, dn_field,
                working_crs=working_crs)
 
        # ── branch B: raster file (GeoTIFF) ──────────────────────────────────
        elif ext in (".tif", ".tiff"):
            norm_arr = _elevation_from_raster(cfg.path,
                                               working_crs=working_crs)
 
        else:
            print(f"  [elevation] Unsupported file type '{ext}'. "
                  "Accepted: .gpkg .shp .geojson .tif .tiff")
 
    # ── fallback: synthetic Gaussian hill ────────────────────────────────────
    if norm_arr is None:
        N = size
        ri, ci = np.mgrid[0:N, 0:N]
        norm_arr = np.exp(
            -(((ri - N / 2) / (N / 3)) ** 2 + ((ci - N / 2) / (N / 3)) ** 2)
        )
        print("  [elevation] No valid elevation file — using synthetic placeholder.")
 
    # ── resize to (size, size) ────────────────────────────────────────────────
    arr_u8  = (norm_arr * 255).clip(0, 255).astype(np.uint8)
    gray    = _PIL.fromarray(arr_u8, "L").resize((size, size), _PIL.LANCZOS)
    arr_rs  = np.array(gray) / 255.0
 
    # ── apply colormap ────────────────────────────────────────────────────────
    try:
        import matplotlib.colormaps as _colormaps
        cmap = _colormaps[cfg.colormap]
    except (AttributeError, ImportError):
        #cmap = cm.get_cmap(cfg.colormap) # matplotlib < 3.7 fallback
        cmap = matplotlib.colormaps.get_cmap(cfg.colormap)
    rgba  = cmap(arr_rs)                   # (size, size, 4) float [0, 1]
    rgba8 = (rgba * 255).astype(np.uint8)
    return _PIL.fromarray(rgba8, "RGBA")
 
 
# ── helper: elevation from vector (GPKG / SHP / GeoJSON) ─────────────────────

def _elevation_from_vector(
    path:           str,
    sq_bounds_3857: Tuple,
    size:           int,
    dn_field:       str = "DN",
    working_crs:    str = CRS_WEB_MERCATOR,
) -> Optional[np.ndarray]:
    """
    Rasterise elevation values from a vector GeoDataFrame's ``dn_field``
    column onto a (*size* × *size*) normalised float array.

    Geometry types supported
    ------------------------
    Polygon / MultiPolygon
        Each polygon is filled with its ``dn_field`` value using
        ``rasterio.features.rasterize``.  Overlapping polygons use the
        value of the last-drawn feature (highest index).
    Point / MultiPoint
        Each point contributes its value to the nearest grid pixel; gaps
        are filled with nearest-neighbour interpolation.
    LineString / MultiLineString
        Lines are burned in the same way as polygons (useful for contour
        lines — every pixel on the line gets the contour elevation).

    Parameters
    ----------
    path           : local path to the vector file
    sq_bounds_3857 : (minx, miny, maxx, maxy) square in EPSG:3857
    size           : output array side length (pixels)
    dn_field       : name of the elevation attribute column

    Returns
    -------
    ndarray (size, size) float64 with values in [0, 1], or None on failure.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("  [elevation/vector] geopandas is required.  pip install geopandas")
        return None

    # ── 1. load & validate ────────────────────────────────────────────────────
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        print(f"  [elevation/vector] Cannot read '{path}': {exc}")
        return None
    print(dn_field)
    if dn_field not in gdf.columns:
        # Try case-insensitive match
        matches = [c for c in gdf.columns if c.upper() == dn_field.upper()]
        if matches:
            dn_field = matches[0]
            print(f"  [elevation/vector] Using column '{dn_field}' "
                  f"(case-insensitive match for 'DN').")
        else:
            print(
                f"  [elevation/vector] Column '{dn_field}' not found.\n"
                f"  Available columns: {list(gdf.columns)}\n"
                f"  Falling back to synthetic placeholder."
            )
            return None

    # Drop rows with missing elevation
    gdf = gdf.dropna(subset=[dn_field]).copy()
    if len(gdf) == 0:
        print("  [elevation/vector] All DN values are NaN.")
        return None

    # ── 2. reproject to working_crs ───────────────────────────────────────────
    src_crs = gdf.crs
    if src_crs is None:
        print(f"  [elevation/vector] ⚠  No CRS found — assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
        src_crs = gdf.crs

    src_name     = _crs_display_name(src_crs)
    working_name = _crs_display_name(working_crs)
    print(f"  [elevation/vector] File CRS   : {src_name}")

    if str(src_crs) != working_crs:
        print(f"  [elevation/vector] Converting  : {src_name}  →  {working_name}")
        try:
            gdf = gdf.to_crs(working_crs)
            print(f"  [elevation/vector] ✓ Conversion complete")
        except Exception as exc:
            print(f"  [elevation/vector] ✗ CRS conversion failed: {exc}")
            return None
    else:
        print(f"  [elevation/vector] ✓ Already in working CRS")

    # ── 3. clip to sq_bounds_3857 ─────────────────────────────────────────────
    from shapely.geometry import box as _box
    clip_box = _box(*sq_bounds_3857)
    try:
        gdf = gdf[gdf.geometry.intersects(clip_box)].copy()
    except Exception:
        pass  # keep everything if clip fails

    if len(gdf) == 0:
        print("  [elevation/vector] No features intersect the target bounding box.")
        return None

    print(f"  [elevation/vector] Rasterising {len(gdf):,} features "
          f"from column '{dn_field}' …")

    # ── 4. determine geometry type ────────────────────────────────────────────
    geom_types = set(gdf.geometry.geom_type.unique())
    has_points = bool(geom_types & {"Point", "MultiPoint"})
    has_lines  = bool(geom_types & {"LineString", "MultiLineString"})
    has_polys  = bool(geom_types & {"Polygon", "MultiPolygon"})

    # Affine transform: maps (row, col) → EPSG:3857 coordinates
    minx, miny, maxx, maxy = sq_bounds_3857
    cell_w = (maxx - minx) / size
    cell_h = (maxy - miny) / size   # positive (top=maxy)

    # rasterio affine: (pixel_width, 0, x_origin, 0, -pixel_height, y_origin)
    try:
        from rasterio.transform import from_bounds as _from_bounds
        from rasterio.features import rasterize as _rasterize
        _HAS_RASTERIO_FEATURES = True
    except ImportError:
        _HAS_RASTERIO_FEATURES = False

    # ── 4a. polygon / line rasterisation ─────────────────────────────────────
    if (has_polys or has_lines) and _HAS_RASTERIO_FEATURES:
        transform = _from_bounds(minx, miny, maxx, maxy, size, size)
        elev_vals = gdf[dn_field].values.astype(np.float64)

        shapes = [
            (geom, val)
            for geom, val in zip(gdf.geometry.__geo_interface__["features"]
                                  if hasattr(gdf.geometry, "__geo_interface__")
                                  else [{"geometry": g.__geo_interface__}
                                        for g in gdf.geometry],
                                 elev_vals)
        ]

        shapes = list(zip(
            [g.__geo_interface__ for g in gdf.geometry],
            elev_vals,
        ))

        grid = _rasterize(
            shapes,
            out_shape=(size, size),
            transform=transform,
            fill=np.nan,
            dtype=np.float64,
            all_touched=True,       # burn pixels touched by edge, not just interior
        )
        grid = np.flipud(grid)      # rasterio uses top-down; flip to row 0 = top

    # ── 4b. point scatter (no rasterio) ──────────────────────────────────────
    else:
        grid = np.full((size, size), np.nan, dtype=np.float64)
        xs   = gdf.geometry.x.values if has_points else np.array([

            g.centroid.x for g in gdf.geometry])
        ys   = gdf.geometry.y.values if has_points else np.array([
            g.centroid.y for g in gdf.geometry])
        vals = gdf[dn_field].values.astype(np.float64)

        cols = np.clip(((xs - minx) / cell_w).astype(int), 0, size - 1)
        rows = np.clip(((maxy - ys) / cell_h).astype(int), 0, size - 1)

        for r, c, v in zip(rows, cols, vals):
            grid[r, c] = v   # last write wins for overlapping points

    # ── 5. interpolate NaN gaps (nearest-neighbour) ───────────────────────────
    nan_mask = np.isnan(grid)
    if nan_mask.any() and not nan_mask.all():
        try:
            from scipy.ndimage import distance_transform_edt
            _, idx = distance_transform_edt(
                nan_mask, return_indices=True)
            grid[nan_mask] = grid[idx[0][nan_mask], idx[1][nan_mask]]
        except ImportError:
            # scipy not available: simple forward-fill along rows
            for r in range(size):
                last_val = np.nanmean(grid[r]) if not np.isnan(grid[r]).all() else 0.0
                for c in range(size):
                    if np.isnan(grid[r, c]):
                        grid[r, c] = last_val
                    else:
                        last_val = grid[r, c]

    if np.isnan(grid).all():
        print("  [elevation/vector] Grid is entirely NaN after rasterisation.")
        return None

    # ── 6. 2nd–98th percentile normalisation ─────────────────────────────────
    lo = np.nanpercentile(grid, 2)
    hi = np.nanpercentile(grid, 98)
    if hi - lo < 1e-9:
        norm = np.zeros((size, size), dtype=np.float64)
    else:
        norm = np.clip((grid - lo) / (hi - lo), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)

    print(f"  [elevation/vector] DN range: [{lo:.1f}, {hi:.1f}]  "
          f"→ normalised [0, 1]  (output {size}×{size})")
    return norm


# ── helper: elevation from raster (GeoTIFF) ──────────────────────────────────

def _elevation_from_raster(
    path:        str,
    working_crs: str = CRS_WEB_MERCATOR,
) -> Optional[np.ndarray]:
    """
    Read band 1 of a GeoTIFF DEM, print its CRS, and return a 2nd–98th
    percentile normalised float64 array.  Returns None on any error.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            native_name  = _crs_display_name(src.crs)
            working_name = _crs_display_name(working_crs)
            print(f"  [elevation/raster] File CRS   : {native_name}")
            if str(src.crs) != working_crs:
                print(f"  [elevation/raster] Note: raster data is in {native_name}; "
                      f"working CRS is {working_name}.  "
                      f"Normalised values only — no geometric warp applied.")
            else:
                print(f"  [elevation/raster] ✓ Matches working CRS")
            arr    = src.read(1).astype(np.float64)
            nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return np.nan_to_num(norm, nan=0.0)
    except ImportError:
        print("  [elevation/raster] rasterio required.  pip install rasterio")
    except Exception as exc:
        print(f"  [elevation/raster] Could not load '{path}': {exc}")
    return None


def _crs_display_name(crs) -> str:
    """
    Return a human-readable CRS label: EPSG code + authority name when
    available, e.g.  ``"EPSG:3857  (WGS 84 / Pseudo-Mercator)"``.
    Accepts a pyproj CRS object, a geopandas CRS, or a plain string.
    """
    try:
        from pyproj import CRS as ProjCRS
        if isinstance(crs, str):
            c = ProjCRS.from_user_input(crs)
        else:
            c = ProjCRS.from_user_input(str(crs))
        code = c.to_epsg()
        name = c.name
        if code:
            return f"EPSG:{code}  ({name})"
        return name or str(crs)
    except Exception:
        return str(crs)


def _load_vector_to_working_crs(
    path: str,
    working_crs: str,
    layer_name: str = "vector",
) -> Optional["gpd.GeoDataFrame"]:
    """
    Load any vector file and reproject it to *working_crs*, printing a
    detailed CRS conversion trace to the console.

    Parameters
    ----------
    path        : path to GPKG / SHP / GeoJSON / JSON
    working_crs : target CRS string (e.g. "EPSG:3857" or "EPSG:31985")
    layer_name  : label used in console messages

    Returns
    -------
    GeoDataFrame in *working_crs*, or None on failure.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print(f"  [{layer_name}] geopandas required.  pip install geopandas")
        return None
 
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        print(f"  [{layer_name}] Cannot read '{path}': {exc}")
        return None

    # ── Print CRS of uploaded file ────────────────────────────────────────────
    src_crs = gdf.crs
    if src_crs is None:
        print(f"  [{layer_name}] ⚠  File '{Path(path).name}' has NO CRS — "
              f"assuming EPSG:4326 (WGS84)")
        gdf = gdf.set_crs("EPSG:4326")
        src_crs = gdf.crs
 
    src_name = _crs_display_name(src_crs)
    print(f"  [{layer_name}] Uploaded CRS  : {src_name}")

    # ── Convert to working CRS if needed ─────────────────────────────────────
    if str(src_crs) != working_crs:
        tgt_name = _crs_display_name(working_crs)
        print(f"  [{layer_name}] Converting    : {src_name}  →  {tgt_name}")
        try:
            gdf = gdf.to_crs(working_crs)
            print(f"  [{layer_name}] ✓ Conversion complete  "
                  f"({len(gdf):,} features, working CRS: {tgt_name})")
        except Exception as exc:
            print(f"  [{layer_name}] ✗ CRS conversion failed: {exc}")
            return None
    else:
        print(f"  [{layer_name}] ✓ Already in working CRS — no conversion needed")

    return gdf


def _render_vector_layer(
    cfg:        LayerItemConfig,
    sq_bounds:  Tuple,
    size:       int,
    layer_name: str = "vector",
    working_crs: str = CRS_WEB_MERCATOR,
) -> Optional[Image.Image]:
    """
    Render a local vector file (roads, cadastral, hydrology) as RGBA overlay.
    Requires a non-empty ``cfg.path``.  No OSM fetching is performed.
    """
    if not cfg.path:
        print(f"  [{layer_name}] No file path configured — skipping.")
        return None
 
    if not Path(cfg.path).exists():
        print(f"  [{layer_name}] File not found: {cfg.path}")
        return None

    gdf = _load_vector_to_working_crs(cfg.path, working_crs, layer_name)
    if gdf is None or len(gdf) == 0:
        return None
 
    fc      = cfg.facecolor if cfg.facecolor else "none"
    ec      = cfg.color
    lw      = cfg.linewidth
    is_line = gdf.geometry.geom_type.str.contains("Line").any()
    is_pt   = gdf.geometry.geom_type.str.contains("Point").any()
 
    def _plot(ax):
        try:
            if is_pt:
                ax.scatter(gdf.geometry.x.values, gdf.geometry.y.values,
                           s=12, color=ec, zorder=3)
            elif is_line:
                gdf.plot(ax=ax, color=ec, linewidth=lw, zorder=3)
            else:
                gdf.plot(ax=ax, facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=3)
        except Exception as exc:
            print(f"  [{layer_name}] Plot error: {exc}")
 
    return _render_matplotlib_to_pil(_plot, sq_bounds, size,
                                      bg_color="#00000000")
 

# =============================================================================
# 5b.  OSM FILE PARSER — extract layers from a local .osm / .osm.pbf file
# =============================================================================

def extract_layers_from_osm(
    osm_path: str,
    *,
    target_crs:          str   = "EPSG:3857",
    road_tags:           Optional[List[str]] = None,
    waterway_tags:       Optional[List[str]] = None,
    elevation_dem_path:  Optional[str]       = None,
    elevation_colormap:  str   = "terrain",
    render_size:         int   = CANVAS_SIZE,
    clip_to_osm_bounds:  bool  = True,
    road_linewidth:      float = 1.6,
    road_color:          str   = "#e63946",
    hydro_linewidth:     float = 1.4,
    hydro_color:         str   = "#4cc9f0",
    verbose:             bool  = True,
) -> Dict[str, object]:
    """
    Parse a local OSM file and return the road network, hydrology network,
    and (optionally) an elevation layer, all aligned to the same spatial extent.

    This is the offline counterpart of the live osmnx download functions.
    It is useful when internet access is unavailable or when a specific
    regional OSM extract (e.g. from Geofabrik) is already on disk.

    Supported input formats
    -----------------------
    .osm          Plain-text OpenStreetMap XML
    .osm.bz2      bzip2-compressed OSM XML (decompressed on the fly)
    .osm.gz       gzip-compressed OSM XML
    .osm.pbf      Protocol-Buffer OSM format  (requires osmium-tool or pyosmium)

    Parameters
    ----------
    osm_path : str
        Path to the OSM file.  Accepted extensions: .osm, .osm.bz2,
        .osm.gz, .osm.pbf.
    target_crs : str
        Output CRS for all returned GeoDataFrames (default: EPSG:3857,
        metre-based pseudo-Mercator used throughout this tool).
    road_tags : list[str] or None
        OSM ``highway`` tag values to include as roads.
        Default: motorway, trunk, primary, secondary, tertiary,
        residential, unclassified, service, path, cycleway, footway, track.
    waterway_tags : list[str] or None
        OSM ``waterway`` tag values to include as hydrology.
        Default: river, stream, canal, drain, ditch.
        ``natural=water`` polygons are always included.
    elevation_dem_path : str or None
        Path to a GeoTIFF DEM raster.  When provided, the raster is
        read, stretched, and rendered as a terrain-coloured PIL image
        clipped to the OSM file bounding box.
        When None, a synthetic placeholder elevation is generated.
    elevation_colormap : str
        Matplotlib colormap name for the elevation layer (default: "terrain").
    render_size : int
        Side length in pixels of the rendered overlay images (default: 800).
    clip_to_osm_bounds : bool
        When True (default), all outputs are spatially bounded by the
        envelope of the OSM file rather than the DEM extent.
    road_linewidth : float
        Line width for the road overlay image.
    road_color : str
        Hex colour for roads (default: "#e63946" — red).
    hydro_linewidth : float
        Line width for the hydrology overlay image.
    hydro_color : str
        Hex colour for waterways (default: "#4cc9f0" — cyan).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with the following keys — all values may be ``None`` if the
    corresponding data was not found or a dependency is missing:

    ``"roads_gdf"``        : geopandas.GeoDataFrame
        Road network edges projected to *target_crs*.
        Columns: geometry (LineString/MultiLineString), name, highway,
        osm_id, and any other OSM tag present in the file.

    ``"hydrology_gdf"``    : geopandas.GeoDataFrame
        Waterway and water-body features projected to *target_crs*.
        Columns: geometry, name, waterway, natural, osm_id, …

    ``"elevation_arr"``    : numpy.ndarray  shape (H, W)  dtype float64
        Normalised elevation values in [0, 1], or ``None`` when no DEM
        is available and synthetic generation was skipped.
        The array covers the same spatial extent as *sq_bounds_3857*.

    ``"roads_img"``        : PIL.Image.Image  RGBA  (render_size × render_size)
        Road network rendered as a transparent overlay.

    ``"hydrology_img"``    : PIL.Image.Image  RGBA  (render_size × render_size)
        Hydrology network rendered as a transparent overlay.

    ``"elevation_img"``    : PIL.Image.Image  RGBA  (render_size × render_size)
        Elevation rendered with the terrain colormap, or ``None``.

    ``"sq_bounds_3857"``   : tuple (minx, miny, maxx, maxy) in EPSG:3857
        Square bounding box covering all parsed features.

    ``"sq_bounds_4326"``   : tuple (west, south, east, north) in EPSG:4326

    ``"crs"``              : str  target CRS string

    Raises
    ------
    FileNotFoundError   if osm_path does not exist.
    ImportError         if geopandas is not installed.
    ValueError          if the file extension is not recognised.

    Examples
    --------
    >>> layers = extract_layers_from_osm("ile_de_france.osm.bz2")
    >>> roads   = layers["roads_gdf"]
    >>> print(roads.geometry.geom_type.value_counts())

    >>> # Feed directly into the annotator renderer:
    >>> renderer.set_layer("roads",     layers["roads_img"])
    >>> renderer.set_layer("hydrology", layers["hydrology_img"])
    >>> renderer.set_layer("elevation", layers["elevation_img"])

    Dependencies
    ------------
    Required : geopandas, shapely
    For .pbf  : npyosmium  (pip install npyosmium)
    For DEM   : rasterio  (pip install rasterio)
    Optional  : matplotlib (for overlay images)
    """
    osm_path = str(osm_path)
    if not Path(osm_path).exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    # ── import guard ──────────────────────────────────────────────────────────
    try:
        import geopandas as gpd
        from shapely.geometry import (LineString, MultiLineString,
                                       Polygon, MultiPolygon, Point)
        from shapely.ops import unary_union
    except ImportError:
        raise ImportError(
            "geopandas is required to parse OSM files.\n"
            "  pip install geopandas shapely"
        )

    # ── default tag lists ────────────────────────────────────────────────────
    if road_tags is None:
        road_tags = [
            "motorway", "motorway_link",
            "trunk", "trunk_link",
            "primary", "primary_link",
            "secondary", "secondary_link",
            "tertiary", "tertiary_link",
            "residential", "living_street",
            "unclassified", "service",
            "path", "cycleway", "footway",
            "bridleway", "track", "steps",
        ]
    if waterway_tags is None:
        waterway_tags = ["river", "stream", "canal", "drain", "ditch",
                         "brook", "tidal_channel"]

    ext = "".join(Path(osm_path).suffixes).lower()   # e.g. ".osm.bz2"

    # ── choose parser ─────────────────────────────────────────────────────────
    if ext.endswith(".pbf"):
        roads_gdf, hydro_gdf = _parse_osm_pbf(
            osm_path, road_tags, waterway_tags, target_crs, verbose)
    else:
        # .osm, .osm.bz2, .osm.gz — XML-based
        roads_gdf, hydro_gdf = _parse_osm_xml(
            osm_path, road_tags, waterway_tags, target_crs, verbose)

    # ── compute square bounding box ──────────────────────────────────────────
    all_geoms = []
    if roads_gdf is not None and len(roads_gdf) > 0:
        all_geoms.append(roads_gdf.total_bounds)
    if hydro_gdf is not None and len(hydro_gdf) > 0:
        all_geoms.append(hydro_gdf.total_bounds)

    if all_geoms:
        bounds_arr = np.array(all_geoms)
        raw_bounds = (
            float(bounds_arr[:, 0].min()),   # minx
            float(bounds_arr[:, 1].min()),   # miny
            float(bounds_arr[:, 2].max()),   # maxx
            float(bounds_arr[:, 3].max()),   # maxy
        )
        sq3 = _make_square_bounds_3857(*raw_bounds)
    else:
        if verbose:
            print("  [osm] No features found — bounds undefined.")
        sq3 = None

    sq4 = _bounds_3857_to_4326(sq3) if sq3 else None

    # ── elevation layer ───────────────────────────────────────────────────────
    elev_arr = None
    elev_img = None

    if sq3 is not None:
        elev_cfg = LayerItemConfig(
            colormap  = elevation_colormap,
            path      = elevation_dem_path or "",
            alpha     = 1.0,
        )
        try:
            elev_img = _render_elevation_layer(elev_cfg, sq3, render_size)
            # Also return the raw normalised array
            elev_arr = np.array(elev_img.convert("L"), dtype=np.float64) / 255.0
        except Exception as exc:
            if verbose:
                print(f"  [osm/elevation] Warning: {exc}")

    # ── render road / hydrology images ────────────────────────────────────────
    road_img  = None
    hydro_img = None

    if sq3 is not None:
        road_cfg = LayerItemConfig(color=road_color,  linewidth=road_linewidth,
                                    path="", alpha=1.0)
        hydro_cfg = LayerItemConfig(color=hydro_color, linewidth=hydro_linewidth,
                                     path="", alpha=1.0)

        # Inject the already-loaded GDFs into _render_vector_layer by
        # temporarily monkey-patching the path to a sentinel and providing
        # the GDF via a closure-based fetch function.
        if roads_gdf is not None and len(roads_gdf) > 0:
            _r = roads_gdf   # capture
            road_img = _render_vector_layer(
                road_cfg, sq3, render_size,
                osm_fetch_fn=lambda b, gdf=_r: gdf,
                layer_name="roads",
            )

        if hydro_gdf is not None and len(hydro_gdf) > 0:
            _h = hydro_gdf
            hydro_img = _render_vector_layer(
                hydro_cfg, sq3, render_size,
                osm_fetch_fn=lambda b, gdf=_h: gdf,
                layer_name="hydrology",
            )

    if verbose:
        n_r = len(roads_gdf) if roads_gdf is not None else 0
        n_h = len(hydro_gdf) if hydro_gdf is not None else 0
        print(f"  [osm] Extracted: {n_r} road features, "
              f"{n_h} hydrology features")
        if sq4 is not None:
            print(f"  [osm] Bounding box (WGS84): "
                  f"W={sq4[0]:.4f}° E={sq4[2]:.4f}° "
                  f"S={sq4[1]:.4f}° N={sq4[3]:.4f}°")

    return dict(
        roads_gdf       = roads_gdf,
        hydrology_gdf   = hydro_gdf,
        elevation_arr   = elev_arr,
        roads_img       = road_img,
        hydrology_img   = hydro_img,
        elevation_img   = elev_img,
        sq_bounds_3857  = sq3,
        sq_bounds_4326  = sq4,
        crs             = target_crs,
    )


# ── XML parser  (.osm / .osm.bz2 / .osm.gz) ──────────────────────────────────

def _parse_osm_xml(
    osm_path:      str,
    road_tags:     List[str],
    waterway_tags: List[str],
    target_crs:    str,
    verbose:       bool,
) -> Tuple[Optional["gpd.GeoDataFrame"], Optional["gpd.GeoDataFrame"]]:
    """
    Parse an OSM XML file (plain, bzip2, or gzip) into two GeoDataFrames.

    Strategy:
      1. Read the XML tree (handling compressed formats transparently).
      2. Build a node-id → (lon, lat) lookup.
      3. Reconstruct ways from their node references.
      4. Filter by highway / waterway / natural tags.
      5. Project to *target_crs*.

    Returns (roads_gdf, hydrology_gdf)  — either may be None.
    """
    import bz2
    import gzip
    import xml.etree.ElementTree as ET
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon, MultiPolygon, Point
    from shapely.ops import unary_union

    ext = "".join(Path(osm_path).suffixes).lower()

    if verbose:
        print(f"  [osm] Parsing XML file: {Path(osm_path).name} …")

    # Open raw or decompressed stream
    if ext.endswith(".bz2"):
        with bz2.open(osm_path, "rb") as fh:
            tree = ET.parse(fh)
    elif ext.endswith(".gz"):
        with gzip.open(osm_path, "rb") as fh:
            tree = ET.parse(fh)
    else:
        tree = ET.parse(osm_path)

    root = tree.getroot()

    # ── 1. Node lookup: {osm_id: (lon, lat)} ─────────────────────────────────
    nodes: Dict[str, Tuple[float, float]] = {}
    for node in root.iter("node"):
        nid = node.get("id")
        lat = node.get("lat")
        lon = node.get("lon")
        if nid and lat and lon:
            nodes[nid] = (float(lon), float(lat))

    if verbose:
        print(f"  [osm] Nodes loaded: {len(nodes):,}")

    # ── 2. Way parsing ────────────────────────────────────────────────────────
    road_records:  list = []
    hydro_records: list = []

    for way in root.iter("way"):
        way_id = way.get("id", "")

        # Collect tag key→value pairs
        tags: Dict[str, str] = {
            tag.get("k", ""): tag.get("v", "")
            for tag in way.iter("tag")
        }

        highway   = tags.get("highway",   "")
        waterway  = tags.get("waterway",  "")
        natural   = tags.get("natural",   "")
        name      = tags.get("name",      "")
        max_speed = tags.get("maxspeed",  "")
        lanes     = tags.get("lanes",     "")
        surface   = tags.get("surface",   "")

        # Node references → coordinate list
        coords = [
            nodes[nd.get("ref")]
            for nd in way.iter("nd")
            if nd.get("ref") in nodes
        ]
        if len(coords) < 2:
            continue

        # ── Road ──────────────────────────────────────────────────────────────
        if highway in road_tags:
            road_records.append({
                "osm_id":   way_id,
                "name":     name,
                "highway":  highway,
                "maxspeed": max_speed,
                "lanes":    lanes,
                "surface":  surface,
                "geometry": LineString(coords),
            })

        # ── Hydrology: waterway lines ──────────────────────────────────────────
        if waterway in waterway_tags:
            hydro_records.append({
                "osm_id":   way_id,
                "name":     name,
                "waterway": waterway,
                "natural":  natural,
                "geometry": LineString(coords),
            })

        # ── Hydrology: natural=water polygons ─────────────────────────────────
        if natural == "water" and len(coords) >= 3 and coords[0] == coords[-1]:
            hydro_records.append({
                "osm_id":   way_id,
                "name":     name,
                "waterway": "",
                "natural":  "water",
                "geometry": Polygon(coords),
            })

    # ── 3. Relation handling (multipolygon water bodies) ─────────────────────
    for rel in root.iter("relation"):
        tags = {t.get("k", ""): t.get("v", "") for t in rel.iter("tag")}
        if tags.get("natural") != "water" and tags.get("waterway") == "":
            continue
        outer_rings: list = []
        for member in rel.iter("member"):
            if member.get("type") == "way" and member.get("role") == "outer":
                ref = member.get("ref", "")
                # Reconstruct ring from already-parsed ways is complex;
                # skip for brevity (most water bodies are simple ways)
                pass

    if verbose:
        print(f"  [osm] Ways parsed: "
              f"{len(road_records)} roads, {len(hydro_records)} hydrology")

    # ── 4. Build GeoDataFrames + project ─────────────────────────────────────
    def _to_gdf(records, crs_in="EPSG:4326"):
        if not records:
            return None
        gdf = gpd.GeoDataFrame(records, crs=crs_in)
        if crs_in != target_crs:
            gdf = gdf.to_crs(target_crs)
        return gdf.reset_index(drop=True)

    roads_gdf = _to_gdf(road_records)
    hydro_gdf = _to_gdf(hydro_records)
    return roads_gdf, hydro_gdf


# ── PBF parser  (.osm.pbf)  ───────────────────────────────────────────────────

def _parse_osm_pbf(
    osm_path:      str,
    road_tags:     List[str],
    waterway_tags: List[str],
    target_crs:    str,
    verbose:       bool,
) -> Tuple[Optional["gpd.GeoDataFrame"], Optional["gpd.GeoDataFrame"]]:
    """
    Parse an OSM PBF file using pyosmium (osmium-tool Python bindings).

    Requires:  pip install osmium

    Falls back gracefully to an empty result if pyosmium is not installed,
    printing an actionable error message.
    """
    try:
        import osmium                         # type: ignore
        import osmium.geom as og              # type: ignore
    except ImportError:
        raise ImportError(
            "pyosmium is required to parse .osm.pbf files.\n"
            "  pip install osmium\n"
            "Alternatively convert to .osm.bz2 with osmconvert:\n"
            "  osmconvert file.osm.pbf -o=file.osm.bz2"
        )

    import geopandas as gpd
    from shapely.geometry import LineString, Polygon, shape as sh_shape
    from shapely.wkb import loads as wkb_loads

    if verbose:
        print(f"  [osm] Parsing PBF file: {Path(osm_path).name} …")

    factory = og.WKBFactory()

    class _Handler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.road_records  = []
            self.hydro_records = []

        def way(self, w):
            tags     = dict(w.tags)
            highway  = tags.get("highway",  "")
            waterway = tags.get("waterway", "")
            natural  = tags.get("natural",  "")
            name     = tags.get("name",     "")

            if highway not in road_tags and waterway not in waterway_tags \
                    and natural != "water":
                return

            try:
                wkb  = factory.create_linestring(w)
                geom = wkb_loads(wkb, hex=True)
            except Exception:
                return

            if highway in road_tags:
                self.road_records.append({
                    "osm_id":   str(w.id),
                    "name":     name,
                    "highway":  highway,
                    "maxspeed": tags.get("maxspeed", ""),
                    "lanes":    tags.get("lanes",    ""),
                    "surface":  tags.get("surface",  ""),
                    "geometry": geom,
                })
            if waterway in waterway_tags or natural == "water":
                self.hydro_records.append({
                    "osm_id":   str(w.id),
                    "name":     name,
                    "waterway": waterway,
                    "natural":  natural,
                    "geometry": geom,
                })

    h = _Handler()
    h.apply_file(osm_path, locations=True)

    if verbose:
        print(f"  [osm] PBF parsed: "
              f"{len(h.road_records)} roads, {len(h.hydro_records)} hydrology")

    def _to_gdf(records):
        if not records:
            return None
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        return gdf.to_crs(target_crs).reset_index(drop=True)

    return _to_gdf(h.road_records), _to_gdf(h.hydro_records)


# =============================================================================
# 6.  GRID RENDERER  (zoom + pan + layer compositing)
# =============================================================================
 
class GridRenderer:
    """
    Renders background + layer overlays + annotation overlays with zoom & pan.
 
    Layer images are stored at _orig_size × _orig_size and cropped/zoomed
    identically to the background, ensuring perfect spatial alignment.
 
    sq_bounds_3857 / sq_bounds_4326 : square spatial extent of the background,
        used by layer rendering functions to align data.  None when the source
        has no georeference (plain PNG/JPG).
    """
 
    def __init__(self, pil_bg: Image.Image, N: int,
                 canvas_size: int = CANVAS_SIZE,
                 sq_bounds_3857: Optional[Tuple] = None,
                 sq_bounds_4326: Optional[Tuple] = None):
        self.N               = N
        self.canvas_size     = canvas_size
        self._bg_orig        = pil_bg.convert("RGBA")
        self._orig_size      = self._bg_orig.width
        self._zoom           = 1.0
        self._pan_x          = 0.0
        self._pan_y          = 0.0
        self.sq_bounds_3857  = sq_bounds_3857
        self.sq_bounds_4326  = sq_bounds_4326
 
        # Layer storage: name → PIL RGBA at _orig_size × _orig_size
        self._layer_imgs:    Dict[str, Optional[Image.Image]] = {n: None for n in LAYER_NAMES}
        self._layer_alpha:   Dict[str, float]  = {n: 0.45  for n in LAYER_NAMES}
        self._layer_visible: Dict[str, bool]   = {n: False for n in LAYER_NAMES}
 
    # ── layer management ──────────────────────────────────────────────────────
 
    def set_layer(self, name: str, img: Optional[Image.Image]):
        """Store a layer image (resized to _orig_size if needed)."""
        if img is not None:
            os_ = self._orig_size
            if img.size != (os_, os_):
                img = img.resize((os_, os_), Image.LANCZOS)
            img = img.convert("RGBA")
        self._layer_imgs[name] = img
 
    def set_layer_alpha(self, name: str, alpha: float):
        self._layer_alpha[name] = max(0.0, min(1.0, alpha))
 
    def set_layer_visible(self, name: str, visible: bool):
        self._layer_visible[name] = visible
 
    def layer_loaded(self, name: str) -> bool:
        return self._layer_imgs.get(name) is not None
 
    # ── zoom / pan ────────────────────────────────────────────────────────────
 
    @property
    def zoom(self) -> float: return self._zoom
 
    @property
    def zoom_pct(self) -> int: return int(self._zoom * 100)
 
    def _visible_size(self) -> float:
        """Side of the visible window in _bg_orig pixels."""
        return self._orig_size / self._zoom
 
    def _scale(self) -> float:
        """Canvas pixels per _bg_orig pixel."""
        return self.canvas_size / self._visible_size()
 
    # ── zoom ──────────────────────────────────────────────────────────────────

    def zoom_at(self, factor: float, canvas_cx: float, canvas_cy: float):
        """Zoom centred on canvas point (canvas_cx, canvas_cy)."""
        opp   = self._visible_size() / self.canvas_size
        img_cx = self._pan_x + canvas_cx * opp
        img_cy = self._pan_y + canvas_cy * opp
        new_z  = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        if new_z == self._zoom:
            return
        self._zoom  = new_z
        new_opp     = self._visible_size() / self.canvas_size
        self._pan_x = img_cx - canvas_cx * new_opp
        self._pan_y = img_cy - canvas_cy * new_opp
        self._clamp_pan()
 
    def zoom_in(self):  self.zoom_at(ZOOM_STEP, self.canvas_size/2, self.canvas_size/2)
    def zoom_out(self): self.zoom_at(1/ZOOM_STEP, self.canvas_size/2, self.canvas_size/2)
    def reset_zoom(self):
        self._zoom = 1.0; self._pan_x = 0.0; self._pan_y = 0.0
 
    def pan(self, dx: float, dy: float):
        opp = self._visible_size() / self.canvas_size
        self._pan_x -= dx * opp
        self._pan_y -= dy * opp
        self._clamp_pan()
 
    def pan_pixels(self, dx: float, dy: float):
        """
        Pan in canvas-pixel units (positive dx → move view right, i.e. image scrolls left).
        Same as pan() but exposed separately for arrow-key bindings.
        """
        self.pan(dx, dy)
 
    def _clamp_pan(self):
        vis = self._visible_size()
        mx  = self._orig_size - vis
        self._pan_x = max(0.0, min(self._pan_x, max(0.0, mx)))
        self._pan_y = max(0.0, min(self._pan_y, max(0.0, mx)))
 
    # ── coordinate conversion ─────────────────────────────────────────────────
 
    def cell_at(self, canvas_x: float, canvas_y: float) -> Tuple[int, int]:
        """Canvas coordinates → (row, col) in the N×N grid."""
        opp   = self._visible_size() / self.canvas_size
        img_x = self._pan_x + canvas_x * opp
        img_y = self._pan_y + canvas_y * opp
        col   = int(img_x * self.N / self._orig_size)
        row   = int(img_y * self.N / self._orig_size)
        return (max(0, min(self.N - 1, row)),
                max(0, min(self.N - 1, col)))
 
    def cell_canvas_bounds(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) canvas pixels for cell (row, col)."""
        opc = self._orig_size / self.N
        sc  = self._scale()
        x0  = int((col     * opc - self._pan_x) * sc)
        y0  = int((row     * opc - self._pan_y) * sc)
        x1  = int(((col+1) * opc - self._pan_x) * sc)
        y1  = int(((row+1) * opc - self._pan_y) * sc)
        return x0, y0, x1, y1
 
    def cell_canvas_center(self, row: int, col: int) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.cell_canvas_bounds(row, col)
        return (x0 + x1) // 2, (y0 + y1) // 2
 
    # ── cell pixel sampling (for auto-annotation) ─────────────────────────────

    def cell_pixels(self, row: int, col: int) -> np.ndarray:
        """
        Return the RGBA pixels (H×W×4 uint8) from _bg_orig for cell (row, col).
        Used by auto-annotation to assess image content.
        """
        opc  = self._orig_size / self.N
        x0, y0 = int(col * opc), int(row * opc)
        x1, y1 = max(x0+1, int((col+1)*opc)), max(y0+1, int((row+1)*opc))
        return np.array(self._bg_orig.crop((x0, y0, x1, y1)), dtype=np.uint8)
 
    # ── internal crop helper ──────────────────────────────────────────────────
 
    def _crop_to_canvas(self, img: Image.Image) -> Image.Image:
        """Crop+resize an _orig_size image to the current canvas view."""
        cs  = self.canvas_size
        vis = self._visible_size()
        px0, py0 = self._pan_x, self._pan_y
        box = (int(px0), int(py0),
               int(min(px0 + vis, self._orig_size)),
               int(min(py0 + vis, self._orig_size)))
        crop = img.crop(box)
        if crop.size != (cs, cs):
            padded = Image.new("RGBA", (cs, cs), (0, 0, 0, 0))
            rw = int(crop.width  * cs / vis)
            rh = int(crop.height * cs / vis)
            if rw > 0 and rh > 0:
                crop = crop.resize((rw, rh), Image.LANCZOS)
            padded.paste(crop, (0, 0))
            return padded
        return crop.resize((cs, cs), Image.LANCZOS)
 
    # ── main render ───────────────────────────────────────────────────────────
 
    def render(self, grid: AnnotationGrid,
               selected: Optional[Tuple[int, int]] = None) -> ImageTk.PhotoImage:
        """Full render: background (cropped+zoomed) + overlays + grid + selection."""
        cs  = self.canvas_size
        vis = self._visible_size()
        px0 = self._pan_x
        py0 = self._pan_y
        opc = self._orig_size / self.N
 
        # 1. Background
        base = self._crop_to_canvas(self._bg_orig)
 
        # 2. Layer overlays (elevation → roads → cadastral → hydrology)
        for name in LAYER_NAMES:
            if not self._layer_visible.get(name, False):
                continue
            limg = self._layer_imgs.get(name)
            if limg is None:
                continue
            alpha = self._layer_alpha.get(name, 0.45)
            layer_crop = self._crop_to_canvas(limg)
            # Scale alpha channel
            r_, g_, b_, a_ = layer_crop.split()
            a_scaled = a_.point(lambda v: int(v * alpha))
            layer_crop = Image.merge("RGBA", (r_, g_, b_, a_scaled))
            base = Image.alpha_composite(base, layer_crop)
 
        # 3. Annotation overlays
        ov   = Image.new("RGBA", (cs, cs), (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)
        sc   = self._scale()
        col_lo = max(0, int(px0 / opc))
        col_hi = min(self.N-1, int((px0 + vis) / opc))
        row_lo = max(0, int(py0 / opc))
        row_hi = min(self.N-1, int((py0 + vis) / opc))
 
        for row in range(row_lo, row_hi + 1):
            for col in range(col_lo, col_hi + 1):
                cls = int(grid.labels[row, col])
                if cls == CLASS_NONE:
                    continue
                meta = CLASS_META[cls]
                r, g, b = meta["rgb"]
                x0, y0, x1, y1 = self.cell_canvas_bounds(row, col)
                x0c, y0c = max(0, x0), max(0, y0)
                x1c, y1c = min(cs-1, x1), min(cs-1, y1)
                if x1c > x0c and y1c > y0c:
                    draw.rectangle([x0c, y0c, x1c, y1c],
                                   fill=(r, g, b, meta["alpha"]))
 
        # 4. Selected cell
        if selected is not None:
            x0, y0, x1, y1 = self.cell_canvas_bounds(*selected)
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(cs-1, x1), min(cs-1, y1)
            if x1c > x0c and y1c > y0c:
                draw.rectangle([x0c, y0c, x1c, y1c],
                               outline=(255, 255, 0, 230), width=SEL_WIDTH)
 
        base = Image.alpha_composite(base, ov)
 
        # 5. Cost labels on CLASS_RA
        gd = ImageDraw.Draw(base)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf",
                                       max(8, int(opc * sc) // 3))
        except (IOError, OSError):
            font = ImageFont.load_default()
 
        for row in range(row_lo, row_hi + 1):
            for col in range(col_lo, col_hi + 1):
                if int(grid.labels[row, col]) != CLASS_RA:
                    continue
                cv = grid.costs[row, col]
                if cv <= 0:
                    continue
                cx, cy = self.cell_canvas_center(row, col)
                lbl = f"{cv:g}"
                for dx, dy in ((-1,-1),(1,-1),(-1,1),(1,1)):
                    gd.text((cx+dx, cy+dy), lbl, font=font,
                            fill=(0,0,0,200), anchor="mm")
                gd.text((cx, cy), lbl, font=font,
                        fill=(255,255,255,240), anchor="mm")
 
        # 6. Grid lines — vertical uses px0 (pan_x), horizontal uses py0 (pan_y)
        gd2 = ImageDraw.Draw(base)
        for i in range(self.N + 1):
            xp = int((i * opc - px0) * sc)
            if 0 <= xp <= cs:
                gd2.line([(xp, 0), (xp, cs)],
                         fill=(255,255,255,GRID_ALPHA), width=1)
            yp = int((i * opc - py0) * sc)
            if 0 <= yp <= cs:
                gd2.line([(0, yp), (cs, yp)],
                         fill=(255,255,255,GRID_ALPHA), width=1)
 
        return ImageTk.PhotoImage(base)
 
    # ── full-res export ───────────────────────────────────────────────────────
 
    def render_to_pil(self, grid: AnnotationGrid) -> Image.Image:
        """Render full-resolution annotated image with all visible layers."""
        size = self._orig_size
        base = self._bg_orig.copy().convert("RGBA")
 
        # Layer overlays
        for name in LAYER_NAMES:
            if not self._layer_visible.get(name, False):
                continue
            limg = self._layer_imgs.get(name)
            if limg is None:
                continue
            alpha = self._layer_alpha.get(name, 0.45)
            r_, g_, b_, a_ = limg.split()
            a_scaled = a_.point(lambda v: int(v * alpha))
            limg_a = Image.merge("RGBA", (r_, g_, b_, a_scaled))
            base = Image.alpha_composite(base, limg_a)
 
        # Annotation overlay
        ov   = Image.new("RGBA", (size, size), (0,0,0,0))
        draw = ImageDraw.Draw(ov)
        opc  = size / self.N
        for row in range(self.N):
            for col in range(self.N):
                cls = int(grid.labels[row, col])
                if cls == CLASS_NONE:
                    continue
                meta = CLASS_META[cls]
                r, g, b = meta["rgb"]
                x0, y0 = int(col * opc), int(row * opc)
                x1, y1 = int((col+1)*opc)-1, int((row+1)*opc)-1
                draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, meta["alpha"]))
        base = Image.alpha_composite(base, ov)
 
        # Grid lines
        gd = ImageDraw.Draw(base)
        for i in range(self.N + 1):
            pos = int(i * opc)
            gd.line([(pos,0),(pos,size)], fill=(255,255,255,GRID_ALPHA), width=1)
            gd.line([(0,pos),(size,pos)], fill=(255,255,255,GRID_ALPHA), width=1)
 
        # Legend
        lh = 26; leg_w = 220; leg_h = (N_CLASSES+1)*lh + 10
        leg = Image.new("RGBA", (leg_w, leg_h), (20,20,40,215))
        ld  = ImageDraw.Draw(leg)
        try:
            lfont = ImageFont.truetype("DejaVuSans.ttf", 14)
        except (IOError, OSError):
            lfont = ImageFont.load_default()
        for i, (cls, meta) in enumerate(CLASS_META.items()):
            y = 5 + i * lh
            r, g, b = meta["rgb"]
            ld.rectangle([5, y+4, 22, y+20], fill=(r,g,b,200))
            ld.text((28, y+5), meta["label"], font=lfont,
                    fill=(220,220,220,255))
        base.paste(leg, (8,8), leg)
        return base
 
 
# =============================================================================
# 7.  MAIN ANNOTATION WINDOW
# =============================================================================
 
class Annotator(tk.Tk):
    """
    Unified interactive annotation window for:
      • Raster images  (PNG, JPEG, GeoTIFF)
      • Vector GPKG files  (rasterised as background, annotated on N×N grid)
      - Zoom / pan / multi-cell paint
      - 4 optional overlay layers  (elevation, roads, cadastral, hydrology)
      - Layer visibility + transparency controls
      - Default restoration cost for CLASS_RA cells
      - Hydra-aware configuration

    Parameters
    ----------
    image_path   : path to image or GPKG file
    N            : grid size (N×N cells)
    layer        : GPKG layer to load (optional)
    pil_image    : pre-computed PIL image (if already loaded)
    source_gdf   : GeoDataFrame when source is a GPKG
    source_label : display label in the title bar
    """
 
    def __init__(
        self,
        image_path:    Optional[str]               = None,
        N:             int                          = 30,
        layer:         Optional[str]                = None,
        pil_image:     Optional[Image.Image]        = None,
        source_gdf:    Optional["gpd.GeoDataFrame"] = None,
        source_label:  Optional[str]                = None,
        sq_bounds_3857: Optional[Tuple]             = None,
        sq_bounds_4326: Optional[Tuple]             = None,
        cfg:           Optional[AnnotatorConfig]    = None,
        working_crs: str = CRS_SIRGAS_UTM25S,
    ):
        super().__init__()
        self.title("FOREMOST — Annotation Tool")
        self.resizable(False, False)
        self.configure(bg=BG_DARK)
 
        self._cfg          = cfg or AnnotatorConfig()
        ui                 = self._cfg.ui
        self._image_path   = image_path
        self._layer        = layer
        self._N            = N
        self._source_gdf   = source_gdf
        self._working_crs  = working_crs           # ← stored for layer loaders
        self._source_label = source_label or (
            Path(image_path).name if image_path else "source")
        self._grid         = AnnotationGrid(N)
        self._sel_row      = None
        self._sel_col      = None
        self._cur_class    = tk.IntVar(value=CLASS_HAB)
        self._tk_img       = None
        self._canvas_img_id = None
        self._inline_entry = self._inline_entry_row = self._inline_entry_col = None
        self._pan_last_x   = self._pan_last_y = None
        self._drag_last_cell: Optional[Tuple[int,int]] = None
        self.result_arrays = None
        self.result_gdf    = None
 
        # Default cost
        self._default_cost_var = tk.DoubleVar(value=self._cfg.cost.default_cost)
 
        # Layer vars (checkbox + alpha scale per layer)
        self._layer_vis_vars:   Dict[str, tk.BooleanVar] = {}
        self._layer_alpha_vars: Dict[str, tk.DoubleVar]  = {}
        self._layer_status:     Dict[str, tk.StringVar]  = {}
        for name in LAYER_NAMES:
            lcfg = getattr(self._cfg.layers, name)
            self._layer_vis_vars[name]   = tk.BooleanVar(value=lcfg.enabled)
            self._layer_alpha_vars[name] = tk.DoubleVar(value=lcfg.alpha * 100)
            self._layer_status[name]     = tk.StringVar(value="Not loaded")
 
        # Load background
        if pil_image is not None:
            bg = pil_image
        elif image_path is not None:
            print(f"Loading {image_path} (working CRS: {working_crs}) …")
            bg, sq_w, sq4, gdf = _load_source_with_meta(
                image_path, layer, CANVAS_SIZE, working_crs=working_crs)
            if gdf is not None and source_gdf is None:
                self._source_gdf = gdf
            if sq_bounds_3857 is None:
                sq_bounds_3857 = sq_w   # generic name reused for any working CRS
            if sq_bounds_4326 is None:
                sq_bounds_4326 = sq4
        else:
            bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (15,15,30,255))
 
        self._renderer = GridRenderer(bg, N, canvas_size=CANVAS_SIZE,
                                       sq_bounds_3857=sq_bounds_3857,
                                       sq_bounds_4326=sq_bounds_4326)
 
        # Pre-load layers if enabled in config
        for name in LAYER_NAMES:
            lcfg = getattr(self._cfg.layers, name)
            if lcfg.enabled:
                self.after(200, lambda n=name: self._load_layer(n, silent=True))
 
        self._build_ui()
        self._bind_keys()
        self._refresh()
 
    # =========================================================================
    # UI CONSTRUCTION
    # =========================================================================
 
    def _build_ui(self):
        left = tk.Frame(self, bg=BG_DARK)
        left.pack(side=tk.LEFT, padx=10, pady=10)
 
        tk.Label(left, text=f"{self._source_label}  —  {self._N}×{self._N} grid",
                 font=FONT_TITLE, fg=FG_LIGHT, bg=BG_DARK).pack(pady=(0, 4))
 
        self._canvas = tk.Canvas(
            left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            cursor="crosshair", highlightthickness=2,
            highlightbackground=ACCENT)
        self._canvas.pack()
 
        self._build_zoom_bar(left)
 
        self._coord_var = tk.StringVar(value="Cell: —")
        tk.Label(left, textvariable=self._coord_var,
                 font=FONT_SMALL, fg=FG_DIM, bg=BG_DARK).pack(pady=(2, 0))
 
        right = tk.Frame(self, bg=BG_DARK, width=PANEL_WIDTH)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=10)
        right.pack_propagate(False)
 
        self._build_class_panel(right)
        self._build_layer_panel(right)
        self._build_cost_panel(right)
        self._build_stats_panel(right)
        self._build_action_panel(right)
 
    # ── zoom bar ──────────────────────────────────────────────────────────────
 
    def _build_zoom_bar(self, parent):
        bar = tk.Frame(parent, bg=BG_MID)
        bar.pack(fill=tk.X, pady=(4, 0))
        btn = dict(font=FONT_SMALL, relief=tk.FLAT, padx=8, pady=3,
                   bg=BG_PANEL, fg=FG_LIGHT,
                   activeforeground=FG_LIGHT, activebackground=BG_DARK)
        tk.Button(bar, text="−", command=self._do_zoom_out, **btn).pack(
            side=tk.LEFT, padx=(4, 0))
        tk.Button(bar, text="+", command=self._do_zoom_in, **btn).pack(
            side=tk.LEFT, padx=2)
        tk.Button(bar, text="⊡  1:1", command=self._do_zoom_reset, **btn).pack(
            side=tk.LEFT, padx=2)
        self._zoom_var = tk.StringVar(value="Zoom: 100 %")
        tk.Label(bar, textvariable=self._zoom_var, font=FONT_SMALL,
                 fg=ACCENT, bg=BG_MID, width=12, anchor=tk.E).pack(
                     side=tk.RIGHT, padx=(0, 6))
        tk.Label(bar, text="right-drag/↑↓←→ = pan",
                 font=FONT_SMALL, fg=FG_DIM, bg=BG_MID).pack(side=tk.RIGHT, padx=4)
 
    # ── class selector ────────────────────────────────────────────────────────
 
    def _build_class_panel(self, parent):
        sep = tk.Frame(parent, bg=BG_MID)
        sep.pack(fill=tk.X, pady=(0, 6))
        tk.Label(sep, text="ANNOTATION", font=FONT_TITLE,
                 fg=FG_LIGHT, bg=BG_MID, pady=6).pack()
 
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=6, pady=2)
 
        for cls, meta in CLASS_META.items():
            if cls == CLASS_NONE:
                continue
            rf = tk.Frame(frame, bg=BG_DARK)
            rf.pack(fill=tk.X, pady=2)
            sw = tk.Canvas(rf, width=16, height=16, bg=BG_DARK, highlightthickness=0)
            sw.pack(side=tk.LEFT, padx=(0, 5))
            r, g, b = meta["rgb"]
            sw.create_rectangle(2, 2, 14, 14, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")
            tk.Radiobutton(rf,
                text=f"[{meta['key']}]  {meta['label']}",
                variable=self._cur_class, value=cls,
                font=FONT_SMALL, fg=FG_LIGHT, bg=BG_DARK,
                selectcolor=BG_MID, activeforeground=FG_LIGHT,
                activebackground=BG_MID, command=self._on_class_change,
            ).pack(side=tk.LEFT)
 
        btn_row = tk.Frame(frame, bg=BG_DARK)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        tk.Button(btn_row, text="✕  Clear cell  [Del]",
                  font=FONT_SMALL, fg="#ff6b6b", bg=BG_MID,
                  activeforeground="#ff6b6b", activebackground=BG_DARK,
                  relief=tk.FLAT, padx=5, pady=3,
                  command=self._clear_selected).pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Auto-annotate button (requires GPKG or image background)
        tk.Button(
            frame, text="🤖 Auto-label Habitat from image",
            font=FONT_SMALL, fg="#333942", bg=BG_MID,
            activeforeground="#7ec8e3", activebackground=BG_DARK,
            relief=tk.FLAT, padx=6, pady=4,
            command=self._auto_annotate_habitat,
        ).pack(fill=tk.X, pady=(4, 0))

        # Auto-annotate button (requires GPKG or image background)
        tk.Button(
            frame, text="🤖 Auto-label Non Restorable area from image",
            font=FONT_SMALL, fg="#333942", bg=BG_MID,
            activeforeground="#7ec8e3", activebackground=BG_DARK,
            relief=tk.FLAT, padx=6, pady=4,
            command=self._auto_annotate_nohabitat,
        ).pack(fill=tk.X, pady=(4, 0))

        # Auto-annotate button (requires GPKG or image background)
        tk.Button(
            frame, text="🤖 Auto-label remaining cells as Restorable",
            font=FONT_SMALL, fg="#333942", bg=BG_MID,
            activeforeground="#7ec8e3", activebackground=BG_DARK,
            relief=tk.FLAT, padx=6, pady=4,
            command=self._auto_annotate_restorable,
        ).pack(fill=tk.X, pady=(4, 0))

    # ── layer control panel ───────────────────────────────────────────────────
 
    def _build_layer_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="  OVERLAY LAYERS",
            font=FONT_SMALL, fg=FG_DIM, bg=BG_DARK,
            bd=1, relief=tk.GROOVE)
        frame.pack(fill=tk.X, padx=6, pady=(6, 4))
 
        georef = self._renderer.sq_bounds_3857 is not None
        if not georef:
            tk.Label(frame, text="⚠  No georeferencing — layers disabled",
                     font=FONT_SMALL, fg="#ffb347", bg=BG_DARK).pack(
                         padx=6, pady=4)
 
        for name in LAYER_NAMES:
            lbl  = LAYER_LABELS[name]
            vis  = self._layer_vis_vars[name]
            alph = self._layer_alpha_vars[name]
            stat = self._layer_status[name]
 
            row_f = tk.Frame(frame, bg=BG_DARK)
            row_f.pack(fill=tk.X, padx=4, pady=2)
 
            # Checkbox
            cb = tk.Checkbutton(
                row_f, text=lbl, variable=vis,
                font=FONT_SMALL, fg=FG_LIGHT, bg=BG_DARK,
                selectcolor=BG_MID, activeforeground=FG_LIGHT,
                activebackground=BG_DARK, width=14, anchor=tk.W,
                state=tk.NORMAL if georef else tk.DISABLED,
                command=lambda n=name: self._on_layer_toggle(n),
            )
            cb.pack(side=tk.LEFT)
 
            # Alpha scale  (0–100)

            sc = tk.Scale(
                row_f, variable=alph, from_=0, to=100,
                orient=tk.HORIZONTAL, length=80, showvalue=True,
                bg=BG_DARK, fg=FG_DIM, troughcolor=BG_MID,
                highlightthickness=1, bd=1,
                state=tk.NORMAL if georef else tk.DISABLED,
                command=lambda v, n=name: self._on_layer_alpha(n, v),
            )

            sc.pack(side=tk.LEFT, padx=2)
            sc.set(100)
 
            # Load button
            tk.Button(
                row_f, text="Load",
                font=("Helvetica", 7), fg="#0a0a0a", bg=BG_PANEL,
                relief=tk.FLAT, padx=4, pady=1,
                state=tk.NORMAL if georef else tk.DISABLED,
                command=lambda n=name: self._load_layer(n),
            ).pack(side=tk.LEFT, padx=(2, 0))
 
            # Change button  (re-opens file dialog even when file already set)
            tk.Button(
                row_f, text="↺",
                font=("Helvetica", 8, "bold"), fg="#0a0a0a", bg=BG_PANEL,
                relief=tk.FLAT, padx=3, pady=1,
                state=tk.NORMAL if georef else tk.DISABLED,
                command=lambda n=name: self._load_layer(n, reload=True),
            ).pack(side=tk.LEFT, padx=(2, 0))
 
            # Status label
            tk.Label(row_f, textvariable=stat, font=("Helvetica", 7),
                     fg=FG_DIM, bg=BG_DARK, width=8, anchor=tk.W).pack(
                         side=tk.LEFT)
 
    # ── default cost panel ────────────────────────────────────────────────────
 
    def _build_cost_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="  DEFAULT RESTORATION COST",
            font=FONT_SMALL, fg=FG_DIM, bg=BG_DARK,
            bd=1, relief=tk.GROOVE)
        frame.pack(fill=tk.X, padx=6, pady=(4, 4))
 
        row = tk.Frame(frame, bg=BG_DARK)
        row.pack(fill=tk.X, padx=6, pady=4)
 
        tk.Label(row, text="$ / cell:", font=FONT_SMALL,
                 fg=FG_DIM, bg=BG_DARK).pack(side=tk.LEFT)
 
        entry = tk.Entry(row, textvariable=self._default_cost_var,
                         font=FONT_MONO, width=8, bg=BG_MID, fg=FG_LIGHT,
                         insertbackground=FG_LIGHT, relief=tk.FLAT, bd=3)
        entry.pack(side=tk.LEFT, padx=4)
 
        tk.Button(row, text="Fill missing costs",
                  font=FONT_SMALL, fg="#0a0a0a", bg=BG_MID,
                  relief=tk.FLAT, padx=5, pady=2,
                  command=self._apply_default_cost).pack(side=tk.LEFT)
 
    # ── statistics panel ──────────────────────────────────────────────────────
 
    def _build_stats_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="  STATISTICS",
            font=FONT_LABEL, fg=FG_LIGHT,
            bg=BG_DARK, bd=1, relief=tk.GROOVE, padx=6, pady=6)
        frame.pack(fill=tk.X, padx=6, pady=(4, 4))
 
        self._stat_vars = {}
        for cls in range(N_CLASSES + 1):
            meta = CLASS_META[cls]
            rf   = tk.Frame(frame, bg=BG_DARK)
            rf.pack(fill=tk.X, pady=1)
            r, g, b = meta["rgb"]
            tk.Label(rf, width=2, bg=f"#{r:02x}{g:02x}{b:02x}",
                     relief=tk.FLAT).pack(side=tk.LEFT, padx=(0,5))
            tk.Label(rf, text=meta["label"][:24], font=FONT_SMALL,
                     fg=FG_DIM, bg=BG_DARK, anchor=tk.W).pack(
                         side=tk.LEFT, fill=tk.X, expand=True)
            var = tk.StringVar(value="0")
            self._stat_vars[cls] = var
            tk.Label(rf, textvariable=var, font=FONT_MONO,
                     fg=FG_LIGHT, bg=BG_DARK, width=5, anchor=tk.E).pack(side=tk.RIGHT)
 
        tk.Frame(frame, bg=BG_MID, height=1).pack(fill=tk.X, pady=(4,2))
        cr = tk.Frame(frame, bg=BG_DARK)
        cr.pack(fill=tk.X)
        tk.Label(cr, text="Coverage:", font=FONT_SMALL, fg=FG_DIM,
                 bg=BG_DARK).pack(side=tk.LEFT)
        self._cov_var = tk.StringVar(value="0.0 %")
        tk.Label(cr, textvariable=self._cov_var, font=FONT_MONO,
                 fg=ACCENT, bg=BG_DARK).pack(side=tk.RIGHT)
        self._cov_bar_frame = tk.Frame(frame, bg=BG_MID, height=5)
        self._cov_bar_frame.pack(fill=tk.X, pady=(2,0))
        self._cov_bar = tk.Frame(self._cov_bar_frame, bg=ACCENT, height=5, width=0)
        self._cov_bar.place(x=0, y=0, height=5)
 
    # ── action buttons ────────────────────────────────────────────────────────
 
    def _build_action_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=6, pady=(8, 4))
 
        for txt, cmd, bg, fg in [
            #("↩  Undo  [Ctrl+Z]",    self._undo,         BG_MID,    #0a0a0a),
            ("💾  Save session",       self._save_session, BG_MID,    "#0a0a0a"),
            ("📂  Load session",       self._load_session, BG_MID,    "#0a0a0a"),
            ("🗑  Clear all",           self._clear_all,   "#3a1a1a", "#0a0a0a"),
        ]:
            tk.Button(frame, text=txt, command=cmd, font=FONT_SMALL, fg=fg, bg=bg,
                      activeforeground=fg, activebackground=BG_DARK,
                      relief=tk.FLAT, padx=6, pady=4, anchor=tk.W,
                      ).pack(fill=tk.X, pady=2)
 
        tk.Frame(frame, bg=ACCENT, height=2).pack(fill=tk.X, pady=(8, 5))
        tk.Button(frame, text="EXPORT NumPy Arrays & Annotated Image",
                  command=self._export,
                  font=("Helvetica", 11, "bold"),
                  fg="#0a0a0a", bg=ACCENT,
                  activeforeground="#ffffff", activebackground="#c73652",
                  relief=tk.FLAT, padx=6, pady=8,
                  ).pack(fill=tk.X)
        tk.Label(frame, text="[Ctrl+S]", font=FONT_SMALL,
                 fg=FG_DIM, bg=BG_DARK).pack(anchor=tk.E)
 
    # =========================================================================
    # KEY & MOUSE BINDINGS
    # =========================================================================
 
    def _bind_keys(self):
        for key, cls in [("1", CLASS_HAB), ("2", CLASS_RA), ("3", CLASS_NR)]:
            self.bind(f"<Key-{key}>",
                      lambda e, c=cls: (self._cur_class.set(c),
                                        self._on_class_change()))
        self.bind("<Delete>",      lambda e: self._clear_selected())
        self.bind("<BackSpace>",   lambda e: self._clear_selected())
        self.bind("<Control-z>",   lambda e: self._undo())
        self.bind("<Control-Z>",   lambda e: self._undo())
        self.bind("<Control-s>",   lambda e: self._export())
        self.bind("<Control-S>",   lambda e: self._export())
        for k in ("<equal>","<plus>","<KP_Add>"):
            self.bind(k, lambda e: self._do_zoom_in())
        for k in ("<minus>","<KP_Subtract>"):
            self.bind(k, lambda e: self._do_zoom_out())
        self.bind("<Key-0>",       lambda e: self._do_zoom_reset())
        self.bind("<Left>",        lambda e: self._do_pan_key(-PAN_KEY_STEP, 0))
        self.bind("<Right>",       lambda e: self._do_pan_key( PAN_KEY_STEP, 0))
        self.bind("<Up>",          lambda e: self._do_pan_key(0, -PAN_KEY_STEP))
        self.bind("<Down>",        lambda e: self._do_pan_key(0,  PAN_KEY_STEP))
 
        self._canvas.bind("<Button-1>",        self._on_canvas_click)
        self._canvas.bind("<B1-Motion>",       self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag_last_cell", None))
        self._canvas.bind("<Button-3>",         self._on_pan_start)
        self._canvas.bind("<B3-Motion>",        self._on_pan_move)
        self._canvas.bind("<ButtonRelease-3>",  self._on_pan_end)
        self._canvas.bind("<MouseWheel>",  self._on_mousewheel)
        self._canvas.bind("<Button-4>",
            lambda e: (self._renderer.zoom_at(ZOOM_STEP, e.x, e.y),
                       self._post_zoom()))
        self._canvas.bind("<Button-5>",
            lambda e: (self._renderer.zoom_at(1/ZOOM_STEP, e.x, e.y),
                       self._post_zoom()))
 
    # ── zoom ──────────────────────────────────────────────────────────────────
 
    def _do_zoom_in(self):  self._renderer.zoom_in();  self._post_zoom()
    def _do_zoom_out(self): self._renderer.zoom_out(); self._post_zoom()
    def _do_zoom_reset(self): self._renderer.reset_zoom(); self._post_zoom()
 
    def _on_mousewheel(self, event):
        self._renderer.zoom_at(
            ZOOM_STEP if event.delta > 0 else 1/ZOOM_STEP, event.x, event.y)
        self._post_zoom()
 
    def _post_zoom(self):
        self._zoom_var.set(f"Zoom: {self._renderer.zoom_pct} %")
        if self._inline_entry is not None:
            self._reposition_inline_entry(
                self._inline_entry_row, self._inline_entry_col)
        self._refresh()
 
    # ── arrow-key pan ─────────────────────────────────────────────────────────
 
    def _do_pan_key(self, dx: float, dy: float):
        if self._renderer.zoom <= 1.0:
            return
        self._renderer.pan_pixels(dx, dy)
        self._refresh()
 
    # ── right-click pan ───────────────────────────────────────────────────────
 
    def _on_pan_start(self, event):
        self._pan_last_x = event.x; self._pan_last_y = event.y
        self._canvas.config(cursor="fleur")
 
    def _on_pan_move(self, event):
        if self._pan_last_x is None: return
        self._renderer.pan(event.x - self._pan_last_x,
                           event.y - self._pan_last_y)
        self._pan_last_x = event.x; self._pan_last_y = event.y
        self._refresh()
 
    def _on_pan_end(self, event):
        self._pan_last_x = None; self._canvas.config(cursor="crosshair")
 
    # ── left-click annotation ─────────────────────────────────────────────────
 
    def _on_canvas_click(self, event):
        if self._inline_entry is not None:
            r, c = self._renderer.cell_at(event.x, event.y)
            if (r, c) != (self._inline_entry_row, self._inline_entry_col):
                self._hide_inline_entry(validate=True)
        row, col = self._renderer.cell_at(event.x, event.y)
        self._drag_last_cell = (row, col)
        self._select_cell(row, col)
        self._annotate_cell(row, col)
 
    def _on_canvas_drag(self, event):
        row, col = self._renderer.cell_at(event.x, event.y)
        if (row, col) == self._drag_last_cell:
            return
        self._drag_last_cell = (row, col)
        cls = self._cur_class.get()
        self._hide_inline_entry(validate=True)
        self._select_cell(row, col)
        existing = (self._grid.costs[row, col]
                    if self._grid.labels[row, col] == CLASS_RA else 0.0)
        self._grid.set_cell(row, col, cls, existing)
        self._refresh()
 
    def _on_class_change(self):
        if self._sel_row is not None:
            cls = self._cur_class.get()
            if cls == CLASS_RA:
                self._grid.set_cell(self._sel_row, self._sel_col, CLASS_RA, 0.0)
                self._refresh()
                self._show_inline_entry(self._sel_row, self._sel_col)
            else:
                self._hide_inline_entry(validate=False)
                self._grid.set_cell(self._sel_row, self._sel_col, cls, 0.0)
                self._refresh()
 
    # =========================================================================
    # CELL OPERATIONS
    # =========================================================================
 
    def _select_cell(self, row: int, col: int):
        self._sel_row = row; self._sel_col = col
        self._coord_var.set(
            f"Cell: R{row+1}, C{col+1}  |  Zoom {self._renderer.zoom_pct} %")
 
    def _annotate_cell(self, row: int, col: int):
        cls = self._cur_class.get()
        existing = (self._grid.costs[row, col]
                    if self._grid.labels[row, col] == CLASS_RA else 0.0)
        self._grid.set_cell(row, col, cls, existing)
        self._refresh()
        if cls == CLASS_RA:
            self._show_inline_entry(row, col)
 
    # ── inline cost entry ─────────────────────────────────────────────────────
 
    def _show_inline_entry(self, row: int, col: int):
        self._hide_inline_entry(validate=False)
        x0, y0, x1, y1 = self._renderer.cell_canvas_bounds(row, col)
        cx = (x0+x1)//2; cy = (y0+y1)//2; cpx = max(1, x1-x0)
        w, h = max(40, cpx-6), max(16, min(22, cpx-6))
        var  = tk.StringVar()
        ev   = self._grid.costs[row, col]
        if ev > 0:
            var.set(str(ev))
        e = tk.Entry(self._canvas, textvariable=var,
                     font=("Helvetica", max(7, cpx//4), "bold"),
                     fg="#ffffff", bg="#7a3d00", insertbackground="#ffffff",
                     selectbackground=_rgb_hex(CLASS_RA), selectforeground="#ffffff",
                     relief=tk.FLAT, justify=tk.CENTER, highlightthickness=2,
                     highlightcolor=_rgb_hex(CLASS_RA),
                     highlightbackground=_rgb_hex(CLASS_RA))
        e.place(x=cx-w//2, y=cy-h//2, width=w, height=h)
        e.select_range(0, tk.END); e.focus_set()
        e.bind("<Return>",   lambda _: self._hide_inline_entry(True))
        e.bind("<KP_Enter>", lambda _: self._hide_inline_entry(True))
        e.bind("<Escape>",   lambda _: self._hide_inline_entry(False))
        e.bind("<FocusOut>", lambda _: self.after(50, lambda: self._hide_inline_entry(True)))
        self._inline_entry = e; self._inline_entry_var = var
        self._inline_entry_row = row; self._inline_entry_col = col
 
    def _reposition_inline_entry(self, row: int, col: int):
        if self._inline_entry is None: return
        x0, y0, x1, y1 = self._renderer.cell_canvas_bounds(row, col)
        cx = (x0+x1)//2; cy = (y0+y1)//2; cpx = max(1, x1-x0)
        w, h = max(40, cpx-6), max(16, min(22, cpx-6))
        self._inline_entry.place(x=cx-w//2, y=cy-h//2, width=w, height=h)
        self._inline_entry.configure(font=("Helvetica", max(7, cpx//4), "bold"))
 
    def _hide_inline_entry(self, validate: bool = True):
        if self._inline_entry is None: return
        if validate:
            try:
                cost = float(self._inline_entry_var.get().strip().replace(",",".") or 0)
                if cost < 0: cost = 0.0
            except ValueError:
                cost = 0.0
            self._grid.set_cell(self._inline_entry_row,
                                 self._inline_entry_col, CLASS_RA, cost)
        try:
            self._inline_entry.destroy()
        except tk.TclError:
            pass
        self._inline_entry = self._inline_entry_var = None
        self._inline_entry_row = self._inline_entry_col = None
        self._refresh()
 
    def _clear_selected(self):
        if self._sel_row is not None:
            self._hide_inline_entry(False)
            self._grid.clear_cell(self._sel_row, self._sel_col)
            self._refresh()
 
    def _clear_all(self):
        if messagebox.askyesno("Confirm", "Clear all annotations?"):
            self._hide_inline_entry(False)
            self._grid = AnnotationGrid(self._N)
            self._sel_row = self._sel_col = None
            self._coord_var.set("Cell: —")
            self._refresh()
 
    def _undo(self):
        result = self._grid.undo()
        if result:
            self._select_cell(*result)
            self._refresh()
 
    # =========================================================================
    # LAYER OPERATIONS
    # =========================================================================
 
    def _on_layer_toggle(self, name: str):
        vis = self._layer_vis_vars[name].get()
        if vis and not self._renderer.layer_loaded(name):
            self._load_layer(name)
        else:
            self._renderer.set_layer_visible(name, vis)
            self._refresh()
 
    def _on_layer_alpha(self, name: str, value):
        alpha = float(value) / 100.0
        self._renderer.set_layer_alpha(name, alpha)
        if self._layer_vis_vars[name].get():
            self._refresh()
 
    def _load_layer(self, name: str, silent: bool = False, reload: bool = False):
        """
        Load (or reload) a layer image and store it in the renderer.

        Parameters
        ----------
        name   : layer name — one of "elevation", "roads", "cadastral", "hydrology"
        silent : suppress dialog boxes (used for auto-load at startup)
        reload : if True, always prompt for a new file even when cfg.path is set
        """
        bounds = self._renderer.sq_bounds_3857
        if bounds is None:
            if not silent:
                messagebox.showwarning(
                    "No georeferencing",
                    "Layers require a georeferenced source (GeoTIFF or GPKG).")
            return
 
        lcfg = getattr(self._cfg.layers, name)
        wcrs = self._working_crs
 
        # ── always offer a file picker for "Change" requests ──────────────────
        if reload or not lcfg.path:
            if name == "elevation":
                filetypes = [("Elevation files",
                              "*.tif *.tiff *.gpkg *.shp *.geojson"),
                             ("GeoTIFF", "*.tif *.tiff"),
                             ("Vector",  "*.gpkg *.shp *.geojson *.json"),
                             ("All",     "*.*")]
                title = "Load / Change elevation file"
            else:
                filetypes = [("Vector files",
                              "*.gpkg *.shp *.geojson *.json"),
                             ("All", "*.*")]
                title = f"Load / Change {name} file"

            path = filedialog.askopenfilename(title=title, filetypes=filetypes)
            if path:
                lcfg.path = path
            elif reload:
                # User cancelled — keep existing layer, do not reload
                return
            elif not lcfg.path:
                self._layer_status[name].set("No file")
                return

        self._layer_status[name].set("⟳ Loading…")
        self.update_idletasks()

        try:
            img = None
            print(f"\n[Layer: {name}]  file = {lcfg.path}")

            if name == "elevation":
                img = _render_elevation_layer(lcfg, bounds, CANVAS_SIZE, # modification to map
                                               working_crs=wcrs)
            else:
                img = _render_vector_layer(lcfg, bounds, CANVAS_SIZE,
                                            layer_name=name,
                                            working_crs=wcrs)
 
            if img is not None:
                self._renderer.set_layer(name, img)
                self._renderer.set_layer_alpha(
                    name, self._layer_alpha_vars[name].get() / 100.0)
                self._renderer.set_layer_visible(
                    name, self._layer_vis_vars[name].get())
                short = Path(lcfg.path).name
                self._layer_status[name].set(f"✓ {short[:14]}")
                print(f"  [layer:{name}] ✓ Loaded  →  {Path(lcfg.path).name}")
            else:
                self._layer_status[name].set("No data")
 
        except Exception as exc:
            self._layer_status[name].set("✗ Error")
            if not silent:
                messagebox.showerror(f"Layer '{name}'", str(exc))
            print(f"  [layer:{name}] {exc}")
 
        self._refresh()
 
    # =========================================================================
    # AUTO-ANNOTATION
    # =========================================================================
 
    def _auto_annotate_habitat(self):
        count = 0
        for row in range(self._N):
            for col in range(self._N):
                if self._grid.labels[row, col] != CLASS_NONE:
                    continue
                pix = self._renderer.cell_pixels(row, col)
                non_black = (pix[:, :, :3].astype(np.int32).max(axis=2) > 20)
                if non_black.sum() / non_black.size > AUTO_HAB_THRESHOLD:
                    self._grid.set_cell(row, col, CLASS_HAB)
                    count += 1
        self._refresh()
        messagebox.showinfo(
            "Auto-annotation complete",
            f"{count} cell(s) automatically labelled as Habitat "
            f"(>{AUTO_HAB_THRESHOLD*100:.0f} % non-black pixels).")

    def _auto_annotate_nohabitat(self):
        """
        Automatically label cells as Non Restorable Habitat when more than AUTO_HAB_THRESHOLD
        (default 95 %) of their pixels are black in the background image.

        Uses the full-resolution background stored in GridRenderer._bg_orig. AUTO_LABEL_THRESHOLD = 50
        Black pixels are defined as RGB < (AUTO_LABEL_THRESHOLD, AUTO_LABEL_THRESHOLD, AUTO_LABEL_THRESHOLD) – i.e. very dark pixels
        used as nodata / background in GPKG renders and some TIF files.

        Only cells currently labeled CLASS_NONE are affected.
        """
        N = self._N
        count = 0

        for row in range(N):
            for col in range(N):
                if self._grid.labels[row, col] != CLASS_NONE:
                    continue  # don't overwrite existing annotations
                pixels = self._renderer.cell_pixels(row, col)  # H×W×4 uint8
                rgb = pixels[:, :, :3].astype(np.int32)
                # Non-black = at least one channel > threshold
                non_black = (rgb.max(axis=2) < AUTO_LABEL_THRESHOLD)
                frac = float(non_black.sum()) / non_black.size
                if frac > AUTO_HAB_THRESHOLD:
                    self._grid.set_cell(row, col, CLASS_NR)
                    count += 1

        self._refresh()
        messagebox.showinfo(
            "Auto-annotation complete",
            f"{count} cell(s) automatically labelled as Non Restorable Area\n"
            f"(threshold: {AUTO_HAB_THRESHOLD * 100:.0f} % black pixels).",
        )


    def _auto_annotate_restorable(self):
        """
        Automatically label remaining cells as Restorable Habitat with a default cost
        Only cells currently labeled CLASS_NONE are affected.
        """
        N = self._N
        count = 0
        for row in range(N):
            for col in range(N):
                if self._grid.labels[row, col] == CLASS_NONE:
                    self._grid.set_cell(row, col, CLASS_RA)
                    count += 1
        self._refresh()
        messagebox.showinfo(
            "Auto-annotation complete",
            f"{count} remaining cell(s) automatically labelled as Restorable Area."
        )

    # =========================================================================
    # DEFAULT COST
    # =========================================================================
 
    def _apply_default_cost(self):
        """Fill cost=0 CLASS_RA cells with the configured default cost."""
        try:
            default = float(self._default_cost_var.get())
            if default < 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showwarning("Invalid cost",
                                   "Please enter a non-negative number.")
            return
        n = self._grid.apply_default_cost(default)
        self._refresh()
        messagebox.showinfo("Default cost applied",
                            f"{n} cell(s) updated with default cost = {default:g} $.")
 
    # =========================================================================
    # SESSION
    # =========================================================================
 
    def _save_session(self):
        path = filedialog.asksaveasfilename(
            title="Save session", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self._grid.save_json(path)
            messagebox.showinfo("Saved", f"Session saved to:\n{path}")
 
    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load session", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path: return
        try:
            g = AnnotationGrid.load_json(path)
            if g.N != self._N:
                messagebox.showerror("Error",
                    f"Session N={g.N} incompatible with grid N={self._N}.")
                return
            self._hide_inline_entry(False)
            self._grid = g
            self._refresh()
            messagebox.showinfo("Loaded", f"Session loaded from:\n{path}")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
 
    # =========================================================================
    # EXPORT
    # =========================================================================
 
    def _export(self):
        # Auto-fill default cost if configured
        if self._cfg.cost.auto_fill:
            try:
                default = float(self._default_cost_var.get())
                n = self._grid.apply_default_cost(default)
                if n > 0:
                    print(f"  Auto-filled default cost {default:g} $ "
                          f"for {n}  cell(s).")
            except (ValueError, tk.TclError):
                pass
 
        arrays   = self._grid.to_arrays()
        coverage = self._grid.coverage()
        counts   = self._grid.counts()
 
        if coverage < 100.0:
            if not messagebox.askyesno(
                "Incomplete export",
                f"Coverage: {coverage:.1f} %\n"
                f"{counts[CLASS_NONE]} cell(s) not annotated.\n\nExport anyway?"):
                return
 
        folder = filedialog.askdirectory(title="Choose output folder", initialdir = "./output")
        if not folder: return
 
        stem  = Path(self._source_label).stem
        saved = []
 
        for name, arr in arrays.items():
            fname = os.path.join(folder, f"{stem}_{name}_N{self._N}.npy")
            np.save(fname, arr)
            saved.append(fname)
 
        json_path = os.path.join(folder, f"{stem}_session_N{self._N}.json")
        self._grid.save_json(json_path)
 
        # Annotated image
        try:
            img_path = os.path.join(folder,
                                     f"{stem}_annotated_N{self._N}.png")
            self._renderer.render_to_pil(self._grid).save(img_path)
            saved.append(img_path)
        except Exception as exc:
            print(f"  [Export image] Warning: {exc}")
 
        # Annotated GPKG
        gpkg_out = None
        if self._source_gdf is not None:
            try:
                out_gdf  = _map_grid_to_gdf(self._grid, self._source_gdf)
                gpkg_out = os.path.join(folder,
                                         f"{stem}_annotated_N{self._N}.gpkg")
                out_gdf.to_file(gpkg_out, driver="GPKG")
                self.result_gdf = out_gdf
            except Exception as exc:
                print(f"  [Export GPKG] Warning: {exc}")
 
        msg = (
            "Export complete!\n\n"
            f"Folder: {folder}\n\n"
            + "\n".join(f"  • {os.path.basename(p)}" for p in saved)
            + f"\n  • {os.path.basename(json_path)} (session)"
            + (f"\n  • {os.path.basename(gpkg_out)} (annotated GeoDataFrame)"
               if gpkg_out else "")
            + f"\n\nStatistics:\n"
            f"  Habitat            : {counts[CLASS_HAB]} with {AUTO_HAB_THRESHOLD *100:.0f}% \n"
            f"  Restorable Access. : {counts[CLASS_RA]}\n"
            f"  Non-Restorable     : {counts[CLASS_NR]}\n"
            f"  Not annotated        : {counts[CLASS_NONE]}\n"
            f"  Coverage           : {coverage:.1f} %"
        )
        messagebox.showinfo("NumPy Export", msg)
        self.result_arrays = arrays
        return arrays
 
    # =========================================================================
    # RENDER
    # =========================================================================
 
    def _refresh(self):
        sel = ((self._sel_row, self._sel_col)
               if self._sel_row is not None else None)
        self._tk_img = self._renderer.render(self._grid, selected=sel)
        if self._canvas_img_id is None:
            self._canvas_img_id = self._canvas.create_image(
                0, 0, anchor=tk.NW, image=self._tk_img)
        else:
            self._canvas.itemconfig(self._canvas_img_id, image=self._tk_img)
 
        counts = self._grid.counts()
        for cls, var in self._stat_vars.items():
            var.set(str(counts.get(cls, 0)))
        cov = self._grid.coverage()
        self._cov_var.set(f"{cov:.1f} %")
        bw = int(self._cov_bar_frame.winfo_width() * cov / 100.0)
        self._cov_bar.place(x=0, y=0, height=5, width=max(0, bw))
 
 
# =============================================================================
# 8.  GPKG UTILITIES
# =============================================================================
 
def _pick_layer_dialog(gpkg_path: str) -> Optional[str]:
    try:
        import geopandas as gpd
    except ImportError:
        return None
    layers = gpd.list_layers(gpkg_path)["name"].tolist()
    if len(layers) == 1:
        return layers[0]
 
    dlg = tk.Tk()
    dlg.title("Select GPKG layer")
    dlg.configure(bg=BG_DARK)
    dlg.resizable(False, False)
    tk.Label(dlg, text="Available layers:", font=FONT_LABEL,
             fg=FG_LIGHT, bg=BG_DARK).pack(padx=16, pady=(12, 4))
    var = tk.StringVar(value=layers[0])
    for lyr in layers:
        tk.Radiobutton(dlg, text=lyr, variable=var, value=lyr,
                       font=FONT_SMALL, fg=FG_LIGHT, bg=BG_DARK,
                       selectcolor=BG_MID).pack(anchor=tk.W, padx=24)
    chosen = [layers[0]]
    def _ok(): chosen[0] = var.get(); dlg.destroy()
    tk.Button(dlg, text="OK", font=FONT_LABEL, fg=FG_LIGHT, bg=ACCENT,
              relief=tk.FLAT, padx=20, pady=6, command=_ok).pack(pady=12)
    dlg.mainloop()
    return chosen[0]
 
 
def _map_grid_to_gdf(grid: AnnotationGrid,
                      gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    from shapely.geometry import box as sbox
    out = gdf.copy()
    minx, miny, maxx, maxy = gdf.total_bounds
    span = max(maxx - minx, maxy - miny, 1e-9)
    N = grid.N; cs = span / N
    classes = np.full(len(gdf), CLASS_NONE, dtype=int)
    costs_out = np.zeros(len(gdf), dtype=float)
 
    for fi in range(len(gdf)):
        geom = gdf.geometry.iloc[fi]
        if geom is None or geom.is_empty:
            continue
        fx0, fy0, fx1, fy1 = geom.bounds
        cl0 = max(0, int((fx0 - minx) / cs))
        cl1 = min(N-1, int((fx1 - minx) / cs))
        rl0 = max(0, int((miny + span - fy1) / cs))
        rl1 = min(N-1, int((miny + span - fy0) / cs))
        votes = {}; cra = 0.0
        for r in range(rl0, rl1+1):
            for c in range(cl0, cl1+1):
                if geom.intersects(sbox(
                    minx+c*cs, miny+span-(r+1)*cs,
                    minx+(c+1)*cs, miny+span-r*cs)):
                    cls = int(grid.labels[r, c])
                    votes[cls] = votes.get(cls, 0) + 1
                    if cls == CLASS_RA:
                        cra = max(cra, grid.costs[r, c])
        if votes:
            best = max(votes, key=votes.get)
            classes[fi]   = best
            costs_out[fi] = cra if best == CLASS_RA else 0.0
 
    out["class"]       = classes
    out["class_label"] = [CLASS_META[c]["label"] for c in classes]
    out["restorable"]  = (classes == CLASS_RA).astype(int)
    out["accessible"]  = (classes == CLASS_RA).astype(int)
    out["cost"]        = costs_out
    return out
 
 
# =============================================================================
# 9.  ROAD-BASED ACCESSIBILITY MATRIX  (from local GPKG road network)
# =============================================================================

def compute_accessibility_from_roads(
    cfg:            "AnnotatorConfig",
    sq_bounds:      Tuple[float, float, float, float],
    N:              int   = 30,
    max_distance_m: float = 500.0, # distanc to road in meters
    working_crs:    str   = CRS_WEB_MERCATOR,
) -> np.ndarray:
    """
    Compute a binary N×N accessibility matrix from a local road-network GPKG.

    Each cell of the matrix is set to **1** when it either:
      * contains at least one road **node** (intersection / endpoint), or
      * is intersected by at least one road **edge** (line segment), or
      * has its centroid within *max_distance_m* metres of the nearest node
        or edge.

    Otherwise the cell is set to **0**.

    The road file is read from ``cfg.layers.roads.path`` (the value configured
    in *conf/annotator.yaml* under ``layers.roads.path``).  The GPKG is treated
    as a **spatial graph**: every LineString / MultiLineString feature becomes
    an edge, and every unique coordinate pair that appears at a line endpoint
    becomes a node.  The graph is built with ``networkx`` so that graph
    algorithms (shortest path, connected components, …) can be applied to the
    same data if needed.
 
    Parameters
    ----------
    cfg : AnnotatorConfig
        Full Hydra / dataclass configuration object.  The roads file path is
        read from ``cfg.layers.roads.path``.
    sq_bounds : (minx, miny, maxx, maxy)
        Square bounding box of the study area in *working_crs*.  This is the
        spatial extent that will be divided into the N×N grid.
    N : int
        Grid resolution (number of rows = number of columns).  Default: 30.
    max_distance_m : float
        Maximum distance in metres from the nearest road node or edge for a
        cell to be considered accessible.  Cells that are farther away receive
        value 0.  Default: 500 m.
    working_crs : str
        Metre-based projected CRS used for all distance calculations.
        Default: ``"EPSG:3857"`` (Web Mercator).
        Use ``"EPSG:31985"`` for SIRGAS 2000 / UTM Zone 25S (Brazil), etc.
 
    Returns
    -------
    accessible : numpy.ndarray, shape (N, N), dtype int, values in {0, 1}
        Binary accessibility matrix aligned to the N×N annotation grid.
        Row 0 corresponds to the **top** of the bounding box (largest y),
        column 0 to the **left** (smallest x).

    Algorithm
    ---------
    1. Read the road GPKG, print its CRS, and reproject to *working_crs*.
    2. Build a ``networkx.Graph`` where every unique coordinate is a node
       (keyed by (x, y)) and every LineString segment between consecutive
       coordinates is an edge.
    3. Extract three types of spatial objects for proximity testing:
         - All **node positions** (coordinate pairs) as a numpy array.
         - All **edge midpoints** (midpoints of every segment between two
           consecutive coordinates) as a numpy array.
         - All **edge geometries** as a list for direct intersection tests.
    4. Construct the N×N grid of cell centroids in *working_crs*.
    5. Run a ``scipy.spatial.cKDTree`` query to find, for each centroid, the
       distance to the nearest node *and* the nearest edge midpoint.
    6. Additionally, intersect each cell polygon with road edge geometries
       to detect cells that are crossed by a road (distance = 0 by definition).
    7. A cell is accessible iff distance_to_nearest_road ≤ max_distance_m.

    Console output
    --------------
    Detailed CRS trace and progress messages are printed so the user can
    follow the loading and conversion steps:

    .. code-block:: text
 
        [accessibility] Road file    : roads.gpkg
        [accessibility] File CRS     : EPSG:4326  (WGS 84)
        [accessibility] Converting   : EPSG:4326  →  EPSG:3857  (WGS 84 / Pseudo-Mercator)
        [accessibility] ✓ Conversion complete  (12 345 features)
        [accessibility] Building road graph …
        [accessibility]   Nodes : 8 210
        [accessibility]   Edges : 12 344
        [accessibility] Computing N×N grid centroids  (N=30) …
        [accessibility] KD-Tree: 900 centroids vs 20 554 spatial samples …
        [accessibility] ✓ Accessibility matrix computed
        [accessibility]   Accessible : 432 / 900 cells (48.0 %)
        [accessibility]   Threshold  : 500.0 m
        [accessibility]   Working CRS: EPSG:3857  (WGS 84 / Pseudo-Mercator)

    Raises
    ------
    FileNotFoundError
        If the road file configured in ``cfg.layers.roads.path`` does not exist.
    ImportError
        If ``geopandas``, ``networkx``, or ``scipy`` are not installed.
    ValueError
        If the road file contains no valid line geometries after loading and
        reprojection.

    Examples
    --------
    Basic usage with a YAML config:


    >>> cfg = AnnotatorConfig()
    >>> cfg.layers.roads.path = "my_roads.gpkg"
    >>> sq = make_square_bounds(680_000, 7_350_000, 720_000, 7_390_000)
    >>> acc = compute_accessibility_from_roads(
    ...     cfg, sq, N=30, max_distance_m=300,
    ...     working_crs=CRS_SIRGAS_UTM25S,
    ... )
    >>> print(acc.shape, acc.sum(), "accessible cells")

    Exporting the matrix alongside annotation arrays:

    >>> arrays = app._grid.to_arrays()
    >>> arrays["accessible"] = acc          # override default accessible layer
    >>> np.save("accessible.npy", acc)

    Dependencies
    ------------
    Required : ``geopandas``, ``networkx``, ``scipy``
    Optional : ``pyproj`` (for full CRS name printing; degrades gracefully)

    .. code-block:: bash

        pip install geopandas networkx scipy
    """
    tag = "[accessibility]"

    # ── 0. Validate path before importing heavy dependencies ──────────────────
    roads_path = cfg.layers.roads.path if cfg.layers.roads.path else ""
    if not roads_path:
        raise ValueError(
            f"{tag} No road file configured.\n"
            "Set 'layers.roads.path' in conf/annotator.yaml or pass it via CLI:\n"
            "  python satellite_annotator.py layers.roads.path=my_roads.gpkg"
        )
    if not Path(roads_path).exists():
        raise FileNotFoundError(
            f"{tag} Road file not found: '{roads_path}'\n"
            "Update 'layers.roads.path' in conf/annotator.yaml."
        )

    # ── 1. Import guards ──────────────────────────────────────────────────────
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            f"{tag} geopandas is required.  pip install geopandas")

    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            f"{tag} networkx is required.  pip install networkx")

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError(
            f"{tag} scipy is required.  pip install scipy")

    print(f"\n{tag} Road file    : {Path(roads_path).name}")

    # ── 2. Load GPKG → reproject with CRS trace ───────────────────────────────
    roads_gdf = _load_vector_to_working_crs(roads_path, working_crs, "accessibility")
    if roads_gdf is None or len(roads_gdf) == 0:
        raise ValueError(
            f"{tag} Road file loaded but is empty or failed reprojection.")

    # Keep only line geometries (drop points, polygons, empties)
    line_mask = roads_gdf.geometry.geom_type.str.contains(
        "LineString", case=False, na=False)
    roads_lines = roads_gdf[line_mask].copy()

    if len(roads_lines) == 0:
        raise ValueError(
            f"{tag} No LineString / MultiLineString geometries found in "
            f"'{Path(roads_path).name}'.  Road networks must contain line features."
        )
 
    # ── 3. Build networkx graph from the road geometries ─────────────────────
    #
    # Each unique (x, y) coordinate pair → graph node
    # Each consecutive pair of coordinates in a LineString → graph edge
    #
    # The graph stores node positions as (x, y) tuples so they can be
    # collected into a numpy array for KD-Tree queries.
    print(f"{tag} Building road graph …")

    G = nx.Graph()
    node_coords: list = []    # list of (x, y) for every unique node
    edge_coords: list = []    # list of (mx, my) midpoints of every edge segment
    edge_geoms:  list = []    # shapely LineString for every edge (for intersection)
    node_index:  dict = {}    # (x, y) → node id

    try:
        from shapely.geometry import LineString as _LS
    except ImportError:
        raise ImportError(f"{tag} shapely is required.  pip install shapely")

    def _add_node(xy: tuple) -> int:
        if xy not in node_index:
            nid = len(node_index)
            node_index[xy] = nid
            G.add_node(nid, x=xy[0], y=xy[1])
            node_coords.append(xy)
        return node_index[xy]

    for geom in roads_lines.geometry:
        if geom is None or geom.is_empty:
            continue
        # Flatten MultiLineString → list of LineStrings
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            prev_id = _add_node(coords[0])
            for i in range(1, len(coords)):
                cur_id = _add_node(coords[i])
                # Edge: store midpoint + shapely geometry for proximity tests
                mx = (coords[i-1][0] + coords[i][0]) / 2
                my = (coords[i-1][1] + coords[i][1]) / 2
                edge_coords.append((mx, my))
                edge_geoms.append(_LS([coords[i-1], coords[i]]))
                if not G.has_edge(prev_id, cur_id):
                    seg_len = part.length / max(1, len(coords) - 1)
                    G.add_edge(prev_id, cur_id, length=seg_len)
                prev_id = cur_id

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"{tag}   Nodes : {n_nodes:,}")
    print(f"{tag}   Edges : {n_edges:,}")

    if n_nodes == 0:
        warnings.warn(f"{tag} No road nodes extracted — returning all-inaccessible.")
        return np.zeros((N, N), dtype=int)

    # ── 4. Build spatial samples for KD-Tree: nodes + edge midpoints ──────────
    node_xy_arr   = np.array(node_coords, dtype=np.float64)       # (n_nodes, 2)
    if edge_coords:
        edge_mid_arr  = np.array(edge_coords, dtype=np.float64)   # (n_edges, 2)
        spatial_pts   = np.vstack([node_xy_arr, edge_mid_arr])    # all samples
    else:
        spatial_pts   = node_xy_arr

    # ── 5. Build N×N grid of cell centroids in working_crs ────────────────────
    minx, miny, maxx, maxy = sq_bounds
    cell_w = (maxx - minx) / N
    cell_h = (maxy - miny) / N

    print(f"{tag} Computing N×N grid centroids  (N={N}) …")

    col_idx = np.arange(N)
    row_idx = np.arange(N)
    cent_x  = minx + (col_idx + 0.5) * cell_w   # shape (N,)
    # Row 0 = top → highest y
    cent_y  = maxy - (row_idx + 0.5) * cell_h   # shape (N,)

    grid_cx, grid_cy = np.meshgrid(cent_x, cent_y)   # both (N, N)
    centroids = np.column_stack([grid_cx.ravel(), grid_cy.ravel()])  # (N², 2)
 
    # ── 6. KD-Tree: distance from each centroid to nearest spatial sample ──────
    print(f"{tag} KD-Tree: {len(centroids):,} centroids vs "
          f"{len(spatial_pts):,} spatial samples …")

    tree  = cKDTree(spatial_pts)
    dists, _ = tree.query(centroids, k=1, workers=-1)   # (N²,)
    dist_grid = dists.reshape(N, N)                      # (N, N), metres

    # ── 7. Direct intersection: cells crossed by any road edge get dist=0 ─────
    #
    # Build a GDF of cell polygons and do a spatial join with road edges.
    # Cells that intersect any road edge are unconditionally accessible.
    try:
        from shapely.geometry import box as _box
        import geopandas as gpd

        cell_boxes = []
        for ri in range(N):
            y1 = maxy - ri       * cell_h
            y0 = maxy - (ri + 1) * cell_h
            for ci in range(N):
                x0 = minx + ci       * cell_w
                x1 = minx + (ci + 1) * cell_w
                cell_boxes.append(_box(x0, y0, x1, y1))

        cells_gdf = gpd.GeoDataFrame(
            {"row": np.repeat(np.arange(N), N),
             "col": np.tile(np.arange(N), N)},
            geometry=cell_boxes,
            crs=working_crs,
        )
        roads_clip = roads_lines.copy()
        joined = cells_gdf.sjoin(roads_clip[["geometry"]],
                                   how="left", predicate="intersects")
        intersected_idx = joined[~joined.index_right.isna()].index.unique()

        # Override distance for intersected cells → force accessible
        for flat_idx in intersected_idx:
            ri = flat_idx // N
            ci = flat_idx %  N
            dist_grid[ri, ci] = 0.0

        print(f"{tag}   Direct intersections: "
              f"{len(intersected_idx):,} cells contain a road edge")

    except Exception as exc:
        # Non-fatal: the KD-Tree distances alone are still used
        print(f"{tag}   Direct intersection check skipped: {exc}")
 
    # ── 8. Apply threshold ────────────────────────────────────────────────────
    accessible = (dist_grid <= max_distance_m).astype(int)
 
    n_acc  = int(accessible.sum())
    pct    = 100.0 * n_acc / (N * N)
    w_name = _crs_display_name(working_crs)

    print(f"{tag} ✓ Accessibility matrix computed")
    print(f"{tag}   Accessible : {n_acc:,} / {N*N} cells ({pct:.1f} %)")
    print(f"{tag}   Threshold  : {max_distance_m:.1f} m")
    print(f"{tag}   Working CRS: {w_name}")

    return accessible
 
 
# =============================================================================
# 10.  PUBLIC API
# =============================================================================
 
def annotate(
    image_path:  Optional[str]           = None,
    N:           int                      = 30,
    ask_N:       bool                     = True,
    cfg:         Optional[AnnotatorConfig] = None,
    working_crs: str                      = CRS_WEB_MERCATOR,
) -> dict:
    """
    Launch interactive annotation for an image (PNG/JPEG/GeoTIFF).

    Parameters
    ----------
    image_path  : path to the image file.  A file picker opens if None.
    N           : grid size (N×N cells).
    ask_N       : show a dialog to choose N if True.
    cfg         : optional Hydra/dataclass configuration.
    working_crs : CRS used for spatial metadata and layer alignment.
                  ``"EPSG:3857"`` (default) or ``"EPSG:31985"`` (SIRGAS).
    """
    if image_path is None:
        r = tk.Tk(); r.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images & vectors",
                        "*.tif *.tiff *.png *.jpg *.jpeg *.gpkg *.geojson"),
                       ("GeoTIFF", "*.tif *.tiff"),
                       ("Images",  "*.png *.jpg *.jpeg"),
                       ("All",     "*.*")])
        r.destroy()
        if not image_path:
            print("No file selected."); return {}
 
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
 
    if ask_N:
        r = tk.Tk(); r.withdraw()
        val = simpledialog.askinteger("Grid N×N", "Grid size (N):",
                                       minvalue=5, maxvalue=500, initialvalue=N,
                                       parent=r)
        r.destroy()
        if val: N = val
 
    app = Annotator(image_path=image_path, N=N, cfg=cfg,
                              working_crs=working_crs)
    app.mainloop()
    return (app.result_arrays if hasattr(app, "result_arrays") and app.result_arrays
            else app._grid.to_arrays())
 
 
def annotate_gpkg(
    gpkg_path:   Optional[str]           = None,
    layer:       Optional[str]           = None,
    N:           int                      = 30,
    ask_N:       bool                     = True,
    cfg:         Optional[AnnotatorConfig] = None,
    working_crs: str                      = CRS_WEB_MERCATOR,
) -> Optional["gpd.GeoDataFrame"]:
    """
    Launch interactive annotation for a GeoPackage file (N×N grid).

    Parameters
    ----------
    gpkg_path   : path to the .gpkg file.  A file picker opens if None.
    layer       : layer name.  A selection dialog opens if None and multi-layer.
    N           : grid size.
    ask_N       : show a dialog to choose N if True.
    cfg         : optional Hydra/dataclass configuration.
    working_crs : CRS for spatial alignment of background and overlay layers.
                  ``"EPSG:3857"`` (default) or ``"EPSG:31985"`` (SIRGAS UTM 25S).
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("pip install geopandas matplotlib")
 
    if gpkg_path is None:
        r = tk.Tk(); r.withdraw()
        gpkg_path = filedialog.askopenfilename(
            title="Select GeoPackage",
            filetypes=[("GeoPackage", "*.gpkg"),
                       ("Vectors",    "*.gpkg *.shp *.geojson *.json"),
                       ("All",        "*.*")])
        r.destroy()
        if not gpkg_path:
            print("No file selected."); return None
 
    if not os.path.isfile(gpkg_path):
        raise FileNotFoundError(f"File not found: {gpkg_path}")
 
    if layer is None:
        layer = _pick_layer_dialog(gpkg_path)
 
    if ask_N:
        r = tk.Tk(); r.withdraw()
        val = simpledialog.askinteger("Grid N×N", "Grid size (N):",
                                       minvalue=5, maxvalue=500, initialvalue=N,
                                       parent=r)
        r.destroy()
        if val: N = val
 
    print(f"Rasterising {gpkg_path} (layer: {layer}, CRS: {working_crs}) …")
    pil_bg, sq_w, sq4, gdf = _load_gpkg_with_meta(
        gpkg_path, layer, CANVAS_SIZE, working_crs=working_crs)
 
    app = Annotator(
        pil_image=pil_bg, N=N, source_gdf=gdf,
        source_label=Path(gpkg_path).name,
        sq_bounds_3857=sq_w, sq_bounds_4326=sq4,
        cfg=cfg, working_crs=working_crs)
    app.mainloop()
 
    if app.result_gdf is not None:
        return app.result_gdf
    print("Window closed without export — mapping grid to features …")
    try:
        return _map_grid_to_gdf(app._grid, gdf)
    except Exception as exc:
        print(f"  Error: {exc}"); return None
 
 
# =============================================================================
# 11.  CLI  (with optional Hydra)
# =============================================================================
 
if _HAS_HYDRA:
    @hydra.main(config_path="conf", config_name="annotator", version_base=None)
    def _hydra_main(hydra_cfg: DictConfig) -> None:
        cfg = _cfg_from_dict(hydra_cfg)
        _run_from_config(cfg)
 
 
def _run_from_config(cfg: AnnotatorConfig):
    """Launch annotation using a fully populated AnnotatorConfig."""
    d = cfg.data
    if d.gpkg_path:
        annotate_gpkg(gpkg_path=d.gpkg_path, layer=d.layer or None,
                      N=d.N, ask_N=False, cfg=cfg)
    elif d.image_path:
        annotate(image_path=d.image_path, N=d.N, ask_N=False, cfg=cfg)
    else:
        annotate(N=d.N, ask_N=True, cfg=cfg)
 
 
def main():
    parser = argparse.ArgumentParser(
        description="annotation.py — GUI for interactive annotation of FOREMOST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image",  "-i", type=str, default=None)
    parser.add_argument("--gpkg",   "-g", type=str, default=None)
    parser.add_argument("--layer",  "-l", type=str, default=None)
    parser.add_argument("--N",      "-n", type=int, default=30)
    parser.add_argument(
        "--working-crs", "-c",
        type=str, default=CRS_WEB_MERCATOR,
        metavar="EPSG",
        help=(
            "Working CRS for spatial alignment of files and layers. "
            f"Default: '{CRS_WEB_MERCATOR}' (Web Mercator). "
            f"Use '{CRS_SIRGAS_UTM25S}' for SIRGAS 2000 / UTM Zone 25S (Brazil). "
            "Any EPSG code understood by pyproj is accepted."
        ),
    )
    parser.add_argument("--write-config", action="store_true", help="Write conf/annotator.yaml and exit")
    args, remaining = parser.parse_known_args()
 
    if args.write_config:
        write_default_yaml("conf")
        return
 
    # Delegate to Hydra if installed and YAML / overrides present
    if _HAS_HYDRA and (remaining or Path("conf/annotator.yaml").exists()):
        sys.argv = [sys.argv[0]] + remaining
        _hydra_main()
        return
 
    # Fall back to argparse + defaults
    cfg = AnnotatorConfig()
    if args.image:  cfg.data.image_path = args.image
    if args.gpkg:   cfg.data.gpkg_path  = args.gpkg
    if args.layer:  cfg.data.layer      = args.layer
    cfg.data.N = args.N
    wcrs = args.working_crs

    if args.gpkg:
        gdf = annotate_gpkg(gpkg_path=args.gpkg, layer=args.layer,
                             N=args.N, ask_N=not args.no_ask_N,
                             cfg=cfg, working_crs=wcrs)
        if gdf is not None:
            print(f"\nAnnotated GeoDataFrame: {len(gdf):,} features")
            for col in ["class", "class_label", "restorable", "accessible", "cost"]:
                if col in gdf.columns:
                    print(f"  {col:14s}: {gdf[col].dtype}")
    else:
        arrays = annotate(image_path=args.image or None,
                                     N=args.N, ask_N=not args.no_ask_N,
                                     cfg=cfg, working_crs=wcrs)
        if arrays:
            print("\nExported arrays:")
            for name, arr in arrays.items():
                print(f"  {name:12s}: shape={arr.shape}  min={arr.min():.3f}  "
                      f"max={arr.max():.3f}")
 
 
if __name__ == "__main__":
    main()
 