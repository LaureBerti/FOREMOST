"""
cost_model.py — Per-cell restoration cost estimation.

Two public functions:
  compute_cell_cost()  — scalar estimate; used for the manual-paint spin preview.
  compute_grid_cost()  — vectorized N×N array matching annotation.py formula:

    n_trees   = cell_size_m² / spacing_m²
    base      = n_trees × tree_unit_cost
    elev_m    = elev_norm × max_elevation_m          (N×N)
    elev_f    = 1 + elevation_slope × max(0, elev_m − elevation_base_m)
    road_f    = 1 + road_penalty_slope  × max(0, dist_road  − road_ref_dist_m)
    water_f   = 1 + water_penalty_slope × max(0, dist_water − water_ref_dist_m)
    access_f  = 1 + inaccessible_surcharge  if dist_road > road_ref_dist_m else 1
    noise     = clip(1 + N(0, σ²), 0.5, 2.5)          (reproducible seed=0)
    cost[r,c] = base × access_f × elev_f × road_f × water_f × noise  (CLASS_RA only)
"""

import math
import random

import numpy as np


# ── scalar preview ─────────────────────────────────────────────────────────────

def compute_cell_cost(
    cell_size_m: float,
    *,
    tree_unit_cost:         float = 15.0,
    tree_spacing_m:         float = 2.5,
    is_accessible:          bool  = True,
    inaccessible_surcharge: float = 0.40,
    elev_norm:              float = 0.0,
    elevation_base_m:       float = 0.0,
    elevation_slope:        float = 0.005,
    max_elevation_m:        float = 1000.0,
    road_dist_m:            float = 0.0,
    road_ref_dist_m:        float = 500.0,
    road_penalty_slope:     float = 0.0002,
    water_dist_m:           float = 0.0,
    water_ref_dist_m:       float = 200.0,
    water_penalty_slope:    float = 0.0001,
    noise_sigma:            float = 0.0,   # 0 → deterministic preview
    rng_seed:               int | None = None,
    **_ignored,
) -> float:
    """Return a single-cell cost estimate (used for the paint-brush preview spin)."""
    if rng_seed is not None:
        random.seed(rng_seed)

    n_trees = cell_size_m ** 2 / max(tree_spacing_m ** 2, 1e-9)
    base    = n_trees * tree_unit_cost

    access_f = (1.0 + inaccessible_surcharge) if not is_accessible else 1.0

    elev_m   = elev_norm * max_elevation_m
    elev_f   = 1.0 + elevation_slope * max(0.0, elev_m - elevation_base_m)

    road_f   = 1.0 + road_penalty_slope  * max(0.0, road_dist_m  - road_ref_dist_m)
    water_f  = 1.0 + water_penalty_slope * max(0.0, water_dist_m - water_ref_dist_m)

    if noise_sigma > 0:
        noise = max(0.5, min(2.5, 1.0 + random.gauss(0.0, noise_sigma)))
    else:
        noise = 1.0

    return base * access_f * elev_f * road_f * water_f * noise


# ── vectorized grid formula (matches annotation.py _fill_computed_costs) ───────

def compute_grid_cost(
    N: int,
    cell_size_m: float,
    ra_mask: "np.ndarray",
    *,
    elev_norm:              "np.ndarray | None" = None,
    dist_road:              "np.ndarray | None" = None,
    dist_water:             "np.ndarray | None" = None,
    tree_unit_cost:         float = 15.0,
    tree_spacing_m:         float = 2.5,
    max_elevation_m:        float = 1000.0,
    elevation_base_m:       float = 0.0,
    elevation_slope:        float = 0.005,
    inaccessible_surcharge: float = 0.40,
    road_ref_dist_m:        float = 500.0,
    road_penalty_slope:     float = 0.0002,
    water_ref_dist_m:       float = 200.0,
    water_penalty_slope:    float = 0.0001,
    noise_sigma:            float = 0.05,
    **_ignored,
) -> "np.ndarray":
    """
    Vectorized cost computation over the full N×N grid.

    Parameters
    ----------
    ra_mask    : N×N bool — cost is written only where True
    elev_norm  : N×N float64 in [0, 1]; zeros used if None
    dist_road  : N×N float64 metres to nearest road point; zeros if None
    dist_water : N×N float64 metres to nearest water point; zeros if None

    Returns N×N float64; non-RA cells are 0.
    """
    zeros = np.zeros((N, N), dtype=np.float64)

    n_trees   = cell_size_m ** 2 / max(tree_spacing_m ** 2, 1e-9)
    base_cost = n_trees * tree_unit_cost

    elev_m      = (elev_norm if elev_norm is not None else zeros) * max_elevation_m
    elev_factor = 1.0 + elevation_slope * np.maximum(0.0, elev_m - elevation_base_m)

    dr            = dist_road  if dist_road  is not None else zeros
    access_factor = np.where(dr <= road_ref_dist_m, 1.0, 1.0 + inaccessible_surcharge)
    road_factor   = 1.0 + road_penalty_slope  * np.maximum(0.0, dr - road_ref_dist_m)

    dw           = dist_water if dist_water is not None else zeros
    water_factor = 1.0 + water_penalty_slope * np.maximum(0.0, dw - water_ref_dist_m)

    rng   = np.random.default_rng(0)
    noise = np.clip(1.0 + rng.normal(0.0, noise_sigma, (N, N)), 0.5, 2.5)

    cost_grid = np.clip(
        base_cost * access_factor * elev_factor * road_factor * water_factor * noise,
        0.0, None,
    )

    result = np.zeros((N, N), dtype=np.float64)
    result[ra_mask] = cost_grid[ra_mask]
    return result


# ── utility ────────────────────────────────────────────────────────────────────

def cost_per_ha(total_cost: float, n_cells: int, cell_size_m: float) -> float:
    """Normalise absolute cost to cost/ha so values are resolution-invariant."""
    phys_ha = n_cells * cell_size_m ** 2 / 1e4
    if phys_ha <= 0:
        return float("nan")
    return total_cost / phys_ha
