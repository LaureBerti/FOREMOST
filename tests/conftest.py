"""
Shared pytest fixtures for the FOREMOST test suite.

All HabitatData instances are built directly via the dataclass constructor
(not via HabitatData.synthetic(), which ignores keyword arguments and reads
ForemostConfig defaults instead).
"""

import numpy as np
import pytest

from foremost import HabitatData
from foremost.core import CostConfig


# ---------------------------------------------------------------------------
# Tiny deterministic landscape (5×5) — fast, used in all unit tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_habitat_data() -> HabitatData:
    """
    5×5 landscape with known properties:
    - 5 habitat cells (main diagonal)
    - remaining non-habitat cells are accessible and restorable
    - one locked-out non-habitat cell at (4, 0)
    - simple Gaussian elevation dome
    """
    rng = np.random.default_rng(42)
    n = 5

    habitat = np.zeros((n, n), dtype=int)
    for i in range(n):
        habitat[i, i] = 1  # diagonal pattern

    restorable = np.where(habitat == 0, 0.8, 0.0)
    accessible = np.where(habitat == 0, 1, 0).astype(int)

    locked_out = np.zeros((n, n), dtype=bool)
    locked_out[4, 0] = True  # non-habitat cell, explicitly locked out

    ri, ci = np.mgrid[0:n, 0:n]
    elevation = 200.0 * np.exp(
        -(((ri - n / 2) / (n / 3)) ** 2 + ((ci - n / 2) / (n / 3)) ** 2)
    )
    cost = np.where(habitat == 0, 5.0 + rng.uniform(0, 2, (n, n)), 0.0)
    cost[locked_out] = 0.0  # keep cost array valid

    return HabitatData(
        habitat=habitat,
        restorable=restorable,
        accessible=accessible,
        cost=cost,
        cell_area=1.0,
        locked_out=locked_out,
        elevation=elevation,
    )


# ---------------------------------------------------------------------------
# Small landscape (10×10) — for connectivity tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_habitat_data() -> HabitatData:
    """10×10 random landscape (seed=7)."""
    rng = np.random.default_rng(7)
    n = 10

    raw = rng.random((n, n))
    habitat = (raw < 0.28).astype(int)
    restorable = np.where(habitat == 0, rng.uniform(0.2, 1.0, (n, n)), 0.0)
    accessible = np.where(habitat == 0, 1, 0).astype(int)
    locked_out = (rng.random((n, n)) < 0.05) & (habitat == 0)

    ri, ci = np.mgrid[0:n, 0:n]
    elevation = 500.0 * np.exp(
        -(((ri - n / 2) / (n / 3)) ** 2 + ((ci - n / 2) / (n / 3)) ** 2)
    )
    cost = np.where(
        (habitat == 0) & (~locked_out),
        10.0 * (0.4 + rng.uniform(0, 0.3, (n, n))),
        0.0,
    )
    return HabitatData(
        habitat=habitat,
        restorable=restorable,
        accessible=accessible,
        cost=cost,
        cell_area=1.0,
        locked_out=locked_out,
        elevation=elevation,
    )


# ---------------------------------------------------------------------------
# Deterministic CostConfig (zero noise for reproducible assertions)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def default_cost_config() -> CostConfig:
    return CostConfig(
        tree_unit_cost=30.0,
        tree_spacing_m=2.0,
        cell_size_m=100.0,
        inaccessible_surcharge=0.40,
        elevation_base_m=0.0,
        elevation_slope=0.005,
        noise_sigma=0.0,
    )
