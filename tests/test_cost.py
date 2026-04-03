"""Tests for compute_restoration_cost() — foremost/core.py lines 643-755."""

import numpy as np
import pytest

from foremost import HabitatData, compute_restoration_cost
from foremost.core import CostConfig


class TestComputeRestorationCostBasic:
    def test_returns_nonnegative(self, tiny_habitat_data, default_cost_config):
        cost = compute_restoration_cost(tiny_habitat_data, default_cost_config)
        assert (cost >= 0).all()

    def test_shape_matches_input(self, tiny_habitat_data, default_cost_config):
        cost = compute_restoration_cost(tiny_habitat_data, default_cost_config)
        assert cost.shape == tiny_habitat_data.shape

    def test_habitat_cells_have_zero_cost(self, tiny_habitat_data, default_cost_config):
        cost = compute_restoration_cost(tiny_habitat_data, default_cost_config)
        hab_mask = tiny_habitat_data.habitat == 1
        assert (cost[hab_mask] == 0.0).all()

    def test_locked_out_cells_have_zero_cost(
        self, tiny_habitat_data, default_cost_config
    ):
        cost = compute_restoration_cost(tiny_habitat_data, default_cost_config)
        assert (cost[tiny_habitat_data.locked_out] == 0.0).all()


class TestCostModelSensitivity:
    def test_higher_unit_cost_increases_total(self, tiny_habitat_data):
        """Doubling tree_unit_cost must double the total cost (noise_sigma=0)."""
        cfg_low = CostConfig(tree_unit_cost=30.0, noise_sigma=0.0)
        cfg_high = CostConfig(tree_unit_cost=60.0, noise_sigma=0.0)
        c_low = compute_restoration_cost(tiny_habitat_data, cfg_low)
        c_high = compute_restoration_cost(tiny_habitat_data, cfg_high)
        assert pytest.approx(c_high.sum(), rel=1e-6) == 2.0 * c_low.sum()

    def test_inaccessible_surcharge_increases_cost(self):
        """Inaccessible cells must cost more than accessible ones."""
        n = 4
        habitat = np.zeros((n, n), dtype=int)
        restorable = np.full((n, n), 0.5)
        accessible = np.zeros((n, n), dtype=int)
        accessible[0, 0] = 1  # only top-left is accessible
        cost_arr = np.zeros((n, n))
        data = HabitatData(
            habitat=habitat,
            restorable=restorable,
            accessible=accessible,
            cost=cost_arr,
        )
        cfg = CostConfig(inaccessible_surcharge=0.40, noise_sigma=0.0)
        cost = compute_restoration_cost(data, cfg)
        assert cost[0, 0] < cost[1, 0]

    def test_elevation_factor_increases_cost_monotonically(self):
        """Higher elevation → higher cost along each row."""
        n = 3
        habitat = np.zeros((n, n), dtype=int)
        restorable = np.ones((n, n)) * 0.5
        accessible = np.ones((n, n), dtype=int)
        cost_arr = np.zeros((n, n))
        elevation = np.array([[0, 100, 200], [0, 100, 200], [0, 100, 200]], dtype=float)
        data = HabitatData(
            habitat=habitat,
            restorable=restorable,
            accessible=accessible,
            cost=cost_arr,
            elevation=elevation,
        )
        cfg = CostConfig(elevation_slope=0.01, noise_sigma=0.0)
        cost = compute_restoration_cost(data, cfg)
        for row in range(n):
            assert cost[row, 0] <= cost[row, 1] <= cost[row, 2]

    def test_no_elevation_data_does_not_crash(self, tiny_habitat_data):
        data = HabitatData(
            habitat=tiny_habitat_data.habitat,
            restorable=tiny_habitat_data.restorable,
            accessible=tiny_habitat_data.accessible,
            cost=np.zeros(tiny_habitat_data.shape),
            elevation=None,
        )
        cost = compute_restoration_cost(data, CostConfig(noise_sigma=0.0))
        assert (cost >= 0).all()
