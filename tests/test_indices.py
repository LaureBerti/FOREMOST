"""Tests for mesh() and iic() — foremost/core.py lines 772-836."""

import numpy as np
import pytest

from foremost import mesh, iic


class TestMesh:
    def test_empty_habitat_returns_zero(self):
        grid = np.zeros((5, 5), dtype=int)
        assert mesh(grid, total_cells=25) == 0.0

    def test_single_patch_analytical(self):
        """
        One 3×3 patch in a 5×5 landscape.
        MESH = (1/A) * sum(a_i²) = (1/25) * 9² = 81/25 = 3.24
        """
        grid = np.zeros((5, 5), dtype=int)
        grid[1:4, 1:4] = 1  # 9-cell connected patch
        result = mesh(grid, total_cells=25, cell_area=1.0)
        assert pytest.approx(result, abs=1e-10) == 81.0 / 25.0

    def test_two_equal_patches_less_than_one_big_patch(self):
        """
        Fragmentation: two equal patches yield lower MESH than one combined patch.
        """
        grid_frag = np.zeros((10, 10), dtype=int)
        grid_frag[1:4, 1:4] = 1
        grid_frag[6:9, 6:9] = 1
        grid_whole = np.zeros((10, 10), dtype=int)
        grid_whole[1:7, 1:7] = 1
        assert mesh(grid_frag, total_cells=100) < mesh(grid_whole, total_cells=100)

    def test_cell_area_scaling(self):
        """MESH scales linearly with cell_area."""
        grid = np.zeros((4, 4), dtype=int)
        grid[1:3, 1:3] = 1
        m1 = mesh(grid, total_cells=16, cell_area=1.0)
        m2 = mesh(grid, total_cells=16, cell_area=2.0)
        assert pytest.approx(m2, rel=1e-9) == 2.0 * m1

    def test_restoration_never_decreases_mesh(self, tiny_habitat_data):
        """Adding restored cells must never decrease MESH."""
        original = tiny_habitat_data.habitat_mask.astype(int)
        restored = tiny_habitat_data.habitat_mask.astype(int).copy()
        cands = np.argwhere(tiny_habitat_data.candidate_mask)
        if len(cands) > 0:
            r, c = cands[0]
            restored[r, c] = 1
        A = int((tiny_habitat_data.habitat != -1).sum())
        assert mesh(restored, A) >= mesh(original, A)


class TestIIC:
    def test_empty_habitat_returns_zero(self):
        assert iic(np.zeros((5, 5), dtype=int), total_cells=25) == 0.0

    def test_single_patch_analytical(self):
        """
        Single patch of area a=9 in landscape A=25.
        IIC = a² / A²  (only i==j term, nl=0)
        """
        grid = np.zeros((5, 5), dtype=int)
        grid[1:4, 1:4] = 1
        a, A = 9.0, 25.0
        expected = (a * a) / (A * A)
        result = iic(grid, total_cells=25, cell_area=1.0, max_dist=3)
        assert pytest.approx(result, rel=1e-6) == expected

    def test_connected_patches_higher_iic_than_disconnected(self):
        """
        Patches within max_dist yield higher IIC than when max_dist is too small.
        """
        grid = np.zeros((10, 5), dtype=int)
        grid[1:3, 1:3] = 1
        grid[7:9, 1:3] = 1
        iic_connected = iic(grid, total_cells=50, max_dist=6)
        iic_disconnected = iic(grid, total_cells=50, max_dist=1)
        assert iic_connected > iic_disconnected

    def test_symmetry_on_transposed_grid(self):
        """IIC must be identical on a transposed landscape."""
        grid = np.zeros((8, 8), dtype=int)
        grid[0:3, 0:2] = 1
        grid[5:8, 5:8] = 1
        assert pytest.approx(iic(grid, total_cells=64), rel=1e-9) == iic(
            grid.T, total_cells=64
        )
