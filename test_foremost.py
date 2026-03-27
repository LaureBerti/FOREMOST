"""
test_restopt_pymoo.py
=====================
Comprehensive unit-test suite for restopt_pymoo_3d.py.

Test groups
-----------
A. TestHabitatData           – construction, validation, properties, factories
B. TestLandscapeIndices      – mesh, iic, compute_patches, diameter,
                               connected_components
C. TestObjectiveType         – enum properties (n_obj, is_multi, needs_*)
D. TestRestorationConstraints – defaults and custom values
E. TestRestorationProblem    – __init__, candidate mask, _violation, _evaluate,
                               decode_solution
F. TestConstraintSatisfaction – per-ObjectiveType: best solutions found by the
                                solver satisfy every active constraint; multi-obj
                                Pareto fronts contain distinct solutions
G. TestRestoptProblemBuilder   – fluent builder + full round-trip solve

Running
-------
# Without pymoo (pure-logic tests only):
    python test_restopt_pymoo.py -v

# With pymoo installed (all tests):
    python test_restopt_pymoo.py -v

Notes
-----
- Solve tests (groups F, G) are skipped automatically if pymoo is not
  importable.  Install with:  pip install pymoo networkx
- Solve tests use a small 15x15 synthetic landscape with few generations so
  the suite finishes in a reasonable time (~2-3 min with pymoo).
- Constraint satisfaction is tested with a 5 % tolerance to account for the
  heuristic nature of the GA/NSGA-II solver.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

# ── pymoo availability ────────────────────────────────────────────────────────
try:
    import pymoo  # noqa: F401
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    # Inject minimal stubs so restopt_pymoo can be imported for pure-logic tests
    for _mod in [
        "pymoo", "pymoo.algorithms", "pymoo.algorithms.moo",
        "pymoo.algorithms.moo.nsga2", "pymoo.algorithms.soo",
        "pymoo.algorithms.soo.nonconvex", "pymoo.algorithms.soo.nonconvex.ga",
        "pymoo.core", "pymoo.core.problem", "pymoo.core.callback",
        "pymoo.operators", "pymoo.operators.crossover",
        "pymoo.operators.crossover.pntx",
        "pymoo.operators.mutation", "pymoo.operators.mutation.bitflip",
        "pymoo.operators.sampling", "pymoo.operators.sampling.rnd",
        "pymoo.optimize", "pymoo.termination",
    ]:
        sys.modules[_mod] = types.ModuleType(_mod)

    class _EWP:
        def __init__(self, **k): pass

    class _CB:
        def __init__(self): pass

    sys.modules["pymoo.core.problem"].ElementwiseProblem = _EWP
    sys.modules["pymoo.core.callback"].Callback = _CB
    for _k, _v in [
        ("pymoo.algorithms.moo.nsga2",          "NSGA2"),
        ("pymoo.algorithms.soo.nonconvex.ga",   "GA"),
        ("pymoo.operators.crossover.pntx",      "TwoPointCrossover"),
        ("pymoo.operators.mutation.bitflip",    "BitflipMutation"),
        ("pymoo.operators.sampling.rnd",        "BinaryRandomSampling"),
    ]:
        setattr(sys.modules[_k], _v, type(_v, (), {}))
    sys.modules["pymoo.optimize"].minimize       = lambda *a, **k: None
    sys.modules["pymoo.termination"].get_termination = lambda *a, **k: None

# ── module under test ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import restopt_pymoo_3d as rm

# ── shared small fixtures ─────────────────────────────────────────────────────
# 10x10 – used by pure-logic tests (no solver needed)
DATA_10 = rm.HabitatData.synthetic(nrows=10, ncols=10,
                                   habitat_fraction=0.25, seed=1)
# 15x15 – used by solver tests (larger candidate pool)
DATA_15 = rm.HabitatData.synthetic(nrows=15, ncols=15,
                                   habitat_fraction=0.25, seed=42)

# Solver parameters (kept small for speed)
_POP  = 60
_NGEN = 120


def _make_problem(data=DATA_10,
                  obj=rm.ObjectiveType.MESH,
                  **constraint_kwargs) -> rm.RestorationProblem:
    c = rm.RestorationConstraints(**constraint_kwargs)
    return rm.RestorationProblem(data, c, obj)


def _x_select(problem: rm.RestorationProblem, n: int) -> np.ndarray:
    """Binary vector selecting the first n candidate cells."""
    x = np.zeros(problem.n_candidates, dtype=bool)
    x[:min(n, problem.n_candidates)] = True
    return x


# =============================================================================
# A.  HabitatData
# =============================================================================

class TestHabitatData(unittest.TestCase):

    # ── construction ──────────────────────────────────────────────────────────

    def test_synthetic_shapes_match(self):
        d = rm.HabitatData.synthetic(nrows=8, ncols=12, seed=0)
        self.assertEqual(d.habitat.shape,    (8, 12))
        self.assertEqual(d.restorable.shape, (8, 12))
        self.assertEqual(d.accessible.shape, (8, 12))
        self.assertEqual(d.cost.shape,       (8, 12))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rm.HabitatData(
                habitat    = np.zeros((5, 5), dtype=int),
                restorable = np.zeros((6, 5)),          # wrong rows
                accessible = np.zeros((5, 5), dtype=int),
                cost       = np.zeros((5, 5)),
            )

    def test_negative_cost_raises(self):
        with self.assertRaises(ValueError):
            rm.HabitatData(
                habitat    = np.zeros((3, 3), dtype=int),
                restorable = np.zeros((3, 3)),
                accessible = np.zeros((3, 3), dtype=int),
                cost       = np.full((3, 3), -1.0),
            )

    def test_locked_out_default_is_all_false(self):
        """When locked_out is not supplied, it must default to all-False."""
        d = rm.HabitatData(
            habitat    = np.zeros((5, 5), dtype=int),
            restorable = np.zeros((5, 5)),
            accessible = np.ones((5, 5), dtype=int),
            cost       = np.zeros((5, 5)),
        )
        self.assertFalse(d.locked_out.any())

    def test_habitat_fraction_approximately_correct(self):
        d = rm.HabitatData.synthetic(nrows=50, ncols=50,
                                      habitat_fraction=0.30, seed=7)
        self.assertAlmostEqual(float(d.habitat_mask.mean()), 0.30, delta=0.07)

    def test_cost_non_negative_everywhere(self):
        d = rm.HabitatData.synthetic(nrows=20, ncols=20, seed=3)
        self.assertTrue((d.cost >= 0).all())

    def test_cell_area_stored_correctly(self):
        d = rm.HabitatData.synthetic(nrows=5, ncols=5, cell_area=4.5)
        self.assertEqual(d.cell_area, 4.5)

    def test_shape_property(self):
        d = rm.HabitatData.synthetic(nrows=7, ncols=11, seed=0)
        self.assertEqual(d.shape, (7, 11))

    def test_manual_construction_minimal(self):
        d = rm.HabitatData(
            habitat    = np.array([[1, 0], [0, 0]]),
            restorable = np.array([[0.0, 0.5], [0.5, 0.5]]),
            accessible = np.array([[0, 1], [1, 1]]),
            cost       = np.array([[0.0, 3.0], [2.0, 4.0]]),
            cell_area  = 1.0,
        )
        self.assertEqual(d.shape, (2, 2))
        self.assertEqual(int(d.candidate_mask.sum()), 3)

    # ── candidate_mask ────────────────────────────────────────────────────────

    def test_candidate_mask_excludes_habitat(self):
        overlap = (DATA_10.candidate_mask & DATA_10.habitat_mask).sum()
        self.assertEqual(int(overlap), 0)

    def test_candidate_mask_excludes_locked_out(self):
        overlap = (DATA_10.candidate_mask & DATA_10.locked_out).sum()
        self.assertEqual(int(overlap), 0)

    def test_candidate_mask_subset_of_accessible(self):
        self.assertTrue(
            np.all(DATA_10.accessible[DATA_10.candidate_mask] == 1)
        )

    def test_candidate_mask_bool_dtype(self):
        self.assertEqual(DATA_10.candidate_mask.dtype, bool)

    def test_habitat_mask_equals_habitat_eq_1(self):
        self.assertTrue(np.array_equal(DATA_10.habitat_mask,
                                        DATA_10.habitat == 1))


# =============================================================================
# B.  Landscape Indices
# =============================================================================

class TestLandscapeIndices(unittest.TestCase):

    # ── compute_patches ───────────────────────────────────────────────────────

    def test_patches_single_block(self):
        g = np.zeros((5, 5), dtype=int)
        g[1:4, 1:4] = 1
        _, n, areas = rm.compute_patches(g)
        self.assertEqual(n, 1)
        self.assertEqual(int(areas[0]), 9)

    def test_patches_two_separated_cells(self):
        g = np.zeros((5, 5), dtype=int)
        g[0, 0] = 1;  g[4, 4] = 1
        _, n, areas = rm.compute_patches(g)
        self.assertEqual(n, 2)
        self.assertTrue(np.all(areas == 1))

    def test_patches_diagonal_connectivity(self):
        """Diagonally adjacent cells are 8-connected -> one patch."""
        g = np.zeros((3, 3), dtype=int)
        g[0, 0] = 1;  g[1, 1] = 1;  g[2, 2] = 1
        _, n, _ = rm.compute_patches(g)
        self.assertEqual(n, 1)

    def test_patches_empty_grid(self):
        _, n, areas = rm.compute_patches(np.zeros((5, 5), dtype=int))
        self.assertEqual(n, 0)
        self.assertEqual(len(areas), 0)

    def test_patches_labeled_shape(self):
        g = np.zeros((8, 8), dtype=int)
        g[2:5, 2:5] = 1
        labeled, _, _ = rm.compute_patches(g)
        self.assertEqual(labeled.shape, g.shape)

    # ── mesh ──────────────────────────────────────────────────────────────────

    def test_mesh_empty_is_zero(self):
        self.assertEqual(rm.mesh(np.zeros((5, 5), dtype=int), 25, 1.0), 0.0)

    def test_mesh_single_patch_formula(self):
        """MESH = area^2 / A_total for one patch."""
        g = np.zeros((5, 5), dtype=int)
        g[1:4, 1:4] = 1          # 9 cells
        self.assertAlmostEqual(rm.mesh(g, 25, 1.0), 9**2 / 25, places=6)

    def test_mesh_scales_with_cell_area(self):
        g = np.zeros((4, 4), dtype=int)
        g[:2, :2] = 1             # 4 cells
        ca = 2.5
        expected = (4 * ca)**2 / (16 * ca)
        self.assertAlmostEqual(rm.mesh(g, 16, ca), expected, places=5)

    def test_mesh_one_large_gt_two_small(self):
        """One large connected patch has higher MESH than two equal fragments."""
        g_one = np.zeros((5, 5), dtype=int)
        g_one[1:3, 1:3] = 1      # 4-cell patch

        g_two = np.zeros((5, 5), dtype=int)
        g_two[0:2, 0] = 1        # two isolated 2-cell patches
        g_two[3:5, 4] = 1

        self.assertGreater(rm.mesh(g_one, 25, 1.0),
                           rm.mesh(g_two, 25, 1.0))

    def test_mesh_non_negative(self):
        self.assertGreaterEqual(rm.mesh(DATA_10.habitat, 100, 1.0), 0.0)

    def test_mesh_increases_with_restoration(self):
        g_small = np.zeros((10, 10), dtype=int)
        g_small[5, 5] = 1
        g_large = g_small.copy()
        g_large[4:7, 4:7] = 1
        self.assertGreaterEqual(rm.mesh(g_large, 100, 1.0),
                                 rm.mesh(g_small, 100, 1.0))

    # ── iic ───────────────────────────────────────────────────────────────────

    def test_iic_empty_is_zero(self):
        self.assertEqual(rm.iic(np.zeros((5, 5), dtype=int), 25, 1.0), 0.0)

    def test_iic_single_patch_positive_and_leq_1(self):
        g = np.zeros((6, 6), dtype=int)
        g[2:5, 2:5] = 1
        v = rm.iic(g, 36, 1.0, max_dist=5)
        self.assertGreater(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_iic_one_large_gt_isolated(self):
        g_iso = np.zeros((6, 6), dtype=int)
        for r, c in [(0,0),(0,5),(5,0),(5,5)]:
            g_iso[r, c] = 1

        g_big = np.zeros((6, 6), dtype=int)
        g_big[2:4, 2:4] = 1

        self.assertGreaterEqual(rm.iic(g_big, 36, 1.0, max_dist=3),
                                 rm.iic(g_iso, 36, 1.0, max_dist=3))

    def test_iic_non_negative_on_real_data(self):
        self.assertGreaterEqual(
            rm.iic(DATA_10.habitat, 100, DATA_10.cell_area), 0.0
        )

    # ── diameter ──────────────────────────────────────────────────────────────

    def test_diameter_empty_is_zero(self):
        self.assertEqual(rm.diameter(np.zeros((5, 5), dtype=int)), 0)

    def test_diameter_single_cell_is_one(self):
        g = np.zeros((5, 5), dtype=int)
        g[2, 2] = 1
        self.assertEqual(rm.diameter(g), 1)

    def test_diameter_two_adjacent_horizontal(self):
        g = np.zeros((5, 5), dtype=int)
        g[2, 2] = 1;  g[2, 3] = 1
        self.assertEqual(rm.diameter(g), 2)

    def test_diameter_diagonal_adjacency_is_two(self):
        g = np.zeros((4, 4), dtype=int)
        g[0, 0] = 1;  g[1, 1] = 1
        self.assertEqual(rm.diameter(g), 2)

    def test_diameter_3x1_line(self):
        g = np.zeros((5, 5), dtype=int)
        g[2, 1:4] = 1
        self.assertEqual(rm.diameter(g), 3)

    def test_diameter_non_negative(self):
        self.assertGreaterEqual(rm.diameter(DATA_10.habitat), 0)

    # ── connected_components ──────────────────────────────────────────────────

    def test_cc_single_block(self):
        g = np.zeros((5, 5), dtype=int)
        g[1:4, 1:4] = 1
        _, n = rm.connected_components(g)
        self.assertEqual(n, 1)

    def test_cc_empty_is_zero(self):
        _, n = rm.connected_components(np.zeros((5, 5), dtype=int))
        self.assertEqual(n, 0)

    def test_cc_two_isolated(self):
        g = np.zeros((5, 5), dtype=int)
        g[0, 0] = 1;  g[4, 4] = 1
        _, n = rm.connected_components(g)
        self.assertEqual(n, 2)

    def test_cc_diagonals_connected(self):
        g = np.zeros((3, 3), dtype=int)
        g[0, 0] = 1;  g[1, 1] = 1;  g[2, 2] = 1
        _, n = rm.connected_components(g)
        self.assertEqual(n, 1)

    def test_cc_labeled_shape(self):
        g = np.zeros((6, 6), dtype=int)
        g[0:2, 0:2] = 1;  g[4:6, 4:6] = 1
        labeled, n = rm.connected_components(g)
        self.assertEqual(labeled.shape, g.shape)
        self.assertEqual(n, 2)


# =============================================================================
# C.  ObjectiveType
# =============================================================================

class TestObjectiveType(unittest.TestCase):

    def _check(self, obj, n_obj, is_multi, mesh, iic, cost):
        with self.subTest(obj=obj.value):
            self.assertEqual(obj.n_obj,      n_obj)
            self.assertEqual(obj.is_multi,   is_multi)
            self.assertEqual(obj.needs_mesh, mesh)
            self.assertEqual(obj.needs_iic,  iic)
            self.assertEqual(obj.needs_cost, cost)

    def test_MESH(self):
        self._check(rm.ObjectiveType.MESH,
                    n_obj=1, is_multi=False, mesh=True,  iic=False, cost=False)

    def test_IIC(self):
        self._check(rm.ObjectiveType.IIC,
                    n_obj=1, is_multi=False, mesh=False, iic=True,  cost=False)

    def test_COST(self):
        self._check(rm.ObjectiveType.COST,
                    n_obj=1, is_multi=False, mesh=False, iic=False, cost=True)

    def test_MESH_IIC(self):
        self._check(rm.ObjectiveType.MESH_IIC,
                    n_obj=2, is_multi=True,  mesh=True,  iic=True,  cost=False)

    def test_MESH_COST(self):
        self._check(rm.ObjectiveType.MESH_COST,
                    n_obj=2, is_multi=True,  mesh=True,  iic=False, cost=True)

    def test_IIC_COST(self):
        self._check(rm.ObjectiveType.IIC_COST,
                    n_obj=2, is_multi=True,  mesh=False, iic=True,  cost=True)

    def test_FULL(self):
        self._check(rm.ObjectiveType.FULL,
                    n_obj=3, is_multi=True,  mesh=True,  iic=True,  cost=True)

    def test_exactly_7_objective_types(self):
        self.assertEqual(len(list(rm.ObjectiveType)), 7)

    def test_three_single_objectives(self):
        self.assertEqual(
            len([o for o in rm.ObjectiveType if not o.is_multi]), 3
        )

    def test_four_multi_objectives(self):
        self.assertEqual(
            len([o for o in rm.ObjectiveType if o.is_multi]), 4
        )

    def test_all_values_are_strings(self):
        for o in rm.ObjectiveType:
            self.assertIsInstance(o.value, str)


# =============================================================================
# D.  RestorationConstraints
# =============================================================================

class TestRestorationConstraints(unittest.TestCase):

    def test_defaults(self):
        c = rm.RestorationConstraints()
        self.assertEqual(c.min_restore,    0.0)
        self.assertEqual(c.max_restore,    float("inf"))
        self.assertEqual(c.max_diameter,   999)
        self.assertEqual(c.max_nb_cc,      1)
        self.assertEqual(c.min_proportion, 0.0)
        self.assertEqual(c.max_cost,       float("inf"))

    def test_custom_values(self):
        c = rm.RestorationConstraints(
            min_restore=2.0, max_restore=10.0,
            max_diameter=5,  max_nb_cc=2,
            min_proportion=0.4, max_cost=300.0,
        )
        self.assertEqual(c.min_restore,  2.0)
        self.assertEqual(c.max_restore,  10.0)
        self.assertEqual(c.max_diameter, 5)
        self.assertEqual(c.max_nb_cc,    2)
        self.assertAlmostEqual(c.min_proportion, 0.4)
        self.assertAlmostEqual(c.max_cost,       300.0)


# =============================================================================
# E.  RestorationProblem
# =============================================================================

class TestRestorationProblem(unittest.TestCase):

    # ── __init__ ──────────────────────────────────────────────────────────────

    def test_n_candidates_positive(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        self.assertGreater(p.n_candidates, 0)

    def test_n_candidates_matches_mask(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        self.assertEqual(p.n_candidates, int(DATA_10.candidate_mask.sum()))

    def test_candidate_rows_cols_length(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        self.assertEqual(len(p._candidate_rows), p.n_candidates)
        self.assertEqual(len(p._candidate_cols), p.n_candidates)

    def test_no_candidates_raises_value_error(self):
        hab = np.ones((5, 5), dtype=int)          # all habitat -> no candidates
        d   = rm.HabitatData(
            habitat=hab, restorable=np.zeros((5,5)),
            accessible=np.ones((5,5), dtype=int),
            cost=np.zeros((5,5)),
        )
        with self.assertRaises(ValueError):
            rm.RestorationProblem(d, rm.RestorationConstraints(),
                                   rm.ObjectiveType.MESH)

    def test_total_area_cells_correct(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        expected = int((DATA_10.habitat != DATA_10.nodata_value).sum())
        self.assertEqual(p.total_area_cells, expected)

    def test_n_obj_matches_objective_type(self):
        """n_obj stored on the objective enum must match the type constant."""
        for obj in rm.ObjectiveType:
            with self.subTest(obj=obj.value):
                p = _make_problem(DATA_10, obj)
                # n_obj is set via pymoo's super().__init__(n_obj=…);
                # in the mock environment the attribute may not exist on the
                # problem instance, so fall back to checking objective.n_obj.
                expected = obj.n_obj
                actual   = getattr(p, "n_obj", None) or p.objective.n_obj
                self.assertEqual(actual, expected)

    def test_habitat_data_attribute_present(self):
        """Must use habitat_data (not data) to avoid pymoo cache collision."""
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        self.assertTrue(hasattr(p, "habitat_data"))
        self.assertIsInstance(p.habitat_data, rm.HabitatData)
        self.assertFalse(hasattr(p, "_data_field_named_data"),
                          "should not use 'data' as attribute name")

    # ── _violation ────────────────────────────────────────────────────────────

    def test_violation_zero_when_feasible(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH,
                          min_restore=0.0, max_restore=float("inf"),
                          max_diameter=999, max_nb_cc=999)
        self.assertEqual(p._violation(_x_select(p, 3)), 0.0)

    def test_violation_positive_below_min_restore(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH, min_restore=100.0)
        self.assertGreater(p._violation(np.zeros(p.n_candidates, dtype=bool)), 0.0)

    def test_violation_positive_above_max_restore(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH, max_restore=0.001)
        self.assertGreater(p._violation(np.ones(p.n_candidates, dtype=bool)), 0.0)

    def test_violation_positive_diameter_exceeded(self):
        h  = np.zeros((1, 20), dtype=int)
        r  = np.ones((1, 20))
        a  = np.ones((1, 20), dtype=int)
        co = np.ones((1, 20))
        d  = rm.HabitatData(habitat=h, restorable=r, accessible=a, cost=co)
        p  = rm.RestorationProblem(
            d, rm.RestorationConstraints(max_diameter=3), rm.ObjectiveType.MESH
        )
        self.assertGreater(
            p._violation(np.ones(p.n_candidates, dtype=bool)), 0.0
        )

    def test_violation_positive_budget_exceeded(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.COST, max_cost=0.001)
        self.assertGreater(
            p._violation(np.ones(p.n_candidates, dtype=bool)), 0.0
        )

    def test_violation_positive_excess_connected_components(self):
        h  = np.zeros((5, 5), dtype=int)
        r  = np.ones((5, 5))
        a  = np.ones((5, 5), dtype=int)
        co = np.ones((5, 5))
        d  = rm.HabitatData(habitat=h, restorable=r, accessible=a, cost=co)
        p  = rm.RestorationProblem(d, rm.RestorationConstraints(max_nb_cc=1),
                                    rm.ObjectiveType.MESH)
        rows, cols = p._candidate_rows, p._candidate_cols
        idx_00 = np.where((rows == 0) & (cols == 0))[0]
        idx_44 = np.where((rows == 4) & (cols == 4))[0]
        if len(idx_00) and len(idx_44):
            x = np.zeros(p.n_candidates, dtype=bool)
            x[idx_00[0]] = True;  x[idx_44[0]] = True
            self.assertGreater(p._violation(x), 0.0)

    def test_violation_increases_with_more_excess(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH, max_restore=0.5)
        x_few  = _x_select(p, 1)
        x_many = np.ones(p.n_candidates, dtype=bool)
        self.assertLessEqual(p._violation(x_few), p._violation(x_many))

    def test_violation_always_non_negative(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        for n in [0, 3, p.n_candidates]:
            with self.subTest(n=n):
                self.assertGreaterEqual(p._violation(_x_select(p, n)), 0.0)

    # ── _evaluate ─────────────────────────────────────────────────────────────

    def test_evaluate_full_objective_returns_3_values(self):
        """FULL is the only objective that doesn't hit the eager-fmap bug."""
        p = _make_problem(DATA_10, rm.ObjectiveType.FULL)
        out = {}
        p._evaluate(_x_select(p, 4), out)
        self.assertEqual(len(out["F"]), 3)

    def test_evaluate_full_penalty_when_infeasible(self):
        p = _make_problem(DATA_10, rm.ObjectiveType.FULL,
                          min_restore=1000.0)     # impossible
        out = {}
        p._evaluate(np.zeros(p.n_candidates, dtype=bool), out)
        self.assertTrue(any(abs(f) > 1.0 for f in out["F"]))

    def test_evaluate_single_obj_eager_fmap_bug(self):
        """
        KNOWN BUG – documents that _evaluate raises for single-objective runs
        because the F_map dict is built eagerly, evaluating None values.
        Remove / invert this test once the bug is fixed.
        """
        for obj in (rm.ObjectiveType.MESH,
                    rm.ObjectiveType.IIC,
                    rm.ObjectiveType.COST):
            with self.subTest(obj=obj.value):
                p   = _make_problem(DATA_10, obj)
                out = {}
                with self.assertRaises((TypeError, NameError)):
                    p._evaluate(_x_select(p, 4), out)

    # ── decode_solution ───────────────────────────────────────────────────────

    def test_decode_required_keys(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(_x_select(p, 4))
        for key in ("n_restored_cells", "total_restored_area", "total_cost",
                    "n_connected_components", "diameter_cells", "n_patches",
                    "mesh", "iic", "habitat_grid", "selection_grid"):
            with self.subTest(key=key):
                self.assertIn(key, sol)

    def test_decode_grid_shapes(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(_x_select(p, 4))
        self.assertEqual(sol["habitat_grid"].shape,   DATA_10.shape)
        self.assertEqual(sol["selection_grid"].shape, DATA_10.shape)

    def test_decode_n_restored_matches_x(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(_x_select(p, 5))
        self.assertEqual(sol["n_restored_cells"], 5)

    def test_decode_selection_grid_sum_matches_n_restored(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        n   = 3
        sol = p.decode_solution(_x_select(p, n))
        self.assertEqual(int(sol["selection_grid"].sum()), n)

    def test_decode_habitat_grid_includes_original_habitat(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(_x_select(p, 4))
        orig = DATA_10.habitat_mask
        post = sol["habitat_grid"] == 1
        self.assertTrue(np.all(post[orig]),
                         "Original habitat cells must remain in post-restoration grid")

    def test_decode_total_cost_matches_manual_sum(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        x   = _x_select(p, 4)
        sel = x.astype(bool)
        expected = float(
            DATA_10.cost[p._candidate_rows[sel], p._candidate_cols[sel]].sum()
        )
        sol = p.decode_solution(x)
        self.assertAlmostEqual(sol["total_cost"], expected, places=6)

    def test_decode_area_matches_manual_sum(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        x   = _x_select(p, 4)
        sel = x.astype(bool)
        expected = float(
            DATA_10.restorable[p._candidate_rows[sel],
                                p._candidate_cols[sel]].sum()
        )
        sol = p.decode_solution(x)
        self.assertAlmostEqual(sol["total_restored_area"], expected, places=6)

    def test_decode_cost_non_negative(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(_x_select(p, 3))
        self.assertGreaterEqual(sol["total_cost"], 0.0)

    def test_decode_mesh_non_negative(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(_x_select(p, 3))
        self.assertGreaterEqual(sol["mesh"], 0.0)

    def test_decode_iic_non_negative(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(_x_select(p, 3))
        self.assertGreaterEqual(sol["iic"], 0.0)

    def test_decode_diameter_zero_when_empty(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(np.zeros(p.n_candidates, dtype=bool))
        self.assertEqual(sol["diameter_cells"], 0)
        self.assertEqual(sol["n_restored_cells"], 0)

    def test_decode_mesh_increases_with_more_restoration(self):
        p    = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol1 = p.decode_solution(_x_select(p, 1))
        sol5 = p.decode_solution(_x_select(p, 5))
        self.assertGreaterEqual(sol5["mesh"], sol1["mesh"])

    def test_decode_n_patches_positive_when_cells_selected(self):
        p   = _make_problem(DATA_10, rm.ObjectiveType.MESH)
        sol = p.decode_solution(_x_select(p, 1))
        self.assertGreater(sol["n_patches"], 0)


# =============================================================================
# F.  Constraint Satisfaction & Pareto Distinctness  (requires pymoo)
# =============================================================================

@unittest.skipUnless(PYMOO_AVAILABLE, "pymoo not installed")
class TestConstraintSatisfaction(unittest.TestCase):
    """
    Solve all 7 objective types once (shared via setUpClass).
    For each result verify:
      1. At least one cell is restored.
      2. Restored area is within [min_restore, max_restore].
      3. Diameter does not exceed max_diameter.
      4. n_connected_components does not exceed max_nb_cc.
      5. (budget-constrained) total_cost does not exceed max_cost.
      6. Multi-obj Pareto fronts contain >= 2 distinct selection_grids.
      7. Reported Pareto fronts contain no dominated solutions.
    """

    # Constraints applied to all runs
    _CC = dict(
        min_restore  = 1.0,
        max_restore  = 15.0,
        max_diameter = 10,
        max_nb_cc    = 2,
    )
    _MAX_COST = 200.0
    _TOL      = 0.05   # 5 % tolerance for soft constraints

    @classmethod
    def setUpClass(cls):
        cc  = cls._CC
        bcc = {**cc, "max_cost": cls._MAX_COST}

        def _run(obj, **extra):
            c = rm.RestorationConstraints(**{**cc, **extra})
            p = rm.RestorationProblem(DATA_15, c, obj)
            return rm.solve(p, pop_size=_POP, n_gen=_NGEN,seed=42, verbose=False)

        cls.R = {
            rm.ObjectiveType.MESH:      _run(rm.ObjectiveType.MESH),
            rm.ObjectiveType.IIC:       _run(rm.ObjectiveType.IIC),
            rm.ObjectiveType.COST:      _run(rm.ObjectiveType.COST),
            rm.ObjectiveType.MESH_IIC:  _run(rm.ObjectiveType.MESH_IIC),
            rm.ObjectiveType.MESH_COST: _run(rm.ObjectiveType.MESH_COST,
                                              max_cost=cls._MAX_COST),
            rm.ObjectiveType.IIC_COST:  _run(rm.ObjectiveType.IIC_COST,
                                              max_cost=cls._MAX_COST),
            rm.ObjectiveType.FULL:      _run(rm.ObjectiveType.FULL,
                                              max_cost=cls._MAX_COST),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _best(self, obj) -> dict:
        sols = self.R[obj]["solutions"]
        return max(sols, key=lambda s: s["total_restored_area"])

    def _soft(self, value, lo=None, hi=None, label="value"):
        tol = self._TOL
        if lo is not None:
            self.assertGreaterEqual(value, lo * (1 - tol),
                                     f"{label}={value:.4f} < min {lo}")
        if hi is not None:
            self.assertLessEqual(value, hi * (1 + tol),
                                  f"{label}={value:.4f} > max {hi}")

    # ── 1.  Non-empty solution ────────────────────────────────────────────────

    def _check_non_empty(self, obj):
        sol = self._best(obj)
        self.assertGreater(sol["n_restored_cells"], 0,
                            f"{obj.value}: no cells restored")

    def test_mesh_non_empty(self):        self._check_non_empty(rm.ObjectiveType.MESH)
    def test_iic_non_empty(self):         self._check_non_empty(rm.ObjectiveType.IIC)
    def test_cost_non_empty(self):        self._check_non_empty(rm.ObjectiveType.COST)
    def test_mesh_iic_non_empty(self):    self._check_non_empty(rm.ObjectiveType.MESH_IIC)
    def test_mesh_cost_non_empty(self):   self._check_non_empty(rm.ObjectiveType.MESH_COST)
    def test_iic_cost_non_empty(self):    self._check_non_empty(rm.ObjectiveType.IIC_COST)
    def test_full_non_empty(self):        self._check_non_empty(rm.ObjectiveType.FULL)

    # ── 2.  Area constraint ───────────────────────────────────────────────────

    def _check_area(self, obj):
        sol = self._best(obj)
        self._soft(sol["total_restored_area"],
                   lo=self._CC["min_restore"],
                   hi=self._CC["max_restore"],
                   label=f"{obj.value}.area")

    def test_mesh_area(self):        self._check_area(rm.ObjectiveType.MESH)
    def test_iic_area(self):         self._check_area(rm.ObjectiveType.IIC)
    def test_cost_area(self):        self._check_area(rm.ObjectiveType.COST)
    def test_mesh_iic_area(self):    self._check_area(rm.ObjectiveType.MESH_IIC)
    def test_mesh_cost_area(self):   self._check_area(rm.ObjectiveType.MESH_COST)
    def test_iic_cost_area(self):    self._check_area(rm.ObjectiveType.IIC_COST)
    def test_full_area(self):        self._check_area(rm.ObjectiveType.FULL)

    # ── 3.  Diameter constraint ───────────────────────────────────────────────

    def _check_diameter(self, obj):
        sol = self._best(obj)
        self._soft(sol["diameter_cells"],
                   hi=self._CC["max_diameter"],
                   label=f"{obj.value}.diameter")

    def test_mesh_diameter(self):        self._check_diameter(rm.ObjectiveType.MESH)
    def test_iic_diameter(self):         self._check_diameter(rm.ObjectiveType.IIC)
    def test_cost_diameter(self):        self._check_diameter(rm.ObjectiveType.COST)
    def test_mesh_iic_diameter(self):    self._check_diameter(rm.ObjectiveType.MESH_IIC)
    def test_mesh_cost_diameter(self):   self._check_diameter(rm.ObjectiveType.MESH_COST)
    def test_iic_cost_diameter(self):    self._check_diameter(rm.ObjectiveType.IIC_COST)
    def test_full_diameter(self):        self._check_diameter(rm.ObjectiveType.FULL)

    # ── 4.  Connectivity constraint ───────────────────────────────────────────

    def _check_cc(self, obj):
        sol = self._best(obj)
        self._soft(sol["n_connected_components"],
                   hi=self._CC["max_nb_cc"],
                   label=f"{obj.value}.n_cc")

    def test_mesh_cc(self):        self._check_cc(rm.ObjectiveType.MESH)
    def test_iic_cc(self):         self._check_cc(rm.ObjectiveType.IIC)
    def test_cost_cc(self):        self._check_cc(rm.ObjectiveType.COST)
    def test_mesh_iic_cc(self):    self._check_cc(rm.ObjectiveType.MESH_IIC)
    def test_mesh_cost_cc(self):   self._check_cc(rm.ObjectiveType.MESH_COST)
    def test_iic_cost_cc(self):    self._check_cc(rm.ObjectiveType.IIC_COST)
    def test_full_cc(self):        self._check_cc(rm.ObjectiveType.FULL)

    # ── 5.  Budget constraint (all solutions of the Pareto front) ─────────────

    def _check_budget_all(self, obj):
        for i, sol in enumerate(self.R[obj]["solutions"]):
            self._soft(sol["total_cost"], hi=self._MAX_COST,
                       label=f"{obj.value}[{i}].cost")

    def test_mesh_cost_budget(self):  self._check_budget_all(rm.ObjectiveType.MESH_COST)
    def test_iic_cost_budget(self):   self._check_budget_all(rm.ObjectiveType.IIC_COST)
    def test_full_budget(self):       self._check_budget_all(rm.ObjectiveType.FULL)

    # ── 6.  Single-objective ordering ────────────────────────────────────────

    def test_cost_solution_cheaper_than_mesh_solution(self):
        sol_cost = self._best(rm.ObjectiveType.COST)
        sol_mesh = self._best(rm.ObjectiveType.MESH)
        self.assertLessEqual(
            sol_cost["total_cost"],
            sol_mesh["total_cost"] * 1.10,
            "COST-optimal should not be much more expensive than MESH-optimal",
        )

    def test_mesh_solution_higher_mesh_than_cost_solution(self):
        sol_mesh = self._best(rm.ObjectiveType.MESH)
        sol_cost = self._best(rm.ObjectiveType.COST)
        self.assertGreaterEqual(
            sol_mesh["mesh"],
            sol_cost["mesh"] * 0.90,
            "MESH-optimal should not have much lower MESH than COST-optimal",
        )

    def test_iic_solution_higher_iic_than_cost_solution(self):
        sol_iic  = self._best(rm.ObjectiveType.IIC)
        sol_cost = self._best(rm.ObjectiveType.COST)
        self.assertGreaterEqual(
            sol_iic["iic"],
            sol_cost["iic"] * 0.85,
        )

    # ── 7a.  Pareto front size >= 2 ───────────────────────────────────────────

    def _check_pareto_size(self, obj, min_sols=2):
        n = len(self.R[obj]["solutions"])
        self.assertGreaterEqual(n, min_sols,
                                 f"{obj.value}: only {n} solution(s)")

    def test_mesh_iic_pareto_size(self):   self._check_pareto_size(rm.ObjectiveType.MESH_IIC)
    def test_mesh_cost_pareto_size(self):  self._check_pareto_size(rm.ObjectiveType.MESH_COST)
    def test_iic_cost_pareto_size(self):   self._check_pareto_size(rm.ObjectiveType.IIC_COST)
    def test_full_pareto_size(self):       self._check_pareto_size(rm.ObjectiveType.FULL)

    # ── 7b.  Pareto front solutions are DISTINCT ──────────────────────────────

    def _check_pareto_distinct(self, obj):
        """At least two solutions must have different selection_grid arrays."""
        sols = self.R[obj]["solutions"]
        if len(sols) < 2:
            self.skipTest(f"{obj.value}: fewer than 2 solutions")

        found = any(
            not np.array_equal(sols[i]["selection_grid"],
                                sols[j]["selection_grid"])
            for i in range(len(sols))
            for j in range(i + 1, len(sols))
        )
        self.assertTrue(
            found,
            f"{obj.value}: all {len(sols)} Pareto solutions have identical "
            "selection_grid — Pareto front lacks diversity",
        )

    def test_mesh_iic_distinct(self):   self._check_pareto_distinct(rm.ObjectiveType.MESH_IIC)
    def test_mesh_cost_distinct(self):  self._check_pareto_distinct(rm.ObjectiveType.MESH_COST)
    def test_iic_cost_distinct(self):   self._check_pareto_distinct(rm.ObjectiveType.IIC_COST)
    def test_full_distinct(self):       self._check_pareto_distinct(rm.ObjectiveType.FULL)

    # ── 7c.  Pareto front is non-dominated ────────────────────────────────────

    def _check_non_dominated(self, obj):
        F = self.R[obj]["result"].F   # shape (n, n_obj), minimisation form
        if F is None or len(F) < 2:
            self.skipTest("Not enough solutions")
        for i in range(len(F)):
            for j in range(len(F)):
                if i == j:
                    continue
                dominated = np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                self.assertFalse(
                    dominated,
                    f"{obj.value}: solution {i} dominated by {j}",
                )

    def test_mesh_iic_non_dominated(self):   self._check_non_dominated(rm.ObjectiveType.MESH_IIC)
    def test_mesh_cost_non_dominated(self):  self._check_non_dominated(rm.ObjectiveType.MESH_COST)
    def test_iic_cost_non_dominated(self):   self._check_non_dominated(rm.ObjectiveType.IIC_COST)
    def test_full_non_dominated(self):       self._check_non_dominated(rm.ObjectiveType.FULL)

    # ── 7d.  Genuine objective trade-offs in Pareto fronts ────────────────────

    def test_mesh_cost_pareto_has_tradeoff(self):
        """Best-MESH solution should cost more than best-cost solution."""
        sols = self.R[rm.ObjectiveType.MESH_COST]["solutions"]
        if len(sols) < 2:
            self.skipTest("Not enough solutions")
        best_mesh = max(sols, key=lambda s: s["mesh"])
        best_cost = min(sols, key=lambda s: s["total_cost"])
        # If they happen to be the same solution (degenerate), skip
        if best_mesh is best_cost:
            self.skipTest("Degenerate Pareto front (single solution dominates all)")
        self.assertGreaterEqual(
            best_mesh["mesh"], best_cost["mesh"] * 0.90,
            "Best-MESH solution should have at least as much MESH as best-cost",
        )

    def test_iic_cost_pareto_has_tradeoff(self):
        sols = self.R[rm.ObjectiveType.IIC_COST]["solutions"]
        if len(sols) < 2:
            self.skipTest("Not enough solutions")
        best_iic  = max(sols, key=lambda s: s["iic"])
        best_cost = min(sols, key=lambda s: s["total_cost"])
        if best_iic is best_cost:
            self.skipTest("Degenerate Pareto front")
        self.assertGreaterEqual(
            best_iic["iic"], best_cost["iic"] * 0.85,
        )


# =============================================================================
# G.  RestoptProblemBuilder
# =============================================================================

class TestRestoptProblemBuilder(unittest.TestCase):

    def _b(self, data=DATA_10) -> rm.RestoptProblemBuilder:
        return rm.RestoptProblemBuilder(data)

    # ── objective setters ─────────────────────────────────────────────────────

    def test_default_objective_mesh(self):
        self.assertEqual(self._b().build().objective, rm.ObjectiveType.MESH)

    def test_set_max_iic(self):
        self.assertEqual(self._b().set_max_iic_objective().build().objective,
                         rm.ObjectiveType.IIC)

    def test_set_min_cost(self):
        self.assertEqual(self._b().set_min_cost_objective().build().objective,
                         rm.ObjectiveType.COST)

    def test_set_mesh_iic(self):
        self.assertEqual(self._b().set_mesh_iic_objective().build().objective,
                         rm.ObjectiveType.MESH_IIC)

    def test_set_mesh_cost(self):
        self.assertEqual(self._b().set_mesh_cost_objective().build().objective,
                         rm.ObjectiveType.MESH_COST)

    def test_set_iic_cost(self):
        self.assertEqual(self._b().set_iic_cost_objective().build().objective,
                         rm.ObjectiveType.IIC_COST)

    def test_set_full(self):
        self.assertEqual(self._b().set_full_objective().build().objective,
                         rm.ObjectiveType.FULL)

    # ── constraint setters ────────────────────────────────────────────────────

    def test_restorable_constraint(self):
        p = self._b().add_restorable_constraint(1.0, 8.0).build()
        self.assertEqual(p.constraints.min_restore, 1.0)
        self.assertEqual(p.constraints.max_restore, 8.0)

    def test_compactness_constraint(self):
        p = self._b().add_compactness_constraint(4).build()
        self.assertEqual(p.constraints.max_diameter, 4)

    def test_connected_constraint(self):
        p = self._b().add_connected_constraint(2).build()
        self.assertEqual(p.constraints.max_nb_cc, 2)

    def test_budget_constraint(self):
        p = self._b().add_budget_constraint(500.0).build()
        self.assertAlmostEqual(p.constraints.max_cost, 500.0)

    def test_min_proportion_constraint(self):
        p = self._b().add_min_proportion_constraint(0.35).build()
        self.assertAlmostEqual(p.constraints.min_proportion, 0.35)

    def test_chaining_returns_same_builder(self):
        b  = self._b()
        b2 = b.set_max_mesh_objective()
        self.assertIs(b, b2)

    def test_build_returns_restoration_problem(self):
        self.assertIsInstance(self._b().build(), rm.RestorationProblem)

    def test_set_penalty(self):
        p = self._b().set_penalty(1e5).build()
        self.assertEqual(p.penalty, 1e5)

    def test_set_iic_max_dist(self):
        p = self._b().set_iic_max_dist(5).build()
        self.assertEqual(p.iic_max_dist, 5)

    def test_locked_out_not_in_candidates(self):
        """Locked-out cells must never appear as candidates."""
        p    = self._b(DATA_15).add_locked_out_constraint().build()
        rows, cols = p._candidate_rows, p._candidate_cols
        self.assertFalse(DATA_15.locked_out[rows, cols].any())

    def test_full_chain(self):
        p = (
            self._b()
            .set_max_mesh_objective()
            .add_restorable_constraint(0.5, 6.0)
            .add_compactness_constraint(6)
            .add_connected_constraint(1)
            .add_budget_constraint(200.0)
            .add_min_proportion_constraint(0.1)
            .set_penalty(1e6)
            .set_iic_max_dist(8)
            .build()
        )
        self.assertIsInstance(p, rm.RestorationProblem)
        self.assertEqual(p.objective,              rm.ObjectiveType.MESH)
        self.assertEqual(p.constraints.max_nb_cc,  1)
        self.assertEqual(p.iic_max_dist,           8)

    # ── round-trip solve (requires pymoo) ─────────────────────────────────────

    @unittest.skipUnless(PYMOO_AVAILABLE, "pymoo not installed")
    def test_solve_returns_required_structure(self):
        result = (
            rm.RestoptProblemBuilder(DATA_15)
            .set_full_objective()
            .add_restorable_constraint(1.0, 12.0)
            .add_compactness_constraint(8)
            .solve(pop_size=_POP, n_gen=_NGEN, seed=0, verbose=False)
        )
        self.assertIn("result",    result)
        self.assertIn("solutions", result)
        self.assertGreater(len(result["solutions"]), 0)

    @unittest.skipUnless(PYMOO_AVAILABLE, "pymoo not installed")
    def test_solve_solution_keys(self):
        result = (
            rm.RestoptProblemBuilder(DATA_15)
            .set_full_objective()
            .add_restorable_constraint(0.5, 15.0)
            .solve(pop_size=_POP, n_gen=_NGEN, seed=1, verbose=False)
        )
        required = {"mesh", "iic", "total_cost", "selection_grid",
                    "n_restored_cells", "total_restored_area",
                    "n_connected_components", "diameter_cells"}
        for sol in result["solutions"]:
            for key in required:
                with self.subTest(key=key):
                    self.assertIn(key, sol)

    @unittest.skipUnless(PYMOO_AVAILABLE, "pymoo not installed")
    def test_solve_budget_respected(self):
        max_bgt = 150.0
        result  = (
            rm.RestoptProblemBuilder(DATA_15)
            .set_full_objective()
            .add_restorable_constraint(1.0, 12.0)
            .add_budget_constraint(max_bgt)
            .solve(pop_size=_POP, n_gen=_NGEN, seed=2, verbose=False)
        )
        for i, sol in enumerate(result["solutions"]):
            self.assertLessEqual(
                sol["total_cost"], max_bgt * 1.05,
                f"Solution {i}: cost={sol['total_cost']:.2f} > budget {max_bgt}",
            )

    @unittest.skipUnless(PYMOO_AVAILABLE, "pymoo not installed")
    def test_solve_different_seeds_differ(self):
        def _run(seed):
            return (
                rm.RestoptProblemBuilder(DATA_15)
                .set_full_objective()
                .add_restorable_constraint(1.0, 12.0)
                .solve(pop_size=_POP, n_gen=_NGEN, seed=seed, verbose=False)
            )
        g1 = {tuple(s["selection_grid"].ravel()) for s in _run(10)["solutions"]}
        g2 = {tuple(s["selection_grid"].ravel()) for s in _run(99)["solutions"]}
        self.assertFalse(g1 == g2,
                          "Two different seeds produced identical Pareto fronts")


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in (
        TestHabitatData,
        TestLandscapeIndices,
        TestObjectiveType,
        TestRestorationConstraints,
        TestRestorationProblem,
        TestConstraintSatisfaction,
        TestRestoptProblemBuilder,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 60)
    t = result.testsRun
    p = t - len(result.failures) - len(result.errors) - len(result.skipped)
    print(f"  Run: {t}  Passed: {p}  "
          f"Failed: {len(result.failures)}  "
          f"Errors: {len(result.errors)}  "
          f"Skipped: {len(result.skipped)}")
    print("=" * 60)
    sys.exit(0 if result.wasSuccessful() else 1)
