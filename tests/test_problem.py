"""Tests for RestorationProblem — foremost/core.py lines 946-1119."""

import numpy as np
import pytest

from foremost import (
    HabitatData,
    ObjectiveType,
    RestorationConstraints,
    RestorationProblem,
)


@pytest.fixture
def simple_problem(tiny_habitat_data):
    constraints = RestorationConstraints(
        min_restore=0.0,
        max_restore=float("inf"),
        max_diameter=20,
        max_nb_cc=10,
        max_cost=float("inf"),
    )
    return RestorationProblem(
        data=tiny_habitat_data,
        constraints=constraints,
        objective=ObjectiveType.MESH,
    )


class TestRestorationProblemInit:
    def test_no_candidates_raises(self):
        """All-habitat grid → no candidates → ValueError."""
        all_habitat = np.ones((4, 4), dtype=int)
        data = HabitatData(
            habitat=all_habitat,
            restorable=np.zeros((4, 4)),
            accessible=np.zeros((4, 4), dtype=int),
            cost=np.zeros((4, 4)),
        )
        with pytest.raises(ValueError, match="No candidate cells"):
            RestorationProblem(
                data=data,
                constraints=RestorationConstraints(),
                objective=ObjectiveType.MESH,
            )

    def test_candidate_count_matches_mask(self, tiny_habitat_data):
        prob = RestorationProblem(
            data=tiny_habitat_data,
            constraints=RestorationConstraints(),
            objective=ObjectiveType.MESH,
        )
        expected = int(tiny_habitat_data.candidate_mask.sum())
        assert prob.n_candidates == expected

    def test_n_var_matches_n_candidates(self, simple_problem):
        assert simple_problem.n_var == simple_problem.n_candidates

    def test_single_obj_has_one_objective(self, simple_problem):
        assert simple_problem.n_obj == 1

    def test_multi_obj_correct_n_obj(self, tiny_habitat_data):
        constraints = RestorationConstraints()
        for obj, expected_n in [
            (ObjectiveType.MESH_IIC, 2),
            (ObjectiveType.MESH_COST, 2),
            (ObjectiveType.FULL, 3),
        ]:
            prob = RestorationProblem(tiny_habitat_data, constraints, obj)
            assert prob.n_obj == expected_n


class TestRestorationProblemEvaluate:
    def test_evaluate_all_zeros_does_not_crash(self, simple_problem):
        x = np.zeros(simple_problem.n_candidates, dtype=bool)
        out = {}
        simple_problem._evaluate(x, out)
        assert "F" in out
        assert len(out["F"]) == 1

    def test_evaluate_all_ones_does_not_crash(self, simple_problem):
        x = np.ones(simple_problem.n_candidates, dtype=bool)
        out = {}
        simple_problem._evaluate(x, out)
        assert "F" in out

    def test_objective_value_is_finite(self, simple_problem):
        x = np.zeros(simple_problem.n_candidates, dtype=bool)
        x[0] = True
        out = {}
        simple_problem._evaluate(x, out)
        assert np.isfinite(out["F"]).all()


class TestDecodeSolution:
    def test_decode_returns_required_keys(self, simple_problem):
        x = np.zeros(simple_problem.n_candidates, dtype=bool)
        x[0] = True
        result = simple_problem.decode_solution(x)
        required_keys = {
            "n_restored_cells",
            "total_restored_area",
            "total_cost",
            "n_connected_components",
            "diameter_cells",
            "n_patches",
            "mesh",
            "iic",
            "habitat_grid",
            "selection_grid",
        }
        assert required_keys.issubset(result.keys())

    def test_decode_empty_selection_zero_cells(self, simple_problem):
        x = np.zeros(simple_problem.n_candidates, dtype=bool)
        result = simple_problem.decode_solution(x)
        assert result["n_restored_cells"] == 0
        assert result["total_restored_area"] == pytest.approx(0.0)

    def test_decode_nonnegative_metrics(self, simple_problem):
        x = np.zeros(simple_problem.n_candidates, dtype=bool)
        x[:2] = True
        result = simple_problem.decode_solution(x)
        assert result["mesh"] >= 0.0
        assert result["iic"] >= 0.0
        assert result["total_cost"] >= 0.0
        assert result["n_restored_cells"] == 2
