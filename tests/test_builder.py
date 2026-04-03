"""Tests for ForemostProblemBuilder — foremost/core.py lines 1785-1885."""

import pytest

from foremost import ForemostProblemBuilder, ObjectiveType, RestorationProblem


class TestBuilderChaining:
    def test_set_objective_returns_self(self, tiny_habitat_data):
        builder = ForemostProblemBuilder(tiny_habitat_data)
        assert builder.set_max_mesh_objective() is builder

    def test_add_constraint_returns_self(self, tiny_habitat_data):
        builder = ForemostProblemBuilder(tiny_habitat_data)
        assert builder.add_compactness_constraint(max_diameter=10) is builder

    def test_full_chain_builds_problem(self, tiny_habitat_data):
        problem = (
            ForemostProblemBuilder(tiny_habitat_data)
            .set_full_objective()
            .add_restorable_constraint(min_restore=0.0, max_restore=100.0)
            .add_compactness_constraint(max_diameter=20)
            .add_connected_constraint(max_nb_cc=5)
            .build()
        )
        assert isinstance(problem, RestorationProblem)
        assert problem.objective == ObjectiveType.FULL

    def test_all_objective_setters(self, tiny_habitat_data):
        mapping = {
            "set_max_mesh_objective": ObjectiveType.MESH,
            "set_max_iic_objective": ObjectiveType.IIC,
            "set_min_cost_objective": ObjectiveType.COST,
            "set_mesh_iic_objective": ObjectiveType.MESH_IIC,
            "set_mesh_cost_objective": ObjectiveType.MESH_COST,
            "set_iic_cost_objective": ObjectiveType.IIC_COST,
            "set_full_objective": ObjectiveType.FULL,
        }
        for method_name, expected_obj in mapping.items():
            b = ForemostProblemBuilder(tiny_habitat_data)
            getattr(b, method_name)()
            prob = b.build()
            assert prob.objective == expected_obj

    def test_budget_constraint_applied(self, tiny_habitat_data):
        prob = (
            ForemostProblemBuilder(tiny_habitat_data)
            .add_budget_constraint(max_cost=5000.0)
            .build()
        )
        assert prob.constraints.max_cost == 5000.0

    def test_penalty_setter(self, tiny_habitat_data):
        prob = ForemostProblemBuilder(tiny_habitat_data).set_penalty(1e4).build()
        assert prob.penalty == pytest.approx(1e4)


class TestBuilderSolve:
    """Minimal solve runs (1 generation, 4 individuals) to verify wiring."""

    @pytest.mark.timeout(30)
    def test_mesh_solve_returns_solutions(self, tiny_habitat_data):
        result = (
            ForemostProblemBuilder(tiny_habitat_data)
            .set_max_mesh_objective()
            .add_restorable_constraint(min_restore=0.0, max_restore=50.0)
            .solve(pop_size=4, n_gen=1, verbose=False, algo="GA")
        )
        assert "solutions" in result
        assert len(result["solutions"]) >= 1

    @pytest.mark.timeout(30)
    def test_nsga2_solve_returns_pareto_front(self, tiny_habitat_data):
        result = (
            ForemostProblemBuilder(tiny_habitat_data)
            .set_mesh_cost_objective()
            .add_restorable_constraint(min_restore=0.0, max_restore=50.0)
            .solve(pop_size=4, n_gen=1, verbose=False, algo="NSGA2")
        )
        assert "solutions" in result
        assert result["algo_name"] == "NSGA2"
