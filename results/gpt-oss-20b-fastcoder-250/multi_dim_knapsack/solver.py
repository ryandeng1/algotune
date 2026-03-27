import sys
from typing import Any, NamedTuple, List
from ortools.sat.python import cp_model


class MultiDimKnapsackInstance(NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]


MultiKnapsackSolution = List[int]


class Solver:
    def solve(
        self,
        problem: MultiDimKnapsackInstance | Any | tuple | list,
    ) -> MultiKnapsackSolution:
        """
        Solve a Multi‑Dimensional Knapsack problem using OR‑Tools CP‑SAT.
        Returns a list of selected item indices (0‑based).  Returns an empty
        list on failure or on invalid input.
        """
        # ------------------------------------------------------------------
        #   Unpack and validate `problem`
        # ------------------------------------------------------------------
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception:   # pragma: no cover
                return []

        n, k = len(problem.value), len(problem.supply)

        # ------------------------------------------------------------------
        #   Build the model
        # ------------------------------------------------------------------
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Constraints: sum_i demand[i][r] * x_i <= supply[r]   for each resource r
        for r in range(k):
            model.Add(
                sum(x[i] * problem.demand[i][r] for i in range(n)) <= problem.supply[r]
            )

        # Objective: maximise total value
        model.Maximize(sum(x[i] * problem.value[i] for i in range(n)))

        # ------------------------------------------------------------------
        #   Solve
        # ------------------------------------------------------------------
        solver = cp_model.CpSolver()
        # give a generous time limit only if the user has not set one
        if not solver.Parameters.HasField("max_time_in_seconds"):
            solver.parameters.max_time_in_seconds = 30.0

        status = solver.Solve(model)

        if status in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
            cp_model.INFEASIBLE,  # CP‑SAT may return INFEASIBLE in rare cases
        ):
            # All items for which the solver put a value of 1
            return [i for i in range(n) if solver.Value(x[i]) == 1]

        return []  # pragma: no cover