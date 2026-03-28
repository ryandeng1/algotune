from typing import NamedTuple, List
from ortools.sat.python import cp_model


class MultiDimKnapsackInstance(NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]
MultiKnapsackSolution = List[int]


class Solver:

    def solve(self, problem: MultiDimKnapsackInstance | list | tuple) -> MultiKnapsackSolution:
        """
        Returns list of selected item indices. Empty list on failure.
        """
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception:
                return []

        n = len(problem.value)
        k = len(problem.supply)

        model = cp_model.CpModel()

        # Decision variables
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Resource constraints
        for r in range(k):
            model.Add(
                sum(demand_r_i * x[i]
                    for i, demand_r_i in enumerate(
                        [problem.demand[i][r] for i in range(n)]
                    )
                ) <= problem.supply[r]
            )

        # Objective
        model.Maximize(sum(v * x[i] for i, v in enumerate(problem.value)))

        solver = cp_model.CpSolver()
        # Optional: set a time limit for very large instances
        solver.parameters.max_time_in_seconds = 1.0

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []