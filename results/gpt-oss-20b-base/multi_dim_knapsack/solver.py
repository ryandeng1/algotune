from typing import NamedTuple, List, Union
from ortools.sat.python import cp_model


class MultiDimKnapsackInstance(NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]


class Solver:
    def solve(self, problem: Union[MultiDimKnapsackInstance, List, tuple]) -> List[int]:
        """
        Returns list of selected item indices. Empty list on failure.
        """
        # Ensure we have a MultiDimKnapsackInstance
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception:
                return []

        n = len(problem.value)
        k = len(problem.supply)

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Add capacity constraints
        for r in range(k):
            coeffs = [problem.demand[i][r] for i in range(n)]
            expr = sum(x[i] * coeffs[i] for i in range(n))
            model.Add(expr <= problem.supply[r])

        # Objective
        model.Maximize(sum(x[i] * problem.value[i] for i in range(n)))

        # Solver with fast settings
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 0  # let OR-Tools decide
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []