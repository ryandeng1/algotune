from typing import Any, NamedTuple, List
from ortools.sat.python import cp_model

class MultiDimKnapsackInstance(NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]

MultiKnapsackSolution = List[int]

class Solver:
    def solve(self, problem: Any) -> MultiKnapsackSolution:
        """Solve a multi‑dimensional knapsack problem."""
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception:
                return []

        n, k = len(problem.value), len(problem.supply)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add capacity constraints
        for r in range(k):
            demand_r = [problem.demand[i][r] * x[i] for i in range(n)]
            model.Add(sum(demand_r) <= problem.supply[r])

        # Maximise value
        model.Maximize(sum(problem.value[i] * x[i] for i in range(n)))

        solver = cp_model.CpSolver()
        # Optional: use multiple workers to speed up search
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []