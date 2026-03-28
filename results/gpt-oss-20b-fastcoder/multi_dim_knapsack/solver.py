from ortools.sat.python import cp_model
from typing import NamedTuple, List, Union

class MultiDimKnapsackInstance(NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]

MultiKnapsackSolution = List[int]

class Solver:
    def solve(self, problem: Union[MultiDimKnapsackInstance, List, tuple]) -> MultiKnapsackSolution:
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception:
                return []

        n, k = len(problem.value), len(problem.supply)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Resources constraints
        for r in range(k):
            model.Add(sum(x[i] * demand_i_r for i, demand_i_r in enumerate(row[r] := [d[i][r] for d in problem.demand])) <= problem.supply[r])

        # Objective
        model.Maximize(sum(x[i] * problem.value[i] for i in range(n)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0   # optional timeout

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []