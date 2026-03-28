from typing import NamedTuple, List
from ortools.sat.python import cp_model

class MultiDimKnapsackInstance(NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]

class Solver:
    def solve(self, problem: MultiDimKnapsackInstance | List | tuple) -> List[int]:
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception:
                return []

        n = len(problem.value)
        k = len(problem.supply)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        for r in range(k):
            model.Add(
                sum(x[i] * problem.demand[i][r] for i in range(n)) <= problem.supply[r]
            )

        model.Maximize(sum(x[i] * problem.value[i] for i in range(n)))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []