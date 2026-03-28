from typing import Any
from typing import NamedTuple
from ortools.sat.python import cp_model


class MultiDimKnapsackInstance(NamedTuple):
    value: list[int]  # item values               (length n)
    demand: list[list[int]]  # demand[i][r] = demand of item i in resource r (n × k)
    supply: list[int]  # resource capacities       (length k)


MultiKnapsackSolution = list[int]


class Solver:
    def solve(
        self,
        problem: MultiDimKnapsackInstance | list | tuple,
    ) -> MultiKnapsackSolution:
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception as e:
                return []

        n = len(problem.value)
        k = len(problem.supply)

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        value_local = problem.value
        demand_local = problem.demand
        supply_local = problem.supply
        x_local = x

        for r in range(k):
            model.Add(sum(x_local[i] * demand_local[i][r] for i in range(n)) <= supply_local[r])
        model.Maximize(sum(x_local[i] * value_local[i] for i in range(n)))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x_local[i])]
        return []