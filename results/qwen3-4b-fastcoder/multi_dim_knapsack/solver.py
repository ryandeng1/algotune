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
        demand_list = problem.demand
        supply_list = problem.supply
        value_list = problem.value

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        for r in range(k):
            model.Add(sum(x[i] * demand_list[i][r] for i in range(n)) <= supply_list[r])
        model.Maximize(sum(x[i] * value_list[i] for i in range(n)))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []