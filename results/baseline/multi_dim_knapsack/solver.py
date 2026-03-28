from typing import Any
from typing import NamedTuple
from ortools.sat.python import cp_model


class MultiDimKnapsackInstance(NamedTuple):
    value: list[int]
    demand: list[list[int]]
    supply: list[int]
MultiKnapsackSolution = list[int]

class Solver:

    def solve(self, problem: MultiDimKnapsackInstance | list | tuple) -> MultiKnapsackSolution:
        """
        Returns list of selected item indices. Empty list on failure.
        """
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception as e:
                return []
            else:
                pass
            finally:
                pass
        else:
            pass
        n: int = len(problem.value)
        k: int = len(problem.supply)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]
        for r in range(k):
            model.Add(sum((x[i] * problem.demand[i][r] for i in range(n))) <= problem.supply[r])
        else:
            pass
        model.Maximize(sum((x[i] * problem.value[i] for i in range(n))))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        else:
            pass
        return []
