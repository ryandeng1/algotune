# solver.py
import typing
from typing import Any, List, Tuple, Iterable
from ortools.sat.python import cp_model

# Define a simple data structure for the problem
class MultiDimKnapsackInstance(typing.NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]


class Solver:
    def solve(self, problem: Tuple[Any, ...] | List[Any] | MultiDimKnapsackInstance, **kwargs) -> List[int]:
        """
        Solves a multi‑dimensional knapsack problem.

        Parameters
        ----------
        problem : tuple | list | MultiDimKnapsackInstance
            A tuple (value, demand, supply) or a prepared instance.

        Returns
        -------
        List[int]
            Indices of the selected items.
        """
        # Normalise input
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                value, demand, supply = problem
                problem = MultiDimKnapsackInstance(list(value), [list(r) for r in demand], list(supply))
            except Exception:
                return []

        n = len(problem.value)
        k = len(problem.supply)

        # Build CP‑satisfiability model
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add resource constraints
        for r in range(k):
            model.Add(sum(x[i] * problem.demand[i][r] for i in range(n)) <= problem.supply[r])

        # Maximise the total value
        model.Maximize(sum(x[i] * problem.value[i] for i in range(n)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = kwargs.get("time_limit", 30.0)
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []
