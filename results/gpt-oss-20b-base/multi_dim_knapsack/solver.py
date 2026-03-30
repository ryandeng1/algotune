# solver.py
from typing import Any, NamedTuple, List

from ortools.sat.python import cp_model


class MultiDimKnapsackInstance(NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]


MultiKnapsackSolution = List[int]


class Solver:
    """
    Very lightweight solver for the multidimensional knapsack.
    It builds an OR‑Tools CP‑SAT model once per solve call and
    immediately asks the solver.  All most‑critical loops are written
    in plain Python with the least possible overhead (no unnecessary
    `else`/`pass` blocks, no intermediate tuples, all list comprehensions
    are computed lazily via generator expressions where they are
    expensive to materialise).
    """

    def solve(self, problem: Any) -> MultiKnapsackSolution:
        # Normalise input
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)  # type: ignore
            except Exception:
                return []

        n = len(problem.value)
        k = len(problem.supply)

        model = cp_model.CpModel()

        # Decision variables
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Capacity constraints
        for r in range(k):
            # Sum of weighted items ≤ capacity
            model.Add(
                sum(x[i] * problem.demand[i][r] for i in range(n))
                <= problem.supply[r]
            )

        # Objective: maximise total value
        model.Maximize(sum(x[i] * problem.value[i] for i in range(n)))

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Return indices of chosen items
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        return []