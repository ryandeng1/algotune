# solver.py
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

from ortools.sat.python import cp_model


class MultiDimKnapsackInstance(tuple):
    """
    A lightweight immutable container for a multi‑dimensional knapsack problem.
    The tuple is interpreted as (value, demand, supply) where:
    - value : list[int] of length n (the item worth)
    - demand : list[list[int]] of size n×k (the resource consumption of each item in each dimension)
    - supply : list[int] of length k (the resource limits)
    """

    @property
    def value(self) -> List[int]:
        return self[0]

    @property
    def demand(self) -> List[List[int]]:
        return self[1]

    @property
    def supply(self) -> List[int]:
        return self[2]


MultiKnapsackSolution = List[int]  # selected item indices


class Solver:
    """
    Implements a fast OR‑Tools CP‑SAT solver for the multi‑dimensional knapsack.
    """

    # Pre‑initialising a solver instance reduces per‑call construction overhead.
    _model: cp_model.CpModel = cp_model.CpModel()
    _solver: cp_model.CpSolver = cp_model.CpSolver()

    def __init__(self) -> None:
        # Fix search parameters for deterministic and fast runs
        sol: cp_model.CpSolver = self._solver
        sol.parameters.log_search_progress = False
        sol.parameters.search_branching = cp_model.CpSolverParameters.BRANCHING_RANDOM
        sol.parameters.cp_model_presolve = True
        sol.parameters.num_search_workers = 4  # use all available cores

    @staticmethod
    def _wrap_instance(problem: Union[MultiDimKnapsackInstance, Sequence]) -> MultiDimKnapsackInstance:
        """Coerce the input into a MultiDimKnapsackInstance."""
        if isinstance(problem, MultiDimKnapsackInstance):
            return problem
        try:
            return MultiDimKnapsackInstance(problem[0], problem[1], problem[2])
        except Exception:  # pragma: no cover
            return MultiDimKnapsackInstance([], [], [])

    def solve(self, problem: Union[MultiDimKnapsackInstance, Sequence]) -> MultiKnapsackSolution:
        """
        Return a list of selected item indices that maximise the total value
        while respecting all dimensional supply constraints.
        If the instance is malformed or no solution exists, an empty list is returned.
        """
        problem = self._wrap_instance(problem)
        if not problem.value or not problem.supply:  # trivial or empty
            return []

        n = len(problem.value)
        k = len(problem.supply)

        # Re‑use a fresh model for each call but keep the solver object
        model = self._model
        model.Clear()  # reset previous data

        xs = [model.NewBoolVar(f"x[{i}]") for i in range(n)]

        # Add capacity constraints
        demand = problem.demand
        for r in range(k):
            # Build the sum as a linear expression for the solver
            expr = sum(xs[i] * demand[i][r] for i in range(n))
            model.Add(expr <= problem.supply[r])

        # Objective
        model.Maximize(sum(xs[i] * problem.value[i] for i in range(n)))

        # Solve
        status = self._solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if self._solver.Value(xs[i])]
        return []