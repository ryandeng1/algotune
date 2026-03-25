#!/usr/bin/env python3
# solver.py

from typing import List, Any

# Import OR-Tools CP-SAT solver (fast ILP solver)
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the set cover problem optimally using OR-Tools CP-SAT solver.

        Parameters
        ----------
        problem : List[List[int]]
            List of subsets; each subset contains integers belonging to the universe.
            Elements are 1‑indexed in the input but internally treated as 0‑indexed.

        Returns
        -------
        List[int]
            1‑indexed list of selected subset indices that form a minimum set cover.
        """
        # Build universe set to map element -> covering sets
        universe = set()
        for subset in problem:
            universe.update(subset)
        if not universe:
            return []

        # Map each element to list of set indices that contain it
        element_to_sets = {e: [] for e in universe}
        for i, subset in enumerate(problem):
            for e in subset:
                element_to_sets[e].append(i)

        # Initialize CP-SAT model
        model = cp_model.CpModel()

        # Boolean variables: x[i] == 1 if set i is selected
        x = [model.NewBoolVar(f"x_{i}") for i in range(len(problem))]

        # Coverage constraints: each element must be covered by at least one selected set
        for e, sets in element_to_sets.items():
            model.Add(sum(x[i] for i in sets) >= 1)

        # Objective: minimize total number of selected sets
        model.Minimize(sum(x))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # safety cap
        solver.parameters.num_search_workers = 0  # let CP-SAT manage workers
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Fallback: return empty list if no solution found
            return []

        # Gather selected indices (1-indexed)
        solution = [i + 1 for i, var in enumerate(x) if solver.Value(var) == 1]
        return solution
