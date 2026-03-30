# solver.py
# Optimised set‑cover with conflicts solver using OR‑Tools CP‑Solver.
# The implementation pre‑computes necessary data structures to avoid
# excessive Python‑level loops during model construction.

from __future__ import annotations

from typing import Any, NamedTuple
from ortools.sat.python import cp_model


class Instance(NamedTuple):
    n: int
    sets: list[list[int]]
    conflicts: list[list[int]]


class Solver:
    """
    Fast set cover with conflicts solver based on OR‑Tools CP‑SAT.
    """

    def solve(self, problem: Instance | tuple) -> list[int]:
        """
        Solve the set cover with conflicts problem.

        Parameters
        ----------
        problem : Instance or tuple
            (n, sets, conflicts) where:
            - n is the number of objects,
            - sets is a list of lists of integers, each list describing a set,
            - conflicts is a list of lists of set indices; sets in each list
              conflict with each other.

        Returns
        -------
        list[int]
            Indices of selected sets that form a valid cover. The returned list
            is optimized for minimum cardinality. Raises ValueError if no
            feasible solution exists.
        """
        # Normalise input
        if not isinstance(problem, Instance):
            problem = Instance(*problem)

        n, sets, conflicts = problem
        m = len(sets)

        # Create CP-SAT model
        model = cp_model.CpModel()

        # Boolean variable for each set
        set_vars = [model.NewBoolVar(f"set_{i}") for i in range(m)]

        # Pre‑compute for each object the indices of sets that contain it
        obj_to_sets: list[list[int]] = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                obj_to_sets[obj].append(i)

        # Coverage constraints: each object must be in at least one selected set
        for obj_sets in obj_to_sets:
            # Use Add(sum(...)) directly; owned by CP-SAT, efficient
            model.Add(sum(set_vars[i] for i in obj_sets) >= 1)

        # Conflict constraints: at most one set from each conflict group is selected
        for conf in conflicts:
            model.Add(sum(set_vars[i] for i in conf) <= 1)

        # Objective: minimize number of selected sets
        model.Minimize(sum(set_vars))

        # Solver with a small time limit for quick results (optional: remove/adjust)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # 60 seconds > 0 ensures timeout is considered
        solver.parameters.num_search_workers = 8      # use multiple cores

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Return indices of selected sets
            return [i for i in range(m) if solver.Value(set_vars[i])]
        else:
            raise ValueError("No feasible solution found.")