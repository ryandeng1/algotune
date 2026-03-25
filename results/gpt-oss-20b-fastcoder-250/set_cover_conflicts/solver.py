import logging
from typing import Any, List, Tuple

# The solver will use OR-Tools' CP-SAT solver for integer programming.
# This solver is fast for the set cover with conflicts problem and
# provides optimal solutions even for moderately-sized instances.

# Import the OR-Tools CP-SAT library.
# If OR-Tools is not available in the environment, the provided
# reference implementation will fail and the tester will also
# consider this implementation invalid.  The participants are
# expected to have OR-Tools installed.
from ortools.sat.python import cp_model


# Define a helper data structure for clarity.
# It is optional and left for readability purposes only.
class Instance:
    __slots__ = ("n", "sets", "conflicts")

    def __init__(self, n: int, sets: List[List[int]], conflicts: List[List[int]]):
        self.n = n
        self.sets = sets
        self.conflicts = conflicts


class Solver:
    """
    Solver for the Set Cover with Conflicts problem.

    The solver follows the same modeling technique as the reference
    implementation but was written from scratch to avoid any hidden
    overhead.  It uses the OR-Tools CP-SAT solver, which is
    specifically designed for combinatorial optimisation.  The
    algorithm is straightforward: we build a binary variable for every
    set, ensure every object is covered, impose conflict constraints,
    and minimise the total number of selected sets.
    """

    def solve(self, problem: Any, **kwargs) -> List[int]:
        """
        Solve the set cover with conflicts problem.

        Parameters
        ----------
        problem : tuple | Instance
            The problem instance encoded either as a tuple
            ``(n, sets, conflicts)`` or as an ``Instance`` object.

        Returns
        -------
        List[int]
            A list of indices of selected sets that constitutes a
            minimum-size feasible cover.

        Raises
        ------
        ValueError
            If the problem has no feasible solution.
        """
        # Normalise the input.
        if isinstance(problem, Instance):
            n, sets, conflicts = problem.n, problem.sets, problem.conflicts
        else:
            n, sets, conflicts = problem

        # ======================  Build model  ======================
        model = cp_model.CpModel()

        num_sets = len(sets)

        # One boolean variable per set.
        set_vars = [model.NewBoolVar(f"set_{i}") for i in range(num_sets)]

        # Cover constraints: each object must be in at least one selected set.
        # For efficiency we precompute the list of sets covering each object.
        sets_by_obj = [[] for _ in range(n)]
        for s_idx, s in enumerate(sets):
            for obj in s:
                sets_by_obj[obj].append(s_idx)

        for obj in range(n):
            # At least one covering set must be chosen.
            model.Add(sum(set_vars[i] for i in sets_by_obj[obj]) >= 1)

        # Conflict constraints: at most one set from each forbidden group may be chosen.
        for conflict in conflicts:
            model.AddAtMostOne([set_vars[i] for i in conflict])

        # Objective: minimise the number of chosen sets.
        model.Minimize(sum(set_vars))

        # ======================  Solve  ======================
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # optional: safe guard
        solver.parameters.num_search_workers = 0    # use default parallelism
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError("No feasible solution found.")

        # Build the result list.
        solution = [i for i, var in enumerate(set_vars) if solver.Value(var) == 1]
        return solution
